import os
import sys
import ctypes
import pathlib
import platform
import numpy as np
from typing import Optional, List, Tuple, Callable

QUANTIZED_FORMAT_NAMES: Tuple[str, str, str, str, str] = (
    'Q4_0',
    'Q4_1',
    'Q4_K',
    'Q5_0',
    'Q5_1',
    'Q5_K',
    'Q8_0'
)

P_FLOAT = ctypes.POINTER(ctypes.c_float)
P_INT = ctypes.POINTER(ctypes.c_int32)


def _native_buffer_pointer(
    data: bytes | bytearray | memoryview,
) -> tuple[ctypes.c_void_p, object, int]:
    """Return a borrowed pointer without copying writable upload buffers.

    Layer uploads are synchronous in the C ABI, so the returned holder only
    needs to stay alive for the duration of the call.  Read-only or
    non-contiguous views are materialized once; bytearrays and contiguous
    writable mmap views are passed directly to ggml.
    """
    if isinstance(data, memoryview):
        if not data.contiguous or data.format not in {"B", "b", "c"}:
            data = data.tobytes()
        else:
            view = data.cast("B") if data.format != "B" else data
            # ``ctypes.from_buffer`` rejects a read-only mmap view.  NumPy
            # still exposes its address without copying, and the retained
            # view/array pair keeps the mmap owner alive through the
            # synchronous native call (or the model's borrowed lifetime).
            try:
                array = np.frombuffer(view, dtype=np.uint8)
                return (
                    ctypes.c_void_p(int(array.ctypes.data)),
                    (view, array),
                    view.nbytes,
                )
            except (TypeError, ValueError):
                pass
        if not data.readonly:
            view = data.cast("B") if data.format != "B" else data
            holder = (ctypes.c_ubyte * view.nbytes).from_buffer(view)
            return ctypes.cast(holder, ctypes.c_void_p), holder, view.nbytes
        else:
            # A read-only slice may start at a non-zero offset inside a bytes
            # object; materialize the exact slice rather than passing the
            # base object's address.
            data = data.tobytes()
    if isinstance(data, bytearray):
        holder = (ctypes.c_ubyte * len(data)).from_buffer(data)
        return ctypes.cast(holder, ctypes.c_void_p), holder, len(data)
    if isinstance(data, bytes):
        holder = ctypes.c_char_p(data)
        return ctypes.cast(holder, ctypes.c_void_p), holder, len(data)
    payload = bytes(data)
    holder = ctypes.c_char_p(payload)
    return ctypes.cast(holder, ctypes.c_void_p), holder, len(payload)

class RWKVContext:

    def __init__(self, ptr: ctypes.pointer) -> None:
        self.ptr: ctypes.pointer = ptr

class RWKVSharedLibrary:
    """
    Python wrapper around rwkv.cpp shared library.
    """

    def __init__(self, shared_library_path: str) -> None:
        """
        Loads the shared library from specified file.
        In case of any error, this method will throw an exception.

        Parameters
        ----------
        shared_library_path : str
            Path to rwkv.cpp shared library. On Windows, it would look like 'rwkv.dll'. On UNIX, 'rwkv.so'.
        """
        #  When Python is greater than 3.8, we need to reprocess the custom dll
        #  according to the documentation to prevent loading failure errors.
        #  https://docs.python.org/3/whatsnew/3.8.html#ctypes
        if platform.system().lower() == 'windows':
            self.library = ctypes.CDLL(shared_library_path, winmode=0)
        else:
            self.library = ctypes.cdll.LoadLibrary(shared_library_path)

        self.library.rwkv_init_from_file.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32]
        self.library.rwkv_init_from_file.restype = ctypes.c_void_p

        self.supports_native_u8_init = hasattr(self.library, "rwkv_init_from_file_ex")
        if self.supports_native_u8_init:
            self.library.rwkv_init_from_file_ex.argtypes = [
                ctypes.c_char_p,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.c_uint32,
            ]
            self.library.rwkv_init_from_file_ex.restype = ctypes.c_void_p

        self.supports_layer_streaming = all(
            hasattr(self.library, name)
            for name in (
                "rwkv_supports_layer_streaming",
                "rwkv_get_layer_state_len",
                "rwkv_layer_begin",
                "rwkv_layer_set_tensor",
                "rwkv_layer_set_tensor_scale_u8_grouped",
                "rwkv_layer_step",
            )
        )
        self.supports_layer_streaming_sequence = bool(
            self.supports_layer_streaming
            and hasattr(self.library, "rwkv_layer_step_sequence")
        )
        self.supports_layer_borrowed_u8 = bool(
            self.supports_layer_streaming
            and hasattr(self.library, "rwkv_layer_set_tensor_scale_u8_grouped_borrowed")
        )
        self.supports_layer_borrowed_dense = bool(
            self.supports_layer_streaming
            and hasattr(self.library, "rwkv_layer_set_tensor_borrowed")
        )
        self.supports_layer_persistent_dense = bool(
            self.supports_layer_borrowed_dense
            and hasattr(self.library, "rwkv_layer_set_tensor_borrowed_persistent")
        )
        self.supports_layer_invalidate_borrowed_cache = bool(
            self.supports_layer_streaming
            and hasattr(self.library, "rwkv_layer_invalidate_borrowed_cache")
        )
        self.supports_layer_persistent_u8 = bool(
            self.supports_layer_borrowed_u8
            and hasattr(
                self.library,
                "rwkv_layer_set_tensor_scale_u8_grouped_borrowed_persistent",
            )
        )
        self.supports_layer_u8_batch = bool(
            self.supports_layer_streaming
            and hasattr(self.library, "rwkv_layer_set_tensor_scale_u8_grouped_batch")
        )
        self.supports_layer_ready = bool(
            self.supports_layer_streaming
            and hasattr(self.library, "rwkv_layer_is_ready")
        )
        self.supports_layer_cached_step = bool(
            self.supports_layer_ready
            and hasattr(self.library, "rwkv_layer_step_cached")
            and hasattr(self.library, "rwkv_layer_cache_ready")
        )
        if self.supports_layer_streaming:
            self.library.rwkv_supports_layer_streaming.argtypes = [ctypes.c_void_p]
            self.library.rwkv_supports_layer_streaming.restype = ctypes.c_bool
            self.library.rwkv_get_layer_state_len.argtypes = [ctypes.c_void_p]
            self.library.rwkv_get_layer_state_len.restype = ctypes.c_size_t
            self.library.rwkv_layer_begin.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            self.library.rwkv_layer_begin.restype = ctypes.c_bool
            if self.supports_layer_invalidate_borrowed_cache:
                self.library.rwkv_layer_invalidate_borrowed_cache.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_uint32,
                ]
                self.library.rwkv_layer_invalidate_borrowed_cache.restype = ctypes.c_bool
            if self.supports_layer_ready:
                self.library.rwkv_layer_is_ready.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_uint32,
                ]
                self.library.rwkv_layer_is_ready.restype = ctypes.c_bool
            if self.supports_layer_cached_step:
                self.library.rwkv_layer_step_cached.argtypes = [
                    ctypes.c_void_p,
                    P_FLOAT,
                    P_FLOAT,
                    P_FLOAT,
                    P_FLOAT,
                    P_FLOAT,
                    P_FLOAT,
                ]
                self.library.rwkv_layer_step_cached.restype = ctypes.c_bool
                self.library.rwkv_layer_cache_ready.argtypes = [ctypes.c_void_p]
                self.library.rwkv_layer_cache_ready.restype = ctypes.c_bool
            self.library.rwkv_layer_set_tensor.argtypes = [
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
            ]
            self.library.rwkv_layer_set_tensor.restype = ctypes.c_bool
            if self.supports_layer_borrowed_dense:
                self.library.rwkv_layer_set_tensor_borrowed.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_char_p,
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                ]
                self.library.rwkv_layer_set_tensor_borrowed.restype = ctypes.c_bool
            if self.supports_layer_persistent_dense:
                self.library.rwkv_layer_set_tensor_borrowed_persistent.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_char_p,
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                ]
                self.library.rwkv_layer_set_tensor_borrowed_persistent.restype = ctypes.c_bool
            self.library.rwkv_layer_set_tensor_scale_u8_grouped.argtypes = [
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
            ]
            self.library.rwkv_layer_set_tensor_scale_u8_grouped.restype = ctypes.c_bool
            if self.supports_layer_borrowed_u8:
                self.library.rwkv_layer_set_tensor_scale_u8_grouped_borrowed.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_char_p,
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                ]
                self.library.rwkv_layer_set_tensor_scale_u8_grouped_borrowed.restype = ctypes.c_bool
            if self.supports_layer_persistent_u8:
                self.library.rwkv_layer_set_tensor_scale_u8_grouped_borrowed_persistent.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_char_p,
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                ]
                self.library.rwkv_layer_set_tensor_scale_u8_grouped_borrowed_persistent.restype = ctypes.c_bool
            if self.supports_layer_u8_batch:
                self.library.rwkv_layer_set_tensor_scale_u8_grouped_batch.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_char_p),
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.POINTER(ctypes.c_size_t),
                    ctypes.c_size_t,
                    ctypes.c_bool,
                ]
                self.library.rwkv_layer_set_tensor_scale_u8_grouped_batch.restype = ctypes.c_bool
            self.library.rwkv_layer_step.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint32,
                P_FLOAT,
                P_FLOAT,
                P_FLOAT,
                P_FLOAT,
                P_FLOAT,
                P_FLOAT,
            ]
            self.library.rwkv_layer_step.restype = ctypes.c_bool
            if self.supports_layer_streaming_sequence:
                self.library.rwkv_layer_step_sequence.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_uint32,
                    P_FLOAT,
                    P_FLOAT,
                    ctypes.c_size_t,
                    P_FLOAT,
                    P_FLOAT,
                    P_FLOAT,
                    P_FLOAT,
                ]
                self.library.rwkv_layer_step_sequence.restype = ctypes.c_bool

            self.supports_layer_cache = all(
                hasattr(self.library, name)
                for name in (
                    "rwkv_set_layer_cache_bytes",
                    "rwkv_get_layer_cache_bytes",
                    "rwkv_get_layer_cache_used_bytes",
                    "rwkv_get_layer_cache_hits",
                    "rwkv_get_layer_cache_misses",
                    "rwkv_get_layer_cache_evictions",
                )
            )
            if self.supports_layer_cache:
                self.library.rwkv_set_layer_cache_bytes.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                ]
                self.library.rwkv_set_layer_cache_bytes.restype = ctypes.c_bool
                for name in (
                    "rwkv_get_layer_cache_bytes",
                    "rwkv_get_layer_cache_used_bytes",
                ):
                    getattr(self.library, name).argtypes = [ctypes.c_void_p]
                    getattr(self.library, name).restype = ctypes.c_size_t
                for name in (
                    "rwkv_get_layer_cache_hits",
                    "rwkv_get_layer_cache_misses",
                    "rwkv_get_layer_cache_evictions",
                ):
                    getattr(self.library, name).argtypes = [ctypes.c_void_p]
                    getattr(self.library, name).restype = ctypes.c_uint64
        else:
            self.supports_layer_cache = False

        self.library.rwkv_eval.argtypes = [
            ctypes.c_void_p, # ctx
            ctypes.c_int32, # token
            P_FLOAT, # state_in
            P_FLOAT, # state_out
            P_FLOAT  # logits_out
        ]
        self.library.rwkv_eval.restype = ctypes.c_bool

        self.library.rwkv_eval_sequence.argtypes = [
            ctypes.c_void_p, # ctx
            P_INT, # tokens
            ctypes.c_size_t, # token count
            P_FLOAT, # state_in
            P_FLOAT, # state_out
            P_FLOAT  # logits_out
        ]
        self.library.rwkv_eval_sequence.restype = ctypes.c_bool

        self.library.rwkv_eval_sequence_in_chunks.argtypes = [
            ctypes.c_void_p, # ctx
            P_INT, # tokens
            ctypes.c_size_t, # token count
            ctypes.c_size_t, # chunk size
            P_FLOAT, # state_in
            P_FLOAT, # state_out
            P_FLOAT  # logits_out
        ]
        self.library.rwkv_eval_sequence_in_chunks.restype = ctypes.c_bool

        self.library.rwkv_get_n_vocab.argtypes = [ctypes.c_void_p]
        self.library.rwkv_get_n_vocab.restype = ctypes.c_size_t

        self.library.rwkv_get_n_embed.argtypes = [ctypes.c_void_p]
        self.library.rwkv_get_n_embed.restype = ctypes.c_size_t

        self.library.rwkv_get_n_layer.argtypes = [ctypes.c_void_p]
        self.library.rwkv_get_n_layer.restype = ctypes.c_size_t

        self.supports_model_state_metadata = all(
            hasattr(self.library, name)
            for name in (
                "rwkv_get_arch_version_major",
                "rwkv_get_head_count",
                "rwkv_get_head_size",
            )
        )
        if self.supports_model_state_metadata:
            self.library.rwkv_get_arch_version_major.argtypes = [ctypes.c_void_p]
            self.library.rwkv_get_arch_version_major.restype = ctypes.c_uint32
            self.library.rwkv_get_head_count.argtypes = [ctypes.c_void_p]
            self.library.rwkv_get_head_count.restype = ctypes.c_int64
            self.library.rwkv_get_head_size.argtypes = [ctypes.c_void_p]
            self.library.rwkv_get_head_size.restype = ctypes.c_int64

        self.library.rwkv_get_state_buffer_element_count.argtypes = [ctypes.c_void_p]
        self.library.rwkv_get_state_buffer_element_count.restype = ctypes.c_uint32

        self.library.rwkv_get_logits_buffer_element_count.argtypes = [ctypes.c_void_p]
        self.library.rwkv_get_logits_buffer_element_count.restype = ctypes.c_uint32

        self.library.rwkv_free.argtypes = [ctypes.c_void_p]
        self.library.rwkv_free.restype = None

        self.library.rwkv_get_last_error.argtypes = [ctypes.c_void_p]
        self.library.rwkv_get_last_error.restype = ctypes.c_int

        self.supports_tensor_upload = all(
            hasattr(self.library, name)
            for name in ("rwkv_set_tensor", "rwkv_get_tensor_nbytes")
        )
        if self.supports_tensor_upload:
            self.library.rwkv_set_tensor.argtypes = [
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
            ]
            self.library.rwkv_set_tensor.restype = ctypes.c_bool

            self.library.rwkv_get_tensor_nbytes.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            self.library.rwkv_get_tensor_nbytes.restype = ctypes.c_size_t

        self.supports_tensor_type = hasattr(self.library, "rwkv_get_tensor_type")
        if self.supports_tensor_type:
            self.library.rwkv_get_tensor_type.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            self.library.rwkv_get_tensor_type.restype = ctypes.c_int32

        self.supports_native_u8_upload = hasattr(
            self.library, "rwkv_set_tensor_scale_u8_grouped"
        )
        if self.supports_native_u8_upload:
            self.library.rwkv_set_tensor_scale_u8_grouped.argtypes = [
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
            ]
            self.library.rwkv_set_tensor_scale_u8_grouped.restype = ctypes.c_bool

        self.supports_tensor_slot_upload = all(
            hasattr(self.library, name)
            for name in ("rwkv_register_tensor_slot", "rwkv_set_tensor_from_slot")
        )
        self._tensor_slot_buffers = {}
        if self.supports_tensor_slot_upload:
            self.library.rwkv_register_tensor_slot.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_size_t]
            self.library.rwkv_register_tensor_slot.restype = ctypes.c_bool
            self.library.rwkv_set_tensor_from_slot.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint64]
            self.library.rwkv_set_tensor_from_slot.restype = ctypes.c_bool

        self.library.rwkv_free.argtypes = [ctypes.c_void_p]
        self.library.rwkv_free.restype = None

        self.library.rwkv_quantize_model_file.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        self.library.rwkv_quantize_model_file.restype = ctypes.c_bool

        self.library.rwkv_get_system_info_string.argtypes = []
        self.library.rwkv_get_system_info_string.restype = ctypes.c_char_p

        self.nullptr = ctypes.cast(0, ctypes.c_void_p)

    def raise_if_failed(
        self, ok: bool, operation: str, ctx: RWKVContext
    ) -> None:
        """Turn the C boolean error contract into an actionable exception."""
        if ok:
            return
        error_flags = int(self.library.rwkv_get_last_error(ctx.ptr))
        raise ValueError(
            f'{operation} failed (error flags {error_flags}), check stderr'
        )

    def rwkv_init_from_file(
        self,
        model_file_path: str,
        thread_count: int,
        offload_layers: int,
        *,
        native_u8: bool = False,
        native_u8_packed_only: bool = False,
        layer_streaming: bool = False,
    ) -> RWKVContext:
        """
        Loads the model from a file and prepares it for inference.
        Throws an exception in case of any error. Error messages would be printed to stderr.

        Parameters
        ----------
        model_file_path : str
            Path to model file in ggml format.
        thread_count : int
            Count of threads to use, must be positive.
        """

        encoded_path = model_file_path.encode('utf-8')
        if layer_streaming and not self.supports_layer_streaming:
            raise NotImplementedError(
                "rwkv.cpp library lacks the native layer-streaming ABI; rebuild "
                "the vendored backend before selecting an F1-F4 rwkv.cpp tier"
            )
        if (native_u8 or layer_streaming) and self.supports_native_u8_init:
            flags = (1 if native_u8 else 0) | (2 if native_u8_packed_only else 0)
            flags |= 4 if layer_streaming else 0
            ptr = self.library.rwkv_init_from_file_ex(
                encoded_path,
                ctypes.c_uint32(thread_count),
                ctypes.c_uint32(offload_layers),
                ctypes.c_uint32(flags),
            )
        else:
            ptr = self.library.rwkv_init_from_file(
                encoded_path,
                ctypes.c_uint32(thread_count),
                ctypes.c_uint32(offload_layers),
            )

        if ptr is None:
            raise ValueError('rwkv_init_from_file failed, check stderr')

        return RWKVContext(ptr)

    def rwkv_supports_layer_streaming(self, ctx: RWKVContext) -> bool:
        if not self.supports_layer_streaming:
            return False
        return bool(self.library.rwkv_supports_layer_streaming(ctx.ptr))

    def rwkv_get_layer_state_len(self, ctx: RWKVContext) -> int:
        if not self.supports_layer_streaming:
            return 0
        return int(self.library.rwkv_get_layer_state_len(ctx.ptr))

    def rwkv_layer_begin(self, ctx: RWKVContext, layer_id: int) -> None:
        if not self.supports_layer_streaming:
            raise NotImplementedError("native layer-streaming ABI is unavailable")
        ok = self.library.rwkv_layer_begin(
            ctx.ptr, ctypes.c_uint32(int(layer_id))
        )
        self.raise_if_failed(ok, "rwkv_layer_begin", ctx)

    def rwkv_layer_invalidate_borrowed_cache(
        self, ctx: RWKVContext, layer_id: int
    ) -> None:
        if not getattr(self, "supports_layer_invalidate_borrowed_cache", False):
            return
        ok = self.library.rwkv_layer_invalidate_borrowed_cache(
            ctx.ptr, ctypes.c_uint32(int(layer_id))
        )
        self.raise_if_failed(ok, "rwkv_layer_invalidate_borrowed_cache", ctx)

    def rwkv_layer_is_ready(self, ctx: RWKVContext, layer_id: int) -> bool:
        if not getattr(self, "supports_layer_ready", False):
            return False
        return bool(
            self.library.rwkv_layer_is_ready(ctx.ptr, ctypes.c_uint32(int(layer_id)))
        )

    def rwkv_layer_cache_ready(self, ctx: RWKVContext) -> bool:
        if not getattr(self, "supports_layer_cached_step", False):
            return False
        return bool(self.library.rwkv_layer_cache_ready(ctx.ptr))

    def rwkv_layer_step_cached(
        self,
        ctx: RWKVContext,
        activation_in_address: int,
        activation_out_address: int,
        state_in_address: int,
        state_out_address: int,
        v_first_in_address: int | None,
        v_first_out_address: int | None,
    ) -> None:
        if not getattr(self, "supports_layer_cached_step", False):
            raise NotImplementedError("native cached layer-step ABI is unavailable")
        ok = self.library.rwkv_layer_step_cached(
            ctx.ptr,
            ctypes.cast(int(activation_in_address), P_FLOAT),
            ctypes.cast(int(activation_out_address), P_FLOAT),
            ctypes.cast(int(state_in_address), P_FLOAT),
            ctypes.cast(int(state_out_address), P_FLOAT),
            ctypes.cast(
                0 if v_first_in_address is None else int(v_first_in_address), P_FLOAT
            ),
            ctypes.cast(
                0 if v_first_out_address is None else int(v_first_out_address), P_FLOAT
            ),
        )
        self.raise_if_failed(ok, "rwkv_layer_step_cached", ctx)

    def rwkv_layer_set_tensor(
        self, ctx: RWKVContext, name: str, data: bytes | memoryview
    ) -> None:
        if not self.supports_layer_streaming:
            raise NotImplementedError("native layer-streaming ABI is unavailable")
        pointer, _holder, nbytes = _native_buffer_pointer(data)
        ok = self.library.rwkv_layer_set_tensor(
            ctx.ptr,
            name.encode("utf-8"),
            pointer,
            ctypes.c_size_t(nbytes),
        )
        self.raise_if_failed(ok, "rwkv_layer_set_tensor", ctx)

    def rwkv_layer_set_tensor_borrowed(
        self,
        ctx: RWKVContext,
        name: str,
        address: int,
        nbytes: int,
        *,
        persistent: bool = False,
    ) -> None:
        supported = (
            getattr(self, "supports_layer_persistent_dense", False)
            if persistent
            else getattr(self, "supports_layer_borrowed_dense", False)
        )
        if not supported:
            raise NotImplementedError("borrowed dense layer ABI is unavailable")
        symbol = (
            self.library.rwkv_layer_set_tensor_borrowed_persistent
            if persistent
            else self.library.rwkv_layer_set_tensor_borrowed
        )
        ok = symbol(
            ctx.ptr,
            str(name).encode("utf-8"),
            ctypes.c_void_p(int(address)),
            ctypes.c_size_t(int(nbytes)),
        )
        self.raise_if_failed(
            ok,
            "rwkv_layer_set_tensor_borrowed_persistent"
            if persistent
            else "rwkv_layer_set_tensor_borrowed",
            ctx,
        )

    def rwkv_layer_set_tensor_scale_u8_grouped(
        self, ctx: RWKVContext, name: str, data: bytes | memoryview
    ) -> None:
        if not self.supports_layer_streaming:
            raise NotImplementedError("native layer-streaming ABI is unavailable")
        pointer, _holder, nbytes = _native_buffer_pointer(data)
        ok = self.library.rwkv_layer_set_tensor_scale_u8_grouped(
            ctx.ptr,
            name.encode("utf-8"),
            pointer,
            ctypes.c_size_t(nbytes),
        )
        self.raise_if_failed(ok, "rwkv_layer_set_tensor_scale_u8_grouped", ctx)

    def rwkv_layer_set_tensor_scale_u8_grouped_borrowed(
        self, ctx: RWKVContext, name: str, data: bytes | memoryview
    ) -> object:
        if not getattr(self, "supports_layer_borrowed_u8", False):
            raise NotImplementedError("borrowed grouped-U8 ABI is unavailable")
        pointer, holder, nbytes = _native_buffer_pointer(data)
        ok = self.library.rwkv_layer_set_tensor_scale_u8_grouped_borrowed(
            ctx.ptr,
            name.encode("utf-8"),
            pointer,
            ctypes.c_size_t(nbytes),
        )
        self.raise_if_failed(
            ok, "rwkv_layer_set_tensor_scale_u8_grouped_borrowed", ctx
        )
        # ``holder`` is part of the borrowed pointer lifetime for read-only
        # bytes and non-buffer-compatible views.  The model retains it until
        # the next layer_begin.
        return holder

    def rwkv_layer_set_tensor_scale_u8_grouped_borrowed_persistent(
        self, ctx: RWKVContext, name: str, data: bytes | memoryview
    ) -> object:
        if not getattr(self, "supports_layer_persistent_u8", False):
            raise NotImplementedError("persistent borrowed grouped-U8 ABI is unavailable")
        pointer, holder, nbytes = _native_buffer_pointer(data)
        ok = self.library.rwkv_layer_set_tensor_scale_u8_grouped_borrowed_persistent(
            ctx.ptr,
            name.encode("utf-8"),
            pointer,
            ctypes.c_size_t(nbytes),
        )
        self.raise_if_failed(
            ok,
            "rwkv_layer_set_tensor_scale_u8_grouped_borrowed_persistent",
            ctx,
        )
        return holder

    def rwkv_layer_set_tensor_scale_u8_grouped_batch(
        self,
        ctx: RWKVContext,
        items: list[tuple[str, bytes | memoryview]],
        *,
        persistent: bool = False,
    ) -> list[tuple[str, object, bytes | memoryview]]:
        if not getattr(self, "supports_layer_u8_batch", False):
            raise NotImplementedError("grouped-U8 batch ABI is unavailable")
        if not items:
            return []
        names = (ctypes.c_char_p * len(items))()
        pointers = (ctypes.c_void_p * len(items))()
        sizes = (ctypes.c_size_t * len(items))()
        owners: list[tuple[str, object, bytes | memoryview]] = []
        for index, (name, data) in enumerate(items):
            pointer, holder, nbytes = _native_buffer_pointer(data)
            names[index] = str(name).encode("utf-8")
            pointers[index] = pointer
            sizes[index] = nbytes
            owners.append((str(name), holder, data))
        ok = self.library.rwkv_layer_set_tensor_scale_u8_grouped_batch(
            ctx.ptr,
            names,
            pointers,
            sizes,
            ctypes.c_size_t(len(items)),
            ctypes.c_bool(bool(persistent)),
        )
        self.raise_if_failed(ok, "rwkv_layer_set_tensor_scale_u8_grouped_batch", ctx)
        return owners

    def rwkv_layer_step(
        self,
        ctx: RWKVContext,
        layer_id: int,
        activation_in_address: int,
        activation_out_address: int,
        state_in_address: int | None,
        state_out_address: int,
        v_first_in_address: int | None,
        v_first_out_address: int | None,
    ) -> None:
        if not self.supports_layer_streaming:
            raise NotImplementedError("native layer-streaming ABI is unavailable")
        ok = self.library.rwkv_layer_step(
            ctx.ptr,
            ctypes.c_uint32(int(layer_id)),
            ctypes.cast(int(activation_in_address), P_FLOAT),
            ctypes.cast(int(activation_out_address), P_FLOAT),
            ctypes.cast(
                0 if state_in_address is None else int(state_in_address), P_FLOAT
            ),
            ctypes.cast(int(state_out_address), P_FLOAT),
            ctypes.cast(
                0 if v_first_in_address is None else int(v_first_in_address), P_FLOAT
            ),
            ctypes.cast(
                0 if v_first_out_address is None else int(v_first_out_address), P_FLOAT
            ),
        )
        self.raise_if_failed(ok, "rwkv_layer_step", ctx)

    def rwkv_layer_step_sequence(
        self,
        ctx: RWKVContext,
        layer_id: int,
        activation_in_address: int,
        activation_out_address: int,
        n_tokens: int,
        state_in_address: int | None,
        state_out_address: int,
        v_first_in_address: int | None,
        v_first_out_address: int | None,
    ) -> None:
        if not self.supports_layer_streaming_sequence:
            raise NotImplementedError(
                "rwkv.cpp library lacks the native layer sequence ABI"
            )
        ok = self.library.rwkv_layer_step_sequence(
            ctx.ptr,
            ctypes.c_uint32(int(layer_id)),
            ctypes.cast(int(activation_in_address), P_FLOAT),
            ctypes.cast(int(activation_out_address), P_FLOAT),
            ctypes.c_size_t(int(n_tokens)),
            ctypes.cast(
                0 if state_in_address is None else int(state_in_address), P_FLOAT
            ),
            ctypes.cast(int(state_out_address), P_FLOAT),
            ctypes.cast(
                0 if v_first_in_address is None else int(v_first_in_address), P_FLOAT
            ),
            ctypes.cast(
                0 if v_first_out_address is None else int(v_first_out_address), P_FLOAT
            ),
        )
        self.raise_if_failed(ok, "rwkv_layer_step_sequence", ctx)

    def rwkv_set_layer_cache_bytes(self, ctx: RWKVContext, bytes_limit: int) -> None:
        if not getattr(self, "supports_layer_cache", False):
            raise NotImplementedError("native layer cache ABI is unavailable")
        ok = self.library.rwkv_set_layer_cache_bytes(
            ctx.ptr, ctypes.c_size_t(max(0, int(bytes_limit)))
        )
        self.raise_if_failed(ok, "rwkv_set_layer_cache_bytes", ctx)

    def rwkv_get_layer_cache_stats(self, ctx: RWKVContext) -> dict[str, int]:
        if not getattr(self, "supports_layer_cache", False):
            return {
                "limit_bytes": 0,
                "used_bytes": 0,
                "hits": 0,
                "misses": 0,
                "evictions": 0,
            }
        return {
            "limit_bytes": int(self.library.rwkv_get_layer_cache_bytes(ctx.ptr)),
            "used_bytes": int(
                self.library.rwkv_get_layer_cache_used_bytes(ctx.ptr)
            ),
            "hits": int(self.library.rwkv_get_layer_cache_hits(ctx.ptr)),
            "misses": int(self.library.rwkv_get_layer_cache_misses(ctx.ptr)),
            "evictions": int(
                self.library.rwkv_get_layer_cache_evictions(ctx.ptr)
            ),
        }

    def rwkv_eval(
            self,
            ctx: RWKVContext,
            token: int,
            state_in_address: Optional[int],
            state_out_address: int,
            logits_out_address: int
    ) -> None:
        """
        Evaluates the model for a single token.
        Throws an exception in case of any error. Error messages would be printed to stderr.
        Not thread-safe. For parallel inference, call rwkv_clone_context to create one rwkv_context for each thread.

        Parameters
        ----------
        ctx : RWKVContext
            RWKV context obtained from rwkv_init_from_file.
        token : int
            Next token index, in range 0 <= token < n_vocab.
        state_in_address : int
            Address of the first element of a FP32 buffer of size rwkv_get_state_buffer_element_count; or None, if this is a first pass.
        state_out_address : int
            Address of the first element of a FP32 buffer of size rwkv_get_state_buffer_element_count. This buffer will be written to.
        logits_out_address : int
            Address of the first element of a FP32 buffer of size rwkv_get_logits_buffer_element_count. This buffer will be written to.
        """

        ok = self.library.rwkv_eval(
            ctx.ptr,
            ctypes.c_int32(token),
            ctypes.cast(0 if state_in_address is None else state_in_address, P_FLOAT),
            ctypes.cast(state_out_address, P_FLOAT),
            ctypes.cast(logits_out_address, P_FLOAT)
        )
        self.raise_if_failed(ok, 'rwkv_eval', ctx)

    def rwkv_set_tensor(self, ctx: RWKVContext, name: str, data: bytes) -> None:
        """Replace one loaded tensor using its exact ggml-native bytes."""
        if not self.supports_tensor_upload:
            raise NotImplementedError(
                "rwkv.cpp library lacks the SSD tensor-upload ABI; rebuild the "
                "vendored backend"
            )
        payload = ctypes.create_string_buffer(data)
        ok = self.library.rwkv_set_tensor(
            ctx.ptr,
            name.encode('utf-8'),
            ctypes.cast(payload, ctypes.c_void_p),
            ctypes.c_size_t(len(data)),
        )
        if not ok:
            raise ValueError(f'rwkv_set_tensor failed for {name!r}, check stderr')

    def rwkv_set_tensor_from_pointer(
        self, ctx: RWKVContext, name: str, address: int, nbytes: int
    ) -> None:
        """Replace a tensor directly from an already-contiguous native buffer.

        The legacy ``rwkv_set_tensor`` wrapper accepts ``bytes`` and therefore
        allocates a second temporary ctypes buffer before ggml copies the
        tensor.  Streaming callers already own a correctly typed CPU tensor;
        passing its address removes that intermediate Python-side copy while
        retaining the same synchronous C ABI and lifetime requirements.
        """
        if not self.supports_tensor_upload:
            raise NotImplementedError(
                "rwkv.cpp library lacks the SSD tensor-upload ABI; rebuild the "
                "vendored backend"
            )
        if int(address) <= 0 or int(nbytes) <= 0:
            raise ValueError("tensor pointer and byte count must be positive")
        ok = self.library.rwkv_set_tensor(
            ctx.ptr,
            name.encode('utf-8'),
            ctypes.c_void_p(int(address)),
            ctypes.c_size_t(int(nbytes)),
        )
        if not ok:
            raise ValueError(f'rwkv_set_tensor failed for {name!r}, check stderr')

    def rwkv_set_tensor_scale_u8_grouped(
        self, ctx: RWKVContext, name: str, data: bytes | memoryview
    ) -> None:
        """Upload one packed SG8 grouped-U8 tensor to the native CPU graph."""
        if not self.supports_native_u8_upload:
            raise NotImplementedError(
                "rwkv.cpp library lacks the grouped-U8 upload ABI; rebuild the "
                "vendored backend"
            )
        payload = ctypes.create_string_buffer(bytes(data))
        ok = self.library.rwkv_set_tensor_scale_u8_grouped(
            ctx.ptr,
            name.encode("utf-8"),
            ctypes.cast(payload, ctypes.c_void_p),
            ctypes.c_size_t(len(payload) - 1),
        )
        if not ok:
            raise ValueError(
                f'rwkv_set_tensor_scale_u8_grouped failed for {name!r}, check stderr'
            )

    def rwkv_get_tensor_nbytes(self, ctx: RWKVContext, name: str) -> int:
        if not self.supports_tensor_upload:
            return 0
        return int(self.library.rwkv_get_tensor_nbytes(ctx.ptr, name.encode('utf-8')))

    def rwkv_get_tensor_type(self, ctx: RWKVContext, name: str) -> int:
        """Return the loaded tensor's numeric ggml type, or -1 if unavailable."""
        if not self.supports_tensor_type:
            return -1
        return int(self.library.rwkv_get_tensor_type(ctx.ptr, name.encode("utf-8")))

    def rwkv_register_tensor_slot(self, ctx: RWKVContext, slot_id: int, buffer: bytearray) -> None:
        view = (ctypes.c_ubyte * len(buffer)).from_buffer(buffer)
        if not self.library.rwkv_register_tensor_slot(ctx.ptr, slot_id, ctypes.cast(view, ctypes.c_void_p), len(buffer)):
            raise ValueError('rwkv_register_tensor_slot failed')
        self._tensor_slot_buffers[(int(ctx.ptr), slot_id)] = view

    def rwkv_set_tensor_from_slot(self, ctx: RWKVContext, name: str, slot_id: int, offset: int, length: int, generation: int) -> None:
        if not self.library.rwkv_set_tensor_from_slot(ctx.ptr, name.encode('utf-8'), slot_id, offset, length, generation):
            raise ValueError(f'rwkv_set_tensor_from_slot failed for {name!r}')

    def rwkv_eval_sequence(
            self,
            ctx: RWKVContext,
            tokens: List[int],
            state_in_address: Optional[int],
            state_out_address: int,
            logits_out_address: int
    ) -> None:
        """
        Evaluates the model for a sequence of tokens.
        Uses a faster algorithm than `rwkv_eval` if you do not need the state and logits for every token. Best used with sequence lengths of 64 or so.
        Has to build a computation graph on the first call for a given sequence, but will use this cached graph for subsequent calls of the same sequence length.

        NOTE ON GGML NODE LIMIT

        ggml has a hard-coded limit on max amount of nodes in a computation graph. The sequence graph is built in a way that quickly exceedes
        this limit when using large models and/or large sequence lengths.
        Fortunately, rwkv.cpp's fork of ggml has increased limit which was tested to work for sequence lengths up to 64 for 14B models.

        If you get `GGML_ASSERT: ...\\ggml.c:16941: cgraph->n_nodes < GGML_MAX_NODES`, this means you've exceeded the limit.
        To get rid of the assertion failure, reduce the model size and/or sequence length.

        Not thread-safe. For parallel inference, call `rwkv_clone_context` to create one rwkv_context for each thread.
        Throws an exception in case of any error. Error messages would be printed to stderr.

        Parameters
        ----------
        ctx : RWKVContext
            RWKV context obtained from rwkv_init_from_file.
        tokens : List[int]
            Next token indices, in range 0 <= token < n_vocab.
        state_in_address : int
            Address of the first element of a FP32 buffer of size rwkv_get_state_buffer_element_count; or None, if this is a first pass.
        state_out_address : int
            Address of the first element of a FP32 buffer of size rwkv_get_state_buffer_element_count. This buffer will be written to.
        logits_out_address : int
            Address of the first element of a FP32 buffer of size rwkv_get_logits_buffer_element_count. This buffer will be written to.
        """

        ok = self.library.rwkv_eval_sequence(
            ctx.ptr,
            ctypes.cast((ctypes.c_int32 * len(tokens))(*tokens), P_INT),
            ctypes.c_size_t(len(tokens)),
            ctypes.cast(0 if state_in_address is None else state_in_address, P_FLOAT),
            ctypes.cast(state_out_address, P_FLOAT),
            ctypes.cast(logits_out_address, P_FLOAT)
        )
        self.raise_if_failed(ok, 'rwkv_eval_sequence', ctx)

    def rwkv_eval_sequence_in_chunks(
            self,
            ctx: RWKVContext,
            tokens: List[int],
            chunk_size: int,
            state_in_address: Optional[int],
            state_out_address: int,
            logits_out_address: int
    ) -> None:
        """
        Evaluates the model for a sequence of tokens using `rwkv_eval_sequence`, splitting a potentially long sequence into fixed-length chunks.
        This function is useful for processing complete prompts and user input in chat & role-playing use-cases.
        It is recommended to use this function instead of `rwkv_eval_sequence` to avoid mistakes and get maximum performance.

        Chunking allows processing sequences of thousands of tokens, while not reaching the ggml's node limit and not consuming too much memory.
        A reasonable and recommended value of chunk size is 16. If you want maximum performance, try different chunk sizes in range [2..64]
        and choose one that works the best in your use case.

        Not thread-safe. For parallel inference, call `rwkv_clone_context` to create one rwkv_context for each thread.
        Throws an exception in case of any error. Error messages would be printed to stderr.

        Parameters
        ----------
        ctx : RWKVContext
            RWKV context obtained from rwkv_init_from_file.
        tokens : List[int]
            Next token indices, in range 0 <= token < n_vocab.
        chunk_size : int
            Size of each chunk in tokens, must be positive.
        state_in_address : int
            Address of the first element of a FP32 buffer of size rwkv_get_state_buffer_element_count; or None, if this is a first pass.
        state_out_address : int
            Address of the first element of a FP32 buffer of size rwkv_get_state_buffer_element_count. This buffer will be written to.
        logits_out_address : int
            Address of the first element of a FP32 buffer of size rwkv_get_logits_buffer_element_count. This buffer will be written to.
        """

        ok = self.library.rwkv_eval_sequence_in_chunks(
            ctx.ptr,
            ctypes.cast((ctypes.c_int32 * len(tokens))(*tokens), P_INT),
            ctypes.c_size_t(len(tokens)),
            ctypes.c_size_t(chunk_size),
            ctypes.cast(0 if state_in_address is None else state_in_address, P_FLOAT),
            ctypes.cast(state_out_address, P_FLOAT),
            ctypes.cast(logits_out_address, P_FLOAT)
        )
        self.raise_if_failed(ok, 'rwkv_eval_sequence_in_chunks', ctx)

    def rwkv_get_n_vocab(self, ctx: RWKVContext) -> int:
        """
        Returns the number of tokens in the given model's vocabulary.
        Useful for telling 20B_tokenizer models (n_vocab = 50277) apart from World models (n_vocab = 65536).

        Parameters
        ----------
        ctx : RWKVContext
            RWKV context obtained from rwkv_init_from_file.
        """

        return self.library.rwkv_get_n_vocab(ctx.ptr)

    def rwkv_get_n_embed(self, ctx: RWKVContext) -> int:
        """
        Returns the number of elements in the given model's embedding.
        Useful for reading individual fields of a model's hidden state.

        Parameters
        ----------
        ctx : RWKVContext
            RWKV context obtained from rwkv_init_from_file.
        """

        return self.library.rwkv_get_n_embed(ctx.ptr)

    def rwkv_get_n_layer(self, ctx: RWKVContext) -> int:
        """
        Returns the number of layers in the given model.
        A layer is a pair of RWKV and FFN operations, stacked multiple times throughout the model.
        Embedding matrix and model head (unembedding matrix) are NOT counted in `n_layer`.
        Useful for always offloading the entire model to GPU.

        Parameters
        ----------
        ctx : RWKVContext
            RWKV context obtained from rwkv_init_from_file.
        """

        return self.library.rwkv_get_n_layer(ctx.ptr)

    def rwkv_get_state_buffer_element_count(self, ctx: RWKVContext) -> int:
        """
        Returns count of FP32 elements in state buffer.

        Parameters
        ----------
        ctx : RWKVContext
            RWKV context obtained from rwkv_init_from_file.
        """

        return self.library.rwkv_get_state_buffer_element_count(ctx.ptr)

    def rwkv_get_logits_buffer_element_count(self, ctx: RWKVContext) -> int:
        """
        Returns count of FP32 elements in logits buffer.

        Parameters
        ----------
        ctx : RWKVContext
            RWKV context obtained from rwkv_init_from_file.
        """

        return self.library.rwkv_get_logits_buffer_element_count(ctx.ptr)

    def rwkv_free(self, ctx: RWKVContext) -> None:
        """
        Frees all allocated memory and the context.

        Parameters
        ----------
        ctx : RWKVContext
            RWKV context obtained from rwkv_init_from_file.
        """

        self.library.rwkv_free(ctx.ptr)

        ctx.ptr = self.nullptr

    def rwkv_quantize_model_file(self, model_file_path_in: str, model_file_path_out: str, format_name: str) -> None:
        """
        Quantizes FP32 or FP16 model to one of INT4 formats.
        Throws an exception in case of any error. Error messages would be printed to stderr.

        Parameters
        ----------
        model_file_path_in : str
            Path to model file in ggml format, must be either FP32 or FP16.
        model_file_path_out : str
            Quantized model will be written here.
        format_name : str
            One of QUANTIZED_FORMAT_NAMES.
        """

        if format_name not in QUANTIZED_FORMAT_NAMES:
            raise ValueError(f'Unknown format name {format_name}, use one of {QUANTIZED_FORMAT_NAMES}')

        if not self.library.rwkv_quantize_model_file(
            model_file_path_in.encode('utf-8'),
            model_file_path_out.encode('utf-8'),
            format_name.encode('utf-8')
        ):
            raise ValueError('rwkv_quantize_model_file failed, check stderr')

    def rwkv_get_system_info_string(self) -> str:
        """
        Returns system information string.
        """

        return self.library.rwkv_get_system_info_string().decode('utf-8')

def load_rwkv_shared_library() -> RWKVSharedLibrary:
    """
    Attempts to find rwkv.cpp shared library and load it.
    To specify exact path to the library, create an instance of RWKVSharedLibrary explicitly.
    """

    # Keep the convenience loader aligned with the engine backend.  Windows
    # builds made by MinGW commonly emit ``librwkv.dll`` while the upstream
    # helper historically searched only for ``rwkv.dll``.  An explicit path
    # is also useful for side-by-side optimized builds and deployment.
    explicit = os.environ.get('RWKVCPP_DLL', '').strip()
    if explicit and os.path.isfile(explicit):
        return RWKVSharedLibrary(explicit)

    if 'win32' in sys.platform or 'cygwin' in sys.platform:
        file_names = ('rwkv.dll', 'librwkv.dll')
    elif 'darwin' in sys.platform:
        file_names = ('librwkv.dylib',)
    else:
        file_names = ('librwkv.so',)

    # Possible sub-paths to the library relative to the repo dir.
    child_paths: List[Callable[[pathlib.Path], pathlib.Path]] = [
        # No lookup for Debug config here.
        # I assume that if a user wants to debug the library,
        # they will be able to find the library and set the exact path explicitly.
        lambda p: p / 'bin' / 'Release' / file_name,
        lambda p: p / 'bin' / file_name,
        # Some people prefer to build in the "build" subdirectory.
        lambda p: p / 'build' / 'bin' / 'Release' / file_name,
        lambda p: p / 'build' / 'bin' / file_name,
        lambda p: p / 'build' / file_name,
        # Fallback.
        lambda p: p / file_name
    ]

    working_dir: pathlib.Path = pathlib.Path(os.path.abspath(os.getcwd()))

    parent_paths: List[pathlib.Path] = [
        # Possible repo dirs relative to the working dir.
        # ./python/rwkv_cpp
        working_dir.parent.parent,
        # ./python
        working_dir.parent,
        # .
        working_dir,
        # Repo dir relative to this Python file.
        pathlib.Path(os.path.abspath(__file__)).parent.parent.parent
    ]

    for parent_path in parent_paths:
        for file_name in file_names:
            for child_path in child_paths:
                full_path: pathlib.Path = child_path(parent_path)

                if os.path.isfile(full_path):
                    return RWKVSharedLibrary(str(full_path))

    raise ValueError(f'Failed to find {file_names[0]} automatically; '
                     f'you need to find the library and create RWKVSharedLibrary specifying the path to it')
