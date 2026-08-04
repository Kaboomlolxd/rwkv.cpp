import os
import multiprocessing

# Pre-import PyTorch, if available.
# This fixes "OSError: [WinError 127] The specified procedure could not be found".
try:
    import torch
except ModuleNotFoundError:
    pass

# I'm sure this is not strictly correct, but let's keep this crutch for now.
try:
    import rwkv_cpp_shared_library
except ModuleNotFoundError:
    from . import rwkv_cpp_shared_library

from typing import TypeVar, Optional, Tuple, List

# A value of this type is either a numpy's ndarray or a PyTorch's Tensor.
NumpyArrayOrPyTorchTensor: TypeVar = TypeVar('NumpyArrayOrPyTorchTensor')

class RWKVModel:
    """
    An RWKV model managed by rwkv.cpp library.
    """

    def __init__(
            self,
            shared_library: rwkv_cpp_shared_library.RWKVSharedLibrary,
            model_path: str,
            thread_count: int = max(1, multiprocessing.cpu_count() // 2),
            gpu_layer_count: int = 0,
            native_u8: bool = False,
            native_u8_packed_only: bool = False,
            layer_streaming: bool = False,
            **kwargs
    ) -> None:
        """
        Loads the model and prepares it for inference.
        In case of any error, this method will throw an exception.

        Parameters
        ----------
        shared_library : RWKVSharedLibrary
            rwkv.cpp shared library.
        model_path : str
            Path to RWKV model file in ggml format.
        thread_count : int
            Thread count to use. If not set, defaults to CPU count / 2.
        gpu_layer_count : int
            Count of layers to offload onto the GPU, must be >= 0.
            See documentation of `gpu_offload_layers` for details about layer offloading.
        """

        if 'gpu_layers_count' in kwargs:
            gpu_layer_count = kwargs['gpu_layers_count']

        if not os.path.isfile(model_path):
            raise ValueError(f'{model_path} is not a file')
        
        if not (thread_count > 0):
            raise ValueError('Thread count must be > 0')  

        if not (gpu_layer_count >= 0):
            raise ValueError('GPU layer count must be >= 0')

        self._library: rwkv_cpp_shared_library.RWKVSharedLibrary = shared_library

        self._ctx: rwkv_cpp_shared_library.RWKVContext = self._library.rwkv_init_from_file(
            model_path,
            thread_count,
            gpu_layer_count,
            native_u8=bool(native_u8),
            native_u8_packed_only=bool(native_u8_packed_only),
            layer_streaming=bool(layer_streaming),
        )
        self._valid: bool = True

        self._native_u8_enabled = bool(
            native_u8
            and getattr(self._library, "supports_native_u8_init", False)
            and getattr(self._library, "supports_native_u8_upload", False)
            and gpu_layer_count == 0
        )
        self._native_u8_packed_only = bool(
            self._native_u8_enabled and native_u8_packed_only
        )
        self._layer_streaming_enabled = bool(
            layer_streaming
            and getattr(self._library, "supports_layer_streaming", False)
            and self._library.rwkv_supports_layer_streaming(self._ctx)
        )
        if layer_streaming and not self._layer_streaming_enabled:
            self.free()
            raise ValueError(
                "rwkv.cpp did not enable the requested native layer-streaming ABI"
            )

        self._supports_layer_cache = bool(
            self.supports_layer_streaming
            and getattr(self._library, "supports_layer_cache", False)
        )

        self._state_buffer_element_count: int = self._library.rwkv_get_state_buffer_element_count(self._ctx)
        self._logits_buffer_element_count: int = self._library.rwkv_get_logits_buffer_element_count(self._ctx)
        self._layer_borrowed_buffers: dict[str, object] = {}
        self._layer_persistent_buffers: dict[str, object] = {}

    @property
    def supports_tensor_upload(self) -> bool:
        return bool(getattr(self._library, "supports_tensor_upload", False))

    @property
    def supports_tensor_slot_upload(self) -> bool:
        return bool(getattr(self._library, "supports_tensor_slot_upload", False))

    @property
    def supports_native_u8(self) -> bool:
        return bool(self._native_u8_enabled)

    @property
    def native_u8_packed_only(self) -> bool:
        return bool(self._native_u8_packed_only)

    @property
    def supports_layer_streaming(self) -> bool:
        return bool(self._layer_streaming_enabled)

    @property
    def supports_layer_cache(self) -> bool:
        return bool(self._supports_layer_cache)

    @property
    def supports_layer_borrowed_dense(self) -> bool:
        return bool(getattr(self._library, "supports_layer_borrowed_dense", False))

    @property
    def supports_layer_persistent_dense(self) -> bool:
        return bool(
            getattr(self._library, "supports_layer_persistent_dense", False)
        )

    def set_layer_cache_bytes(self, bytes_limit: int) -> None:
        if not self.supports_layer_cache:
            raise NotImplementedError("native layer cache ABI is unavailable")
        self._library.rwkv_set_layer_cache_bytes(self._ctx, int(bytes_limit))

    def layer_cache_stats(self) -> dict[str, int]:
        if not self.supports_layer_cache:
            return {
                "limit_bytes": 0,
                "used_bytes": 0,
                "hits": 0,
                "misses": 0,
                "evictions": 0,
            }
        return self._library.rwkv_get_layer_cache_stats(self._ctx)

    @property
    def layer_state_len(self) -> int:
        if not self.supports_layer_streaming:
            return 0
        return int(self._library.rwkv_get_layer_state_len(self._ctx))

    def layer_begin(self, layer_id: int) -> None:
        if not self.supports_layer_streaming:
            raise NotImplementedError("native layer-streaming ABI is unavailable")
        self._layer_borrowed_buffers.clear()
        self._library.rwkv_layer_begin(self._ctx, int(layer_id))

    def layer_invalidate_borrowed_cache(self, layer_id: int) -> None:
        prefix = f"blocks.{int(layer_id)}."
        for owner_map in (
            self._layer_borrowed_buffers,
            self._layer_persistent_buffers,
        ):
            for name in list(owner_map):
                if str(name).startswith(prefix):
                    owner_map.pop(name, None)
        self._library.rwkv_layer_invalidate_borrowed_cache(
            self._ctx, int(layer_id)
        )

    def layer_ready(self, layer_id: int) -> bool:
        if not bool(getattr(self._library, "supports_layer_ready", False)):
            return False
        return bool(self._library.rwkv_layer_is_ready(self._ctx, int(layer_id)))

    @property
    def supports_layer_cached_step(self) -> bool:
        return bool(getattr(self._library, "supports_layer_cached_step", False))

    def layer_cache_ready(self) -> bool:
        if not self.supports_layer_cached_step:
            return False
        return bool(self._library.rwkv_layer_cache_ready(self._ctx))

    def layer_step_cached(
        self,
        activation_in: NumpyArrayOrPyTorchTensor,
        activation_out: NumpyArrayOrPyTorchTensor,
        state_in: NumpyArrayOrPyTorchTensor,
        state_out: NumpyArrayOrPyTorchTensor,
        v_first_in: NumpyArrayOrPyTorchTensor | None = None,
        v_first_out: NumpyArrayOrPyTorchTensor | None = None,
    ) -> None:
        if not self.supports_layer_cached_step:
            raise NotImplementedError("native cached layer-step ABI is unavailable")
        self._validate_layer_tensor(activation_in, "activation_in")
        self._validate_layer_tensor(activation_out, "activation_out")
        self._validate_layer_full_state_tensor(state_in, "state_in")
        self._validate_layer_full_state_tensor(state_out, "state_out")
        if v_first_in is not None:
            self._validate_layer_tensor(v_first_in, "v_first_in")
        if v_first_out is not None:
            self._validate_layer_tensor(v_first_out, "v_first_out")
        self._library.rwkv_layer_step_cached(
            self._ctx,
            self._get_data_ptr(activation_in),
            self._get_data_ptr(activation_out),
            self._get_data_ptr(state_in),
            self._get_data_ptr(state_out),
            None if v_first_in is None else self._get_data_ptr(v_first_in),
            None if v_first_out is None else self._get_data_ptr(v_first_out),
        )

    def layer_set_tensor(self, name: str, data: bytes | memoryview) -> None:
        if not self.supports_layer_streaming:
            raise NotImplementedError("native layer-streaming ABI is unavailable")
        self._library.rwkv_layer_set_tensor(self._ctx, name, data)

    def _layer_set_tensor_borrowed(
        self,
        name: str,
        tensor: NumpyArrayOrPyTorchTensor,
        *,
        persistent: bool,
    ) -> None:
        supported = (
            self.supports_layer_persistent_dense
            if persistent
            else self.supports_layer_borrowed_dense
        )
        if not supported:
            raise NotImplementedError("borrowed dense layer ABI is unavailable")
        if self._is_pytorch_tensor(tensor):
            if tensor.device != torch.device("cpu"):
                raise ValueError("borrowed layer tensor is not on CPU")
            if not tensor.is_contiguous():
                raise ValueError("borrowed layer tensor is not contiguous")
            nbytes = int(tensor.numel()) * int(tensor.element_size())
        else:
            if not bool(tensor.data.contiguous):
                raise ValueError("borrowed layer tensor is not contiguous")
            nbytes = int(tensor.nbytes)
        self._library.rwkv_layer_set_tensor_borrowed(
            self._ctx,
            str(name),
            int(self._get_data_ptr(tensor)),
            nbytes,
            persistent=persistent,
        )
        owner_map = (
            self._layer_persistent_buffers
            if persistent
            else self._layer_borrowed_buffers
        )
        owner_map[str(name)] = tensor

    def layer_set_tensor_borrowed(
        self, name: str, tensor: NumpyArrayOrPyTorchTensor
    ) -> None:
        self._layer_set_tensor_borrowed(name, tensor, persistent=False)

    def layer_set_tensor_borrowed_persistent(
        self, name: str, tensor: NumpyArrayOrPyTorchTensor
    ) -> None:
        self._layer_set_tensor_borrowed(name, tensor, persistent=True)

    def layer_set_tensor_scale_u8_grouped(
        self, name: str, data: bytes | memoryview
    ) -> None:
        if not self.supports_layer_streaming:
            raise NotImplementedError("native layer-streaming ABI is unavailable")
        self._library.rwkv_layer_set_tensor_scale_u8_grouped(
            self._ctx, name, data
        )

    def layer_set_tensor_scale_u8_grouped_borrowed(
        self, name: str, data: bytes | memoryview
    ) -> None:
        if not getattr(self._library, "supports_layer_borrowed_u8", False):
            self.layer_set_tensor_scale_u8_grouped(name, data)
            return
        # Keep the Python owner alive until the next layer_begin.  The native
        # descriptor only borrows the underlying bytes on this path.
        holder = self._library.rwkv_layer_set_tensor_scale_u8_grouped_borrowed(
            self._ctx, str(name), data
        )
        self._layer_borrowed_buffers[str(name)] = (data, holder)

    def layer_set_tensor_scale_u8_grouped_borrowed_persistent(
        self, name: str, data: bytes | memoryview
    ) -> None:
        if not getattr(self._library, "supports_layer_persistent_u8", False):
            self.layer_set_tensor_scale_u8_grouped_borrowed(name, data)
            return
        holder = self._library.rwkv_layer_set_tensor_scale_u8_grouped_borrowed_persistent(
            self._ctx, str(name), data
        )
        self._layer_persistent_buffers[str(name)] = (data, holder)

    def layer_set_tensors_scale_u8_grouped(
        self,
        items: list[tuple[str, bytes | memoryview]],
        *,
        persistent: bool = False,
    ) -> None:
        if not items:
            return
        if not getattr(self._library, "supports_layer_u8_batch", False):
            setter = (
                self.layer_set_tensor_scale_u8_grouped_borrowed_persistent
                if persistent
                else self.layer_set_tensor_scale_u8_grouped_borrowed
            )
            for name, data in items:
                setter(name, data)
            return
        owners = self._library.rwkv_layer_set_tensor_scale_u8_grouped_batch(
            self._ctx,
            items,
            persistent=persistent,
        )
        target = (
            self._layer_persistent_buffers
            if persistent
            else self._layer_borrowed_buffers
        )
        for name, holder, data in owners:
            target[str(name)] = (data, holder)

    def layer_step(
        self,
        layer_id: int,
        activation_in: NumpyArrayOrPyTorchTensor,
        activation_out: NumpyArrayOrPyTorchTensor,
        state_in: NumpyArrayOrPyTorchTensor | None,
        state_out: NumpyArrayOrPyTorchTensor,
        v_first_in: NumpyArrayOrPyTorchTensor | None = None,
        v_first_out: NumpyArrayOrPyTorchTensor | None = None,
    ) -> None:
        if not self.supports_layer_streaming:
            raise NotImplementedError("native layer-streaming ABI is unavailable")
        for tensor, name in (
            (activation_in, "activation_in"),
            (activation_out, "activation_out"),
            (state_out, "state_out"),
        ):
            self._validate_layer_tensor(tensor, name)
        if state_in is not None:
            self._validate_layer_tensor(state_in, "state_in")
        if v_first_in is not None:
            self._validate_layer_tensor(v_first_in, "v_first_in")
        if v_first_out is not None:
            self._validate_layer_tensor(v_first_out, "v_first_out")
        self._library.rwkv_layer_step(
            self._ctx,
            int(layer_id),
            self._get_data_ptr(activation_in),
            self._get_data_ptr(activation_out),
            None if state_in is None else self._get_data_ptr(state_in),
            self._get_data_ptr(state_out),
            None if v_first_in is None else self._get_data_ptr(v_first_in),
            None if v_first_out is None else self._get_data_ptr(v_first_out),
        )

    @property
    def supports_layer_streaming_sequence(self) -> bool:
        return bool(getattr(self._library, "supports_layer_streaming_sequence", False))

    def layer_step_sequence(
        self,
        layer_id: int,
        activation_in: NumpyArrayOrPyTorchTensor,
        activation_out: NumpyArrayOrPyTorchTensor,
        n_tokens: int,
        state_in: NumpyArrayOrPyTorchTensor | None,
        state_out: NumpyArrayOrPyTorchTensor,
        v_first_in: NumpyArrayOrPyTorchTensor | None = None,
        v_first_out: NumpyArrayOrPyTorchTensor | None = None,
    ) -> None:
        if not self.supports_layer_streaming:
            raise NotImplementedError("native layer-streaming ABI is unavailable")
        if not self.supports_layer_streaming_sequence:
            raise NotImplementedError("native layer sequence ABI is unavailable")
        count = int(n_tokens)
        if count <= 0:
            raise ValueError("n_tokens must be positive")
        for tensor, name in (
            (activation_in, "activation_in"),
            (activation_out, "activation_out"),
        ):
            self._validate_layer_sequence_tensor(tensor, name, count, self.n_embed)
        self._validate_layer_tensor(state_out, "state_out")
        if state_in is not None:
            self._validate_layer_tensor(state_in, "state_in")
        if v_first_in is not None:
            self._validate_layer_sequence_tensor(v_first_in, "v_first_in", count, self.n_embed)
        if v_first_out is not None:
            self._validate_layer_sequence_tensor(v_first_out, "v_first_out", count, self.n_embed)
        self._library.rwkv_layer_step_sequence(
            self._ctx,
            int(layer_id),
            self._get_data_ptr(activation_in),
            self._get_data_ptr(activation_out),
            count,
            None if state_in is None else self._get_data_ptr(state_in),
            self._get_data_ptr(state_out),
            None if v_first_in is None else self._get_data_ptr(v_first_in),
            None if v_first_out is None else self._get_data_ptr(v_first_out),
        )

    def register_tensor_slot(self, slot_id: int, buffer: bytearray) -> None:
        self._library.rwkv_register_tensor_slot(self._ctx, slot_id, buffer)

    def set_tensor_from_slot(self, name: str, slot_id: int, offset: int, length: int, generation: int) -> None:
        self._library.rwkv_set_tensor_from_slot(self._ctx, name, slot_id, offset, length, generation)

    @property
    def n_vocab(self) -> int:
        return self._library.rwkv_get_n_vocab(self._ctx)

    @property
    def n_embed(self) -> int:
        return self._library.rwkv_get_n_embed(self._ctx)

    @property
    def n_layer(self) -> int:
        return self._library.rwkv_get_n_layer(self._ctx)

    @property
    def arch_version_major(self) -> int:
        if not getattr(self._library, "supports_model_state_metadata", False):
            return 0
        return int(self._library.library.rwkv_get_arch_version_major(self._ctx.ptr))

    @property
    def head_count(self) -> int:
        if not getattr(self._library, "supports_model_state_metadata", False):
            return 0
        return int(self._library.library.rwkv_get_head_count(self._ctx.ptr))

    @property
    def head_size(self) -> int:
        if not getattr(self._library, "supports_model_state_metadata", False):
            return 0
        return int(self._library.library.rwkv_get_head_size(self._ctx.ptr))

    def tensor_nbytes(self, name: str) -> int:
        """Return the native byte size of a loaded tensor, or 0 if absent."""
        if not self._valid:
            raise ValueError('Model was freed')
        return self._library.rwkv_get_tensor_nbytes(self._ctx, name)

    @property
    def supports_tensor_type(self) -> bool:
        return bool(getattr(self._library, "supports_tensor_type", False))

    def tensor_type(self, name: str) -> int:
        """Return the numeric ggml type for a loaded tensor, or -1 if unknown."""
        if not self._valid:
            raise ValueError('Model was freed')
        return self._library.rwkv_get_tensor_type(self._ctx, name)

    def set_tensor(self, name: str, data: bytes) -> None:
        """Replace one tensor in the loaded ggml model."""
        if not self._valid:
            raise ValueError('Model was freed')
        self._library.rwkv_set_tensor(self._ctx, name, data)

    def set_tensor_from_pointer(self, name: str, address: int, nbytes: int) -> None:
        """Replace one tensor from a live contiguous native CPU buffer."""
        if not self._valid:
            raise ValueError('Model was freed')
        self._library.rwkv_set_tensor_from_pointer(self._ctx, name, address, nbytes)

    def set_tensor_scale_u8_grouped(self, name: str, data: bytes | memoryview) -> None:
        """Replace a matrix with a native SG8 grouped-U8 record."""
        if not self._valid:
            raise ValueError('Model was freed')
        self._library.rwkv_set_tensor_scale_u8_grouped(self._ctx, name, data)

    def eval(
            self,
            token: int,
            state_in: Optional[NumpyArrayOrPyTorchTensor],
            state_out: Optional[NumpyArrayOrPyTorchTensor] = None,
            logits_out: Optional[NumpyArrayOrPyTorchTensor] = None,
            use_numpy: bool = False
    ) -> Tuple[NumpyArrayOrPyTorchTensor, NumpyArrayOrPyTorchTensor]:
        """
        Evaluates the model for a single token.
        In case of any error, this method will throw an exception.

        Parameters
        ----------
        token : int
            Index of next token to be seen by the model. Must be in range 0 <= token < n_vocab.
        state_in : Optional[NumpyArrayOrTorchTensor]
            State from previous call of this method. If this is a first pass, set it to None.
        state_out : Optional[NumpyArrayOrTorchTensor]
            Optional output tensor for state. If provided, must be of type float32, contiguous and of shape (state_buffer_element_count).
        logits_out : Optional[NumpyArrayOrTorchTensor]
            Optional output tensor for logits. If provided, must be of type float32, contiguous and of shape (logits_buffer_element_count).
        use_numpy : bool
            If set to True, numpy's ndarrays will be created instead of PyTorch's Tensors.
            This parameter is ignored if any tensor parameter is not None; in such case,
            type of returned tensors will match the type of received tensors.

        Returns
        -------
        logits, state
            Logits vector of shape (n_vocab); state for the next step.
        """

        if not self._valid:
            raise ValueError('Model was freed')

        use_numpy = self._detect_numpy_usage([state_in, state_out, logits_out], use_numpy)

        if state_in is not None:
            self._validate_tensor(state_in, 'state_in', self._state_buffer_element_count)

            state_in_ptr = self._get_data_ptr(state_in)
        else:
            state_in_ptr = 0

        if state_out is not None:
            self._validate_tensor(state_out, 'state_out', self._state_buffer_element_count)
        else:
            state_out = self._zeros_float32(self._state_buffer_element_count, use_numpy)

        if logits_out is not None:
            self._validate_tensor(logits_out, 'logits_out', self._logits_buffer_element_count)
        else:
            logits_out = self._zeros_float32(self._logits_buffer_element_count, use_numpy)

        self._library.rwkv_eval(
            self._ctx,
            token,
            state_in_ptr,
            self._get_data_ptr(state_out),
            self._get_data_ptr(logits_out)
        )

        return logits_out, state_out
    
    def eval_sequence(
            self,
            tokens: List[int],
            state_in: Optional[NumpyArrayOrPyTorchTensor],
            state_out: Optional[NumpyArrayOrPyTorchTensor] = None,
            logits_out: Optional[NumpyArrayOrPyTorchTensor] = None,
            use_numpy: bool = False
    ) -> Tuple[NumpyArrayOrPyTorchTensor, NumpyArrayOrPyTorchTensor]:
        """
        Evaluates the model for a sequence of tokens.

        NOTE ON GGML NODE LIMIT

        ggml has a hard-coded limit on max amount of nodes in a computation graph. The sequence graph is built in a way that quickly exceedes
        this limit when using large models and/or large sequence lengths.
        Fortunately, rwkv.cpp's fork of ggml has increased limit which was tested to work for sequence lengths up to 64 for 14B models.

        If you get `GGML_ASSERT: ...\\ggml.c:16941: cgraph->n_nodes < GGML_MAX_NODES`, this means you've exceeded the limit.
        To get rid of the assertion failure, reduce the model size and/or sequence length.

        In case of any error, this method will throw an exception.

        Parameters
        ----------
        tokens : List[int]
            Indices of the next tokens to be seen by the model. Must be in range 0 <= token < n_vocab.
        state_in : Optional[NumpyArrayOrTorchTensor]
            State from previous call of this method. If this is a first pass, set it to None.
        state_out : Optional[NumpyArrayOrTorchTensor]
            Optional output tensor for state. If provided, must be of type float32, contiguous and of shape (state_buffer_element_count).
        logits_out : Optional[NumpyArrayOrTorchTensor]
            Optional output tensor for logits. If provided, must be of type float32, contiguous and of shape (logits_buffer_element_count).
        use_numpy : bool
            If set to True, numpy's ndarrays will be created instead of PyTorch's Tensors.
            This parameter is ignored if any tensor parameter is not None; in such case,
            type of returned tensors will match the type of received tensors.

        Returns
        -------
        logits, state
            Logits vector of shape (n_vocab); state for the next step.
        """

        if not self._valid:
            raise ValueError('Model was freed')

        use_numpy = self._detect_numpy_usage([state_in, state_out, logits_out], use_numpy)

        if state_in is not None:
            self._validate_tensor(state_in, 'state_in', self._state_buffer_element_count)

            state_in_ptr = self._get_data_ptr(state_in)
        else:
            state_in_ptr = 0

        if state_out is not None:
            self._validate_tensor(state_out, 'state_out', self._state_buffer_element_count)
        else:
            state_out = self._zeros_float32(self._state_buffer_element_count, use_numpy)

        if logits_out is not None:
            self._validate_tensor(logits_out, 'logits_out', self._logits_buffer_element_count)
        else:
            logits_out = self._zeros_float32(self._logits_buffer_element_count, use_numpy)

        self._library.rwkv_eval_sequence(
            self._ctx,
            tokens,
            state_in_ptr,
            self._get_data_ptr(state_out),
            self._get_data_ptr(logits_out)
        )

        return logits_out, state_out

    def eval_sequence_in_chunks(
            self,
            tokens: List[int],
            state_in: Optional[NumpyArrayOrPyTorchTensor],
            state_out: Optional[NumpyArrayOrPyTorchTensor] = None,
            logits_out: Optional[NumpyArrayOrPyTorchTensor] = None,
            chunk_size: int = 16,
            use_numpy: bool = False
    ) -> Tuple[NumpyArrayOrPyTorchTensor, NumpyArrayOrPyTorchTensor]:
        """
        Evaluates the model for a sequence of tokens using `eval_sequence`, splitting a potentially long sequence into fixed-length chunks.
        This function is useful for processing complete prompts and user input in chat & role-playing use-cases.
        It is recommended to use this function instead of `eval_sequence` to avoid mistakes and get maximum performance.

        Chunking allows processing sequences of thousands of tokens, while not reaching the ggml's node limit and not consuming too much memory.
        A reasonable and recommended value of chunk size is 16. If you want maximum performance, try different chunk sizes in range [2..64]
        and choose one that works the best in your use case.

        In case of any error, this method will throw an exception.

        Parameters
        ----------
        tokens : List[int]
            Indices of the next tokens to be seen by the model. Must be in range 0 <= token < n_vocab.
        chunk_size : int
            Size of each chunk in tokens, must be positive.
        state_in : Optional[NumpyArrayOrTorchTensor]
            State from previous call of this method. If this is a first pass, set it to None.
        state_out : Optional[NumpyArrayOrTorchTensor]
            Optional output tensor for state. If provided, must be of type float32, contiguous and of shape (state_buffer_element_count).
        logits_out : Optional[NumpyArrayOrTorchTensor]
            Optional output tensor for logits. If provided, must be of type float32, contiguous and of shape (logits_buffer_element_count).
        use_numpy : bool
            If set to True, numpy's ndarrays will be created instead of PyTorch's Tensors.
            This parameter is ignored if any tensor parameter is not None; in such case,
            type of returned tensors will match the type of received tensors.

        Returns
        -------
        logits, state
            Logits vector of shape (n_vocab); state for the next step.
        """

        if not self._valid:
            raise ValueError('Model was freed')

        use_numpy = self._detect_numpy_usage([state_in, state_out, logits_out], use_numpy)

        if state_in is not None:
            self._validate_tensor(state_in, 'state_in', self._state_buffer_element_count)

            state_in_ptr = self._get_data_ptr(state_in)
        else:
            state_in_ptr = 0

        if state_out is not None:
            self._validate_tensor(state_out, 'state_out', self._state_buffer_element_count)
        else:
            state_out = self._zeros_float32(self._state_buffer_element_count, use_numpy)

        if logits_out is not None:
            self._validate_tensor(logits_out, 'logits_out', self._logits_buffer_element_count)
        else:
            logits_out = self._zeros_float32(self._logits_buffer_element_count, use_numpy)

        self._library.rwkv_eval_sequence_in_chunks(
            self._ctx,
            tokens,
            chunk_size,
            state_in_ptr,
            self._get_data_ptr(state_out),
            self._get_data_ptr(logits_out)
        )

        return logits_out, state_out

    def free(self) -> None:
        """
        Frees all allocated resources.
        In case of any error, this method will throw an exception.
        The object must not be used anymore after calling this method.
        """

        if not self._valid:
            raise ValueError('Already freed')

        # Borrowed grouped-U8 layer views may point into a provider mmap.  Drop
        # every Python owner before releasing the native context so the pack
        # store can close its mapping on Windows.
        self._layer_borrowed_buffers.clear()
        self._layer_persistent_buffers.clear()
        self._valid = False

        self._library.rwkv_free(self._ctx)

    def __del__(self) -> None:
        # Free the context on GC in case user forgot to call free() explicitly.
        if hasattr(self, '_valid') and self._valid:
            self.free()

    def _is_pytorch_tensor(self, tensor: NumpyArrayOrPyTorchTensor) -> bool:
        return hasattr(tensor, '__module__') and tensor.__module__ == 'torch'

    def _detect_numpy_usage(self, tensors: List[Optional[NumpyArrayOrPyTorchTensor]], use_numpy_by_default: bool) -> bool:
        for tensor in tensors:
            if tensor is not None:
                return False if self._is_pytorch_tensor(tensor) else True

        return use_numpy_by_default

    def _validate_tensor(self, tensor: NumpyArrayOrPyTorchTensor, name: str, size: int) -> None:
        if self._is_pytorch_tensor(tensor):
            tensor: torch.Tensor = tensor
            
            if tensor.device != torch.device('cpu'):
                raise ValueError(f'{name} is not on CPU')
            if tensor.dtype != torch.float32:
                raise ValueError(f'{name} is not of type float32')
            if tensor.shape != (size,):
                raise ValueError(f'{name} has invalid shape {tensor.shape}, expected ({size})')
            if not tensor.is_contiguous():
                raise ValueError(f'{name} is not contiguous')
        else:
            import numpy as np
            tensor: np.ndarray = tensor

            if tensor.dtype != np.float32:
                raise ValueError(f'{name} is not of type float32')
            if tensor.shape != (size,):
                raise ValueError(f'{name} has invalid shape {tensor.shape}, expected ({size})')
            if not tensor.data.contiguous:
                raise ValueError(f'{name} is not contiguous')

    def _validate_layer_tensor(
        self, tensor: NumpyArrayOrPyTorchTensor, name: str
    ) -> None:
        if name.startswith("state"):
            size = self.layer_state_len
        else:
            size = self.n_embed
        self._validate_tensor(tensor, name, size)

    def _validate_layer_full_state_tensor(
        self, tensor: NumpyArrayOrPyTorchTensor, name: str
    ) -> None:
        self._validate_tensor(tensor, name, self._state_buffer_element_count)

    def _validate_layer_sequence_tensor(
        self,
        tensor: NumpyArrayOrPyTorchTensor,
        name: str,
        n_tokens: int,
        width: int,
    ) -> None:
        expected = int(n_tokens) * int(width)
        if self._is_pytorch_tensor(tensor):
            if tensor.dtype != torch.float32:
                raise ValueError(f'{name} is not of type float32')
            if tensor.numel() != expected or not tensor.is_contiguous():
                raise ValueError(
                    f'{name} has invalid shape/contiguity; expected {expected} contiguous elements'
                )
        else:
            import numpy as np
            array: np.ndarray = tensor
            if array.dtype != np.float32:
                raise ValueError(f'{name} is not of type float32')
            if array.size != expected or not array.flags.c_contiguous:
                raise ValueError(
                    f'{name} has invalid shape/contiguity; expected {expected} contiguous elements'
                )

    def _get_data_ptr(self, tensor: NumpyArrayOrPyTorchTensor):
        if self._is_pytorch_tensor(tensor):
            return tensor.data_ptr()
        else:
            return tensor.ctypes.data

    def _zeros_float32(self, element_count: int, use_numpy: bool) -> NumpyArrayOrPyTorchTensor:
        if use_numpy:
            import numpy as np
            return np.zeros(element_count, dtype=np.float32)
        else:
            return torch.zeros(element_count, dtype=torch.float32, device='cpu')
