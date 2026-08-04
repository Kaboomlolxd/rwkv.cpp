#include "rwkv.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-impl.h"

#include "ggml-cpu.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#ifdef GGML_USE_BLAS
#include "ggml-blas.h"
#endif

#include <string>
#include <vector>
#include <cstring>
#include <cinttypes>
#include <cmath>
#include <fstream>
#include <unordered_map>
#include <memory>
#include <utility>

#define _FILE_OFFSET_BITS 64
// Puts an optional break point, if debug is enabled.
#define RWKV_MAYBE_BREAK

#include <sys/stat.h>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#    define stat _stat64
#    define fstat _fstat64
#    define ftell _ftelli64
#    define fseek _fseeki64
#    if !defined(NDEBUG)
#        include <intrin.h>
#        define RWKV_MAYBE_BREAK __debugbreak()
#    endif
#else
#    if !defined(__APPLE__)
#        define ftell ftello
#        define fseek fseeko
#    endif
#endif

static_assert(sizeof(stat::st_size) >= 8, "File offsets should be 64-bit or else rwkv.cpp will not be able to load model files over 2 GB");
static_assert(sizeof(decltype(ftell(NULL))) >= 8, "File offsets should be 64-bit or else rwkv.cpp will not be able to load model files over 2 GB");

#define RWKV_MAX_NODES 80000

#include "rwkv_error_handling.inc"

#include "rwkv_utilities.inc"

#include "rwkv_file_format.inc"

#include "rwkv_model_loading.inc"

#include "rwkv_native_u8.inc"

#include "rwkv_operators.inc"

#include "rwkv_graph.inc"
#include "rwkv_layer.inc"

static struct rwkv_context * rwkv_init_from_file_impl(
    const char * file_path,
    const uint32_t n_threads,
    const uint32_t n_gpu_layers,
    const bool native_u8,
    const bool native_u8_packed_only,
    const bool layer_streaming
) {
    global_last_error = RWKV_ERROR_NONE;

    std::unique_ptr<struct rwkv_context> ctx(new(std::nothrow) struct rwkv_context());
    RWKV_ASSERT_NULL_MSG(RWKV_ERROR_CTX | RWKV_ERROR_ALLOC, ctx, "Failed to allocate rwkv_context");
    ctx->print_errors = true;

    ctx->model = new(std::nothrow) struct rwkv_model();
    ctx->model->reference_count++;
    // Native grouped-U8 custom nodes are CPU-only.  Keep the option disabled
    // when a caller asks for accelerator offload so the existing GGML graph
    // remains the safe path for CUDA/Metal/BLAS builds.
    ctx->model->native_u8_enabled = native_u8 && n_gpu_layers == 0;
    ctx->model->native_u8_packed_only =
        ctx->model->native_u8_enabled && native_u8_packed_only;
    ctx->model->layer_streaming = layer_streaming && n_gpu_layers == 0;

    ctx->n_threads = n_threads;

    if (n_gpu_layers) {
        ggml_backend_t backend = nullptr;

#ifdef GGML_USE_CUDA
        backend = ggml_backend_cuda_init(0);
        RWKV_ENSURE_OR_NULL(backend);
#endif

#ifdef GGML_USE_METAL
        backend = ggml_backend_metal_init();
        RWKV_ENSURE_OR_NULL(backend);
#endif

#ifdef GGML_USE_BLAS
        backend = ggml_backend_blas_init();
        RWKV_ENSURE_OR_NULL(backend);
        ggml_backend_blas_set_n_threads(backend, ctx->n_threads);
#endif
        if (backend != nullptr) {
            ctx->model->backends.push_back(backend);
        }
    }

    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    RWKV_ENSURE_OR_NULL(cpu_backend);
    ggml_backend_cpu_set_n_threads(cpu_backend, n_threads);
    ctx->model->backends.push_back(cpu_backend);

    int ngl = n_gpu_layers;
    if (ctx->model->backends.size() == 1) {
        ngl = 0;
    }

    RWKV_ENSURE_OR_NULL(
        rwkv_load_model_from_file(file_path, *ctx->model, ngl, ctx->model->layer_streaming)
    );

    rwkv_native_u8_prepare_model(*ctx->model);

    if (!ctx->model->layer_streaming) {
        RWKV_ENSURE_OR_NULL(rwkv_measure_and_build_serial_context(*ctx->model, ctx->serial_graph));
    } else {
        RWKV_ENSURE_OR_NULL(rwkv_layer_plans_create(*ctx));
    }

    return ctx.release();
}

// API function.
struct rwkv_context * rwkv_init_from_file(const char * file_path, const uint32_t n_threads, const uint32_t n_gpu_layers) {
    return rwkv_init_from_file_impl(
        file_path, n_threads, n_gpu_layers, false, false, false
    );
}

// API function.  ``flags & 1`` enables the CPU-native grouped-U8 graph;
// ``flags & 2`` additionally omits dense matrix payloads and requires packed
// records to be uploaded before evaluation.
struct rwkv_context * rwkv_init_from_file_ex(
    const char * file_path,
    const uint32_t n_threads,
    const uint32_t n_gpu_layers,
    const uint32_t flags
) {
    return rwkv_init_from_file_impl(
        file_path,
        n_threads,
        n_gpu_layers,
        (flags & 1u) != 0u,
        (flags & 2u) != 0u,
        (flags & 4u) != 0u
    );
}

// API function.
struct rwkv_context * rwkv_clone_context(struct rwkv_context * ctx, const uint32_t n_threads) {
    std::unique_ptr<struct rwkv_context> clone(new(std::nothrow) struct rwkv_context());
    RWKV_ASSERT_NULL_MSG(RWKV_ERROR_CTX | RWKV_ERROR_ALLOC, clone, "Failed to allocate rwkv_context");

    clone->model = ctx->model;
    clone->model->reference_count++;

    clone->n_threads = n_threads;

    if (!clone->model->layer_streaming) {
        RWKV_ENSURE_OR_NULL(rwkv_measure_and_build_serial_context(*clone->model, clone->serial_graph));
    } else {
        RWKV_ENSURE_OR_NULL(rwkv_layer_plans_create(*clone));
    }

    clone->last_used_sequence_length = 0;

    clone->print_errors = ctx->print_errors;

    return clone.release();
}

#include "rwkv_eval.inc"

// API function.
// Provided for backwards compatibility.
extern "C" RWKV_API uint32_t rwkv_get_state_buffer_element_count(const struct rwkv_context * ctx) {
    return rwkv_get_state_len(ctx);
}

// API function.
// Provided for backwards compatibility.
extern "C" RWKV_API uint32_t rwkv_get_logits_buffer_element_count(const struct rwkv_context * ctx) {
    return rwkv_get_logits_len(ctx);
}

// API function.
size_t rwkv_get_n_vocab(const struct rwkv_context * ctx) {
    return (size_t) ctx->model->header.n_vocab;
}

// API function.
size_t rwkv_get_n_embed(const struct rwkv_context * ctx) {
    return (size_t) ctx->model->header.n_embed;
}

// API function.
size_t rwkv_get_n_layer(const struct rwkv_context * ctx) {
    return (size_t) ctx->model->header.n_layer;
}

// API function.  These metadata getters are additive and intentionally keep
// the existing state/eval buffers opaque to callers.
uint32_t rwkv_get_arch_version_major(const struct rwkv_context * ctx) {
    return ctx == nullptr || ctx->model == nullptr
        ? 0u
        : ctx->model->arch_version_major;
}

int64_t rwkv_get_head_count(const struct rwkv_context * ctx) {
    return ctx == nullptr || ctx->model == nullptr
        ? 0
        : ctx->model->head_count;
}

int64_t rwkv_get_head_size(const struct rwkv_context * ctx) {
    return ctx == nullptr || ctx->model == nullptr
        ? 0
        : ctx->model->head_size;
}

// API function.
size_t rwkv_get_state_len(const struct rwkv_context * ctx) {
    const struct rwkv_file_header & header = ctx->model->header;

    if (ctx->model->arch_version_major >= 5) {
        return (size_t) header.n_embed * (2 + ctx->model->head_size) * (size_t) header.n_layer;
    } else {
        return (size_t) header.n_embed * 5 * (size_t) header.n_layer;
    }
}

// API function.
size_t rwkv_get_logits_len(const struct rwkv_context * ctx) {
    return (size_t) ctx->model->header.n_vocab;
}

// API function.
void rwkv_free(struct rwkv_context * ctx) {
    if (ctx == NULL) {
        return;
    }

    // Layer plans use the model's CPU backend and must be destroyed before
    // the last context releases that backend.
    ctx->active_layer_plan = nullptr;
    ctx->layer_plan_zero.reset();
    ctx->layer_plan_rest.reset();

    if (--ctx->model->reference_count == 0) {
        for (auto buffer : ctx->model->buffers_w) {
            ggml_backend_buffer_free(buffer);
        }

        for (auto backend : ctx->model->backends) {
            ggml_backend_free(backend);
        }

        ggml_free(ctx->model->ggml_ctx);

        delete ctx->model;
    }

    ggml_backend_sched_free(ctx->serial_graph.sched);
    ggml_free(ctx->serial_graph.ggml_ctx);

    if (ctx->last_used_sequence_length > 0) {
        ggml_backend_sched_free(ctx->sequential_graph.sched);
        ggml_free(ctx->sequential_graph.ggml_ctx);
    }

    delete ctx;
}

// API function.
bool rwkv_set_tensor(
    struct rwkv_context * ctx,
    const char * name,
    const void * data,
    const size_t nbytes
) {
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_ARGS, ctx != nullptr, "Context is null");
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_ARGS, name != nullptr, "Tensor name is null");
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_ARGS, data != nullptr, "Tensor data is null");

    auto it = ctx->model->tensors.find(name);
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_KEY, it != ctx->model->tensors.end(), "Unknown tensor %s", name);
    struct ggml_tensor * tensor = it->second;
    const size_t expected = ggml_nbytes(tensor);
    RWKV_ASSERT_FALSE_MSG(
        RWKV_ERROR_DATA,
        nbytes == expected,
        "Invalid byte count for tensor %s (%llu, expected %llu)",
        name,
        static_cast<unsigned long long>(nbytes),
        static_cast<unsigned long long>(expected)
    );
    ggml_backend_tensor_set(tensor, data, 0, nbytes);
    return true;
}

// API function.
size_t rwkv_get_tensor_nbytes(
    const struct rwkv_context * ctx,
    const char * name
) {
    if (ctx == nullptr || name == nullptr) {
        return 0;
    }

    auto it = ctx->model->tensors.find(name);
    if (it == ctx->model->tensors.end()) {
        return 0;
    }
    return ggml_nbytes(it->second);
}

// API function.
int32_t rwkv_get_tensor_type(
    const struct rwkv_context * ctx,
    const char * name
) {
    if (ctx == nullptr || name == nullptr) {
        return -1;
    }

    auto it = ctx->model->tensors.find(name);
    if (it == ctx->model->tensors.end()) {
        return -1;
    }
    return static_cast<int32_t>(it->second->type);
}

bool rwkv_register_tensor_slot(
    struct rwkv_context * ctx,
    const uint32_t slot_id,
    const void * data,
    const size_t nbytes
) {
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_ARGS, ctx != nullptr, "Context is null");
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_ARGS, data != nullptr, "Slot data is null");
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_ARGS, nbytes > 0, "Slot is empty");
    ctx->model->tensor_slots[slot_id] = {
        static_cast<const uint8_t *>(data), nbytes, 0
    };
    return true;
}

bool rwkv_set_tensor_from_slot(
    struct rwkv_context * ctx,
    const char * name,
    const uint32_t slot_id,
    const size_t offset,
    const size_t nbytes,
    const uint64_t generation
) {
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_ARGS, ctx != nullptr, "Context is null");
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_ARGS, name != nullptr, "Tensor name is null");
    auto slot_it = ctx->model->tensor_slots.find(slot_id);
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_KEY, slot_it != ctx->model->tensor_slots.end(), "Unknown tensor slot");
    auto & slot = slot_it->second;
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_DATA, generation >= slot.generation, "Stale tensor slot generation");
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_DATA, offset <= slot.nbytes && nbytes <= slot.nbytes - offset, "Tensor slot range is invalid");
    auto tensor_it = ctx->model->tensors.find(name);
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_KEY, tensor_it != ctx->model->tensors.end(), "Unknown tensor %s", name);
    struct ggml_tensor * tensor = tensor_it->second;
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_DATA, ggml_nbytes(tensor) == nbytes, "Tensor slot byte count mismatch");
    ggml_backend_tensor_set(tensor, slot.data + offset, 0, nbytes);
    slot.generation = generation;
    return true;
}

// API function.
bool rwkv_set_tensor_scale_u8_grouped(
    struct rwkv_context * ctx,
    const char * name,
    const void * data,
    const size_t nbytes
) {
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_ARGS, ctx != nullptr, "Context is null");
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_ARGS, name != nullptr, "Tensor name is null");
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_ARGS, data != nullptr, "Tensor data is null");
    RWKV_ASSERT_FALSE_MSG(
        RWKV_ERROR_UNSUPPORTED,
        ctx->model->native_u8_enabled,
        "Native grouped-U8 is not enabled for this context"
    );
    RWKV_ASSERT_FALSE_MSG(
        RWKV_ERROR_DATA,
        rwkv_native_u8_store_tensor(*ctx->model, name, data, nbytes),
        "Invalid grouped-U8 blob for tensor %s",
        name
    );
    return true;
}

// API function.
void rwkv_set_print_errors(struct rwkv_context * ctx, const bool print_errors) {
    bool * ptr = ctx ? &ctx->print_errors : &global_print_errors;
    *ptr = print_errors;
}

// API function.
bool rwkv_get_print_errors(const struct rwkv_context * ctx) {
    return ctx ? ctx->print_errors : global_print_errors;
}

// API function.
enum rwkv_error_flags rwkv_get_last_error(struct rwkv_context * ctx) {
    enum rwkv_error_flags * ptr = ctx ? &ctx->last_error : &global_last_error;
    enum rwkv_error_flags value = *ptr;
    *ptr = RWKV_ERROR_NONE;
    return value;
}

#include "rwkv_quantize.inc"

// API function.
const char * rwkv_get_system_info_string(void) {
    static std::string s;

    if (s.empty()) {
        s  = "";
        s += "AVX="       + std::to_string(ggml_cpu_has_avx())       + " ";
        s += "AVX2="      + std::to_string(ggml_cpu_has_avx2())      + " ";
        s += "AVX512="    + std::to_string(ggml_cpu_has_avx512())    + " ";
        s += "FMA="       + std::to_string(ggml_cpu_has_fma())       + " ";
        s += "NEON="      + std::to_string(ggml_cpu_has_neon())      + " ";
        s += "ARM_FMA="   + std::to_string(ggml_cpu_has_arm_fma())   + " ";
        s += "F16C="      + std::to_string(ggml_cpu_has_f16c())      + " ";
        s += "FP16_VA="   + std::to_string(ggml_cpu_has_fp16_va())   + " ";
        s += "WASM_SIMD=" + std::to_string(ggml_cpu_has_wasm_simd()) + " ";
        s += "SSE3="      + std::to_string(ggml_cpu_has_sse3())      + " ";
        s += "VSX="       + std::to_string(ggml_cpu_has_vsx());
    }

    return s.c_str();
}
