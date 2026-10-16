/* CPU-native grouped-U8 graph nodes used by the SSD provider bridge.

   The packed format is intentionally the same SG8\x01 record emitted by
   rwkv_ssd.runtime.pack_codec:

       magic[4], group_size[u32], (min,max)[groups] as f32, q[numel] as u8

   GGML stores matrices as [input, output], while the pack stores the
   checkpoint's row-major PyTorch view.  Most RWKV-7 matrices have the same
   logical order for the GEMV used by rwkv.cpp.  The small w/a/v/g adapters
   are the eight explicitly transposed targets; their descriptor keeps the
   packed bytes unchanged and walks them as a row-major x @ W map.

   This file is included into rwkv.cpp after rwkv_model_loading.inc.  It is
   deliberately CPU-only: CUDA/Metal schedulers continue to use the original
   GGML graph and upload ABI.
*/

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <new>

#if defined(__AVX2__) || defined(_M_AVX2)
#    include <immintrin.h>
#    define RWKV_NATIVE_U8_AVX2 1
#else
#    define RWKV_NATIVE_U8_AVX2 0
#endif

struct rwkv_u8_grouped_matrix {
    struct ggml_tensor * dense = nullptr;
    std::string name;
    std::vector<uint8_t> blob;
    // Cached shape/offset metadata avoids recomputing GGML element counts and
    // scale-section offsets inside every GEMV.  The two float arrays also
    // avoid unaligned memcpy loads from the packed record's byte vector.
    size_t numel = 0;
    size_t group_count = 0;
    size_t quant_offset = 0;
    std::vector<float> group_min;
    std::vector<float> group_scale;
    std::shared_ptr<const std::vector<float>> shared_group_min;
    std::shared_ptr<const std::vector<float>> shared_group_scale;
    uint32_t group_size = 0;
    bool blob_valid = false;
    // Layer-local callers may lend a packed record whose owner is retained by
    // the Python model until the next layer_begin.  Resident uploads continue
    // to use the owning vector below.
    const uint8_t * borrowed_data = nullptr;
    size_t borrowed_nbytes = 0;
    bool transpose = false;
    // r_k is a small elementwise tensor, not a matmul operand.  Decode it
    // into its existing GGML storage while keeping the same descriptor shape.
    bool dense_required = false;

    const uint8_t * payload_data() const {
        return borrowed_data != nullptr ? borrowed_data : blob.data();
    }

    const float * group_min_data() const {
        return shared_group_min ? shared_group_min->data() : group_min.data();
    }

    const float * group_scale_data() const {
        return shared_group_scale
            ? shared_group_scale->data()
            : group_scale.data();
    }
};

// Layer plans own descriptors for cloned one-block tensors.  The definition
// lives in rwkv_layer.inc (included after this file); keeping the lookup as a
// narrow forward declaration lets the same native GEMV node serve resident
// model tensors and bounded layer-plan tensors.
static rwkv_u8_grouped_matrix * rwkv_native_u8_layer_lookup(
    struct ggml_tensor * tensor
);

static bool rwkv_native_u8_parse_blob(
    const void * data,
    const size_t nbytes,
    const size_t numel,
    uint32_t & group_size
);

static bool rwkv_native_u8_validate_blob_shape(
    const void * data,
    const size_t nbytes,
    const size_t numel,
    uint32_t & group_size
) {
    if (data == nullptr || nbytes < 8) {
        return false;
    }

    const uint8_t * raw = static_cast<const uint8_t *>(data);
    if (memcmp(raw, "SG8\x01", 4) != 0) {
        return false;
    }

    uint32_t group = 0;
    memcpy(&group, raw + 4, sizeof(group));
    if (group == 0) {
        return false;
    }
    const size_t groups = numel / static_cast<size_t>(group) +
        (numel % static_cast<size_t>(group) != 0u ? 1u : 0u);
    if (groups > (SIZE_MAX - 8u) / 8u) {
        return false;
    }
    const size_t metadata = 8u + groups * 8u;
    if (metadata > nbytes || nbytes - metadata != numel) {
        return false;
    }
    group_size = group;
    return true;
}

static bool rwkv_native_u8_assign_blob_impl(
    rwkv_u8_grouped_matrix & matrix,
    const void * data,
    const size_t nbytes,
    const size_t numel,
    const bool borrow
) {
    uint32_t group_size = 0;
    if (!rwkv_native_u8_parse_blob(data, nbytes, numel, group_size)) {
        return false;
    }
    try {
        if (borrow) {
            matrix.blob.clear();
            matrix.borrowed_data = static_cast<const uint8_t *>(data);
            matrix.borrowed_nbytes = nbytes;
        } else {
            matrix.borrowed_data = nullptr;
            matrix.borrowed_nbytes = 0;
            matrix.blob.assign(
                static_cast<const uint8_t *>(data),
                static_cast<const uint8_t *>(data) + nbytes
            );
        }
        matrix.numel = numel;
        matrix.group_size = group_size;
        matrix.group_count = numel / static_cast<size_t>(group_size) +
            (numel % static_cast<size_t>(group_size) != 0u ? 1u : 0u);
        matrix.quant_offset = 8u + matrix.group_count * 8u;
        matrix.group_min.resize(matrix.group_count);
        matrix.group_scale.resize(matrix.group_count);
        matrix.shared_group_min.reset();
        matrix.shared_group_scale.reset();
        const uint8_t * raw = matrix.payload_data() + 8u;
        for (size_t group = 0; group < matrix.group_count; ++group) {
            float mn;
            float mx;
            memcpy(&mn, raw + group * 8u, sizeof(mn));
            memcpy(&mx, raw + group * 8u + 4u, sizeof(mx));
            matrix.group_min[group] = mn;
            matrix.group_scale[group] = mx - mn;
        }
        matrix.blob_valid = true;
    } catch (...) {
        matrix.blob.clear();
        matrix.borrowed_data = nullptr;
        matrix.borrowed_nbytes = 0;
        matrix.group_min.clear();
        matrix.group_scale.clear();
        matrix.shared_group_min.reset();
        matrix.shared_group_scale.reset();
        matrix.group_count = 0;
        matrix.quant_offset = 0;
        matrix.blob_valid = false;
        return false;
    }
    return true;
}

static bool rwkv_native_u8_assign_blob(
    rwkv_u8_grouped_matrix & matrix,
    const void * data,
    const size_t nbytes,
    const size_t numel
) {
    return rwkv_native_u8_assign_blob_impl(matrix, data, nbytes, numel, false);
}

static bool rwkv_native_u8_assign_borrowed_blob(
    rwkv_u8_grouped_matrix & matrix,
    const void * data,
    const size_t nbytes,
    const size_t numel
) {
    return rwkv_native_u8_assign_blob_impl(matrix, data, nbytes, numel, true);
}

static bool rwkv_native_u8_transposed_name(const std::string & name) {
    return name.find(".att.w1") != std::string::npos ||
           name.find(".att.w2") != std::string::npos ||
           name.find(".att.a1") != std::string::npos ||
           name.find(".att.a2") != std::string::npos ||
           name.find(".att.g1") != std::string::npos ||
           name.find(".att.g2") != std::string::npos ||
           name.find(".att.v1") != std::string::npos ||
           name.find(".att.v2") != std::string::npos;
}

static bool rwkv_native_u8_parse_blob(
    const void * data,
    const size_t nbytes,
    const size_t numel,
    uint32_t & group_size
) {
    if (!rwkv_native_u8_validate_blob_shape(data, nbytes, numel, group_size)) {
        return false;
    }

    const uint8_t * raw = static_cast<const uint8_t *>(data);
    const size_t groups = numel / static_cast<size_t>(group_size) +
        (numel % static_cast<size_t>(group_size) != 0u ? 1u : 0u);

    // Reject NaN/Inf metadata and inverted ranges before the record can
    // enter a hot GEMV.  Without this check a malformed upload could poison
    // every output while still satisfying the byte-count check above.
    for (size_t index = 0; index < groups; ++index) {
        float mn;
        float mx;
        memcpy(&mn, raw + 8u + index * 8u, sizeof(mn));
        memcpy(&mx, raw + 8u + index * 8u + 4u, sizeof(mx));
        if (!std::isfinite(mn) || !std::isfinite(mx) || mx < mn) {
            return false;
        }
    }

    return true;
}

static inline float rwkv_native_u8_value(
    const rwkv_u8_grouped_matrix & matrix,
    const size_t logical
) {
    const size_t group = logical / matrix.group_size;
    const uint8_t * quant = matrix.payload_data() + matrix.quant_offset;
    return matrix.group_min_data()[group] + matrix.group_scale_data()[group] *
        (static_cast<float>(quant[logical]) * (1.0f / 255.0f));
}

static inline const uint8_t * rwkv_native_u8_quant(
    const rwkv_u8_grouped_matrix & matrix
) {
    return matrix.payload_data() + matrix.quant_offset;
}

static inline float rwkv_native_u8_dense_value(
    const rwkv_u8_grouped_matrix & matrix,
    const size_t row,
    const size_t col
) {
    if (matrix.dense == nullptr || matrix.dense->data == nullptr) {
        return 0.0f;
    }

    const uint8_t * ptr = static_cast<const uint8_t *>(matrix.dense->data) +
        row * matrix.dense->nb[1] + col * matrix.dense->nb[0];
    switch (matrix.dense->type) {
        case GGML_TYPE_F32:
            return *reinterpret_cast<const float *>(ptr);
        case GGML_TYPE_F16:
            return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(ptr));
        case GGML_TYPE_BF16:
            return ggml_bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t *>(ptr));
        default:
            // Native U8 is enabled for FP16/BF16 skeleton-compatible GGML
            // files.  Leave an explicit zero for an unsupported quantized
            // fallback instead of reading a block as a scalar.
            return 0.0f;
    }
}

static inline float rwkv_native_u8_activation_value(
    const struct ggml_tensor * tensor,
    const uint8_t * ptr
) {
    switch (tensor->type) {
        case GGML_TYPE_F32:
            return *reinterpret_cast<const float *>(ptr);
        case GGML_TYPE_F16:
            return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(ptr));
        case GGML_TYPE_BF16:
            return ggml_bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t *>(ptr));
        default:
            return 0.0f;
    }
}

static bool rwkv_native_u8_dense_type_supported(const enum ggml_type type) {
    return type == GGML_TYPE_F32 ||
           type == GGML_TYPE_F16 ||
           type == GGML_TYPE_BF16;
}

static inline size_t rwkv_native_u8_vector_offset(
    const struct ggml_tensor * tensor,
    size_t vector
) {
    const size_t n1 = static_cast<size_t>(tensor->ne[1]);
    const size_t n2 = static_cast<size_t>(tensor->ne[2]);
    const size_t i1 = n1 ? vector % n1 : 0u;
    vector = n1 ? vector / n1 : 0u;
    const size_t i2 = n2 ? vector % n2 : 0u;
    vector = n2 ? vector / n2 : 0u;
    const size_t i3 = static_cast<size_t>(tensor->ne[3]) ?
        vector % static_cast<size_t>(tensor->ne[3]) : 0u;
    return i1 * tensor->nb[1] + i2 * tensor->nb[2] + i3 * tensor->nb[3];
}

#if RWKV_NATIVE_U8_AVX2
static inline float rwkv_native_u8_hsum256(__m256 value) {
    const __m128 low = _mm256_castps256_ps128(value);
    const __m128 high = _mm256_extractf128_ps(value, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}
#endif

// Aligned rows reuse the same activation groups. Match the direct kernel's
// reduction order, but compute each sum once per worker/vector instead of
// once per output row. No matrix weights are decoded or retained here.
static void rwkv_native_u8_activation_sums(
    const float * x,
    const size_t group_size,
    const size_t groups,
    float * sums
) {
    for (size_t group = 0; group < groups; ++group) {
        const float * input = x + group * group_size;
        float sum = 0.0f;
        size_t j = 0;
#if RWKV_NATIVE_U8_AVX2
        if ((group_size & 7u) == 0u) {
            __m256 accum = _mm256_setzero_ps();
            for (; j + 8u <= group_size; j += 8u) {
                accum = _mm256_add_ps(accum, _mm256_loadu_ps(input + j));
            }
            sum = rwkv_native_u8_hsum256(accum);
        }
#endif
        for (; j < group_size; ++j) {
            sum += input[j];
        }
        sums[group] = sum;
    }
}

static float rwkv_native_u8_dot_aligned(
    const rwkv_u8_grouped_matrix & matrix,
    const float * x,
    const float * sums,
    const size_t groups_per_row,
    const size_t row
) {
    const size_t group_size = static_cast<size_t>(matrix.group_size);
    const size_t first_group = row * groups_per_row;
    const uint8_t * quant = rwkv_native_u8_quant(matrix) + first_group * group_size;
    const float * minima = matrix.group_min_data() + first_group;
    const float * ranges = matrix.group_scale_data() + first_group;
    const float inv_255 = 1.0f / 255.0f;
    float acc = 0.0f;
    for (size_t group = 0; group < groups_per_row; ++group) {
        const size_t col = group * group_size;
        float sum_qx = 0.0f;
        size_t j = 0;
#if RWKV_NATIVE_U8_AVX2
        if ((group_size & 7u) == 0u) {
            __m256 accum = _mm256_setzero_ps();
            for (; j + 8u <= group_size; j += 8u) {
                const __m256 vx = _mm256_loadu_ps(x + col + j);
                const __m128i q8 = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i *>(quant + col + j)
                );
                const __m256 vq = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(q8));
#if defined(__FMA__) || defined(_M_FMA)
                accum = _mm256_fmadd_ps(vq, vx, accum);
#else
                accum = _mm256_add_ps(accum, _mm256_mul_ps(vq, vx));
#endif
            }
            sum_qx = rwkv_native_u8_hsum256(accum);
        }
#endif
        for (; j < group_size; ++j) {
            sum_qx += static_cast<float>(quant[col + j]) * x[col + j];
        }
        acc += minima[group] * sums[group] + ranges[group] * inv_255 * sum_qx;
    }
    return acc;
}

static float rwkv_native_u8_dot_direct(
    const rwkv_u8_grouped_matrix & matrix,
    const float * x,
    const int64_t in_features,
    const size_t row
) {
    const size_t row_base = row * static_cast<size_t>(in_features);
    const uint8_t * quant = rwkv_native_u8_quant(matrix);
    const float inv_255 = 1.0f / 255.0f;
    const size_t group_size = static_cast<size_t>(matrix.group_size);
    size_t col = 0;
    size_t logical = row_base;
    float acc = 0.0f;

    while (col < static_cast<size_t>(in_features)) {
        const size_t group = logical / group_size;
        const size_t group_end = (group + 1u) * group_size;
        const size_t take = std::min(
            group_end - logical,
            static_cast<size_t>(in_features) - col
        );

        const float mn = matrix.group_min_data()[group];
        const float range = matrix.group_scale_data()[group];

        float sum_x = 0.0f;
        float sum_qx = 0.0f;
#if RWKV_NATIVE_U8_AVX2
        size_t j = 0;
        __m256 vx_sum = _mm256_setzero_ps();
        __m256 vqx_sum = _mm256_setzero_ps();
        if ((matrix.group_size & 7u) == 0u) {
            for (; j + 8u <= take; j += 8u) {
                const __m256 vx = _mm256_loadu_ps(x + col + j);
                const __m128i q8 = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i *>(quant + logical + j)
                );
                const __m256i qi = _mm256_cvtepu8_epi32(q8);
                const __m256 vq = _mm256_cvtepi32_ps(qi);
                vx_sum = _mm256_add_ps(vx_sum, vx);
#if defined(__FMA__) || defined(_M_FMA)
                vqx_sum = _mm256_fmadd_ps(vq, vx, vqx_sum);
#else
                vqx_sum = _mm256_add_ps(vqx_sum, _mm256_mul_ps(vq, vx));
#endif
            }
            sum_x = rwkv_native_u8_hsum256(vx_sum);
            sum_qx = rwkv_native_u8_hsum256(vqx_sum);
        } else {
            j = 0;
        }
        for (; j < take; ++j) {
            const float value = x[col + j];
            sum_x += value;
            sum_qx += static_cast<float>(quant[logical + j]) * value;
        }
#else
        for (size_t j = 0; j < take; ++j) {
            const float value = x[col + j];
            sum_x += value;
            sum_qx += static_cast<float>(quant[logical + j]) * value;
        }
#endif
        acc += mn * sum_x + range * inv_255 * sum_qx;
        col += take;
        logical += take;
    }
    return acc;
}

static float rwkv_native_u8_dot_transposed(
    const rwkv_u8_grouped_matrix & matrix,
    const float * x,
    const int64_t in_features,
    const int64_t out_features,
    const size_t output
) {
    const uint8_t * quant = rwkv_native_u8_quant(matrix);
    const float inv_255 = 1.0f / 255.0f;
    float acc = 0.0f;

    // The blob is the source row-major [in_features, out_features] view.
    // This is the exact index map of the Python transposed fused kernel.
    for (int64_t row = 0; row < in_features; ++row) {
        const size_t logical = static_cast<size_t>(row) *
            static_cast<size_t>(out_features) + output;
        const size_t group = logical / static_cast<size_t>(matrix.group_size);
        const float weight = matrix.group_min_data()[group] + matrix.group_scale_data()[group] *
            (static_cast<float>(quant[logical]) * inv_255);
        acc += weight * x[row];
    }
    return acc;
}

static void rwkv_native_u8_dot_transposed_block8(
    const rwkv_u8_grouped_matrix & matrix,
    const float * x,
    const int64_t in_features,
    const int64_t out_features,
    const size_t output,
    float * result
) {
    for (size_t j = 0; j < 8u; ++j) {
        result[j] = 0.0f;
    }

    const uint8_t * quant = rwkv_native_u8_quant(matrix);
    const float inv_255 = 1.0f / 255.0f;
    const size_t group_size = static_cast<size_t>(matrix.group_size);

    for (int64_t row = 0; row < in_features; ++row) {
        const size_t logical = static_cast<size_t>(row) *
            static_cast<size_t>(out_features) + output;
        const float input = x[row];
        const size_t first_group = logical / group_size;
        const size_t last_group = (logical + 7u) / group_size;

#if RWKV_NATIVE_U8_AVX2
        if (first_group == last_group) {
            const __m128i q8 = _mm_loadl_epi64(
                reinterpret_cast<const __m128i *>(quant + logical)
            );
            const __m256i qi = _mm256_cvtepu8_epi32(q8);
            const __m256 vq = _mm256_cvtepi32_ps(qi);
            const __m256 vmin = _mm256_set1_ps(matrix.group_min_data()[first_group]);
            const __m256 vrange = _mm256_set1_ps(
                matrix.group_scale_data()[first_group] * inv_255
            );
#if defined(__FMA__) || defined(_M_FMA)
            __m256 weights = _mm256_fmadd_ps(vq, vrange, vmin);
#else
            __m256 weights = _mm256_add_ps(
                vmin, _mm256_mul_ps(vq, vrange)
            );
#endif
            weights = _mm256_mul_ps(weights, _mm256_set1_ps(input));
            __m256 accum = _mm256_loadu_ps(result);
            accum = _mm256_add_ps(accum, weights);
            _mm256_storeu_ps(result, accum);
            continue;
        }
#endif

        for (size_t j = 0; j < 8u; ++j) {
            const size_t item = logical + j;
            const size_t group = item / group_size;
            result[j] += (
                matrix.group_min_data()[group] + matrix.group_scale_data()[group] *
                    (static_cast<float>(quant[item]) * inv_255)
            ) * input;
        }
    }
}

static float rwkv_native_u8_dot_dense(
    const rwkv_u8_grouped_matrix & matrix,
    const float * x,
    const int64_t in_features,
    const size_t row
) {
    float acc = 0.0f;
    for (int64_t col = 0; col < in_features; ++col) {
        acc += rwkv_native_u8_dense_value(matrix, row, static_cast<size_t>(col)) * x[col];
    }
    return acc;
}

static void rwkv_native_u8_matmul_op(
    struct ggml_tensor * dst,
    const struct ggml_tensor * activation,
    const int ith,
    const int nth,
    void * userdata
) {
    const rwkv_u8_grouped_matrix & matrix =
        *static_cast<const rwkv_u8_grouped_matrix *>(userdata);
    const int64_t in_features = matrix.dense ? matrix.dense->ne[0] : dst->src[0]->ne[0];
    const int64_t out_features = matrix.dense ? matrix.dense->ne[1] : dst->ne[0];
    const size_t n_vectors = static_cast<size_t>(activation->ne[1]) *
        static_cast<size_t>(activation->ne[2]) *
        static_cast<size_t>(activation->ne[3]);
    const bool convert_input = activation->type != GGML_TYPE_F32 ||
        activation->nb[0] != sizeof(float);
    std::vector<float> scratch;
    if (convert_input) {
        scratch.resize(static_cast<size_t>(in_features));
    }
    // g32/g64 groups for the usual RWKV widths fit in 2 KiB of stack
    // scratch, avoiding a new heap allocation on every native GEMV task.
    float local_group_sums[512];
    std::vector<float> large_group_sums;
    float * group_sums = nullptr;
    size_t groups_per_row = 0u;
    const bool aligned_groups = matrix.blob_valid && !matrix.transpose &&
        matrix.group_size != 0u && in_features > 0 &&
        static_cast<size_t>(in_features) % matrix.group_size == 0u &&
        out_features > nth;
    if (aligned_groups) {
        groups_per_row = static_cast<size_t>(in_features) / matrix.group_size;
        if (groups_per_row <= sizeof(local_group_sums) / sizeof(local_group_sums[0])) {
            group_sums = local_group_sums;
        } else {
            try {
                large_group_sums.resize(groups_per_row);
                group_sums = large_group_sums.data();
            } catch (const std::bad_alloc &) {
                // Sum reuse is optional. Preserve the direct computation if
                // the scratch allocation cannot be satisfied.
            }
        }
    }

    // Work is distributed over output rows (or eight-row blocks for the
    // transposed adapter maps), while one activation conversion is reused by
    // all rows handled by this task.  The previous implementation constructed
    // a std::vector inside every output element, which was especially costly
    // for the many small RWKV-7 adapter GEMVs.
    for (size_t vector = 0; vector < n_vectors; ++vector) {
        const size_t input_offset = rwkv_native_u8_vector_offset(activation, vector);
        const size_t output_offset = rwkv_native_u8_vector_offset(dst, vector);
        const uint8_t * input_bytes = static_cast<const uint8_t *>(activation->data) + input_offset;
        float * output = reinterpret_cast<float *>(static_cast<uint8_t *>(dst->data) + output_offset);

        const float * input_f32 = reinterpret_cast<const float *>(input_bytes);
        if (convert_input) {
            for (int64_t col = 0; col < in_features; ++col) {
                scratch[static_cast<size_t>(col)] = rwkv_native_u8_activation_value(
                    activation,
                    input_bytes + static_cast<size_t>(col) * activation->nb[0]
                );
            }
            input_f32 = scratch.data();
        }

        if (matrix.blob_valid && matrix.transpose) {
            const size_t block_count = (
                static_cast<size_t>(out_features) + 7u
            ) / 8u;
            for (size_t block = static_cast<size_t>(ith);
                 block < block_count;
                 block += static_cast<size_t>(nth)) {
                const size_t row = block * 8u;
                if (row + 8u <= static_cast<size_t>(out_features)) {
                    float values[8];
                    rwkv_native_u8_dot_transposed_block8(
                        matrix,
                        input_f32,
                        in_features,
                        out_features,
                        row,
                        values
                    );
                    for (size_t j = 0; j < 8u; ++j) {
                        output[row + j] = values[j];
                    }
                } else {
                    for (size_t j = row;
                         j < static_cast<size_t>(out_features);
                         ++j) {
                        output[j] = rwkv_native_u8_dot_transposed(
                            matrix,
                            input_f32,
                            in_features,
                            out_features,
                            j
                        );
                    }
                }
            }
            continue;
        }

        if (group_sums != nullptr) {
            rwkv_native_u8_activation_sums(
                input_f32, matrix.group_size, groups_per_row, group_sums
            );
            // Each worker streams a contiguous weight range rather than
            // visiting every nth row of a large mmap-backed matrix.
            const size_t rows = static_cast<size_t>(out_features);
            const size_t rows_per_task = (rows + static_cast<size_t>(nth) - 1u) /
                static_cast<size_t>(nth);
            const size_t row_begin = std::min(rows, static_cast<size_t>(ith) * rows_per_task);
            const size_t row_end = std::min(rows, row_begin + rows_per_task);
            for (size_t row = row_begin; row < row_end; ++row) {
                output[row] = rwkv_native_u8_dot_aligned(
                    matrix, input_f32, group_sums, groups_per_row, row
                );
            }
            continue;
        }

        for (size_t row = static_cast<size_t>(ith);
             row < static_cast<size_t>(out_features);
             row += static_cast<size_t>(nth)) {
            const float value = matrix.blob_valid
                ? rwkv_native_u8_dot_direct(
                    matrix, input_f32, in_features, row
                )
                : rwkv_native_u8_dot_dense(
                    matrix, input_f32, in_features, row
                );
            output[row] = value;
        }
    }
}

static void rwkv_native_u8_get_rows_op(
    struct ggml_tensor * dst,
    const struct ggml_tensor * tokens,
    const int ith,
    const int nth,
    void * userdata
) {
    const rwkv_u8_grouped_matrix & matrix =
        *static_cast<const rwkv_u8_grouped_matrix *>(userdata);
    const int64_t n_embed = matrix.dense ? matrix.dense->ne[0] : dst->ne[0];
    const size_t n_tokens = static_cast<size_t>(dst->ne[1]) *
        static_cast<size_t>(dst->ne[2]) *
        static_cast<size_t>(dst->ne[3]);
    const size_t total = static_cast<size_t>(n_embed) * n_tokens;

    for (size_t index = static_cast<size_t>(ith); index < total; index += static_cast<size_t>(nth)) {
        const size_t token_index = index / static_cast<size_t>(n_embed);
        const size_t element = index % static_cast<size_t>(n_embed);
        const uint8_t * token_ptr = static_cast<const uint8_t *>(tokens->data) +
            token_index * tokens->nb[0];
        const int32_t token = *reinterpret_cast<const int32_t *>(token_ptr);
        float value = 0.0f;
        if (token >= 0 && matrix.dense != nullptr && token < matrix.dense->ne[1]) {
            if (matrix.blob_valid) {
                const size_t logical = static_cast<size_t>(token) *
                    static_cast<size_t>(n_embed) + element;
                value = rwkv_native_u8_value(matrix, logical);
            } else {
                value = rwkv_native_u8_dense_value(
                    matrix, static_cast<size_t>(token), element
                );
            }
        }
        reinterpret_cast<float *>(dst->data)[index] = value;
    }
}

static struct ggml_tensor * rwkv_native_u8_make_matmul(
    struct ggml_context * ctx,
    rwkv_u8_grouped_matrix * matrix,
    struct ggml_tensor * activation
) {
    // Reuse GGML's shape validation/output layout, then replace only the
    // compute node.  Removing matrix from src[0] keeps the dense GGML weight
    // out of the native custom node's dependency set; it remains available
    // for the explicit dense fallback and for old/non-native paths.
    struct ggml_tensor * result = ggml_mul_mat(ctx, matrix->dense, activation);
    struct ggml_map_custom1_op_params params = {
        /*.fun      =*/ rwkv_native_u8_matmul_op,
        /*.n_tasks  =*/ GGML_N_TASKS_MAX,
        /*.userdata =*/ matrix,
    };
    result->op = GGML_OP_MAP_CUSTOM1;
    result->src[0] = activation;
    result->src[1] = nullptr;
    ggml_set_op_params(result, &params, sizeof(params));
    return result;
}

static struct ggml_tensor * rwkv_native_u8_make_get_rows(
    struct ggml_context * ctx,
    rwkv_u8_grouped_matrix * matrix,
    struct ggml_tensor * tokens
) {
    struct ggml_tensor * result = ggml_get_rows(ctx, matrix->dense, tokens);
    struct ggml_map_custom1_op_params params = {
        /*.fun      =*/ rwkv_native_u8_get_rows_op,
        /*.n_tasks  =*/ GGML_N_TASKS_MAX,
        /*.userdata =*/ matrix,
    };
    result->op = GGML_OP_MAP_CUSTOM1;
    result->src[0] = tokens;
    result->src[1] = nullptr;
    ggml_set_op_params(result, &params, sizeof(params));
    return result;
}

static rwkv_model * rwkv_active_graph_model = nullptr;

struct rwkv_graph_model_scope {
    rwkv_model * previous;

    explicit rwkv_graph_model_scope(rwkv_model & model):
        previous(rwkv_active_graph_model) {
        rwkv_active_graph_model = &model;
    }

    ~rwkv_graph_model_scope() {
        rwkv_active_graph_model = previous;
    }
};

static struct ggml_tensor * rwkv_mul_mat(
    struct ggml_context * ctx,
    struct ggml_tensor * matrix,
    struct ggml_tensor * activation
) {
    rwkv_u8_grouped_matrix * layer_matrix =
        rwkv_native_u8_layer_lookup(matrix);
    if (layer_matrix != nullptr) {
        return rwkv_native_u8_make_matmul(ctx, layer_matrix, activation);
    }
    if (rwkv_active_graph_model != nullptr &&
        rwkv_active_graph_model->native_u8_enabled) {
        auto it = rwkv_active_graph_model->native_u8_by_tensor.find(matrix);
        if (it != rwkv_active_graph_model->native_u8_by_tensor.end()) {
            return rwkv_native_u8_make_matmul(ctx, it->second.get(), activation);
        }
    }
    return ggml_mul_mat(ctx, matrix, activation);
}

static struct ggml_tensor * rwkv_get_rows(
    struct ggml_context * ctx,
    struct ggml_tensor * matrix,
    struct ggml_tensor * tokens
) {
    rwkv_u8_grouped_matrix * layer_matrix =
        rwkv_native_u8_layer_lookup(matrix);
    if (layer_matrix != nullptr) {
        return rwkv_native_u8_make_get_rows(ctx, layer_matrix, tokens);
    }
    if (rwkv_active_graph_model != nullptr &&
        rwkv_active_graph_model->native_u8_enabled) {
        auto it = rwkv_active_graph_model->native_u8_by_tensor.find(matrix);
        if (it != rwkv_active_graph_model->native_u8_by_tensor.end()) {
            return rwkv_native_u8_make_get_rows(ctx, it->second.get(), tokens);
        }
    }
    return ggml_get_rows(ctx, matrix, tokens);
}

static void rwkv_native_u8_prepare_model(rwkv_model & model) {
    model.native_u8_by_tensor.clear();
    model.native_u8_by_name.clear();
    if (!model.native_u8_enabled) {
        return;
    }

    for (const auto & item : model.tensors) {
        struct ggml_tensor * tensor = item.second;
        if (tensor == nullptr || ggml_n_dims(tensor) != 2) {
            continue;
        }
        std::unique_ptr<rwkv_u8_grouped_matrix> matrix(
            new(std::nothrow) rwkv_u8_grouped_matrix()
        );
        if (!matrix) {
            continue;
        }
        matrix->dense = tensor;
        matrix->name = item.first;
        matrix->transpose = rwkv_native_u8_transposed_name(item.first);
        matrix->numel = static_cast<size_t>(ggml_nelements(tensor));
        matrix->dense_required = rwkv_native_u8_dense_control_name(item.first);
        rwkv_u8_grouped_matrix * raw = matrix.get();
        model.native_u8_by_tensor.emplace(tensor, std::move(matrix));
        model.native_u8_by_name[item.first] = raw;
    }
}

static bool rwkv_native_u8_store_tensor(
    rwkv_model & model,
    const char * name,
    const void * data,
    const size_t nbytes
) {
    if (name == nullptr || data == nullptr) {
        return false;
    }
    auto it = model.native_u8_by_name.find(name);
    if (it == model.native_u8_by_name.end() || it->second == nullptr ||
        it->second->dense == nullptr) {
        return false;
    }

    rwkv_u8_grouped_matrix & matrix = *it->second;
    const size_t numel = it->second->numel;
    uint32_t group_size = 0;
    if (!rwkv_native_u8_parse_blob(
            data, nbytes, numel, group_size)) {
        return false;
    }

    try {
        matrix.blob.assign(
            static_cast<const uint8_t *>(data),
            static_cast<const uint8_t *>(data) + nbytes
        );
        matrix.group_size = group_size;
        matrix.group_count = numel / static_cast<size_t>(group_size) +
            (numel % static_cast<size_t>(group_size) != 0u ? 1u : 0u);
        matrix.quant_offset = 8u + matrix.group_count * 8u;
        matrix.group_min.resize(matrix.group_count);
        matrix.group_scale.resize(matrix.group_count);
        const uint8_t * raw = matrix.blob.data() + 8u;
        for (size_t group = 0; group < matrix.group_count; ++group) {
            float mn;
            float mx;
            memcpy(&mn, raw + group * 8u, sizeof(mn));
            memcpy(&mx, raw + group * 8u + 4u, sizeof(mx));
            matrix.group_min[group] = mn;
            matrix.group_scale[group] = mx - mn;
        }
        matrix.blob_valid = true;
    } catch (...) {
        matrix.blob.clear();
        matrix.group_min.clear();
        matrix.group_scale.clear();
        matrix.group_count = 0;
        matrix.quant_offset = 0;
        matrix.blob_valid = false;
        return false;
    }

    // Elementwise 2-D controls (r_k, k_k, k_a and x_rwkvag) still participate in normal
    // GGML ops.  Decode them natively, so Python never materializes the
    // grouped record and old GGML elementwise kernels remain untouched.
    if (matrix.dense_required || !model.native_u8_enabled) {
        if (!rwkv_native_u8_dense_type_supported(matrix.dense->type)) {
            matrix.blob.clear();
            matrix.group_min.clear();
            matrix.group_scale.clear();
            matrix.shared_group_min.reset();
            matrix.shared_group_scale.reset();
            matrix.group_count = 0;
            matrix.quant_offset = 0;
            matrix.blob_valid = false;
            return false;
        }
        const size_t rows = static_cast<size_t>(matrix.dense->ne[1]);
        const size_t cols = static_cast<size_t>(matrix.dense->ne[0]);
        const size_t total = rows * cols;
        std::vector<uint8_t> decoded(total * ggml_type_size(matrix.dense->type));
        for (size_t row = 0; row < rows; ++row) {
            for (size_t col = 0; col < cols; ++col) {
                const size_t ggml_index = row * cols + col;
                const size_t source_index = matrix.transpose
                    ? col * rows + row
                    : ggml_index;
                const float value = rwkv_native_u8_value(matrix, source_index);
                uint8_t * ptr = decoded.data() + ggml_index * ggml_type_size(matrix.dense->type);
                switch (matrix.dense->type) {
                    case GGML_TYPE_F32:
                        *reinterpret_cast<float *>(ptr) = value;
                        break;
                    case GGML_TYPE_F16:
                        *reinterpret_cast<ggml_fp16_t *>(ptr) = ggml_fp32_to_fp16(value);
                        break;
                    case GGML_TYPE_BF16:
                        *reinterpret_cast<ggml_bf16_t *>(ptr) = ggml_fp32_to_bf16(value);
                        break;
                    default:
                        return false;
                }
            }
        }
        ggml_backend_tensor_set(matrix.dense, decoded.data(), 0, decoded.size());
    }
    return true;
}

static bool rwkv_native_u8_model_ready(const rwkv_model & model) {
    if (!model.native_u8_enabled || !model.native_u8_packed_only) {
        return true;
    }

    for (const auto & item : model.native_u8_by_name) {
        const rwkv_u8_grouped_matrix * matrix = item.second;
        if (matrix == nullptr) {
            return false;
        }
        if (matrix->dense_required) {
            if (matrix->dense == nullptr || matrix->dense->data == nullptr) {
                return false;
            }
        } else if (!matrix->blob_valid) {
            return false;
        }
    }
    return true;
}
