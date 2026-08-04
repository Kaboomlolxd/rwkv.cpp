#include "rwkv.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char * const layer_suffixes[] = {
    "ln1.weight", "ln1.bias",
    "att.time_mix_k", "att.time_mix_v", "att.time_mix_r",
    "att.time_first", "att.time_decay", "att.key.weight",
    "att.value.weight", "att.receptance.weight", "att.output.weight",
    "att.ln_x.weight", "att.ln_x.bias", "att.time_faaaa",
    "att.time_mix_g", "att.gate.weight", "att.time_maa_x",
    "att.time_maa_w", "att.time_maa_k", "att.time_maa_v",
    "att.time_maa_r", "att.time_maa_g", "att.time_maa_w1",
    "att.time_maa_w2", "att.time_decay_w1", "att.time_decay_w2",
    "att.w0", "att.w1", "att.w2", "att.a0", "att.a1", "att.a2",
    "att.g1", "att.g2", "att.v0", "att.v1", "att.v2", "att.r_k",
    "att.k_k", "att.k_a", "att.x_rwkvag", "ln2.weight", "ln2.bias",
    "ffn.time_mix_k", "ffn.time_mix_r", "ffn.time_maa_k",
    "ffn.time_maa_r", "ffn.x_k", "ffn.key.weight", "ffn.value.weight",
    "ffn.receptance.weight"
};

static void * persistent_grouped_blobs[2] = { NULL, NULL };

static int upload_zero_layer(struct rwkv_context * ctx, uint32_t layer_id) {
    char name[128];
    size_t uploaded = 0;
    if (!rwkv_layer_begin(ctx, layer_id)) {
        return 0;
    }
    for (size_t i = 0; i < sizeof(layer_suffixes) / sizeof(layer_suffixes[0]); ++i) {
        int written = snprintf(
            name, sizeof(name), "blocks.%u.%s", (unsigned int) layer_id, layer_suffixes[i]
        );
        if (written <= 0 || (size_t) written >= sizeof(name)) {
            return 0;
        }
        size_t nbytes = rwkv_get_tensor_nbytes(ctx, name);
        if (nbytes == 0) {
            continue;
        }
        void * payload = calloc(1, nbytes);
        if (payload == NULL) {
            return 0;
        }
        int ok = rwkv_layer_set_tensor(ctx, name, payload, nbytes);
        free(payload);
        if (!ok) {
            fprintf(stderr, "upload failed for %s (error=%d)\n", name, rwkv_get_last_error(ctx));
            return 0;
        }
        ++uploaded;
    }
    return uploaded > 0;
}

static int make_zero_grouped_blob(
    size_t dense_nbytes,
    void ** blob_out,
    size_t * blob_nbytes_out
) {
    const uint32_t group_size = 8u;
    if (dense_nbytes == 0 || dense_nbytes % 2u != 0u) {
        return 0;
    }
    const size_t numel = dense_nbytes / 2u;
    const size_t groups = (numel + group_size - 1u) / group_size;
    const size_t metadata = 8u + groups * 8u;
    if (groups > (SIZE_MAX - metadata) || metadata + numel < metadata) {
        return 0;
    }
    const size_t blob_nbytes = metadata + numel;
    unsigned char * blob = (unsigned char *) calloc(1, blob_nbytes);
    if (blob == NULL) {
        return 0;
    }
    memcpy(blob, "SG8\x01", 4);
    memcpy(blob + 4, &group_size, sizeof(group_size));
    *blob_out = blob;
    *blob_nbytes_out = blob_nbytes;
    return 1;
}

static int upload_grouped_zero_layer(
    struct rwkv_context * ctx,
    uint32_t layer_id,
    int mode,
    void ** borrowed_blob_out,
    size_t * borrowed_blob_nbytes_out
) {
    const char * const grouped_suffix = "att.key.weight";
    char name[128];
    char grouped_name_buf[128] = { 0 };
    const char * grouped_name = NULL;
    void * grouped_blob = NULL;
    size_t grouped_nbytes = 0;
    size_t uploaded = 0;
    if (borrowed_blob_out != NULL) {
        *borrowed_blob_out = NULL;
    }
    if (borrowed_blob_nbytes_out != NULL) {
        *borrowed_blob_nbytes_out = 0;
    }
    if (!rwkv_layer_begin(ctx, layer_id)) {
        return 0;
    }
    for (size_t i = 0; i < sizeof(layer_suffixes) / sizeof(layer_suffixes[0]); ++i) {
        int written = snprintf(
            name, sizeof(name), "blocks.%u.%s", (unsigned int) layer_id, layer_suffixes[i]
        );
        if (written <= 0 || (size_t) written >= sizeof(name)) {
            free(grouped_blob);
            return 0;
        }
        size_t nbytes = rwkv_get_tensor_nbytes(ctx, name);
        if (nbytes == 0) {
            continue;
        }
        if (strcmp(layer_suffixes[i], grouped_suffix) == 0) {
            if (!make_zero_grouped_blob(nbytes, &grouped_blob, &grouped_nbytes)) {
                return 0;
            }
            memcpy(grouped_name_buf, name, sizeof(grouped_name_buf));
            grouped_name = grouped_name_buf;
            ++uploaded;
            continue;
        }
        void * payload = calloc(1, nbytes);
        if (payload == NULL) {
            free(grouped_blob);
            return 0;
        }
        int ok = rwkv_layer_set_tensor(ctx, name, payload, nbytes);
        free(payload);
        if (!ok) {
            free(grouped_blob);
            return 0;
        }
        ++uploaded;
    }
    if (grouped_name == NULL || grouped_blob == NULL) {
        free(grouped_blob);
        return 0;
    }
    int grouped_ok = 0;
    if (mode == 0) {
        grouped_ok = rwkv_layer_set_tensor_scale_u8_grouped(
            ctx, grouped_name, grouped_blob, grouped_nbytes
        );
    } else if (mode == 1) {
        grouped_ok = rwkv_layer_set_tensor_scale_u8_grouped_borrowed(
            ctx, grouped_name, grouped_blob, grouped_nbytes
        );
    } else if (mode == 2) {
        grouped_ok = rwkv_layer_set_tensor_scale_u8_grouped_borrowed_persistent(
            ctx, grouped_name, grouped_blob, grouped_nbytes
        );
    } else if (mode == 3 || mode == 4) {
        const char * names[1] = { grouped_name };
        const void * data[1] = { grouped_blob };
        const size_t sizes[1] = { grouped_nbytes };
        grouped_ok = rwkv_layer_set_tensor_scale_u8_grouped_batch(
            ctx, names, data, sizes, 1, mode == 4
        );
    }
    if (!grouped_ok) {
        free(grouped_blob);
        return 0;
    }
    if (borrowed_blob_out != NULL && (mode == 1 || mode == 2 || mode == 3 || mode == 4)) {
        *borrowed_blob_out = grouped_blob;
        if (borrowed_blob_nbytes_out != NULL) {
            *borrowed_blob_nbytes_out = grouped_nbytes;
        }
    } else {
        free(grouped_blob);
    }
    return uploaded > 0;
}

static int run_grouped_zero_step(struct rwkv_context * ctx, int mode) {
    void * grouped_blob = NULL;
    size_t grouped_nbytes = 0;
    if (!upload_grouped_zero_layer(ctx, 0, mode, &grouped_blob, &grouped_nbytes)) {
        return 0;
    }
    float * activation_in = (float *) calloc(rwkv_get_n_embed(ctx), sizeof(float));
    float * activation_out = (float *) calloc(rwkv_get_n_embed(ctx), sizeof(float));
    float * state_in = (float *) calloc(rwkv_get_layer_state_len(ctx), sizeof(float));
    float * state_out = (float *) calloc(rwkv_get_layer_state_len(ctx), sizeof(float));
    float * v_first = (float *) calloc(rwkv_get_n_embed(ctx), sizeof(float));
    int ok = activation_in != NULL && activation_out != NULL && state_in != NULL &&
        state_out != NULL && v_first != NULL && rwkv_layer_step(
            ctx, 0, activation_in, activation_out, state_in, state_out, NULL, v_first
        );
    free(activation_in);
    free(activation_out);
    free(state_in);
    free(state_out);
    free(v_first);
    if (mode == 2) {
        persistent_grouped_blobs[0] = grouped_blob;
    } else if (mode == 4) {
        persistent_grouped_blobs[1] = grouped_blob;
    } else {
        free(grouped_blob);
    }
    (void) grouped_nbytes;
    return ok;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <rwkv7-model>\n", argv[0]);
        return 2;
    }
    struct rwkv_context * ctx = rwkv_init_from_file_ex(argv[1], 1, 0, 4);
    if (ctx == NULL) {
        fprintf(stderr, "layer-streaming model initialization failed\n");
        return 1;
    }
    int result = 1;
    float * activation_in = NULL;
    float * activation_out = NULL;
    float * state_in = NULL;
    float * state_out = NULL;
    float * v_first = NULL;
    float * activation_seq = NULL;
    float * activation_seq_out = NULL;
    float * v_first_seq = NULL;
    float * cached_activation_in = NULL;
    float * cached_activation_out = NULL;
    float * cached_state_in = NULL;
    float * cached_state_out = NULL;
    float * cached_v_first = NULL;
    float * direct_activation = NULL;
    float * direct_activation_out = NULL;
    float * direct_state_in = NULL;
    float * direct_state_out = NULL;
    float * direct_v_first = NULL;
    struct rwkv_context * u8_ctx = NULL;

    if (!rwkv_supports_layer_streaming(ctx) || rwkv_get_n_layer(ctx) < 2) {
        fprintf(stderr, "layer-streaming capability or two-layer fixture missing\n");
        goto cleanup;
    }
    if (rwkv_get_tensor_type(ctx, "emb.weight") != 1) {
        fprintf(stderr, "tensor dtype introspection did not report GGML_TYPE_F16\n");
        goto cleanup;
    }
    if (!rwkv_set_layer_cache_bytes(ctx, 64u * 1024u * 1024u)) {
        fprintf(stderr, "failed to enable bounded layer cache (error=%d)\n", rwkv_get_last_error(ctx));
        goto cleanup;
    }
    if (!upload_zero_layer(ctx, 0) || !upload_zero_layer(ctx, 1)) {
        goto cleanup;
    }
    {
        const char * borrowed_name = "blocks.0.ln1.weight";
        const size_t borrowed_nbytes = rwkv_get_tensor_nbytes(ctx, borrowed_name);
        void * borrowed_payload = NULL;
        if (borrowed_nbytes == 0 ||
            !rwkv_layer_begin(ctx, 0) ||
            (borrowed_payload = calloc(1, borrowed_nbytes)) == NULL ||
            !rwkv_layer_set_tensor_borrowed(
                ctx, borrowed_name, borrowed_payload, borrowed_nbytes
            ) ||
            !rwkv_layer_set_tensor_borrowed_persistent(
                ctx, borrowed_name, borrowed_payload, borrowed_nbytes
            ) ||
            !rwkv_layer_invalidate_borrowed_cache(ctx, 0)) {
            free(borrowed_payload);
            fprintf(stderr, "dense borrowed layer ABI failed (error=%d)\n", rwkv_get_last_error(ctx));
            goto cleanup;
        }
        free(borrowed_payload);
    }
    {
        uint64_t hits_before = rwkv_get_layer_cache_hits(ctx);
        if (!upload_zero_layer(ctx, 0) ||
            rwkv_get_layer_cache_hits(ctx) <= hits_before ||
            rwkv_get_layer_cache_used_bytes(ctx) > rwkv_get_layer_cache_bytes(ctx)) {
            fprintf(stderr, "bounded layer cache did not reuse/retain payloads\n");
            goto cleanup;
        }
    }
    for (uint32_t layer_id = 2; layer_id < rwkv_get_n_layer(ctx); ++layer_id) {
        if (!upload_zero_layer(ctx, layer_id)) {
            goto cleanup;
        }
    }
    cached_activation_in = (float *) calloc(rwkv_get_n_embed(ctx), sizeof(float));
    cached_activation_out = (float *) calloc(rwkv_get_n_embed(ctx), sizeof(float));
    cached_state_in = (float *) calloc(rwkv_get_state_len(ctx), sizeof(float));
    cached_state_out = (float *) calloc(rwkv_get_state_len(ctx), sizeof(float));
    cached_v_first = (float *) calloc(rwkv_get_n_embed(ctx), sizeof(float));
    direct_activation = (float *) calloc(rwkv_get_n_embed(ctx), sizeof(float));
    direct_activation_out = (float *) calloc(rwkv_get_n_embed(ctx), sizeof(float));
    direct_state_in = (float *) calloc(rwkv_get_state_len(ctx), sizeof(float));
    direct_state_out = (float *) calloc(rwkv_get_state_len(ctx), sizeof(float));
    direct_v_first = (float *) calloc(rwkv_get_n_embed(ctx), sizeof(float));
    if (cached_activation_in == NULL || cached_activation_out == NULL ||
        cached_state_in == NULL || cached_state_out == NULL || cached_v_first == NULL ||
        direct_activation == NULL || direct_activation_out == NULL ||
        direct_state_in == NULL || direct_state_out == NULL || direct_v_first == NULL) {
        goto cleanup;
    }
    if (!rwkv_layer_cache_ready(ctx) || !rwkv_layer_step_cached(
            ctx,
            cached_activation_in,
            cached_activation_out,
            cached_state_in,
            cached_state_out,
            NULL,
            cached_v_first)) {
        fprintf(stderr, "cached layer step failed (error=%d)\n", rwkv_get_last_error(ctx));
        goto cleanup;
    }
    {
        float * direct_current = direct_activation;
        float * direct_next = direct_activation_out;
        for (uint32_t layer_id = 0; layer_id < rwkv_get_n_layer(ctx); ++layer_id) {
            if (!rwkv_layer_begin(ctx, layer_id) || !rwkv_layer_step(
                    ctx,
                    layer_id,
                    direct_current,
                    direct_next,
                    direct_state_in + (size_t) layer_id * rwkv_get_layer_state_len(ctx),
                    direct_state_out + (size_t) layer_id * rwkv_get_layer_state_len(ctx),
                    layer_id == 0 ? NULL : direct_v_first,
                    layer_id == 0 ? direct_v_first : NULL)) {
                fprintf(stderr, "direct layer step for cached comparison failed (error=%d)\n", rwkv_get_last_error(ctx));
                goto cleanup;
            }
            float * swap = direct_current;
            direct_current = direct_next;
            direct_next = swap;
        }
        for (size_t i = 0; i < rwkv_get_n_embed(ctx); ++i) {
            if (fabsf(cached_activation_out[i] - direct_current[i]) > 1e-5f ||
                fabsf(cached_v_first[i] - direct_v_first[i]) > 1e-5f) {
                fprintf(stderr, "cached and direct activation outputs differ\n");
                goto cleanup;
            }
        }
    }
    for (size_t i = 0; i < rwkv_get_state_len(ctx); ++i) {
        if (fabsf(cached_state_out[i] - direct_state_out[i]) > 1e-5f) {
            fprintf(stderr, "cached and direct state outputs differ\n");
            goto cleanup;
        }
    }
    u8_ctx = rwkv_init_from_file_ex(argv[1], 1, 0, 5);
    if (u8_ctx == NULL || !rwkv_supports_layer_streaming(u8_ctx) ||
        !rwkv_set_layer_cache_bytes(u8_ctx, 64u * 1024u * 1024u)) {
        fprintf(stderr, "grouped-U8 context initialization failed\n");
        goto cleanup;
    }
    for (int mode = 0; mode <= 4; ++mode) {
        if (!run_grouped_zero_step(u8_ctx, mode) ||
            rwkv_get_layer_cache_used_bytes(u8_ctx) > rwkv_get_layer_cache_bytes(u8_ctx)) {
            fprintf(stderr, "grouped-U8 layer mode %d failed (error=%d)\n", mode, rwkv_get_last_error(u8_ctx));
            goto cleanup;
        }
    }
    // A persistent grouped-U8 upload may borrow stable packed bytes while
    // the provider drops its decoded layer view.  The invalidation callback
    // must preserve that packed record and restore it on the next begin;
    // otherwise a large mmap-backed pack silently falls back to a full
    // per-token re-upload sweep.
    const int ready_before_invalidate = rwkv_layer_is_ready(u8_ctx, 0);
    if (!rwkv_layer_invalidate_borrowed_cache(u8_ctx, 0) ||
        !rwkv_layer_begin(u8_ctx, 0) ||
        !run_grouped_zero_step(u8_ctx, 2) ||
        !rwkv_layer_is_ready(u8_ctx, 0)) {
        fprintf(stderr, "persistent grouped-U8 cache was lost on invalidation (before=%d, used=%llu, limit=%llu, error=%d)\n",
            ready_before_invalidate,
            (unsigned long long) rwkv_get_layer_cache_used_bytes(u8_ctx),
            (unsigned long long) rwkv_get_layer_cache_bytes(u8_ctx),
            rwkv_get_last_error(u8_ctx));
        goto cleanup;
    }
    rwkv_free(u8_ctx);
    u8_ctx = NULL;
    if (!rwkv_set_layer_cache_bytes(ctx, 0) ||
        rwkv_get_layer_cache_used_bytes(ctx) != 0) {
        fprintf(stderr, "bounded layer cache did not clear\n");
        goto cleanup;
    }
    if (!upload_zero_layer(ctx, 0) || !upload_zero_layer(ctx, 1)) {
        goto cleanup;
    }
    activation_in = (float *) calloc(rwkv_get_n_embed(ctx), sizeof(float));
    activation_out = (float *) calloc(rwkv_get_n_embed(ctx), sizeof(float));
    state_in = (float *) calloc(rwkv_get_layer_state_len(ctx), sizeof(float));
    state_out = (float *) calloc(rwkv_get_layer_state_len(ctx), sizeof(float));
    v_first = (float *) calloc(rwkv_get_n_embed(ctx), sizeof(float));
    activation_seq = (float *) calloc(rwkv_get_n_embed(ctx) * 2, sizeof(float));
    activation_seq_out = (float *) calloc(rwkv_get_n_embed(ctx) * 2, sizeof(float));
    v_first_seq = (float *) calloc(rwkv_get_n_embed(ctx) * 2, sizeof(float));
    if (activation_in == NULL || activation_out == NULL || state_in == NULL ||
        state_out == NULL || v_first == NULL || activation_seq == NULL ||
        activation_seq_out == NULL || v_first_seq == NULL) {
        goto cleanup;
    }
    if (!rwkv_layer_step(ctx, 0, activation_in, activation_out, state_in, state_out, NULL, v_first) ||
        !rwkv_layer_step(ctx, 1, activation_out, activation_in, state_out, state_in, v_first, v_first)) {
        fprintf(stderr, "layer step failed (error=%d)\n", rwkv_get_last_error(ctx));
        goto cleanup;
    }
    for (size_t i = 0; i < rwkv_get_n_embed(ctx); ++i) {
        if (!isfinite(activation_in[i]) || !isfinite(state_in[i])) {
            fprintf(stderr, "layer step produced a non-finite value\n");
            goto cleanup;
        }
    }
    if (!upload_zero_layer(ctx, 0) || !rwkv_layer_step_sequence(
            ctx, 0, activation_seq, activation_seq_out, 2, state_in, state_out,
            NULL, v_first_seq) || !upload_zero_layer(ctx, 1) ||
        !rwkv_layer_step_sequence(
            ctx, 1, activation_seq_out, activation_seq, 2, state_out, state_in,
            v_first_seq, NULL)) {
        fprintf(stderr, "layer sequence step failed (error=%d)\n", rwkv_get_last_error(ctx));
        goto cleanup;
    }
    result = 0;

cleanup:
    if (u8_ctx != NULL) {
        rwkv_free(u8_ctx);
    }
    free(persistent_grouped_blobs[0]);
    free(persistent_grouped_blobs[1]);
    free(activation_in);
    free(activation_out);
    free(state_in);
    free(state_out);
    free(v_first);
    free(activation_seq);
    free(activation_seq_out);
    free(v_first_seq);
    free(cached_activation_in);
    free(cached_activation_out);
    free(cached_state_in);
    free(cached_state_out);
    free(cached_v_first);
    free(direct_activation);
    free(direct_activation_out);
    free(direct_state_in);
    free(direct_state_out);
    free(direct_v_first);
    rwkv_free(ctx);
    return result;
}
