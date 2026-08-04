/*
 * Native layer-local execution for the CPU rwkv.cpp backend.
 *
 * The ordinary rwkv.cpp graph keeps every model tensor in a GGML weight
 * buffer.  That is the right F5/resident fast path, but it is not a bounded
 * F1-F4 implementation.  This file builds two reusable one-block graphs:
 * one for layer zero (RWKV-7 has no v_first adapter there) and one for the
 * remaining layers.  The graph descriptors are retained, while the weight
 * payloads live in one bounded backend buffer per plan and are replaced by
 * the layer upload ABI before each step.
 *
 * The public ABI deliberately keeps activations and recurrent state in
 * caller-owned FP32 buffers.  A provider can therefore read/decode one
 * layer, upload it, execute it, and reuse the same plan for the next layer.
 */

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <unordered_set>
#include <vector>

struct rwkv_layer_u8_cache_record {
    const uint8_t * borrowed_data = nullptr;
    size_t borrowed_nbytes = 0;
    size_t numel = 0;
    size_t group_count = 0;
    size_t quant_offset = 0;
    uint32_t group_size = 0;
    bool transpose = false;
    bool dense_required = false;
    std::shared_ptr<const std::vector<float>> group_min;
    std::shared_ptr<const std::vector<float>> group_scale;
    std::vector<uint8_t> dense_bytes;

    size_t metadata_bytes() const {
        const size_t min_bytes = group_min ? group_min->size() * sizeof(float) : 0;
        const size_t scale_bytes = group_scale ? group_scale->size() * sizeof(float) : 0;
        return min_bytes + scale_bytes + dense_bytes.size();
    }
};

struct rwkv_layer_dense_cache_record {
    const uint8_t * borrowed_data = nullptr;
    size_t borrowed_nbytes = 0;
};

struct rwkv_layer_plan {
    // The model pointer is metadata only.  The plan owns its GGML context,
    // graph scheduler, and active-layer weight buffer.
    rwkv_model * model = nullptr;
    ggml_backend_t backend = nullptr;
    bool first_layer = false;
    uint32_t active_layer = 0;
    bool begun = false;
    bool native_u8_cached_ready = false;

    struct ggml_context * ggml_ctx = nullptr;
    struct ggml_cgraph * cgraph = nullptr;
    ggml_backend_sched_t sched = nullptr;
    // The layer plan is allocated by the scheduler, but it is always bound to
    // one CPU backend.  Keeping this bit explicit lets the hot step bypass
    // scheduler routing after allocation while retaining the scheduler as the
    // compatibility/allocation path.
    bool direct_graph_compute = false;
    // The CPU backend's graph_compute entry point rebuilds a cplan for every
    // invocation. A streamed decode invokes the same one-block graph once per
    // layer, so retaining the backend graph plan removes that repeated
    // planning work without retaining any additional model weights.
    ggml_backend_graph_plan_t graph_plan = nullptr;
    ggml_backend_buffer_t weights = nullptr;
    struct ggml_tallocr weight_alloc{};

    rwkv_layer layer{};
    std::unordered_map<std::string, struct ggml_tensor *> tensors;
    std::vector<struct ggml_tensor *> weight_tensors;
    std::unordered_set<std::string> uploaded;
    // Native grouped-U8 descriptors are attached to the cloned tensors used
    // by this one-block plan.  They retain only the active packed layer and
    // never create a second resident copy of the model graph.
    std::unordered_map<
        struct ggml_tensor *,
        std::unique_ptr<rwkv_u8_grouped_matrix>
    > native_u8_by_tensor;
    std::unordered_map<
        uint32_t,
        std::unordered_map<std::string, rwkv_layer_u8_cache_record>
    > native_u8_layer_cache;
    // Provider-owned dense payloads are referenced, never copied.  The
    // provider bridge invalidates a layer before releasing its backing
    // tensors, so this map is metadata-only and does not change the native
    // decoded-byte budget.
    std::unordered_map<
        uint32_t,
        std::unordered_map<std::string, rwkv_layer_dense_cache_record>
    > dense_borrowed_layer_cache;
    // ggml_backend_tensor_set/get use tensor->data on the CPU backend.  Keep
    // the original one-block allocation so begin() can detach a stale
    // provider pointer before a new upload or cache restore.
    std::unordered_map<struct ggml_tensor *, void *> owned_tensor_data;
    size_t native_u8_cache_bytes = 0;
    size_t native_u8_cache_limit_bytes = 0;

    struct ggml_tensor * activation_in = nullptr;
    struct ggml_tensor * activation_out = nullptr;
    struct ggml_tensor * state_in = nullptr;
    struct ggml_tensor * state_out = nullptr;
    struct ggml_tensor * v_first_in = nullptr;
    struct ggml_tensor * v_first_out = nullptr;

    size_t n_embed = 0;
    size_t state_len = 0;

    ~rwkv_layer_plan() {
        // The scheduler may own compute buffers that reference the weight
        // tensors, so release it before the weight buffer and GGML context.
        if (graph_plan != nullptr && backend != nullptr) {
            ggml_backend_graph_plan_free(backend, graph_plan);
            graph_plan = nullptr;
        }
        if (sched != nullptr) {
            ggml_backend_sched_free(sched);
            sched = nullptr;
        }
        if (weights != nullptr) {
            ggml_backend_buffer_free(weights);
            weights = nullptr;
        }
        if (ggml_ctx != nullptr) {
            ggml_free(ggml_ctx);
            ggml_ctx = nullptr;
        }
    }

    void begin(const uint32_t layer_id) {
        for (const auto & item : owned_tensor_data) {
            if (item.first != nullptr) {
                item.first->data = item.second;
            }
        }
        active_layer = layer_id;
        begun = true;
        native_u8_cached_ready = false;
        uploaded.clear();
    }
};

static rwkv_layer_plan * rwkv_active_layer_plan_for_u8 = nullptr;

static rwkv_u8_grouped_matrix * rwkv_native_u8_layer_lookup(
    struct ggml_tensor * tensor
) {
    if (rwkv_active_layer_plan_for_u8 == nullptr || tensor == nullptr) {
        return nullptr;
    }
    auto it = rwkv_active_layer_plan_for_u8->native_u8_by_tensor.find(tensor);
    return it == rwkv_active_layer_plan_for_u8->native_u8_by_tensor.end()
        ? nullptr
        : it->second.get();
}

struct rwkv_layer_u8_scope {
    rwkv_layer_plan * previous;

    explicit rwkv_layer_u8_scope(rwkv_layer_plan & plan):
        previous(rwkv_active_layer_plan_for_u8) {
        rwkv_active_layer_plan_for_u8 = &plan;
    }

    ~rwkv_layer_u8_scope() {
        rwkv_active_layer_plan_for_u8 = previous;
    }
};

// Keep the field list in one place.  The layout mirrors rwkv_layer in
// rwkv_model_loading.inc and intentionally includes optional architecture
// fields so the same plan builder can qualify RWKV-4 through RWKV-7.
#define RWKV_LAYER_FIELDS(X) \
    X(ln1_weight) \
    X(ln1_bias) \
    X(att_time_mix_k) \
    X(att_time_mix_v) \
    X(att_time_mix_r) \
    X(att_time_first) \
    X(att_time_decay) \
    X(att_key) \
    X(att_value) \
    X(att_receptance) \
    X(att_output) \
    X(att_ln_x_weight) \
    X(att_ln_x_bias) \
    X(att_time_faaaa) \
    X(att_time_mix_g) \
    X(att_gate) \
    X(att_time_maa_x) \
    X(att_time_maa_w) \
    X(att_time_maa_k) \
    X(att_time_maa_v) \
    X(att_time_maa_r) \
    X(att_time_maa_g) \
    X(att_time_maa_w1) \
    X(att_time_maa_w2) \
    X(att_time_decay_w1) \
    X(att_time_decay_w2) \
    X(att_w0) \
    X(att_w1) \
    X(att_w2) \
    X(att_a0) \
    X(att_a1) \
    X(att_a2) \
    X(att_g1) \
    X(att_g2) \
    X(att_v0) \
    X(att_v1) \
    X(att_v2) \
    X(att_r_k) \
    X(att_k_k) \
    X(att_k_a) \
    X(att_x_rwkvag) \
    X(ln2_weight) \
    X(ln2_bias) \
    X(ffn_time_mix_k) \
    X(ffn_time_mix_r) \
    X(ffn_time_maa_k) \
    X(ffn_time_maa_r) \
    X(ffn_x_k) \
    X(ffn_key) \
    X(ffn_value) \
    X(ffn_receptance)

static std::string rwkv_layer_suffix(const struct ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return std::string();
    }
    const std::string name = ggml_get_name(tensor);
    const size_t first = name.find('.');
    if (first == std::string::npos) {
        return name;
    }
    const size_t second = name.find('.', first + 1);
    if (second == std::string::npos || second + 1 >= name.size()) {
        return name;
    }
    return name.substr(second + 1);
}

static std::string rwkv_layer_upload_suffix(const char * name) {
    if (name == nullptr) {
        return std::string();
    }
    const std::string full(name);
    if (full.compare(0, 7, "blocks.") != 0) {
        return std::string();
    }
    const size_t layer_dot = full.find('.', 7);
    if (layer_dot == std::string::npos || layer_dot + 1 >= full.size()) {
        return std::string();
    }
    return full.substr(layer_dot + 1);
}

static bool rwkv_layer_clone(
    rwkv_layer_plan & plan,
    const rwkv_layer & source
) {
    plan.layer = rwkv_layer{};

#define RWKV_CLONE_FIELD(field) \
    do { \
        if (source.field != nullptr) { \
            plan.layer.field = ggml_dup_tensor(plan.ggml_ctx, source.field); \
            if (plan.layer.field == nullptr) { \
                return false; \
            } \
            const std::string suffix = rwkv_layer_suffix(source.field); \
            if (suffix.empty() || !plan.tensors.emplace(suffix, plan.layer.field).second) { \
                return false; \
            } \
            plan.weight_tensors.push_back(plan.layer.field); \
        } \
    } while (0);

    RWKV_LAYER_FIELDS(RWKV_CLONE_FIELD);

#undef RWKV_CLONE_FIELD
    return true;
}

static bool rwkv_layer_plan_prepare_native_u8(rwkv_layer_plan & plan) {
    if (plan.model == nullptr || !plan.model->native_u8_enabled) {
        return true;
    }
    for (const auto & item : plan.tensors) {
        struct ggml_tensor * tensor = item.second;
        if (tensor == nullptr || ggml_n_dims(tensor) != 2) {
            continue;
        }
        std::unique_ptr<rwkv_u8_grouped_matrix> descriptor(
            new(std::nothrow) rwkv_u8_grouped_matrix()
        );
        if (!descriptor) {
            return false;
        }
        // The plan uses suffix names shared by every block.  Reconstruct a
        // canonical full name so the resident U8 orientation/control rules
        // apply identically to cloned layer tensors.
        const std::string full_name = "blocks.0." + item.first;
        descriptor->dense = tensor;
        descriptor->name = full_name;
        descriptor->numel = static_cast<size_t>(ggml_nelements(tensor));
        descriptor->transpose = rwkv_native_u8_transposed_name(full_name);
        descriptor->dense_required = rwkv_native_u8_dense_control_name(full_name);
        plan.native_u8_by_tensor.emplace(tensor, std::move(descriptor));
    }
    return true;
}

static size_t rwkv_layer_state_len_for_model(const rwkv_model & model) {
    if (model.arch_version_major >= 5) {
        return static_cast<size_t>(model.header.n_embed) *
            (2u + static_cast<size_t>(model.head_size));
    }
    return static_cast<size_t>(model.header.n_embed) * 5u;
}

static size_t rwkv_layer_padded_size(
    const size_t value,
    const size_t alignment
) {
    if (alignment == 0) {
        return value;
    }
    const size_t remainder = value % alignment;
    if (remainder == 0) {
        return value;
    }
    const size_t padding = alignment - remainder;
    if (value > std::numeric_limits<size_t>::max() - padding) {
        return 0;
    }
    return value + padding;
}

static bool rwkv_layer_plan_build_graph(
    rwkv_layer_plan & plan
) {
    rwkv_layer_u8_scope layer_u8_scope(plan);
    const size_t n_embed = static_cast<size_t>(plan.model->header.n_embed);
    const size_t state_len = plan.state_len;
    struct ggml_context * ctx = plan.ggml_ctx;

    plan.activation_in = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, static_cast<int64_t>(n_embed), 1
    );
    plan.activation_out = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, static_cast<int64_t>(n_embed), 1
    );
    plan.state_in = ggml_new_tensor_1d(
        ctx, GGML_TYPE_F32, static_cast<int64_t>(state_len)
    );
    plan.state_out = ggml_new_tensor_1d(
        ctx, GGML_TYPE_F32, static_cast<int64_t>(state_len)
    );
    plan.v_first_in = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, static_cast<int64_t>(n_embed), 1
    );
    plan.v_first_out = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, static_cast<int64_t>(n_embed), 1
    );
    if (plan.activation_in == nullptr || plan.activation_out == nullptr ||
        plan.state_in == nullptr || plan.state_out == nullptr ||
        plan.v_first_in == nullptr || plan.v_first_out == nullptr) {
        return false;
    }

    ggml_set_name(plan.activation_in, "layer.activation.in");
    ggml_set_name(plan.activation_out, "layer.activation.out");
    ggml_set_name(plan.state_in, "layer.state.in");
    ggml_set_name(plan.state_out, "layer.state.out");
    ggml_set_name(plan.v_first_in, "layer.v_first.in");
    ggml_set_name(plan.v_first_out, "layer.v_first.out");
    ggml_set_input(plan.activation_in);
    ggml_set_input(plan.state_in);
    ggml_set_input(plan.v_first_in);
    ggml_set_output(plan.activation_out);
    ggml_set_output(plan.state_out);
    ggml_set_output(plan.v_first_out);

    plan.cgraph = ggml_new_graph_custom(ctx, RWKV_MAX_NODES, false);
    if (plan.cgraph == nullptr) {
        return false;
    }

    rwkv_layer_state input_state{};
    rwkv_layer_state output_state{};
    rwkv_create_input_and_output_views(
        ctx,
        &input_state,
        &output_state,
        plan.state_in,
        plan.state_out,
        1,
        n_embed,
        plan.model->arch_version_major,
        plan.model->head_count,
        plan.model->head_size
    );

    struct ggml_tensor * x = plan.activation_in;
    struct ggml_tensor * v_first = nullptr;
    rwkv_computation_graph layer_graph{};
    layer_graph.cgraph = plan.cgraph;
    if (plan.model->arch_version_major == 7 && !plan.first_layer) {
        v_first = plan.v_first_in;
    }

    switch (plan.model->arch_version_major) {
        case 7:
            x = ggml_add(
                ctx,
                x,
                rwkv_att_v7(
                    ctx,
                    x,
                    v_first,
                    plan.layer,
                    input_state,
                    plan.model->head_count,
                    plan.model->head_size
                )
            );
            x = ggml_add(ctx, x, rwkv_ffn_v7(ctx, x, plan.layer, input_state));
            break;
        case 6:
            x = ggml_add(
                ctx,
                x,
                rwkv_att_v6(
                    ctx,
                    x,
                    plan.layer,
                    input_state,
                    plan.model->head_count,
                    plan.model->head_size
                )
            );
            x = ggml_add(ctx, x, rwkv_ffn_v6(ctx, x, plan.layer, input_state));
            break;
        case 5:
            x = ggml_add(
                ctx,
                x,
                rwkv_att_v5(
                    ctx,
                    x,
                    plan.layer,
                    input_state,
                    plan.model->head_count,
                    plan.model->head_size,
                    plan.model->arch_version_minor
                )
            );
            x = ggml_add(ctx, x, rwkv_ffn_v4_v5(ctx, x, plan.layer, input_state));
            break;
        case 4:
            x = ggml_add(ctx, x, rwkv_att_v4(ctx, x, plan.layer, input_state, layer_graph));
            x = ggml_add(ctx, x, rwkv_ffn_v4_v5(ctx, x, plan.layer, input_state));
            break;
        default:
            return false;
    }

    ggml_build_forward_expand(plan.cgraph, ggml_cpy(ctx, x, plan.activation_out));
    ggml_build_forward_expand(
        plan.cgraph,
        ggml_cpy(ctx, input_state.ffn_xx, output_state.ffn_xx)
    );
    ggml_build_forward_expand(
        plan.cgraph,
        ggml_cpy(ctx, input_state.att_xx, output_state.att_xx)
    );
    if (plan.model->arch_version_major >= 5) {
        ggml_build_forward_expand(
            plan.cgraph,
            ggml_cpy(ctx, input_state.att_heads, output_state.att_heads)
        );
    } else {
        ggml_build_forward_expand(
            plan.cgraph,
            ggml_cpy(ctx, input_state.att_aa, output_state.att_aa)
        );
        ggml_build_forward_expand(
            plan.cgraph,
            ggml_cpy(ctx, input_state.att_bb, output_state.att_bb)
        );
        ggml_build_forward_expand(
            plan.cgraph,
            ggml_cpy(ctx, input_state.att_pp, output_state.att_pp)
        );
    }

    if (plan.model->arch_version_major == 7) {
        if (plan.first_layer) {
            if (v_first == nullptr) {
                return false;
            }
            ggml_build_forward_expand(
                plan.cgraph,
                ggml_cpy(ctx, v_first, plan.v_first_out)
            );
        } else {
            ggml_build_forward_expand(
                plan.cgraph,
                ggml_cpy(ctx, plan.v_first_in, plan.v_first_out)
            );
        }
    } else {
        // Keep the output deterministic for callers that use the common ABI
        // for older architectures even though v_first is not part of their
        // recurrent state.
        ggml_build_forward_expand(
            plan.cgraph,
            ggml_cpy(ctx, plan.v_first_in, plan.v_first_out)
        );
    }
    return true;
}

static bool rwkv_layer_plan_allocate_weights(
    rwkv_layer_plan & plan
) {
    const size_t alignment = ggml_backend_get_alignment(plan.backend);
    size_t bytes = alignment;
    for (struct ggml_tensor * tensor : plan.weight_tensors) {
        const size_t nbytes = ggml_nbytes(tensor);
        const size_t padded = rwkv_layer_padded_size(nbytes, alignment);
        if (padded == 0 || bytes > std::numeric_limits<size_t>::max() - padded) {
            return false;
        }
        bytes += padded;
    }

    plan.weights = ggml_backend_alloc_buffer(plan.backend, bytes);
    if (plan.weights == nullptr) {
        return false;
    }
    ggml_backend_buffer_set_usage(plan.weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    plan.weight_alloc = ggml_tallocr_new(plan.weights);
    for (struct ggml_tensor * tensor : plan.weight_tensors) {
        ggml_tallocr_alloc(&plan.weight_alloc, tensor);
    }
    return true;
}

static bool rwkv_layer_plan_create(
    rwkv_model & model,
    const rwkv_layer & template_layer,
    const bool first_layer,
    std::unique_ptr<rwkv_layer_plan> & result
) {
    std::unique_ptr<rwkv_layer_plan> plan(
        new(std::nothrow) rwkv_layer_plan()
    );
    if (!plan) {
        return false;
    }
    plan->model = &model;
    plan->backend = model.backends.back();
    plan->first_layer = first_layer;
    plan->n_embed = static_cast<size_t>(model.header.n_embed);
    plan->state_len = rwkv_layer_state_len_for_model(model);
    if (plan->backend == nullptr || plan->state_len == 0) {
        return false;
    }

    plan->ggml_ctx = rwkv_init_ggml_context(rwkv_ggml_overhead(), true);
    if (plan->ggml_ctx == nullptr || !rwkv_layer_clone(*plan, template_layer) ||
        !rwkv_layer_plan_prepare_native_u8(*plan)) {
        return false;
    }
    if (!rwkv_layer_plan_build_graph(*plan) ||
        !rwkv_layer_plan_allocate_weights(*plan)) {
        return false;
    }
    for (struct ggml_tensor * tensor : plan->weight_tensors) {
        if (tensor == nullptr || tensor->data == nullptr) {
            return false;
        }
        plan->owned_tensor_data.emplace(tensor, tensor->data);
    }

    plan->sched = ggml_backend_sched_new(
        &plan->backend,
        nullptr,
        1,
        RWKV_MAX_NODES,
        false
    );
    if (plan->sched == nullptr ||
        plan->cgraph == nullptr) {
        return false;
    }
    // The one-backend layer plan still contains custom/view nodes and
    // external weight leaves.  Assign every graph object explicitly so the
    // scheduler cannot leave a leaf with buffer_id == -1 during allocation.
    for (int i = 0; i < plan->cgraph->n_nodes; ++i) {
        ggml_backend_sched_set_tensor_backend(
            plan->sched,
            plan->cgraph->nodes[i],
            plan->backend
        );
    }
    for (int i = 0; i < plan->cgraph->n_leafs; ++i) {
        ggml_backend_sched_set_tensor_backend(
            plan->sched,
            plan->cgraph->leafs[i],
            plan->backend
        );
    }
    if (!ggml_backend_sched_alloc_graph(plan->sched, plan->cgraph)) {
        return false;
    }
    // Layer-local execution is created only for the CPU path by the public
    // model loader.  Once the one-backend graph has been allocated, routing
    // through the scheduler on every block adds avoidable placement work.
    // Keep an environment escape hatch for older/custom backends while making
    // the direct path the native default.
    const char * direct_raw = std::getenv("RWKVCPP_LAYER_DIRECT_GRAPH");
    plan->direct_graph_compute = direct_raw == nullptr ||
        std::string(direct_raw) != "0";
    if (plan->direct_graph_compute && ggml_backend_is_cpu(plan->backend)) {
        plan->graph_plan = ggml_backend_graph_plan_create(
            plan->backend,
            plan->cgraph
        );
        // Keep the scheduler/direct graph compatibility path available if a
        // custom CPU backend declines reusable graph plans.
        if (plan->graph_plan == nullptr) {
            plan->direct_graph_compute = false;
        }
    }
    result = std::move(plan);
    return true;
}

static struct ggml_tensor * rwkv_layer_find_tensor(
    rwkv_layer_plan & plan,
    const char * name
) {
    const std::string suffix = rwkv_layer_upload_suffix(name);
    if (suffix.empty()) {
        return nullptr;
    }
    auto it = plan.tensors.find(suffix);
    return it == plan.tensors.end() ? nullptr : it->second;
}

static bool rwkv_layer_cache_restore(
    rwkv_context & ctx,
    rwkv_layer_plan & plan,
    const char * name,
    struct ggml_tensor * tensor,
    const size_t nbytes
);

static void rwkv_layer_plan_restore_borrowed_dense(
    rwkv_layer_plan & plan
);

static void rwkv_layer_plan_cache_u8_record(
    rwkv_layer_plan & plan,
    const char * name,
    struct ggml_tensor * tensor,
    const rwkv_u8_grouped_matrix & descriptor,
    const bool persistent
) {
    if (plan.native_u8_cache_limit_bytes == 0 ||
        name == nullptr || tensor == nullptr || !descriptor.blob_valid ||
        descriptor.borrowed_data == nullptr) {
        return;
    }
    const std::string suffix = rwkv_layer_upload_suffix(name);
    if (suffix.empty()) {
        return;
    }
    auto & layer_cache = plan.native_u8_layer_cache[plan.active_layer];
    auto old = layer_cache.find(suffix);
    if (old != layer_cache.end()) {
        const size_t old_bytes = old->second.metadata_bytes();
        plan.native_u8_cache_bytes = old_bytes > plan.native_u8_cache_bytes
            ? 0
            : plan.native_u8_cache_bytes - old_bytes;
        layer_cache.erase(old);
    }

    rwkv_layer_u8_cache_record record;
    record.borrowed_data = persistent ? descriptor.borrowed_data : nullptr;
    record.borrowed_nbytes = persistent ? descriptor.borrowed_nbytes : 0;
    record.numel = descriptor.numel;
    record.group_count = descriptor.group_count;
    record.quant_offset = descriptor.quant_offset;
    record.group_size = descriptor.group_size;
    record.transpose = descriptor.transpose;
    record.dense_required = descriptor.dense_required;
    try {
        record.group_min = descriptor.shared_group_min
            ? descriptor.shared_group_min
            : std::make_shared<const std::vector<float>>(descriptor.group_min);
        record.group_scale = descriptor.shared_group_scale
            ? descriptor.shared_group_scale
            : std::make_shared<const std::vector<float>>(descriptor.group_scale);
        if (descriptor.dense_required) {
            record.dense_bytes.resize(ggml_nbytes(tensor));
            ggml_backend_tensor_get(
                tensor,
                record.dense_bytes.data(),
                0,
                record.dense_bytes.size()
            );
        }
    } catch (...) {
        return;
    }
    const size_t bytes = record.metadata_bytes();
    if (bytes == 0 || bytes > plan.native_u8_cache_limit_bytes ||
        plan.native_u8_cache_bytes > plan.native_u8_cache_limit_bytes - bytes) {
        if (layer_cache.empty()) {
            plan.native_u8_layer_cache.erase(plan.active_layer);
        }
        return;
    }
    plan.native_u8_cache_bytes += bytes;
    layer_cache.emplace(suffix, std::move(record));
}

static void rwkv_layer_plan_restore_native_u8_cached(
    rwkv_context & ctx,
    rwkv_layer_plan & plan
) {
    rwkv_layer_plan_restore_borrowed_dense(plan);
    auto layer_it = plan.native_u8_layer_cache.find(plan.active_layer);
    if (layer_it != plan.native_u8_layer_cache.end()) {
        for (const auto & item : plan.tensors) {
            auto descriptor_it = plan.native_u8_by_tensor.find(item.second);
            if (descriptor_it == plan.native_u8_by_tensor.end()) {
                continue;
            }
            auto record_it = layer_it->second.find(item.first);
            if (record_it == layer_it->second.end()) {
                continue;
            }
            const rwkv_layer_u8_cache_record & record = record_it->second;
            if (record.borrowed_data == nullptr || !record.group_min ||
                !record.group_scale) {
                continue;
            }
            rwkv_u8_grouped_matrix & descriptor = *descriptor_it->second;
            descriptor.blob.clear();
            descriptor.borrowed_data = record.borrowed_data;
            descriptor.borrowed_nbytes = record.borrowed_nbytes;
            descriptor.numel = record.numel;
            descriptor.group_count = record.group_count;
            descriptor.quant_offset = record.quant_offset;
            descriptor.group_size = record.group_size;
            descriptor.transpose = record.transpose;
            descriptor.dense_required = record.dense_required;
            descriptor.group_min.clear();
            descriptor.group_scale.clear();
            descriptor.shared_group_min = record.group_min;
            descriptor.shared_group_scale = record.group_scale;
            descriptor.blob_valid = true;
            if (record.dense_required) {
                if (record.dense_bytes.size() != ggml_nbytes(item.second)) {
                    continue;
                }
                ggml_backend_tensor_set(
                    item.second,
                    record.dense_bytes.data(),
                    0,
                    record.dense_bytes.size()
                );
            }
            plan.uploaded.insert(item.first);
        }
    }

    // Non-matrix vectors and any dense controls not represented by a packed
    // record use the existing bounded dense cache.  This keeps the complete
    // layer-ready check honest without making U8 matrices dense again.
    for (const auto & item : plan.tensors) {
        if (plan.uploaded.find(item.first) != plan.uploaded.end()) {
            continue;
        }
        const std::string name =
            "blocks." + std::to_string(static_cast<unsigned long long>(plan.active_layer)) +
            "." + item.first;
        if (rwkv_layer_cache_restore(
                ctx,
                plan,
                name.c_str(),
                item.second,
                ggml_nbytes(item.second)
        )) {
            plan.uploaded.insert(item.first);
        }
    }
    plan.native_u8_cached_ready = plan.uploaded.size() == plan.tensors.size();
}

static bool rwkv_layer_plan_bind_u8_cached_metadata(
    rwkv_layer_plan & plan,
    const char * name,
    struct ggml_tensor * tensor,
    const void * data,
    const size_t nbytes,
    rwkv_u8_grouped_matrix & descriptor
) {
    if (name == nullptr || tensor == nullptr || data == nullptr ||
        !descriptor.blob.empty() || plan.native_u8_cache_limit_bytes == 0) {
        // ``blob`` is empty for a borrowed upload; do not replace an owned
        // descriptor with a non-owning pointer.
        return false;
    }
    const std::string suffix = rwkv_layer_upload_suffix(name);
    auto layer_it = plan.native_u8_layer_cache.find(plan.active_layer);
    if (suffix.empty() || layer_it == plan.native_u8_layer_cache.end()) {
        return false;
    }
    auto record_it = layer_it->second.find(suffix);
    if (record_it == layer_it->second.end()) {
        return false;
    }
    const rwkv_layer_u8_cache_record & record = record_it->second;
    if (!record.group_min || !record.group_scale || record.borrowed_data != nullptr) {
        return false;
    }
    uint32_t group_size = 0;
    // The record was fully validated when it entered the bounded cache.  A
    // layer rebind only needs to verify the immutable SG8 shape before
    // reusing the cached group minima/scales; rescanning every metadata float
    // here made each streamed token pay the validation cost again.
    if (!rwkv_native_u8_validate_blob_shape(
            data,
            nbytes,
            static_cast<size_t>(ggml_nelements(tensor)),
            group_size
        ) || group_size != record.group_size) {
        return false;
    }
    descriptor.blob.clear();
    descriptor.borrowed_data = static_cast<const uint8_t *>(data);
    descriptor.borrowed_nbytes = nbytes;
    descriptor.numel = record.numel;
    descriptor.group_count = record.group_count;
    descriptor.quant_offset = record.quant_offset;
    descriptor.group_size = record.group_size;
    descriptor.transpose = record.transpose;
    descriptor.dense_required = record.dense_required;
    descriptor.group_min.clear();
    descriptor.group_scale.clear();
    descriptor.shared_group_min = record.group_min;
    descriptor.shared_group_scale = record.group_scale;
    descriptor.blob_valid = true;
    return true;
}

static inline void rwkv_layer_store_scalar(
    struct ggml_tensor * tensor,
    const size_t offset,
    const float value
) {
    uint8_t * dst = static_cast<uint8_t *>(tensor->data) + offset;
    switch (tensor->type) {
        case GGML_TYPE_F32:
            *reinterpret_cast<float *>(dst) = value;
            break;
        case GGML_TYPE_F16:
            *reinterpret_cast<ggml_fp16_t *>(dst) = ggml_fp32_to_fp16(value);
            break;
        case GGML_TYPE_BF16:
            *reinterpret_cast<ggml_bf16_t *>(dst) = ggml_fp32_to_bf16(value);
            break;
        default:
            break;
    }
}

static bool rwkv_layer_decode_grouped(
    struct ggml_tensor * tensor,
    const void * data,
    const size_t nbytes,
    const bool transpose
) {
    if (tensor == nullptr || tensor->data == nullptr || ggml_n_dims(tensor) != 2 ||
        (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16 &&
         tensor->type != GGML_TYPE_BF16)) {
        return false;
    }

    const size_t numel = static_cast<size_t>(ggml_nelements(tensor));
    uint32_t group_size = 0;
    if (!rwkv_native_u8_parse_blob(data, nbytes, numel, group_size)) {
        return false;
    }

    rwkv_u8_grouped_matrix matrix;
    matrix.dense = tensor;
    matrix.numel = numel;
    matrix.group_size = group_size;
    matrix.group_count = numel / static_cast<size_t>(group_size) +
        (numel % static_cast<size_t>(group_size) != 0u ? 1u : 0u);
    matrix.quant_offset = 8u + matrix.group_count * 8u;
    try {
        matrix.blob.assign(
            static_cast<const uint8_t *>(data),
            static_cast<const uint8_t *>(data) + nbytes
        );
        matrix.group_min.resize(matrix.group_count);
        matrix.group_scale.resize(matrix.group_count);
        const uint8_t * meta = matrix.blob.data() + 8u;
        for (size_t group = 0; group < matrix.group_count; ++group) {
            float mn;
            float mx;
            memcpy(&mn, meta + group * 8u, sizeof(mn));
            memcpy(&mx, meta + group * 8u + 4u, sizeof(mx));
            matrix.group_min[group] = mn;
            matrix.group_scale[group] = mx - mn;
        }
    } catch (...) {
        return false;
    }

    const size_t n0 = static_cast<size_t>(tensor->ne[0]);
    const size_t n1 = static_cast<size_t>(tensor->ne[1]);
    // Direct records are row-major [out, in], which is the GGML physical
    // order when flattened as [in, out].  Adapter records are stored by the
    // source pack as row-major [in, out] and need the explicit index map used
    // by the resident native-U8 operator.
    for (size_t i1 = 0; i1 < n1; ++i1) {
        for (size_t i0 = 0; i0 < n0; ++i0) {
            const size_t physical = i1 * n0 + i0;
            const size_t logical = transpose ? i0 * n1 + i1 : physical;
            const size_t byte_offset = i0 * tensor->nb[0] + i1 * tensor->nb[1];
            rwkv_layer_store_scalar(
                tensor,
                byte_offset,
                rwkv_native_u8_value(matrix, logical)
            );
        }
    }
    return true;
}

static bool rwkv_layer_plan_ready(const rwkv_layer_plan & plan) {
    // Every tensor present in the template must have been uploaded after the
    // most recent layer_begin.  Optional RWKV-7 v0/v1/v2 tensors are absent
    // from the layer-zero template and therefore do not enter this set.
    return plan.begun && (
        plan.native_u8_cached_ready ||
        plan.uploaded.size() == plan.tensors.size()
    );
}

static std::string rwkv_layer_cache_key(
    const rwkv_layer_plan & plan,
    const char * name
) {
    const std::string suffix = rwkv_layer_upload_suffix(name);
    if (suffix.empty()) {
        return std::string();
    }
    return std::to_string(static_cast<unsigned long long>(plan.active_layer)) +
        ":" + suffix;
}

static void rwkv_layer_cache_remove(
    rwkv_context & ctx,
    const std::string & key
) {
    auto it = ctx.layer_cache.find(key);
    if (it != ctx.layer_cache.end()) {
        const size_t bytes = it->second.size();
        ctx.layer_cache_used_bytes = bytes > ctx.layer_cache_used_bytes
            ? 0
            : ctx.layer_cache_used_bytes - bytes;
        ctx.layer_cache.erase(it);
    }
    ctx.layer_cache_lru.erase(key);
}

static void rwkv_layer_cache_evict_to_limit(rwkv_context & ctx) {
    while (ctx.layer_cache_used_bytes > ctx.layer_cache_limit_bytes &&
        !ctx.layer_cache.empty()) {
        auto oldest = ctx.layer_cache_lru.end();
        for (auto it = ctx.layer_cache_lru.begin();
             it != ctx.layer_cache_lru.end(); ++it) {
            if (oldest == ctx.layer_cache_lru.end() || it->second < oldest->second) {
                oldest = it;
            }
        }
        if (oldest == ctx.layer_cache_lru.end()) {
            break;
        }
        const std::string key = oldest->first;
        rwkv_layer_cache_remove(ctx, key);
        ++ctx.layer_cache_evictions;
    }
}

static bool rwkv_layer_cache_restore(
    rwkv_context & ctx,
    rwkv_layer_plan & plan,
    const char * name,
    struct ggml_tensor * tensor,
    const size_t nbytes
) {
    if (ctx.layer_cache_limit_bytes == 0 || tensor == nullptr) {
        return false;
    }
    const std::string key = rwkv_layer_cache_key(plan, name);
    if (key.empty()) {
        return false;
    }
    auto it = ctx.layer_cache.find(key);
    if (it == ctx.layer_cache.end()) {
        ++ctx.layer_cache_misses;
        return false;
    }
    if (it->second.size() != nbytes) {
        rwkv_layer_cache_remove(ctx, key);
        ++ctx.layer_cache_misses;
        return false;
    }
    // The production layer-local ABI is CPU-only.  Rebinding the GGML weight
    // tensor directly to the bounded cache payload avoids copying every
    // cached layer back into the active one-block arena on every token.  Keep
    // the backend-tensor copy for any custom/accelerated backend so this
    // optimization cannot expose a host pointer to a device plan.
    if (ggml_backend_is_cpu(plan.backend)) {
        tensor->data = const_cast<uint8_t *>(it->second.data());
    } else {
        ggml_backend_tensor_set(tensor, it->second.data(), 0, nbytes);
    }
    ++ctx.layer_cache_clock;
    ctx.layer_cache_lru[key] = ctx.layer_cache_clock;
    ++ctx.layer_cache_hits;
    return true;
}

static void rwkv_layer_cache_store(
    rwkv_context & ctx,
    rwkv_layer_plan & plan,
    const char * name,
    const void * data,
    const size_t nbytes
) {
    if (ctx.layer_cache_limit_bytes == 0 || data == nullptr || nbytes == 0 ||
        nbytes > ctx.layer_cache_limit_bytes) {
        return;
    }
    const std::string key = rwkv_layer_cache_key(plan, name);
    if (key.empty()) {
        return;
    }
    rwkv_layer_cache_remove(ctx, key);
    try {
        std::vector<uint8_t> payload(
            static_cast<const uint8_t *>(data),
            static_cast<const uint8_t *>(data) + nbytes
        );
        ctx.layer_cache_used_bytes += payload.size();
        ctx.layer_cache[key] = std::move(payload);
        ++ctx.layer_cache_clock;
        ctx.layer_cache_lru[key] = ctx.layer_cache_clock;
        rwkv_layer_cache_evict_to_limit(ctx);
    } catch (...) {
        // The cache is an optimization.  An allocation failure must not turn
        // a correct bounded layer execution into a failed inference request.
        rwkv_layer_cache_remove(ctx, key);
    }
}

static void rwkv_layer_plan_restore_borrowed_dense(
    rwkv_layer_plan & plan
) {
    auto layer_it = plan.dense_borrowed_layer_cache.find(plan.active_layer);
    if (layer_it == plan.dense_borrowed_layer_cache.end()) {
        return;
    }
    for (const auto & item : plan.tensors) {
        auto record_it = layer_it->second.find(item.first);
        if (record_it == layer_it->second.end()) {
            continue;
        }
        const rwkv_layer_dense_cache_record & record = record_it->second;
        if (record.borrowed_data == nullptr ||
            record.borrowed_nbytes != ggml_nbytes(item.second)) {
            continue;
        }
        item.second->data = const_cast<uint8_t *>(record.borrowed_data);
        plan.uploaded.insert(item.first);
    }
}

static void rwkv_layer_plan_restore_cached(
    rwkv_context & ctx,
    rwkv_layer_plan & plan
) {
    if (ctx.layer_cache_limit_bytes == 0) {
        return;
    }
    const std::string prefix =
        "blocks." + std::to_string(static_cast<unsigned long long>(plan.active_layer)) + ".";
    for (const auto & item : plan.tensors) {
        if (plan.uploaded.find(item.first) != plan.uploaded.end()) {
            continue;
        }
        const std::string name = prefix + item.first;
        if (rwkv_layer_cache_restore(
                ctx,
                plan,
                name.c_str(),
                item.second,
                ggml_nbytes(item.second))) {
            plan.uploaded.insert(item.first);
        }
    }
}

static bool rwkv_layer_plan_native_u8_cache_ready(
    const rwkv_context & ctx,
    const rwkv_layer_plan & plan,
    const uint32_t layer_id
) {
    auto layer_it = plan.native_u8_layer_cache.find(layer_id);
    auto dense_layer_it = plan.dense_borrowed_layer_cache.find(layer_id);
    for (const auto & item : plan.tensors) {
        const auto descriptor_it = plan.native_u8_by_tensor.find(item.second);
        if (descriptor_it != plan.native_u8_by_tensor.end()) {
            if (layer_it == plan.native_u8_layer_cache.end()) {
                return false;
            }
            const auto record_it = layer_it->second.find(item.first);
            if (record_it == layer_it->second.end()) {
                return false;
            }
            const rwkv_layer_u8_cache_record & record = record_it->second;
            if (record.borrowed_data == nullptr ||
                record.borrowed_nbytes == 0 ||
                !record.group_min || !record.group_scale ||
                (record.dense_required &&
                 record.dense_bytes.size() != ggml_nbytes(item.second))) {
                return false;
            }
            continue;
        }

        if (dense_layer_it != plan.dense_borrowed_layer_cache.end()) {
            const auto borrowed_it = dense_layer_it->second.find(item.first);
            if (borrowed_it != dense_layer_it->second.end() &&
                borrowed_it->second.borrowed_data != nullptr &&
                borrowed_it->second.borrowed_nbytes == ggml_nbytes(item.second)) {
                continue;
            }
        }

        // Non-matrix controls can also remain in the ordinary bounded dense
        // cache.  A cached step is valid only when they can be restored
        // without asking the provider to materialize them again.
        const std::string key =
            std::to_string(static_cast<unsigned long long>(layer_id)) +
            ":" + item.first;
        auto dense_it = ctx.layer_cache.find(key);
        if (dense_it == ctx.layer_cache.end() ||
            dense_it->second.size() != ggml_nbytes(item.second)) {
            return false;
        }
    }
    return true;
}

static bool rwkv_layer_plan_run(
    rwkv_layer_plan & plan,
    const float * activation_in,
    float * activation_out,
    const float * state_in,
    float * state_out,
    const float * v_first_in,
    float * v_first_out
) {
    if (!rwkv_layer_plan_ready(plan) || activation_in == nullptr ||
        activation_out == nullptr || state_out == nullptr) {
        return false;
    }
    if (plan.model->arch_version_major == 7 &&
        ((plan.first_layer && v_first_out == nullptr) ||
         (!plan.first_layer && v_first_in == nullptr))) {
        return false;
    }

    // The public layer-local ABI is CPU-first.  Avoid entering the generic
    // backend tensor set/get wrappers for the small activation/state transfers
    // around each block when the plan is backed by ggml-cpu.  Those wrappers
    // are still used for non-CPU/custom backends, where the host pointer may
    // not be the tensor's storage address.
    const bool direct_host_io = ggml_backend_is_cpu(plan.backend);
    auto tensor_set = [direct_host_io](
        struct ggml_tensor * tensor,
        const void * data,
        const size_t offset,
        const size_t nbytes
    ) {
        if (direct_host_io && tensor != nullptr && tensor->data != nullptr) {
            std::memcpy(
                static_cast<uint8_t *>(tensor->data) + offset,
                data,
                nbytes
            );
        } else {
            ggml_backend_tensor_set(tensor, data, offset, nbytes);
        }
    };
    auto tensor_get = [direct_host_io](
        const struct ggml_tensor * tensor,
        void * data,
        const size_t offset,
        const size_t nbytes
    ) {
        if (direct_host_io && tensor != nullptr && tensor->data != nullptr) {
            std::memcpy(
                data,
                static_cast<const uint8_t *>(tensor->data) + offset,
                nbytes
            );
        } else {
            ggml_backend_tensor_get(tensor, data, offset, nbytes);
        }
    };

    tensor_set(
        plan.activation_in,
        activation_in,
        0,
        ggml_nbytes(plan.activation_in)
    );
    if (state_in != nullptr) {
        tensor_set(
            plan.state_in,
            state_in,
            0,
            ggml_nbytes(plan.state_in)
        );
    } else {
        std::vector<float> initial(plan.state_len, 0.0f);
        if (plan.model->arch_version_major < 5) {
            const size_t layer_zero = plan.n_embed * 4u;
            for (size_t index = layer_zero; index < plan.state_len; ++index) {
                initial[index] = -1e30F;
            }
        }
        tensor_set(
            plan.state_in,
            initial.data(),
            0,
            ggml_nbytes(plan.state_in)
        );
    }
    const bool uses_v_first_input = !(
        plan.model->arch_version_major == 7 && plan.first_layer
    );
    if (uses_v_first_input && v_first_in != nullptr) {
        tensor_set(
            plan.v_first_in,
            v_first_in,
            0,
            ggml_nbytes(plan.v_first_in)
        );
    } else if (uses_v_first_input) {
        std::vector<float> initial(plan.n_embed, 0.0f);
        tensor_set(
            plan.v_first_in,
            initial.data(),
            0,
            ggml_nbytes(plan.v_first_in)
        );
    }

    // Re-enter the already allocated one-backend scheduler so custom-op
    // dispatch follows the resident engine's hot path.  The graph still owns
    // only one block's weights; this changes scheduling, not residency.
    const enum ggml_status status = plan.direct_graph_compute
        ? (plan.graph_plan != nullptr
            ? ggml_backend_graph_plan_compute(plan.backend, plan.graph_plan)
            : ggml_backend_graph_compute(plan.backend, plan.cgraph))
        : ggml_backend_sched_graph_compute(plan.sched, plan.cgraph);
    if (status != GGML_STATUS_SUCCESS) {
        RWKV_MSG("rwkv layer graph compute failed (layer=%u, status=%d)\n",
            static_cast<unsigned int>(plan.active_layer), static_cast<int>(status));
        return false;
    }
    tensor_get(
        plan.activation_out,
        activation_out,
        0,
        ggml_nbytes(plan.activation_out)
    );
    tensor_get(
        plan.state_out,
        state_out,
        0,
        ggml_nbytes(plan.state_out)
    );
    if (v_first_out != nullptr) {
        tensor_get(
            plan.v_first_out,
            v_first_out,
            0,
            ggml_nbytes(plan.v_first_out)
        );
    }
    return true;
}

static bool rwkv_layer_plans_create(rwkv_context & ctx) {
    if (!ctx.model->layer_streaming || ctx.model->layers == nullptr ||
        ctx.model->header.n_layer == 0 || ctx.model->backends.empty()) {
        return false;
    }
    if (!rwkv_layer_plan_create(
            *ctx.model,
            ctx.model->layers[0],
            true,
            ctx.layer_plan_zero)) {
        return false;
    }
    if (ctx.model->header.n_layer > 1 &&
        !rwkv_layer_plan_create(
            *ctx.model,
            ctx.model->layers[1],
            false,
            ctx.layer_plan_rest)) {
        return false;
    }
    return true;
}

static rwkv_layer_plan * rwkv_layer_select_plan(
    rwkv_context & ctx,
    const uint32_t layer_id
) {
    if (layer_id == 0) {
        return ctx.layer_plan_zero.get();
    }
    return ctx.layer_plan_rest.get();
}

static bool rwkv_layer_plan_activate_cached(
    rwkv_context & ctx,
    rwkv_layer_plan & plan,
    const uint32_t layer_id
) {
    // The cached-step path already proved that every layer has a valid
    // provider-owned record.  Avoid resetting the one-block arena back to its
    // owned allocation on every token; rebind the complete cached layer
    // directly and leave no opportunity for a transient upload to overwrite
    // the pointer.  The ordinary begin path remains the recovery path when a
    // provider invalidation races a diagnostic call.
    plan.active_layer = layer_id;
    plan.begun = true;
    plan.native_u8_cached_ready = false;
    plan.uploaded.clear();
    ctx.active_layer_plan = &plan;
    if (ctx.model->native_u8_enabled) {
        rwkv_layer_plan_restore_native_u8_cached(ctx, plan);
    } else {
        rwkv_layer_plan_restore_borrowed_dense(plan);
        rwkv_layer_plan_restore_cached(ctx, plan);
    }
    return rwkv_layer_plan_ready(plan);
}

// API function.
bool rwkv_supports_layer_streaming(const struct rwkv_context * ctx) {
    return ctx != nullptr && ctx->model != nullptr &&
        ctx->model->layer_streaming && ctx->layer_plan_zero != nullptr;
}

// API function.
size_t rwkv_get_layer_state_len(const struct rwkv_context * ctx) {
    if (ctx == nullptr || ctx->model == nullptr || !ctx->model->layer_streaming) {
        return 0;
    }
    return rwkv_layer_state_len_for_model(*ctx->model);
}

// API function.
bool rwkv_layer_begin(
    struct rwkv_context * ctx,
    const uint32_t layer_id
) {
    RWKV_CTX_ASSERT_FALSE_MSG(ctx, RWKV_ERROR_ARGS, ctx != nullptr, "Context is null");
    RWKV_CTX_ASSERT_FALSE_MSG(
        ctx,
        RWKV_ERROR_UNSUPPORTED,
        rwkv_supports_layer_streaming(ctx),
        "Layer-local ABI is not enabled"
    );
    RWKV_CTX_ASSERT_FALSE_MSG(
        ctx,
        RWKV_ERROR_ARGS,
        layer_id < ctx->model->header.n_layer,
        "Layer id %u is out of range",
        static_cast<unsigned int>(layer_id)
    );
    rwkv_layer_plan * plan = rwkv_layer_select_plan(*ctx, layer_id);
    RWKV_CTX_ASSERT_FALSE_MSG(
        ctx,
        RWKV_ERROR_CTX,
        plan != nullptr,
        "No layer plan is available for layer %u",
        static_cast<unsigned int>(layer_id)
    );
    plan->begin(layer_id);
    ctx->active_layer_plan = plan;
    if (ctx->model->native_u8_enabled) {
        rwkv_layer_plan_restore_native_u8_cached(*ctx, *plan);
    } else {
        rwkv_layer_plan_restore_borrowed_dense(*plan);
        rwkv_layer_plan_restore_cached(*ctx, *plan);
    }
    return true;
}

bool rwkv_layer_is_ready(
    const struct rwkv_context * ctx,
    const uint32_t layer_id
) {
    if (!rwkv_supports_layer_streaming(ctx) ||
        layer_id >= ctx->model->header.n_layer) {
        return false;
    }
    rwkv_layer_plan * plan = rwkv_layer_select_plan(
        *const_cast<rwkv_context *>(ctx),
        layer_id
    );
    return plan != nullptr && plan->begun && plan->active_layer == layer_id &&
        rwkv_layer_plan_ready(*plan);
}

static void rwkv_layer_plan_clear_borrowed_cache(
    rwkv_layer_plan & plan,
    const uint32_t layer_id
) {
    auto dense_it = plan.dense_borrowed_layer_cache.find(layer_id);
    if (dense_it != plan.dense_borrowed_layer_cache.end()) {
        plan.dense_borrowed_layer_cache.erase(dense_it);
    }
    auto u8_it = plan.native_u8_layer_cache.find(layer_id);
    if (u8_it != plan.native_u8_layer_cache.end()) {
        // Persistent grouped-U8 records borrow stable mmap bytes and are not
        // invalidated when the provider evicts its decoded view.  They are
        // precisely the mechanism that lets a large packed model avoid
        // re-uploading gigabytes of quantized weights on every token.  Drop
        // only transient records whose payload owner was not promised to
        // outlive the layer switch.
        for (auto item = u8_it->second.begin(); item != u8_it->second.end();) {
            if (item->second.borrowed_data != nullptr) {
                ++item;
                continue;
            }
            const size_t bytes = item->second.metadata_bytes();
            plan.native_u8_cache_bytes = bytes > plan.native_u8_cache_bytes
                ? 0
                : plan.native_u8_cache_bytes - bytes;
            item = u8_it->second.erase(item);
        }
        if (u8_it->second.empty()) {
            plan.native_u8_layer_cache.erase(u8_it);
            u8_it = plan.native_u8_layer_cache.end();
        }
    }
    if (plan.active_layer != layer_id) {
        return;
    }
    for (const auto & item : plan.native_u8_by_tensor) {
        rwkv_u8_grouped_matrix & descriptor = *item.second;
        const rwkv_layer_u8_cache_record * persistent_record = nullptr;
        if (u8_it != plan.native_u8_layer_cache.end()) {
            const auto record_it = u8_it->second.find(
                item.first == nullptr ? std::string() : rwkv_layer_suffix(item.first)
            );
            if (record_it != u8_it->second.end()) {
                persistent_record = &record_it->second;
            }
        }
        if (persistent_record != nullptr &&
            persistent_record->borrowed_data != nullptr &&
            persistent_record->group_min && persistent_record->group_scale) {
            // Rebind the descriptor to its still-valid packed pointer while
            // retaining the cached affine metadata.  This also covers an
            // eviction callback that arrived while this plan was active.
            const rwkv_layer_u8_cache_record & record = *persistent_record;
            descriptor.blob.clear();
            descriptor.borrowed_data = record.borrowed_data;
            descriptor.borrowed_nbytes = record.borrowed_nbytes;
            descriptor.numel = record.numel;
            descriptor.group_count = record.group_count;
            descriptor.quant_offset = record.quant_offset;
            descriptor.group_size = record.group_size;
            descriptor.transpose = record.transpose;
            descriptor.dense_required = record.dense_required;
            descriptor.group_min.clear();
            descriptor.group_scale.clear();
            descriptor.shared_group_min = record.group_min;
            descriptor.shared_group_scale = record.group_scale;
            descriptor.blob_valid = true;
            continue;
        }
        descriptor.blob.clear();
        descriptor.borrowed_data = nullptr;
        descriptor.borrowed_nbytes = 0;
        descriptor.group_min.clear();
        descriptor.group_scale.clear();
        descriptor.shared_group_min.reset();
        descriptor.shared_group_scale.reset();
        descriptor.group_count = 0;
        descriptor.quant_offset = 0;
        descriptor.blob_valid = false;
    }
    for (const auto & item : plan.owned_tensor_data) {
        if (item.first != nullptr) {
            item.first->data = item.second;
        }
    }
    plan.uploaded.clear();
    if (u8_it != plan.native_u8_layer_cache.end()) {
        for (const auto & item : u8_it->second) {
            if (item.second.borrowed_data != nullptr) {
                plan.uploaded.insert(item.first);
            }
        }
    }
    plan.native_u8_cached_ready = false;
}

bool rwkv_layer_invalidate_borrowed_cache(
    struct rwkv_context * ctx,
    const uint32_t layer_id
) {
    RWKV_CTX_ASSERT_FALSE_MSG(ctx, RWKV_ERROR_ARGS, ctx != nullptr, "Context is null");
    RWKV_CTX_ASSERT_FALSE_MSG(
        ctx,
        RWKV_ERROR_UNSUPPORTED,
        rwkv_supports_layer_streaming(ctx),
        "Layer-local ABI is not enabled"
    );
    RWKV_CTX_ASSERT_FALSE_MSG(
        ctx,
        RWKV_ERROR_ARGS,
        layer_id < ctx->model->header.n_layer,
        "Layer id %u is out of range",
        static_cast<unsigned int>(layer_id)
    );
    rwkv_layer_plan * plan = rwkv_layer_select_plan(*ctx, layer_id);
    RWKV_CTX_ASSERT_FALSE_MSG(
        ctx,
        RWKV_ERROR_CTX,
        plan != nullptr,
        "No layer plan is available for layer %u",
        static_cast<unsigned int>(layer_id)
    );
    rwkv_layer_plan_clear_borrowed_cache(*plan, layer_id);
    ctx->layer_cache_ready_cached = false;
    return true;
}

static bool rwkv_layer_cache_has_all_layers(const rwkv_context & ctx) {
    if (!ctx.model->layer_streaming || ctx.model->header.n_layer == 0 ||
        ctx.layer_plan_zero == nullptr) {
        return false;
    }
    for (uint32_t layer_id = 0; layer_id < ctx.model->header.n_layer; ++layer_id) {
        const rwkv_layer_plan * plan = layer_id == 0
            ? ctx.layer_plan_zero.get()
            : ctx.layer_plan_rest.get();
        if (plan == nullptr) {
            return false;
        }
        if (rwkv_layer_plan_native_u8_cache_ready(ctx, *plan, layer_id)) {
            continue;
        }
        for (const auto & item : plan->tensors) {
            const std::string key =
                std::to_string(static_cast<unsigned long long>(layer_id)) +
                ":" + item.first;
            if (ctx.layer_cache.find(key) == ctx.layer_cache.end()) {
                return false;
            }
        }
    }
    return true;
}

bool rwkv_layer_cache_ready(const struct rwkv_context * ctx) {
    if (ctx == nullptr || !rwkv_supports_layer_streaming(ctx)) {
        return false;
    }
    if (ctx->layer_cache_ready_cached) {
        return true;
    }
    const bool ready = rwkv_layer_cache_has_all_layers(*ctx);
    if (ready) {
        const_cast<rwkv_context *>(ctx)->layer_cache_ready_cached = true;
    }
    return ready;
}

bool rwkv_layer_step_cached(
    struct rwkv_context * ctx,
    const float * activation_in,
    float * activation_out,
    const float * state_in,
    float * state_out,
    const float * v_first_in,
    float * v_first_out
) {
    RWKV_CTX_ASSERT_FALSE_MSG(ctx, RWKV_ERROR_ARGS, ctx != nullptr, "Context is null");
    RWKV_CTX_ASSERT_FALSE_MSG(
        ctx,
        RWKV_ERROR_UNSUPPORTED,
        rwkv_supports_layer_streaming(ctx),
        "Layer-local ABI is not enabled"
    );
    RWKV_CTX_ASSERT_FALSE_MSG(
        ctx,
        RWKV_ERROR_DATA,
        rwkv_layer_cache_has_all_layers(*ctx),
        "Native layer cache does not contain every model layer"
    );
    RWKV_CTX_ASSERT_FALSE_MSG(
        ctx,
        RWKV_ERROR_ARGS,
        activation_in != nullptr && activation_out != nullptr &&
            state_in != nullptr && state_out != nullptr,
        "Cached layer activation/state buffers are invalid"
    );

    const size_t n_embed = static_cast<size_t>(ctx->model->header.n_embed);
    const size_t state_len = rwkv_get_layer_state_len(ctx);
    try {
        if (ctx->cached_activation_a.size() != n_embed) {
            ctx->cached_activation_a.resize(n_embed);
            ctx->cached_activation_b.resize(n_embed);
            ctx->cached_v_first_a.resize(n_embed);
            ctx->cached_v_first_b.resize(n_embed);
        }
    } catch (...) {
        RWKV_CTX_ASSERT_FALSE_MSG(
            ctx,
            RWKV_ERROR_ALLOC,
            false,
            "Failed to allocate cached layer-step scratch"
        );
    }
    std::vector<float> & current_activation = ctx->cached_activation_a;
    std::vector<float> & next_activation = ctx->cached_activation_b;
    std::vector<float> & current_v_first = ctx->cached_v_first_a;
    std::vector<float> & next_v_first = ctx->cached_v_first_b;
    std::memcpy(current_activation.data(), activation_in, n_embed * sizeof(float));
    std::fill(current_v_first.begin(), current_v_first.end(), 0.0F);
    std::fill(next_v_first.begin(), next_v_first.end(), 0.0F);
    if (v_first_in != nullptr) {
        std::memcpy(current_v_first.data(), v_first_in, n_embed * sizeof(float));
    }

    for (uint32_t layer_id = 0; layer_id < ctx->model->header.n_layer; ++layer_id) {
        rwkv_layer_plan * plan = rwkv_layer_select_plan(*ctx, layer_id);
        RWKV_CTX_ASSERT_FALSE_MSG(
            ctx,
            RWKV_ERROR_CTX,
            plan != nullptr && rwkv_layer_plan_ready(*plan),
            "Cached layer %u is not ready",
            static_cast<unsigned int>(layer_id)
        );
        RWKV_CTX_ASSERT_FALSE_MSG(
            ctx,
            RWKV_ERROR_DATA,
            rwkv_layer_plan_activate_cached(*ctx, *plan, layer_id),
            "Failed to activate cached layer %u",
            static_cast<unsigned int>(layer_id)
        );
        RWKV_CTX_ASSERT_FALSE_MSG(
            ctx,
            RWKV_ERROR_DATA,
            rwkv_layer_plan_run(
                *plan,
                current_activation.data(),
                next_activation.data(),
                state_in + static_cast<size_t>(layer_id) * state_len,
                state_out + static_cast<size_t>(layer_id) * state_len,
                layer_id == 0 ? (v_first_in == nullptr ? nullptr : current_v_first.data())
                              : current_v_first.data(),
                layer_id == 0 ? next_v_first.data() : nullptr
            ),
            "Cached layer %u execution failed",
            static_cast<unsigned int>(layer_id)
        );
        current_activation.swap(next_activation);
        if (ctx->model->arch_version_major == 7 && layer_id == 0) {
            current_v_first.swap(next_v_first);
        }
    }

    std::memcpy(activation_out, current_activation.data(), n_embed * sizeof(float));
    if (v_first_out != nullptr && ctx->model->arch_version_major == 7) {
        std::memcpy(v_first_out, current_v_first.data(), n_embed * sizeof(float));
    }
    return true;
}

// API function.
bool rwkv_layer_set_tensor(
    struct rwkv_context * ctx,
    const char * name,
    const void * data,
    const size_t nbytes
) {
    RWKV_CTX_ASSERT_FALSE_MSG(ctx, RWKV_ERROR_ARGS, ctx != nullptr, "Context is null");
    ctx->layer_cache_ready_cached = false;
    RWKV_CTX_ASSERT_FALSE_MSG(ctx, RWKV_ERROR_ARGS, name != nullptr, "Tensor name is null");
    RWKV_CTX_ASSERT_FALSE_MSG(ctx, RWKV_ERROR_ARGS, data != nullptr, "Tensor data is null");
    RWKV_CTX_ASSERT_FALSE_MSG(
        ctx,
        RWKV_ERROR_UNSUPPORTED,
        rwkv_supports_layer_streaming(ctx),
        "Layer-local ABI is not enabled"
    );
    // Uploads are associated with the most recently begun plan.  The two
    // plans cannot otherwise be distinguished from a tensor name alone.
    rwkv_layer_plan * plan = ctx->active_layer_plan;
    RWKV_CTX_ASSERT_FALSE_MSG(ctx, RWKV_ERROR_CTX, plan != nullptr && plan->begun, "Call rwkv_layer_begin before upload");
    struct ggml_tensor * tensor = rwkv_layer_find_tensor(*plan, name);
    RWKV_CTX_ASSERT_FALSE_MSG(ctx, RWKV_ERROR_KEY, tensor != nullptr, "Unknown layer tensor %s", name);
    const std::string suffix = rwkv_layer_upload_suffix(name);
    if (plan->uploaded.find(suffix) != plan->uploaded.end()) {
        return true;
    }
    RWKV_CTX_ASSERT_FALSE_MSG(
        ctx,
        RWKV_ERROR_DATA,
        nbytes == ggml_nbytes(tensor),
        "Invalid byte count for layer tensor %s",
        name
    );
    if (!ctx->model->native_u8_enabled &&
        rwkv_layer_cache_restore(*ctx, *plan, name, tensor, nbytes)) {
        plan->uploaded.insert(suffix);
        return true;
    }
    ggml_backend_tensor_set(tensor, data, 0, nbytes);
    auto descriptor = plan->native_u8_by_tensor.find(tensor);
    if (descriptor != plan->native_u8_by_tensor.end()) {
        descriptor->second->blob.clear();
        descriptor->second->borrowed_data = nullptr;
        descriptor->second->borrowed_nbytes = 0;
        descriptor->second->group_min.clear();
        descriptor->second->group_scale.clear();
        descriptor->second->shared_group_min.reset();
        descriptor->second->shared_group_scale.reset();
        descriptor->second->group_count = 0;
        descriptor->second->quant_offset = 0;
        descriptor->second->blob_valid = false;
    }
    if (!ctx->model->native_u8_enabled ||
        descriptor == plan->native_u8_by_tensor.end()) {
        rwkv_layer_cache_store(*ctx, *plan, name, data, nbytes);
    }
    plan->uploaded.insert(suffix);
    return true;
}

static bool rwkv_layer_set_tensor_borrowed_impl(
    struct rwkv_context * ctx,
    const char * name,
    const void * data,
    const size_t nbytes,
    const bool persistent
) {
    RWKV_CTX_ASSERT_FALSE_MSG(ctx, RWKV_ERROR_ARGS, ctx != nullptr, "Context is null");
    ctx->layer_cache_ready_cached = false;
    RWKV_CTX_ASSERT_FALSE_MSG(ctx, RWKV_ERROR_ARGS, name != nullptr, "Tensor name is null");
    RWKV_CTX_ASSERT_FALSE_MSG(ctx, RWKV_ERROR_ARGS, data != nullptr, "Tensor data is null");
    RWKV_CTX_ASSERT_FALSE_MSG(
        ctx,
        RWKV_ERROR_UNSUPPORTED,
        rwkv_supports_layer_streaming(ctx),
        "Layer-local ABI is not enabled"
    );
    rwkv_layer_plan * plan = ctx->active_layer_plan;
    RWKV_CTX_ASSERT_FALSE_MSG(
        ctx,
        RWKV_ERROR_CTX,
        plan != nullptr && plan->begun,
        "Call rwkv_layer_begin before upload"
    );
    struct ggml_tensor * tensor = rwkv_layer_find_tensor(*plan, name);
    RWKV_CTX_ASSERT_FALSE_MSG(ctx, RWKV_ERROR_KEY, tensor != nullptr, "Unknown layer tensor %s", name);
    // A grouped-U8 matrix has a shape-only dense tensor in packed-only mode;
    // binding raw bytes to it would reinterpret the packed record as FP16 or
    // BF16.  Keep the representations explicit at the ABI boundary.
    RWKV_CTX_ASSERT_FALSE_MSG(
        ctx,
        RWKV_ERROR_DATA,
        plan->native_u8_by_tensor.find(tensor) == plan->native_u8_by_tensor.end(),
        "Dense borrowed upload cannot target a grouped-U8 tensor %s",
        name
    );
    RWKV_CTX_ASSERT_FALSE_MSG(
        ctx,
        RWKV_ERROR_DATA,
        nbytes == ggml_nbytes(tensor),
        "Invalid byte count for dense borrowed tensor %s",
        name
    );

    tensor->data = const_cast<void *>(data);
    const std::string suffix = rwkv_layer_upload_suffix(name);
    RWKV_CTX_ASSERT_FALSE_MSG(
        ctx,
        RWKV_ERROR_DATA,
        !suffix.empty(),
        "Invalid layer tensor name %s",
        name
    );
    if (persistent) {
        rwkv_layer_dense_cache_record record;
        record.borrowed_data = static_cast<const uint8_t *>(data);
        record.borrowed_nbytes = nbytes;
        auto & layer_cache = plan->dense_borrowed_layer_cache[plan->active_layer];
        layer_cache[suffix] = record;
    } else {
        auto layer_it = plan->dense_borrowed_layer_cache.find(plan->active_layer);
        if (layer_it != plan->dense_borrowed_layer_cache.end()) {
            layer_it->second.erase(suffix);
            if (layer_it->second.empty()) {
                plan->dense_borrowed_layer_cache.erase(layer_it);
            }
        }
    }
    plan->uploaded.insert(suffix);
    return true;
}

bool rwkv_layer_set_tensor_borrowed(
    struct rwkv_context * ctx,
    const char * name,
    const void * data,
    const size_t nbytes
) {
    return rwkv_layer_set_tensor_borrowed_impl(
        ctx, name, data, nbytes, false
    );
}

bool rwkv_layer_set_tensor_borrowed_persistent(
    struct rwkv_context * ctx,
    const char * name,
    const void * data,
    const size_t nbytes
) {
    return rwkv_layer_set_tensor_borrowed_impl(
        ctx, name, data, nbytes, true
    );
}

static bool rwkv_layer_set_tensor_scale_u8_grouped_impl(
    struct rwkv_context * ctx,
    const char * name,
    const void * data,
    const size_t nbytes,
    const bool borrow,
    const bool persistent
) {
    RWKV_CTX_ASSERT_FALSE_MSG(ctx, RWKV_ERROR_ARGS, ctx != nullptr, "Context is null");
    ctx->layer_cache_ready_cached = false;
    RWKV_CTX_ASSERT_FALSE_MSG(ctx, RWKV_ERROR_ARGS, name != nullptr, "Tensor name is null");
    RWKV_CTX_ASSERT_FALSE_MSG(ctx, RWKV_ERROR_ARGS, data != nullptr, "Tensor data is null");
    RWKV_CTX_ASSERT_FALSE_MSG(
        ctx,
        RWKV_ERROR_UNSUPPORTED,
        rwkv_supports_layer_streaming(ctx),
        "Layer-local ABI is not enabled"
    );
    // Keep grouped-U8 uploads on the same explicit plan as dense uploads.
    rwkv_layer_plan * plan = ctx->active_layer_plan;
    RWKV_CTX_ASSERT_FALSE_MSG(ctx, RWKV_ERROR_CTX, plan != nullptr && plan->begun, "Call rwkv_layer_begin before upload");
    struct ggml_tensor * tensor = rwkv_layer_find_tensor(*plan, name);
    RWKV_CTX_ASSERT_FALSE_MSG(ctx, RWKV_ERROR_KEY, tensor != nullptr, "Unknown layer tensor %s", name);
    const std::string suffix = rwkv_layer_upload_suffix(name);
    if (!borrow && plan->uploaded.find(suffix) != plan->uploaded.end()) {
        return true;
    }
    if (!ctx->model->native_u8_enabled &&
        rwkv_layer_cache_restore(*ctx, *plan, name, tensor, ggml_nbytes(tensor))) {
        plan->uploaded.insert(suffix);
        return true;
    }
    auto descriptor = plan->native_u8_by_tensor.find(tensor);
    if (descriptor != plan->native_u8_by_tensor.end()) {
        const bool reused_metadata = borrow &&
            rwkv_layer_plan_bind_u8_cached_metadata(
                *plan,
                name,
                tensor,
                data,
                nbytes,
                *descriptor->second
            );
        RWKV_CTX_ASSERT_FALSE_MSG(
            ctx,
            RWKV_ERROR_DATA,
            reused_metadata || (
                borrow
                ? rwkv_native_u8_assign_borrowed_blob(
                    *descriptor->second,
                    data,
                    nbytes,
                    static_cast<size_t>(ggml_nelements(tensor))
                )
                : rwkv_native_u8_assign_blob(
                    *descriptor->second,
                    data,
                    nbytes,
                    static_cast<size_t>(ggml_nelements(tensor))
                )
            ),
            "Invalid grouped-U8 layer tensor %s",
            name
        );
        if (descriptor->second->dense_required) {
            RWKV_CTX_ASSERT_FALSE_MSG(
                ctx,
                RWKV_ERROR_DATA,
                rwkv_layer_decode_grouped(
                    tensor,
                    data,
                    nbytes,
                    descriptor->second->transpose
                ),
                "Invalid grouped-U8 control tensor %s",
                name
            );
        }
        // A non-persistent borrowed upload only changes the payload pointer;
        // the validated shape, group count, affine metadata, and transpose
        // decision are immutable for this tensor.  Rebuilding the bounded
        // metadata record on every autoregressive revisit used to allocate,
        // copy, and discard the same vectors once per layer/token.  Keep the
        // existing record when the fast rebind succeeded; persistent uploads
        // still refresh ownership so an explicit provider cache can retain
        // the payload for the cached-step path.
        if (!reused_metadata || persistent) {
            rwkv_layer_plan_cache_u8_record(
                *plan,
                name,
                tensor,
                *descriptor->second,
                persistent
            );
        }
        // Matrix payloads stay packed in the active descriptor.  In
        // particular, do not cache a dense tensor-sized copy: that was the
        // source of the per-token 181 MiB restore/upload cost in F1-F4.
        plan->uploaded.insert(suffix);
        return true;
    }
    RWKV_CTX_ASSERT_FALSE_MSG(
        ctx,
        RWKV_ERROR_DATA,
        rwkv_layer_decode_grouped(
            tensor,
            data,
            nbytes,
        rwkv_native_u8_transposed_name(name)
        ),
        "Invalid grouped-U8 layer tensor %s",
        name
    );
    if (!ctx->model->native_u8_enabled) {
        rwkv_layer_cache_store(
            *ctx,
            *plan,
            name,
            tensor->data,
            ggml_nbytes(tensor)
        );
    }
    plan->uploaded.insert(suffix);
    return true;
}

// API function.
bool rwkv_layer_set_tensor_scale_u8_grouped(
    struct rwkv_context * ctx,
    const char * name,
    const void * data,
    const size_t nbytes
) {
    return rwkv_layer_set_tensor_scale_u8_grouped_impl(
        ctx, name, data, nbytes, false, false
    );
}

// API function.  The caller retains ownership of ``data`` until the next
// rwkv_layer_begin call.  The Python binding holds the corresponding
// memoryview on the model, so this path avoids copying a packed layer into a
// second native vector while preserving bounded active-layer residency.
bool rwkv_layer_set_tensor_scale_u8_grouped_borrowed(
    struct rwkv_context * ctx,
    const char * name,
    const void * data,
    const size_t nbytes
) {
    return rwkv_layer_set_tensor_scale_u8_grouped_impl(
        ctx, name, data, nbytes, true, false
    );
}

// API function.  The caller promises that the packed bytes remain valid for
// the lifetime of the context.  The Python bridge uses this only when its
// provider cache already owns the corresponding layer views.
bool rwkv_layer_set_tensor_scale_u8_grouped_borrowed_persistent(
    struct rwkv_context * ctx,
    const char * name,
    const void * data,
    const size_t nbytes
) {
    return rwkv_layer_set_tensor_scale_u8_grouped_impl(
        ctx, name, data, nbytes, true, true
    );
}

// API function.  The batch ABI keeps the public validation semantics of the
// scalar upload while removing one Python/ctypes transition per tensor.
bool rwkv_layer_set_tensor_scale_u8_grouped_batch(
    struct rwkv_context * ctx,
    const char * const * names,
    const void * const * data,
    const size_t * nbytes,
    const size_t count,
    const bool persistent
) {
    RWKV_CTX_ASSERT_FALSE_MSG(ctx, RWKV_ERROR_ARGS, ctx != nullptr, "Context is null");
    RWKV_CTX_ASSERT_FALSE_MSG(
        ctx,
        RWKV_ERROR_ARGS,
        names != nullptr && data != nullptr && nbytes != nullptr && count > 0,
        "Grouped-U8 batch upload arguments are invalid"
    );
    for (size_t index = 0; index < count; ++index) {
        RWKV_CTX_ASSERT_FALSE_MSG(
            ctx,
            RWKV_ERROR_ARGS,
            names[index] != nullptr && data[index] != nullptr,
            "Grouped-U8 batch item %llu is invalid",
            static_cast<unsigned long long>(index)
        );
        RWKV_CTX_ASSERT_FALSE_MSG(
            ctx,
            RWKV_ERROR_DATA,
            rwkv_layer_set_tensor_scale_u8_grouped_impl(
                ctx,
                names[index],
                data[index],
                nbytes[index],
                true,
                persistent
            ),
            "Grouped-U8 batch item %llu failed",
            static_cast<unsigned long long>(index)
        );
    }
    return true;
}

bool rwkv_set_layer_cache_bytes(
    struct rwkv_context * ctx,
    const size_t bytes
) {
    RWKV_CTX_ASSERT_FALSE_MSG(ctx, RWKV_ERROR_ARGS, ctx != nullptr, "Context is null");
    ctx->layer_cache_ready_cached = false;
    RWKV_CTX_ASSERT_FALSE_MSG(
        ctx,
        RWKV_ERROR_UNSUPPORTED,
        rwkv_supports_layer_streaming(ctx),
        "Layer-local ABI is not enabled"
    );
    ctx->layer_cache_limit_bytes = bytes;
    if (ctx->layer_plan_zero != nullptr) {
        ctx->layer_plan_zero->native_u8_cache_limit_bytes = bytes;
        if (bytes == 0 ||
            ctx->layer_plan_zero->native_u8_cache_bytes > bytes) {
            ctx->layer_plan_zero->native_u8_layer_cache.clear();
            ctx->layer_plan_zero->native_u8_cache_bytes = 0;
        }
    }
    if (ctx->layer_plan_rest != nullptr) {
        ctx->layer_plan_rest->native_u8_cache_limit_bytes = bytes;
        if (bytes == 0 ||
            ctx->layer_plan_rest->native_u8_cache_bytes > bytes) {
            ctx->layer_plan_rest->native_u8_layer_cache.clear();
            ctx->layer_plan_rest->native_u8_cache_bytes = 0;
        }
    }
    if (bytes == 0) {
        ctx->layer_cache.clear();
        ctx->layer_cache_lru.clear();
        ctx->layer_cache_used_bytes = 0;
    } else {
        rwkv_layer_cache_evict_to_limit(*ctx);
    }
    return true;
}

size_t rwkv_get_layer_cache_bytes(const struct rwkv_context * ctx) {
    return ctx == nullptr ? 0 : ctx->layer_cache_limit_bytes;
}

size_t rwkv_get_layer_cache_used_bytes(const struct rwkv_context * ctx) {
    return ctx == nullptr ? 0 : ctx->layer_cache_used_bytes;
}

uint64_t rwkv_get_layer_cache_hits(const struct rwkv_context * ctx) {
    return ctx == nullptr ? 0 : ctx->layer_cache_hits;
}

uint64_t rwkv_get_layer_cache_misses(const struct rwkv_context * ctx) {
    return ctx == nullptr ? 0 : ctx->layer_cache_misses;
}

uint64_t rwkv_get_layer_cache_evictions(const struct rwkv_context * ctx) {
    return ctx == nullptr ? 0 : ctx->layer_cache_evictions;
}

// API function.
bool rwkv_layer_step(
    struct rwkv_context * ctx,
    const uint32_t layer_id,
    const float * activation_in,
    float * activation_out,
    const float * state_in,
    float * state_out,
    const float * v_first_in,
    float * v_first_out
) {
    RWKV_CTX_ASSERT_FALSE_MSG(ctx, RWKV_ERROR_ARGS, ctx != nullptr, "Context is null");
    RWKV_CTX_ASSERT_FALSE_MSG(
        ctx,
        RWKV_ERROR_UNSUPPORTED,
        rwkv_supports_layer_streaming(ctx),
        "Layer-local ABI is not enabled"
    );
    RWKV_CTX_ASSERT_FALSE_MSG(
        ctx,
        RWKV_ERROR_ARGS,
        layer_id < ctx->model->header.n_layer,
        "Layer id is out of range"
    );
    rwkv_layer_plan * plan = rwkv_layer_select_plan(*ctx, layer_id);
    RWKV_CTX_ASSERT_FALSE_MSG(ctx, RWKV_ERROR_CTX, plan != nullptr, "No layer plan is available");
    RWKV_CTX_ASSERT_FALSE_MSG(
        ctx,
        RWKV_ERROR_DATA,
        plan->begun && plan->active_layer == layer_id,
        "Layer %u was not begun",
        static_cast<unsigned int>(layer_id)
    );
    RWKV_CTX_ASSERT_FALSE_MSG(
        ctx,
        RWKV_ERROR_ARGS,
        activation_in != nullptr && activation_out != nullptr && state_out != nullptr,
        "Activation/state buffers are invalid"
    );
    RWKV_CTX_ASSERT_FALSE_MSG(
        ctx,
        RWKV_ERROR_DATA,
        rwkv_layer_plan_run(
            *plan,
            activation_in,
            activation_out,
            state_in,
            state_out,
            v_first_in,
            v_first_out
        ),
        "Layer %u execution failed",
        static_cast<unsigned int>(layer_id)
    );
    return true;
}

// API function.  The graph remains the single-token graph used by the decode
// path, but the active layer is uploaded only once while a prompt chunk is
// consumed.  This is a bounded layer-local prefill ABI; it deliberately keeps
// the caller-owned activation/state buffers outside the native weight arena.
bool rwkv_layer_step_sequence(
    struct rwkv_context * ctx,
    const uint32_t layer_id,
    const float * activation_in,
    float * activation_out,
    const size_t n_tokens,
    const float * state_in,
    float * state_out,
    const float * v_first_in,
    float * v_first_out
) {
    RWKV_CTX_ASSERT_FALSE_MSG(ctx, RWKV_ERROR_ARGS, ctx != nullptr, "Context is null");
    RWKV_CTX_ASSERT_FALSE_MSG(
        ctx,
        RWKV_ERROR_UNSUPPORTED,
        rwkv_supports_layer_streaming(ctx),
        "Layer-local ABI is not enabled"
    );
    RWKV_CTX_ASSERT_FALSE_MSG(
        ctx,
        RWKV_ERROR_ARGS,
        layer_id < ctx->model->header.n_layer,
        "Layer id is out of range"
    );
    RWKV_CTX_ASSERT_FALSE_MSG(
        ctx,
        RWKV_ERROR_ARGS,
        n_tokens > 0 && activation_in != nullptr && activation_out != nullptr &&
            state_out != nullptr,
        "Sequence activation/state buffers are invalid"
    );
    const size_t n_embed = static_cast<size_t>(ctx->model->header.n_embed);
    const size_t state_len = rwkv_get_layer_state_len(ctx);
    RWKV_CTX_ASSERT_FALSE_MSG(
        ctx,
        RWKV_ERROR_DATA,
        n_tokens <= std::numeric_limits<size_t>::max() / n_embed,
        "Layer sequence is too large"
    );
    rwkv_layer_plan * plan = rwkv_layer_select_plan(*ctx, layer_id);
    RWKV_CTX_ASSERT_FALSE_MSG(ctx, RWKV_ERROR_CTX, plan != nullptr, "No layer plan is available");
    RWKV_CTX_ASSERT_FALSE_MSG(
        ctx,
        RWKV_ERROR_DATA,
        plan->begun && plan->active_layer == layer_id && ctx->active_layer_plan == plan,
        "Layer %u was not begun",
        static_cast<unsigned int>(layer_id)
    );
    RWKV_CTX_ASSERT_FALSE_MSG(
        ctx,
        RWKV_ERROR_DATA,
        ctx->model->arch_version_major != 7 ||
            (plan->first_layer ? v_first_out != nullptr : v_first_in != nullptr),
        "RWKV-7 sequence execution requires v_first buffers for this layer"
    );

    std::vector<float> current_state(state_len, 0.0F);
    std::vector<float> next_state(state_len, 0.0F);
    if (state_in != nullptr) {
        std::memcpy(current_state.data(), state_in, state_len * sizeof(float));
    }

    for (size_t token = 0; token < n_tokens; ++token) {
        const float * state_source = token == 0 && state_in == nullptr
            ? nullptr
            : current_state.data();
        const float * token_v_first_in = v_first_in == nullptr
            ? nullptr
            : v_first_in + token * n_embed;
        float * token_v_first_out = v_first_out == nullptr
            ? nullptr
            : v_first_out + token * n_embed;
        RWKV_CTX_ASSERT_FALSE_MSG(
            ctx,
            RWKV_ERROR_DATA,
            rwkv_layer_plan_run(
                *plan,
                activation_in + token * n_embed,
                activation_out + token * n_embed,
                state_source,
                next_state.data(),
                token_v_first_in,
                token_v_first_out
            ),
            "Layer %u sequence execution failed at token %llu",
            static_cast<unsigned int>(layer_id),
            static_cast<unsigned long long>(token)
        );
        current_state.swap(next_state);
    }
    std::memcpy(state_out, current_state.data(), state_len * sizeof(float));
    return true;
}

#undef RWKV_LAYER_FIELDS
