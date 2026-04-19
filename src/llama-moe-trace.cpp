#include "llama-moe-trace.h"

#include "llama-impl.h"

#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <string_view>

#define LLAMA_MOE_TRACE_TOPK_PREFIX "ffn_moe_topk-"
#define LLAMA_MOE_TRACE_EXPERT_IN_PREFIX "ffn_moe_expert_in-"

static bool llama_moe_trace_supports_input_type(enum ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16;
}

static float llama_moe_trace_read_input_value(const ggml_tensor * t, size_t offset) {
    const char * data = static_cast<const char *>(t->data);
    switch (t->type) {
        case GGML_TYPE_F32: {
            float v;
            memcpy(&v, data + offset, sizeof(v));
            return v;
        }
        case GGML_TYPE_F16: {
            ggml_fp16_t v;
            memcpy(&v, data + offset, sizeof(v));
            return ggml_fp16_to_fp32(v);
        }
        case GGML_TYPE_BF16: {
            ggml_bf16_t v;
            memcpy(&v, data + offset, sizeof(v));
            return ggml_bf16_to_fp32(v);
        }
        default:
            GGML_ABORT("fatal error");
    }
}

static std::string llama_moe_trace_json_escape(std::string_view s) {
    std::string out;
    out.reserve(s.size());

    for (char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                out += c;
                break;
        }
    }

    return out;
}

static int64_t llama_moe_trace_now_ms() {
    const auto now = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
}

static uint64_t llama_moe_trace_key(int32_t layer, int32_t expert) {
    return (uint64_t(uint32_t(layer)) << 32) | uint32_t(expert);
}

std::unique_ptr<llama_moe_trace> llama_moe_trace::init(
        const llama_moe_trace_config & config,
        const llama_hparams          & hparams,
        const std::string            & model_name,
        uint64_t                       seed) {
    std::unique_ptr<llama_moe_trace> trace(new llama_moe_trace(config, hparams, model_name, seed));
    if (!trace->enabled()) {
        return nullptr;
    }
    return trace;
}

llama_moe_trace::llama_moe_trace(
        const llama_moe_trace_config & config,
        const llama_hparams          & hparams,
        const std::string            & model_name,
        uint64_t                       seed) :
    config(config),
    hparams(hparams),
    model_name(model_name),
    rng((uint32_t) seed),
    bernoulli(0.0f, 1.0f) {
    if (!config.enabled) {
        return;
    }

    if (hparams.n_expert == 0 || hparams.n_expert_used == 0) {
        LLAMA_LOG_WARN("%s: MoE trace enabled but model has no MoE layers - no rows will be emitted\n", __func__);
    }

    if (config.format != "jsonl") {
        LLAMA_LOG_WARN("%s: unsupported moe trace format '%s' - expected 'jsonl'\n", __func__, config.format.c_str());
        if (config.strict) {
            throw std::runtime_error("invalid moe trace format");
        }
        return;
    }

    if (config.path.empty()) {
        LLAMA_LOG_WARN("%s: MoE trace enabled but no output path configured\n", __func__);
        if (config.strict) {
            throw std::runtime_error("missing moe trace path");
        }
        return;
    }

    if (config.precision != "f16" && config.precision != "f32") {
        LLAMA_LOG_WARN("%s: unsupported moe trace precision '%s' - expected 'f16' or 'f32'\n", __func__, config.precision.c_str());
        if (config.strict) {
            throw std::runtime_error("invalid moe trace precision");
        }
        return;
    }

    if (config.sample_rate < 0.0f || config.sample_rate > 1.0f) {
        LLAMA_LOG_WARN("%s: invalid moe trace sample rate %.6f - expected range [0, 1]\n", __func__, (double) config.sample_rate);
        if (config.strict) {
            throw std::runtime_error("invalid moe trace sample rate");
        }
        return;
    }

    if (!init_writer()) {
        if (config.strict) {
            throw std::runtime_error("failed to initialize moe trace writer");
        }
        return;
    }

    active = true;
    LLAMA_LOG_WARN("%s: MoE routed-input trace enabled; captured vectors may contain sensitive prompt semantics\n", __func__);
}

llama_moe_trace::~llama_moe_trace() {
    flush();
    if (active || rows_emitted > 0 || rows_dropped_total > 0) {
        LLAMA_LOG_INFO(
                "%s: rows_emitted=%lld rows_dropped_total=%lld dropped{cap_total=%lld,cap_layer=%lld,cap_expert=%lld,sample=%lld,shape=%lld,io=%lld} flush_count=%lld flush_error_count=%lld\n",
                __func__,
                (long long) rows_emitted,
                (long long) rows_dropped_total,
                (long long) rows_dropped_cap_total,
                (long long) rows_dropped_cap_layer,
                (long long) rows_dropped_cap_expert,
                (long long) rows_dropped_sample,
                (long long) rows_dropped_shape,
                (long long) rows_dropped_io,
                (long long) flush_count,
                (long long) flush_error_count);
    }
}

bool llama_moe_trace::enabled() const {
    return active;
}

bool llama_moe_trace::wants_tensor(const char * tensor_name) const {
    if (!active || degraded || tensor_name == nullptr) {
        return false;
    }
    return strcmp(tensor_name, "ffn_moe_topk") == 0 || strcmp(tensor_name, "ffn_moe_expert_in") == 0;
}

void llama_moe_trace::set_ubatch(const llama_ubatch & ubatch) {
    if (!active || degraded) {
        return;
    }
    build_token_meta(ubatch, ubatch_meta);
    has_ubatch_meta = true;
}

bool llama_moe_trace::capture_eval(ggml_tensor * t) {
    if (!active || degraded || t == nullptr) {
        return true;
    }

    const char * name = t->name;
    if (strncmp(name, LLAMA_MOE_TRACE_TOPK_PREFIX, strlen(LLAMA_MOE_TRACE_TOPK_PREFIX)) == 0) {
        const int il = std::atoi(name + strlen(LLAMA_MOE_TRACE_TOPK_PREFIX));
        return on_topk(t, il);
    }
    if (strncmp(name, LLAMA_MOE_TRACE_EXPERT_IN_PREFIX, strlen(LLAMA_MOE_TRACE_EXPERT_IN_PREFIX)) == 0) {
        const int il = std::atoi(name + strlen(LLAMA_MOE_TRACE_EXPERT_IN_PREFIX));
        return on_expert_in(t, il);
    }
    return true;
}

bool llama_moe_trace::init_writer() {
    writer.open(config.path, std::ios::out | std::ios::app);
    if (!writer.is_open()) {
        LLAMA_LOG_WARN("%s: failed to open MoE trace output '%s'\n", __func__, config.path.c_str());
        return false;
    }
    flush_last_ms = llama_moe_trace_now_ms();
    return true;
}

bool llama_moe_trace::on_topk(ggml_tensor * t, int il) {
    if (t == nullptr || t->data == nullptr || t->type != GGML_TYPE_I32) {
        rows_dropped_shape += 1;
        rows_dropped_total += 1;
        return true;
    }

    const int64_t n_expert_used = t->ne[0];
    const int64_t n_tokens = t->ne[1];
    if (n_expert_used <= 0 || n_tokens <= 0) {
        return true;
    }

    const int32_t * topk = static_cast<const int32_t *>(t->data);
    layer_state & st = states[il];
    st.topk.resize((size_t) (n_expert_used * n_tokens));
    std::copy(topk, topk + st.topk.size(), st.topk.begin());
    st.has_topk = true;
    return true;
}

void llama_moe_trace::build_token_meta(const llama_ubatch & ubatch, std::vector<token_meta> & out) const {
    out.resize(ubatch.n_tokens);
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        token_meta tm;
        tm.ubatch_index = (int32_t) i;
        tm.token_pos = ubatch.pos ? ubatch.pos[i*ubatch.n_pos] : -1;
        if (ubatch.n_seq_id && ubatch.n_seq_id[i] > 0 && ubatch.seq_id && ubatch.seq_id[i] != nullptr) {
            tm.seq_id = ubatch.seq_id[i][0];
        }
        out[i] = tm;
    }
}

bool llama_moe_trace::should_keep(int32_t layer, int32_t expert) {
    if (config.max_rows_total > 0 && rows_emitted >= config.max_rows_total) {
        rows_dropped_total += 1;
        rows_dropped_cap_total += 1;
        return false;
    }

    auto & by_layer = count_by_layer[layer];
    if (config.max_rows_per_layer > 0 && by_layer >= config.max_rows_per_layer) {
        rows_dropped_total += 1;
        rows_dropped_cap_layer += 1;
        return false;
    }

    const uint64_t key = llama_moe_trace_key(layer, expert);
    auto & by_expert = count_by_layer_expert[key];
    if (config.max_rows_per_expert > 0 && by_expert >= config.max_rows_per_expert) {
        rows_dropped_total += 1;
        rows_dropped_cap_expert += 1;
        return false;
    }

    if (config.sample_rate < 1.0f && bernoulli(rng) > config.sample_rate) {
        rows_dropped_total += 1;
        rows_dropped_sample += 1;
        return false;
    }

    by_layer += 1;
    by_expert += 1;
    return true;
}

void llama_moe_trace::append_row(row && out_row) {
    row_buffer.push_back(std::move(out_row));
    rows_emitted += 1;
    flush_if_needed();
}

bool llama_moe_trace::on_expert_in(ggml_tensor * t, int il) {
    if (t == nullptr || t->data == nullptr || !llama_moe_trace_supports_input_type(t->type)) {
        rows_dropped_shape += 1;
        rows_dropped_total += 1;
        return true;
    }

    auto it = states.find(il);
    if (it == states.end() || !it->second.has_topk) {
        rows_dropped_shape += 1;
        rows_dropped_total += 1;
        return true;
    }

    const int64_t d_model = t->ne[0];
    const int64_t n_expert_used = t->ne[1];
    const int64_t n_tokens = t->ne[2];
    if (d_model <= 0 || n_expert_used <= 0 || n_tokens <= 0) {
        return true;
    }

    if (!has_ubatch_meta || (int64_t) ubatch_meta.size() != n_tokens) {
        rows_dropped_shape += 1;
        rows_dropped_total += 1;
        return true;
    }

    const auto & st = it->second;
    if ((int64_t) st.topk.size() != n_expert_used*n_tokens) {
        rows_dropped_shape += 1;
        rows_dropped_total += 1;
        return true;
    }

    const size_t s0 = (size_t) t->nb[0];
    const size_t s1 = (size_t) t->nb[1];
    const size_t s2 = (size_t) t->nb[2];

    for (int64_t tok = 0; tok < n_tokens; ++tok) {
        for (int64_t rank = 0; rank < n_expert_used; ++rank) {
            const int64_t topk_idx = rank + tok*n_expert_used;
            const int32_t expert = st.topk[(size_t) topk_idx];
            if (!should_keep(il, expert)) {
                continue;
            }

            row out;
            out.layer = il;
            out.expert = expert;
            out.inputs.resize((size_t) d_model);
            out.seq_id = ubatch_meta[(size_t) tok].seq_id;
            out.token_pos = ubatch_meta[(size_t) tok].token_pos;
            out.ubatch_index = ubatch_meta[(size_t) tok].ubatch_index;
            out.expert_rank = (int32_t) rank;

            for (int64_t i = 0; i < d_model; ++i) {
                const size_t offset = (size_t) i*s0 + (size_t) rank*s1 + (size_t) tok*s2;
                float v = llama_moe_trace_read_input_value(t, offset);
                if (config.precision == "f16") {
                    const ggml_fp16_t h = ggml_fp32_to_fp16(v);
                    v = ggml_fp16_to_fp32(h);
                }
                out.inputs[(size_t) i] = v;
            }

            append_row(std::move(out));
        }
    }

    return true;
}

void llama_moe_trace::flush_if_needed() {
    const int64_t now_ms = llama_moe_trace_now_ms();
    const bool by_rows = config.buffer_rows > 0 && (int32_t) row_buffer.size() >= config.buffer_rows;
    const bool by_time = config.flush_interval_ms > 0 && (now_ms - flush_last_ms) >= config.flush_interval_ms;
    if (by_rows || by_time) {
        flush_impl();
    }
}

bool llama_moe_trace::flush_impl() {
    if (!active || degraded || row_buffer.empty()) {
        return !degraded;
    }

    for (const auto & r : row_buffer) {
        writer << "{\"schema_version\":\"moe_routed_input_jsonl.v1\",\"event\":\"moe_routed_input\",\"layer\":" << r.layer
               << ",\"expert\":" << r.expert
               << ",\"inputs\":[";
        for (size_t i = 0; i < r.inputs.size(); ++i) {
            if (i > 0) {
                writer << ',';
            }
            writer << r.inputs[i];
        }
        writer << "]"
               << ",\"seq_id\":" << r.seq_id
               << ",\"token_pos\":" << r.token_pos
               << ",\"ubatch_index\":" << r.ubatch_index
               << ",\"expert_rank\":" << r.expert_rank
               << ",\"model\":\"" << llama_moe_trace_json_escape(model_name) << "\""
               << ",\"ts_unix_ms\":" << llama_moe_trace_now_ms()
               << "}\n";
    }

    writer.flush();
    flush_count += 1;
    flush_last_ms = llama_moe_trace_now_ms();

    if (!writer.good()) {
        flush_error_count += 1;
        rows_dropped_io += (int64_t) row_buffer.size();
        rows_dropped_total += (int64_t) row_buffer.size();
        LLAMA_LOG_WARN("%s: failed flushing MoE trace output, disabling further writes\n", __func__);
        degraded = true;
        row_buffer.clear();
        return false;
    }

    row_buffer.clear();
    return true;
}

void llama_moe_trace::flush() {
    flush_impl();
}
