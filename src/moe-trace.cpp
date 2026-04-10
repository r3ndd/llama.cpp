#include "moe-trace.h"

#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <sstream>
#include <utility>

namespace {

static uint32_t crc32_calc(const uint8_t * data, size_t n) {
    static std::array<uint32_t, 256> table = [] {
        std::array<uint32_t, 256> t{};
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1) ? (0xedb88320u ^ (c >> 1)) : (c >> 1);
            }
            t[i] = c;
        }
        return t;
    }();

    uint32_t c = 0xffffffffu;
    for (size_t i = 0; i < n; ++i) {
        c = table[(c ^ data[i]) & 0xffu] ^ (c >> 8);
    }
    return c ^ 0xffffffffu;
}

static void write_u16(std::ofstream & out, uint16_t v) {
    out.put((char) ( v        & 0xff));
    out.put((char) ((v >> 8)  & 0xff));
}

static void write_u32(std::ofstream & out, uint32_t v) {
    out.put((char) ( v        & 0xff));
    out.put((char) ((v >> 8)  & 0xff));
    out.put((char) ((v >> 16) & 0xff));
    out.put((char) ((v >> 24) & 0xff));
}

static std::string npy_make_header(const char * descr, const std::vector<uint32_t> & shape) {
    std::ostringstream shape_ss;
    shape_ss << "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        shape_ss << shape[i];
        if (shape.size() == 1) {
            shape_ss << ",";
        } else if (i + 1 < shape.size()) {
            shape_ss << ", ";
        }
    }
    shape_ss << ")";

    std::string dict = "{'descr': '" + std::string(descr) + "', 'fortran_order': False, 'shape': " + shape_ss.str() + ", }";
    dict.push_back('\n');

    const size_t base = 10; // magic + version + header_len
    const size_t rem = (base + dict.size()) % 16;
    if (rem != 0) {
        dict.insert(dict.end() - 1, 16 - rem, ' ');
    }

    std::string out;
    out.reserve(base + dict.size());
    out.append("\x93NUMPY", 6);
    out.push_back(1);
    out.push_back(0);

    const uint16_t hlen = (uint16_t) dict.size();
    out.push_back((char) ( hlen       & 0xff));
    out.push_back((char) ((hlen >> 8) & 0xff));
    out += dict;

    return out;
}

template<typename T>
static std::vector<uint8_t> npy_build(const char * descr, const std::vector<uint32_t> & shape, const std::vector<T> & data) {
    std::string header = npy_make_header(descr, shape);
    std::vector<uint8_t> bytes;
    bytes.resize(header.size() + data.size() * sizeof(T));
    std::memcpy(bytes.data(), header.data(), header.size());
    if (!data.empty()) {
        std::memcpy(bytes.data() + header.size(), data.data(), data.size() * sizeof(T));
    }
    return bytes;
}

struct zip_entry {
    std::string name;
    uint32_t crc32 = 0;
    uint32_t size = 0;
    uint32_t local_offset = 0;
};

static bool zip_write_store(
        const std::string & path,
        const std::vector<std::pair<std::string, std::vector<uint8_t>>> & files) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        return false;
    }

    std::vector<zip_entry> entries;
    entries.reserve(files.size());

    for (const auto & f : files) {
        const std::string & name = f.first;
        const std::vector<uint8_t> & data = f.second;
        zip_entry e;
        e.name = name;
        e.crc32 = crc32_calc(data.data(), data.size());
        e.size = (uint32_t) data.size();
        e.local_offset = (uint32_t) out.tellp();

        write_u32(out, 0x04034b50u);
        write_u16(out, 20);
        write_u16(out, 0);
        write_u16(out, 0);
        write_u16(out, 0);
        write_u16(out, 0);
        write_u32(out, e.crc32);
        write_u32(out, e.size);
        write_u32(out, e.size);
        write_u16(out, (uint16_t) e.name.size());
        write_u16(out, 0);
        out.write(e.name.data(), e.name.size());
        if (!data.empty()) {
            out.write((const char *) data.data(), data.size());
        }

        entries.push_back(std::move(e));
    }

    const uint32_t central_offset = (uint32_t) out.tellp();

    for (const auto & e : entries) {
        write_u32(out, 0x02014b50u);
        write_u16(out, 20);
        write_u16(out, 20);
        write_u16(out, 0);
        write_u16(out, 0);
        write_u16(out, 0);
        write_u16(out, 0);
        write_u32(out, e.crc32);
        write_u32(out, e.size);
        write_u32(out, e.size);
        write_u16(out, (uint16_t) e.name.size());
        write_u16(out, 0);
        write_u16(out, 0);
        write_u16(out, 0);
        write_u16(out, 0);
        write_u32(out, 0);
        write_u32(out, e.local_offset);
        out.write(e.name.data(), e.name.size());
    }

    const uint32_t central_size = (uint32_t) out.tellp() - central_offset;

    write_u32(out, 0x06054b50u);
    write_u16(out, 0);
    write_u16(out, 0);
    write_u16(out, (uint16_t) entries.size());
    write_u16(out, (uint16_t) entries.size());
    write_u32(out, central_size);
    write_u32(out, central_offset);
    write_u16(out, 0);

    return out.good();
}

} // namespace

llama_moe_trace_writer::llama_moe_trace_writer(const llama_model & model, const std::string & output_path)
    : output_path(output_path) {
    if (output_path.empty()) {
        LLAMA_LOG_WARN("%s: MoE trace enabled but output path is empty, disabling trace\n", __func__);
        return;
    }

    if (model.arch != LLM_ARCH_QWEN35MOE) {
        LLAMA_LOG_WARN("%s: MoE trace supports Qwen3.5-MoE only, disabling trace\n", __func__);
        return;
    }

    model_id = model.arch_name();
    n_layer = (int32_t) model.hparams.n_layer;
    n_expert = (int32_t) model.hparams.n_expert;
    n_expert_used = (int32_t) model.hparams.n_expert_used;

    is_valid = true;
}

llama_moe_trace_writer::~llama_moe_trace_writer() {
    if (!is_valid || flushed) {
        return;
    }
    flush_npz();
}

bool llama_moe_trace_writer::valid() const {
    return is_valid;
}

void llama_moe_trace_writer::begin_graph() {
    pending_by_layer.clear();
}

void llama_moe_trace_writer::reset_registry() {
    tensor_registry.clear();
}

void llama_moe_trace_writer::register_tensor(const ggml_tensor * t, const char * name, int il) {
    if (!is_valid || il < 0 || t == nullptr || name == nullptr) {
        return;
    }

    tensor_meta meta;
    meta.layer = il;

    if (strcmp(name, "ffn_moe_h_pre") == 0) {
        meta.kind = tensor_kind::H_PRE;
    } else if (strcmp(name, "ffn_moe_topk") == 0) {
        meta.kind = tensor_kind::TOPK;
    } else if (strcmp(name, "ffn_moe_argsort") == 0) {
        meta.kind = tensor_kind::ARGSORT;
    } else if (strcmp(name, "ffn_moe_weights") == 0) {
        meta.kind = tensor_kind::WEIGHTS;
    } else if (strcmp(name, "ffn_moe_expert_out") == 0) {
        meta.kind = tensor_kind::TOPK_EXPERT_OUTPUTS;
    } else if (strcmp(name, "ffn_moe_out") == 0) {
        meta.kind = tensor_kind::Y_FULL;
    } else {
        return;
    }

    tensor_registry[t] = meta;
}

bool llama_moe_trace_writer::wants_tensor(const ggml_tensor * t) const {
    if (!is_valid || t == nullptr) {
        return false;
    }
    return tensor_registry.find(t) != tensor_registry.end();
}

bool llama_moe_trace_writer::read_tensor_f32(const ggml_tensor * t, std::vector<float> & out) const {
    out.resize((size_t) ggml_nelements(t));
    if (out.empty()) {
        return true;
    }

    switch (t->type) {
        case GGML_TYPE_F32:
            ggml_backend_tensor_get(const_cast<ggml_tensor *>(t), out.data(), 0, out.size() * sizeof(float));
            return true;
        case GGML_TYPE_F16: {
            std::vector<ggml_fp16_t> tmp(out.size());
            ggml_backend_tensor_get(const_cast<ggml_tensor *>(t), tmp.data(), 0, tmp.size() * sizeof(ggml_fp16_t));
            for (size_t i = 0; i < out.size(); ++i) {
                out[i] = ggml_fp16_to_fp32(tmp[i]);
            }
            return true;
        }
        case GGML_TYPE_BF16: {
            std::vector<ggml_bf16_t> tmp(out.size());
            ggml_backend_tensor_get(const_cast<ggml_tensor *>(t), tmp.data(), 0, tmp.size() * sizeof(ggml_bf16_t));
            for (size_t i = 0; i < out.size(); ++i) {
                out[i] = ggml_bf16_to_fp32(tmp[i]);
            }
            return true;
        }
        default:
            return false;
    }
}

bool llama_moe_trace_writer::read_tensor_i32(const ggml_tensor * t, std::vector<int32_t> & out) const {
    if (t->type != GGML_TYPE_I32) {
        return false;
    }
    out.resize((size_t) ggml_nelements(t));
    if (out.empty()) {
        return true;
    }
    ggml_backend_tensor_get(const_cast<ggml_tensor *>(t), out.data(), 0, out.size() * sizeof(int32_t));
    return true;
}

bool llama_moe_trace_writer::ingest(const tensor_meta & meta, const ggml_tensor * t) {
    auto & p = pending_by_layer[meta.layer];

    switch (meta.kind) {
        case tensor_kind::H_PRE:
            if (!read_tensor_f32(t, p.h_pre)) {
                return false;
            }
            p.n_embd = (int) t->ne[0];
            p.n_tokens = (int) t->ne[1];
            p.has_h_pre = true;
            break;
        case tensor_kind::TOPK:
            if (!read_tensor_i32(t, p.topk_ids)) {
                return false;
            }
            p.n_topk = (int) t->ne[0];
            p.n_tokens = (int) t->ne[1];
            p.has_topk = true;
            break;
        case tensor_kind::ARGSORT:
            if (!read_tensor_i32(t, p.argsort_ids)) {
                return false;
            }
            p.n_tokens = (int) t->ne[1];
            p.has_argsort = true;
            break;
        case tensor_kind::WEIGHTS:
            if (!read_tensor_f32(t, p.topk_weights)) {
                return false;
            }
            if (t->ne[0] == 1) {
                p.n_topk = (int) t->ne[1];
                p.n_tokens = (int) t->ne[2];
            } else {
                p.n_topk = (int) t->ne[0];
                p.n_tokens = (int) t->ne[1];
            }
            p.has_weights = true;
            break;
        case tensor_kind::TOPK_EXPERT_OUTPUTS:
            if (!read_tensor_f32(t, p.topk_expert_outputs)) {
                return false;
            }
            p.n_embd = (int) t->ne[0];
            p.n_topk = (int) t->ne[1];
            p.n_tokens = (int) t->ne[2];
            p.has_topk_expert_outputs = true;
            break;
        case tensor_kind::Y_FULL:
            if (!read_tensor_f32(t, p.y_full)) {
                return false;
            }
            p.n_embd = (int) t->ne[0];
            p.n_tokens = (int) t->ne[1];
            p.has_y_full = true;
            break;
    }

    return true;
}

bool llama_moe_trace_writer::try_finalize_layer(int layer) {
    auto it = pending_by_layer.find(layer);
    if (it == pending_by_layer.end()) {
        return false;
    }

    const auto drop_pending = [&]() {
        pending_by_layer.erase(it);
    };

    layer_pending & p = it->second;
    if (!(p.has_h_pre && p.has_topk && p.has_weights && p.has_topk_expert_outputs && p.has_y_full)) {
        return false;
    }

    if (p.n_tokens <= 0 || p.n_embd <= 0 || p.n_topk <= 0) {
        drop_pending();
        return false;
    }

    const size_t n_tok = (size_t) p.n_tokens;
    const size_t n_emb = (size_t) p.n_embd;
    const size_t n_k = (size_t) p.n_topk;

    if (p.h_pre.size() != n_tok*n_emb ||
        p.topk_ids.size() != n_tok*n_k ||
        p.topk_weights.size() != n_tok*n_k ||
        p.topk_expert_outputs.size() != n_tok*n_k*n_emb ||
        p.y_full.size() != n_tok*n_emb) {
        drop_pending();
        return false;
    }

    if (!llama_moe_trace_validate_topk_consistency(
            p.topk_ids.data(),
            p.topk_weights.data(),
            p.n_topk,
            p.n_tokens,
            n_expert,
            nullptr)) {
        if (!warn_once_bad_parity) {
            LLAMA_LOG_WARN("%s: dropping layer %d trace row batch due to invalid top-k IDs/weights consistency\n", __func__, layer);
            warn_once_bad_parity = true;
        }
        drop_pending();
        return false;
    }

    if (p.has_argsort) {
        const int n_argsort = (int) (p.argsort_ids.size() / n_tok);
        std::string parity_err;
        if (n_argsort <= 0 || (size_t) n_argsort * n_tok != p.argsort_ids.size() ||
            !llama_moe_trace_validate_topk_parity(
                p.topk_ids.data(),
                p.n_topk,
                p.argsort_ids.data(),
                n_argsort,
                p.n_tokens,
                &parity_err)) {
            if (!warn_once_bad_parity) {
                LLAMA_LOG_WARN("%s: dropping layer %d trace row batch due to top-k parity mismatch (%s)\n", __func__, layer, parity_err.c_str());
                warn_once_bad_parity = true;
            }
            drop_pending();
            return false;
        }
    }

    if (!llama_moe_trace_validate_topk_expert_outputs(
            p.topk_expert_outputs.data(),
            p.n_topk,
            p.n_tokens,
            p.n_embd,
            nullptr)) {
        if (!warn_once_bad_parity) {
            LLAMA_LOG_WARN("%s: dropping layer %d trace row batch due to invalid top-k expert outputs\n", __func__, layer);
            warn_once_bad_parity = true;
        }
        drop_pending();
        return false;
    }

    if (n_embd < 0) {
        n_embd = (int32_t) p.n_embd;
    } else if (n_embd != p.n_embd) {
        if (!warn_once_bad_parity) {
            LLAMA_LOG_WARN("%s: dropping layer %d trace row batch due to n_embd mismatch (%d vs %d)\n", __func__, layer, (int) n_embd, p.n_embd);
            warn_once_bad_parity = true;
        }
        drop_pending();
        return false;
    }
    if (n_topk < 0) {
        n_topk = (int32_t) p.n_topk;
    } else if (n_topk != p.n_topk) {
        if (!warn_once_bad_parity) {
            LLAMA_LOG_WARN("%s: dropping layer %d trace row batch due to n_topk mismatch (%d vs %d)\n", __func__, layer, (int) n_topk, p.n_topk);
            warn_once_bad_parity = true;
        }
        drop_pending();
        return false;
    }

    for (size_t i = 0; i < n_tok; ++i) {
        layer_ids.push_back(layer);
        token_ids.push_back((int32_t) i);
    }

    for (float v : p.h_pre) {
        h_pre_moe.push_back(ggml_fp32_to_fp16(v));
    }
    topk_ids.insert(topk_ids.end(), p.topk_ids.begin(), p.topk_ids.end());
    for (float v : p.topk_weights) {
        topk_weights.push_back(ggml_fp32_to_fp16(v));
    }
    for (float v : p.topk_expert_outputs) {
        topk_expert_outputs.push_back(ggml_fp32_to_fp16(v));
    }
    for (float v : p.y_full) {
        y_full.push_back(ggml_fp32_to_fp16(v));
    }

    pending_by_layer.erase(it);
    return true;
}

bool llama_moe_trace_writer::observe_tensor(const ggml_tensor * t) {
    if (!is_valid || t == nullptr) {
        return true;
    }

    auto it = tensor_registry.find(t);
    if (it == tensor_registry.end()) {
        return true;
    }

    if (!ingest(it->second, t)) {
        if (!warn_once_bad_tensor) {
            LLAMA_LOG_WARN("%s: failed to capture MoE trace tensor '%s'\n", __func__, ggml_get_name(t));
            warn_once_bad_tensor = true;
        }
        return true;
    }

    try_finalize_layer(it->second.layer);
    return true;
}

bool llama_moe_trace_writer::flush_npz() {
    flushed = true;

    if (layer_ids.empty()) {
        LLAMA_LOG_WARN("%s: MoE trace captured no rows, nothing written\n", __func__);
        return true;
    }

    const uint32_t n_rows = (uint32_t) layer_ids.size();

    std::ostringstream meta;
    meta << "{"
         << "\"format_version\":1,"
         << "\"model_id\":\"" << model_id << "\","
         << "\"commit_hash\":\"unknown\"," 
         << "\"n_layer\":" << n_layer << ","
         << "\"n_embd\":" << n_embd << ","
         << "\"n_expert\":" << n_expert << ","
         << "\"n_expert_used\":" << n_expert_used << ","
         << "\"routing_mode\":\"topk\","
         << "\"prompt_source\":\"unknown\","
         << "\"trace_sampling_policy\":\"all\""
         << "}";

    std::vector<std::pair<std::string, std::vector<uint8_t>>> files;
    files.reserve(8);

    files.emplace_back("layer_ids.npy", npy_build<int32_t>("<i4", { n_rows }, layer_ids));
    files.emplace_back("token_ids.npy", npy_build<int32_t>("<i4", { n_rows }, token_ids));
    files.emplace_back("h_pre_moe.npy", npy_build<ggml_fp16_t>("<f2", { n_rows, (uint32_t) n_embd }, h_pre_moe));
    files.emplace_back("topk_ids.npy", npy_build<int32_t>("<i4", { n_rows, (uint32_t) n_topk }, topk_ids));
    files.emplace_back("topk_weights.npy", npy_build<ggml_fp16_t>("<f2", { n_rows, (uint32_t) n_topk }, topk_weights));
    files.emplace_back("topk_expert_outputs.npy", npy_build<ggml_fp16_t>("<f2", { n_rows, (uint32_t) n_topk, (uint32_t) n_embd }, topk_expert_outputs));
    files.emplace_back("y_full.npy", npy_build<ggml_fp16_t>("<f2", { n_rows, (uint32_t) n_embd }, y_full));
    const std::string meta_str = meta.str();
    files.emplace_back("metadata.json", std::vector<uint8_t>(meta_str.begin(), meta_str.end()));

    if (!zip_write_store(output_path, files)) {
        LLAMA_LOG_WARN("%s: failed writing MoE trace NPZ to '%s'\n", __func__, output_path.c_str());
        return false;
    }

    LLAMA_LOG_INFO("%s: wrote MoE trace NPZ v1: %s (%u rows)\n", __func__, output_path.c_str(), n_rows);
    return true;
}
