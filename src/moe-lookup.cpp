#include "moe-lookup.h"

#include "llama-cparams.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

namespace {

constexpr uint32_t LLAMA_MOE_LOOKUP_MAGIC = 0x31544c45u; // "ELT1"
constexpr uint32_t LLAMA_MOE_LOOKUP_FORMAT_VERSION = 1;

static bool read_bytes(std::istream & in, void * dst, size_t size) {
    if (size == 0) {
        return true;
    }
    in.read(reinterpret_cast<char *>(dst), (std::streamsize) size);
    return in.good();
}

static std::string layer_count_error(uint32_t layer_id, uint32_t n_keys, uint32_t expected_bytes) {
    std::ostringstream oss;
    oss << "layer " << layer_id << ": invalid payload size for n_keys=" << n_keys << " (expected bytes=" << expected_bytes << ")";
    return oss.str();
}

} // namespace

bool llama_moe_lookup_layer::valid() const {
    if (n_keys == 0) {
        return false;
    }

    if (centroids.empty() || centroids.size() != contributions.size() || centroids.size() % n_keys != 0) {
        return false;
    }

    const size_t n_embd = centroids.size() / n_keys;
    if (n_embd == 0) {
        return false;
    }

    return n_keys > 0
        && !centroid_l2_sq.empty()
        && centroid_l2_sq.size() == (size_t) n_keys
        && !replaced_mask.empty();
}

std::unique_ptr<llama_moe_lookup_table> llama_moe_lookup_table::load(
        const llama_model & model,
        const llama_cparams & cparams,
        std::string & warning_out) {
    warning_out.clear();

    if (!cparams.moe_lookup_enable) {
        warning_out = "lookup disabled";
        return nullptr;
    }

    if (cparams.moe_lookup_file.empty()) {
        warning_out = "--moe-lookup-file is required when --moe-lookup-enable is set";
        return nullptr;
    }

    if (cparams.moe_lookup_replaced_experts.empty()) {
        warning_out = "--moe-lookup-replaced-experts is required when --moe-lookup-enable is set";
        return nullptr;
    }

    std::ifstream in(cparams.moe_lookup_file, std::ios::binary);
    if (!in.is_open()) {
        warning_out = std::string("failed to open lookup sidecar '") + cparams.moe_lookup_file + "': " + std::strerror(errno);
        return nullptr;
    }

    {
        std::ifstream replaced_in(cparams.moe_lookup_replaced_experts);
        if (!replaced_in.is_open()) {
            warning_out = std::string("failed to open replaced experts file '") + cparams.moe_lookup_replaced_experts + "': " + std::strerror(errno);
            return nullptr;
        }
    }

    auto table = std::unique_ptr<llama_moe_lookup_table>(new llama_moe_lookup_table());

    llama_moe_lookup_header_v1 header = {};
    if (!read_bytes(in, &header, sizeof(header))) {
        warning_out = "failed to read sidecar header";
        return nullptr;
    }

    const uint32_t magic = header.magic;
    const uint32_t fmt = header.format_version;
    const uint32_t model_id_len = header.model_id_len;
    const uint32_t n_layer = header.n_layer;
    const uint32_t n_embd = header.n_embd;
    const uint32_t n_expert = header.n_expert;
    const uint32_t n_expert_used = header.n_expert_used;
    const uint32_t dtype = header.vector_dtype;
    const uint32_t scaling = header.scaling_mode;
    const uint32_t n_layers_payload = header.n_layers_payload;

    if (magic != LLAMA_MOE_LOOKUP_MAGIC) {
        std::ostringstream oss;
        oss << "invalid sidecar magic: 0x" << std::hex << magic << ", expected 0x" << LLAMA_MOE_LOOKUP_MAGIC;
        warning_out = oss.str();
        return nullptr;
    }

    if (fmt != LLAMA_MOE_LOOKUP_FORMAT_VERSION) {
        std::ostringstream oss;
        oss << "unsupported sidecar format_version=" << fmt << ", expected " << LLAMA_MOE_LOOKUP_FORMAT_VERSION;
        warning_out = oss.str();
        return nullptr;
    }

    if (model_id_len == 0 || model_id_len > 4096) {
        warning_out = "invalid sidecar model_id length";
        return nullptr;
    }

    table->model_id.resize(model_id_len);
    if (!read_bytes(in, table->model_id.data(), model_id_len)) {
        warning_out = "failed to read sidecar model_id";
        return nullptr;
    }

    if (dtype != (uint32_t) llama_moe_lookup_vector_dtype::FP16) {
        warning_out = "unsupported sidecar vector_dtype (v1 requires fp16)";
        return nullptr;
    }

    if (scaling != (uint32_t) llama_moe_lookup_scaling_mode::S_MISSING) {
        warning_out = "unsupported sidecar scaling_mode (v1 requires s_missing)";
        return nullptr;
    }

    if (n_layer != model.hparams.n_layer) {
        std::ostringstream oss;
        oss << "sidecar n_layer mismatch: sidecar=" << n_layer << " model=" << model.hparams.n_layer;
        warning_out = oss.str();
        return nullptr;
    }

    if (n_embd != model.hparams.n_embd) {
        std::ostringstream oss;
        oss << "sidecar n_embd mismatch: sidecar=" << n_embd << " model=" << model.hparams.n_embd;
        warning_out = oss.str();
        return nullptr;
    }

    if (n_expert != model.hparams.n_expert) {
        std::ostringstream oss;
        oss << "sidecar n_expert mismatch: sidecar=" << n_expert << " model=" << model.hparams.n_expert;
        warning_out = oss.str();
        return nullptr;
    }

    if (n_expert_used != model.hparams.n_expert_used) {
        std::ostringstream oss;
        oss << "sidecar n_expert_used mismatch: sidecar=" << n_expert_used << " model=" << model.hparams.n_expert_used;
        warning_out = oss.str();
        return nullptr;
    }

    if (table->model_id != model.arch_name()) {
        std::ostringstream oss;
        oss << "sidecar model_id mismatch: sidecar='" << table->model_id << "' model='" << model.arch_name() << "'";
        warning_out = oss.str();
        return nullptr;
    }

    table->fmt_version = fmt;
    table->dtype = llama_moe_lookup_vector_dtype::FP16;
    table->scaling = llama_moe_lookup_scaling_mode::S_MISSING;
    table->n_layer = n_layer;
    table->n_embd = n_embd;
    table->n_expert = n_expert;
    table->n_expert_used = n_expert_used;

    if (n_layers_payload > n_layer) {
        warning_out = "invalid sidecar: payload layer count exceeds n_layer";
        return nullptr;
    }

    std::vector<std::string> skipped_layers;

    for (uint32_t i = 0; i < n_layers_payload; ++i) {
        llama_moe_lookup_layer_header_v1 layer_header = {};
        if (!read_bytes(in, &layer_header, sizeof(layer_header))) {
            warning_out = "failed reading layer header";
            return nullptr;
        }

        const uint32_t layer_id = layer_header.layer_id;
        const uint32_t n_keys = layer_header.n_keys;
        const uint32_t replaced_count = layer_header.replaced_count;

        const bool bad_layer_id = layer_id >= n_layer;
        const bool bad_n_keys = n_keys == 0;
        const bool duplicate_layer = table->layers.find(layer_id) != table->layers.end();

        const size_t vec_sz = (size_t) n_keys * (size_t) n_embd;
        if (vec_sz > (size_t) std::numeric_limits<uint32_t>::max()) {
            warning_out = layer_count_error(layer_id, n_keys, (uint32_t) vec_sz);
            return nullptr;
        }

        llama_moe_lookup_layer layer;
        layer.layer_id = layer_id;
        layer.n_keys = n_keys;
        layer.centroids.resize(vec_sz);
        layer.contributions.resize(vec_sz);
        layer.centroid_l2_sq.resize(n_keys);
        layer.replaced_mask.assign(n_expert, 0);

        if (!read_bytes(in, layer.centroids.data(), vec_sz * sizeof(ggml_fp16_t)) ||
            !read_bytes(in, layer.contributions.data(), vec_sz * sizeof(ggml_fp16_t))) {
            warning_out = "failed reading layer tensor payload";
            return nullptr;
        }

        std::vector<uint32_t> replaced_ids(replaced_count);
        if (replaced_count > 0 && !read_bytes(in, replaced_ids.data(), replaced_count * sizeof(uint32_t))) {
            warning_out = "failed reading replaced expert IDs";
            return nullptr;
        }

        bool skip_layer = bad_layer_id || bad_n_keys || duplicate_layer;

        if (bad_layer_id) {
            std::ostringstream oss;
            oss << "layer " << layer_id << " skipped: invalid layer_id (n_layer=" << n_layer << ")";
            skipped_layers.push_back(oss.str());
        } else if (bad_n_keys) {
            std::ostringstream oss;
            oss << "layer " << layer_id << " skipped: n_keys=0";
            skipped_layers.push_back(oss.str());
        } else if (duplicate_layer) {
            std::ostringstream oss;
            oss << "layer " << layer_id << " skipped: duplicate payload";
            skipped_layers.push_back(oss.str());
        }

        for (uint32_t eid : replaced_ids) {
            if (eid >= n_expert) {
                skip_layer = true;
                std::ostringstream oss;
                oss << "layer " << layer_id << " skipped: replaced expert id " << eid << " out of range [0, " << n_expert << ")";
                skipped_layers.push_back(oss.str());
                break;
            }
            layer.replaced_mask[eid] = 1;
        }

        if (!skip_layer && replaced_count > n_expert - n_expert_used) {
            skip_layer = true;
            std::ostringstream oss;
            oss << "layer " << layer_id << " skipped: replaced_count=" << replaced_count
                << " exceeds fill-safe limit " << (n_expert - n_expert_used);
            skipped_layers.push_back(oss.str());
        }

        for (uint32_t k = 0; k < n_keys; ++k) {
            float sum = 0.0f;
            const size_t base = (size_t) k * (size_t) n_embd;
            for (uint32_t d = 0; d < n_embd; ++d) {
                const float v = ggml_fp16_to_fp32(layer.centroids[base + d]);
                if (!std::isfinite(v)) {
                    skip_layer = true;
                    std::ostringstream oss;
                    oss << "layer " << layer_id << " skipped: centroid contains non-finite value";
                    skipped_layers.push_back(oss.str());
                    break;
                }
                sum += v * v;
            }
            if (skip_layer) {
                break;
            }
            if (!std::isfinite(sum)) {
                skip_layer = true;
                std::ostringstream oss;
                oss << "layer " << layer_id << " skipped: centroid norm overflow";
                skipped_layers.push_back(oss.str());
                break;
            }
            layer.centroid_l2_sq[k] = sum;
        }

        if (!skip_layer && layer.valid()) {
            table->layers.emplace(layer_id, std::move(layer));
        }
    }

    char trailing = 0;
    if (in.read(&trailing, 1)) {
        warning_out = "invalid sidecar: trailing payload bytes detected";
        return nullptr;
    }

    table->is_valid = !table->layers.empty();
    if (!table->is_valid) {
        warning_out = "sidecar contains no valid layer payloads";
        return nullptr;
    }

    if (!skipped_layers.empty() || n_layers_payload < n_layer) {
        std::ostringstream oss;
        oss << "loaded with ";
        if (n_layers_payload < n_layer) {
            oss << "partial coverage (" << n_layers_payload << "/" << n_layer << " layers)";
            if (!skipped_layers.empty()) {
                oss << ", ";
            }
        }
        if (!skipped_layers.empty()) {
            oss << skipped_layers.size() << " skipped layer payload(s): " << skipped_layers[0];
        }
        warning_out = oss.str();
    }

    return table;
}

bool llama_moe_lookup_table::valid() const {
    return is_valid;
}

uint32_t llama_moe_lookup_table::format_version() const {
    return fmt_version;
}

llama_moe_lookup_vector_dtype llama_moe_lookup_table::vector_dtype() const {
    return dtype;
}

llama_moe_lookup_scaling_mode llama_moe_lookup_table::scaling_mode() const {
    return scaling;
}

const llama_moe_lookup_layer * llama_moe_lookup_table::layer(uint32_t layer_id) const {
    auto it = layers.find(layer_id);
    if (it == layers.end()) {
        return nullptr;
    }
    return &it->second;
}

bool llama_moe_lookup_table::has_any_active_layers() const {
    for (const auto & kv : layers) {
        const auto & replaced_mask = kv.second.replaced_mask;
        if (std::any_of(replaced_mask.begin(), replaced_mask.end(), [](uint8_t v) { return v != 0; })) {
            return true;
        }
    }
    return false;
}
