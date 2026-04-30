#include "moe-cov-io.h"

#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <map>
#include <set>

static enum ggml_type moe_cov_precision_to_ggml_type(enum imatrix_cov_precision precision) {
    switch (precision) {
        case IMATRIX_COV_F8:  return GGML_TYPE_I8;
        case IMATRIX_COV_F16: return GGML_TYPE_F16;
        case IMATRIX_COV_F32: return GGML_TYPE_F32;
        case IMATRIX_COV_F64: return GGML_TYPE_F64;
        default:              return GGML_TYPE_F32;
    }
}

static const char * moe_cov_precision_to_str(enum imatrix_cov_precision precision) {
    switch (precision) {
        case IMATRIX_COV_F8:  return "f8";
        case IMATRIX_COV_F16: return "f16";
        case IMATRIX_COV_F32: return "f32";
        case IMATRIX_COV_F64: return "f64";
        default:              return "unknown";
    }
}

struct moe_cov_merge_key {
    uint32_t layer = 0;
    uint32_t expert = 0;
    std::string role;
    std::string role_variant;
    std::string tensor_name;

    bool operator<(const moe_cov_merge_key & other) const {
        if (layer != other.layer) {
            return layer < other.layer;
        }
        if (expert != other.expert) {
            return expert < other.expert;
        }
        if (role != other.role) {
            return role < other.role;
        }
        if (role_variant != other.role_variant) {
            return role_variant < other.role_variant;
        }
        return tensor_name < other.tensor_name;
    }
};

struct moe_cov_write_metadata {
    std::string created_at;
    std::vector<std::string> sources;
};

static bool gguf_read_required_str(
        const gguf_context * ctx_gguf,
        const std::string & key,
        std::string & out,
        std::string & error_msg) {
    const int64_t kid = gguf_find_key(ctx_gguf, key.c_str());
    if (kid < 0) {
        error_msg = string_format("missing required key '%s'", key.c_str());
        return false;
    }
    if (gguf_get_kv_type(ctx_gguf, kid) != GGUF_TYPE_STRING) {
        error_msg = string_format("invalid type for key '%s' (expected string)", key.c_str());
        return false;
    }
    out = gguf_get_val_str(ctx_gguf, kid);
    return true;
}

static bool gguf_read_required_u32(
        const gguf_context * ctx_gguf,
        const std::string & key,
        uint32_t & out,
        std::string & error_msg) {
    const int64_t kid = gguf_find_key(ctx_gguf, key.c_str());
    if (kid < 0) {
        error_msg = string_format("missing required key '%s'", key.c_str());
        return false;
    }
    if (gguf_get_kv_type(ctx_gguf, kid) != GGUF_TYPE_UINT32) {
        error_msg = string_format("invalid type for key '%s' (expected uint32)", key.c_str());
        return false;
    }
    out = gguf_get_val_u32(ctx_gguf, kid);
    return true;
}

static bool gguf_read_optional_str(
        const gguf_context * ctx_gguf,
        const std::string & key,
        std::string & out,
        std::string & error_msg) {
    const int64_t kid = gguf_find_key(ctx_gguf, key.c_str());
    if (kid < 0) {
        return true;
    }
    if (gguf_get_kv_type(ctx_gguf, kid) != GGUF_TYPE_STRING) {
        error_msg = string_format("invalid type for key '%s' (expected string)", key.c_str());
        return false;
    }
    out = gguf_get_val_str(ctx_gguf, kid);
    return true;
}

static bool gguf_read_optional_sources(
        const gguf_context * ctx_gguf,
        std::vector<std::string> & sources,
        std::string & error_msg) {
    sources.clear();
    const int64_t kid = gguf_find_key(ctx_gguf, "moe_cov.sources");
    if (kid < 0) {
        return true;
    }
    if (gguf_get_kv_type(ctx_gguf, kid) != GGUF_TYPE_ARRAY || gguf_get_arr_type(ctx_gguf, kid) != GGUF_TYPE_STRING) {
        error_msg = "invalid type for key 'moe_cov.sources' (expected string array)";
        return false;
    }
    const size_t n = gguf_get_arr_n(ctx_gguf, kid);
    sources.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        sources.emplace_back(gguf_get_arr_str(ctx_gguf, kid, i));
    }
    return true;
}

static bool read_target_tensor(
        const gguf_context * ctx_gguf,
        const ggml_context * ctx_data,
        const std::string & name,
        enum ggml_type expected_type,
        int64_t expected_elements,
        const ggml_tensor *& out,
        std::string & error_msg) {
    const int64_t tid = gguf_find_tensor(ctx_gguf, name.c_str());
    if (tid < 0) {
        error_msg = string_format("missing tensor '%s'", name.c_str());
        return false;
    }
    if (gguf_get_tensor_type(ctx_gguf, tid) != expected_type) {
        error_msg = string_format("tensor '%s' type mismatch", name.c_str());
        return false;
    }
    out = ggml_get_tensor(const_cast<ggml_context *>(ctx_data), name.c_str());
    if (out == nullptr) {
        error_msg = string_format("failed to access tensor '%s' data", name.c_str());
        return false;
    }
    if (ggml_nelements(out) != expected_elements) {
        error_msg = string_format("tensor '%s' element count mismatch", name.c_str());
        return false;
    }
    return true;
}

static bool read_target_from_file(
        const gguf_context * ctx_gguf,
        const ggml_context * ctx_data,
        const std::string & tid,
        enum imatrix_cov_precision precision,
        moe_cov_target_data & out,
        std::string & error_msg) {
    if (!gguf_read_required_u32(ctx_gguf, string_format("moe_cov.target.%s.layer", tid.c_str()), out.layer, error_msg) ||
        !gguf_read_required_u32(ctx_gguf, string_format("moe_cov.target.%s.expert", tid.c_str()), out.expert, error_msg) ||
        !gguf_read_required_u32(ctx_gguf, string_format("moe_cov.target.%s.dim", tid.c_str()), out.dim, error_msg) ||
        !gguf_read_required_str(ctx_gguf, string_format("moe_cov.target.%s.role", tid.c_str()), out.role, error_msg) ||
        !gguf_read_required_str(ctx_gguf, string_format("moe_cov.target.%s.tensor_name", tid.c_str()), out.tensor_name, error_msg) ||
        !gguf_read_optional_str(ctx_gguf, string_format("moe_cov.target.%s.role_variant", tid.c_str()), out.role_variant, error_msg)) {
        return false;
    }

    const ggml_tensor * t_n = nullptr;
    if (!read_target_tensor(
                ctx_gguf,
                ctx_data,
                string_format("moe_cov.%s.n", tid.c_str()),
                GGML_TYPE_I64,
                1,
                t_n,
                error_msg)) {
        return false;
    }

    const int64_t n_raw = ((const int64_t *) t_n->data)[0];
    if (n_raw < 0) {
        error_msg = string_format("tensor 'moe_cov.%s.n' has negative value", tid.c_str());
        return false;
    }
    out.n = (uint64_t) n_raw;

    const int64_t dim = out.dim;
    const int64_t dim2 = dim * dim;
    const enum ggml_type tensor_type = moe_cov_precision_to_ggml_type(precision);

    const ggml_tensor * t_sum = nullptr;
    const ggml_tensor * t_outer = nullptr;
    if (!read_target_tensor(
                ctx_gguf,
                ctx_data,
                string_format("moe_cov.%s.sum", tid.c_str()),
                tensor_type,
                dim,
                t_sum,
                error_msg) ||
        !read_target_tensor(
                ctx_gguf,
                ctx_data,
                string_format("moe_cov.%s.outer", tid.c_str()),
                tensor_type,
                dim2,
                t_outer,
                error_msg)) {
        return false;
    }

    if (precision == IMATRIX_COV_F8) {
        out.sum_f8.resize((size_t) dim);
        out.outer_f8.resize((size_t) dim2);
        std::memcpy(out.sum_f8.data(), t_sum->data, out.sum_f8.size() * sizeof(int8_t));
        std::memcpy(out.outer_f8.data(), t_outer->data, out.outer_f8.size() * sizeof(int8_t));
    } else if (precision == IMATRIX_COV_F16) {
        out.sum_f16.resize((size_t) dim);
        out.outer_f16.resize((size_t) dim2);
        std::memcpy(out.sum_f16.data(), t_sum->data, out.sum_f16.size() * sizeof(ggml_fp16_t));
        std::memcpy(out.outer_f16.data(), t_outer->data, out.outer_f16.size() * sizeof(ggml_fp16_t));
    } else if (precision == IMATRIX_COV_F32) {
        out.sum_f32.resize((size_t) dim);
        out.outer_f32.resize((size_t) dim2);
        std::memcpy(out.sum_f32.data(), t_sum->data, out.sum_f32.size() * sizeof(float));
        std::memcpy(out.outer_f32.data(), t_outer->data, out.outer_f32.size() * sizeof(float));
    } else {
        out.sum_f64.resize((size_t) dim);
        out.outer_f64.resize((size_t) dim2);
        std::memcpy(out.sum_f64.data(), t_sum->data, out.sum_f64.size() * sizeof(double));
        std::memcpy(out.outer_f64.data(), t_outer->data, out.outer_f64.size() * sizeof(double));
    }

    return true;
}

static bool load_existing_covariance(
        const std::filesystem::path & file_path,
        enum imatrix_cov_precision precision,
        const std::string & expected_fingerprint,
        std::vector<moe_cov_target_data> & out_targets,
        moe_cov_write_metadata & out_meta,
        std::string & error_msg) {
    out_targets.clear();
    out_meta = {};

    struct ggml_context * ctx_data = nullptr;
    struct gguf_init_params params = {
        /* .no_alloc = */ false,
        /* .ctx      = */ &ctx_data,
    };

    struct gguf_context * ctx_gguf = gguf_init_from_file(file_path.string().c_str(), params);
    if (ctx_gguf == nullptr) {
        error_msg = string_format("failed to open covariance file '%s'", file_path.string().c_str());
        return false;
    }

    bool ok = false;
    do {
        std::string type;
        std::string convention;
        std::string precision_str;
        std::string fingerprint;
        uint32_t version = 0;
        uint32_t target_count = 0;

        if (!gguf_read_required_str(ctx_gguf, "general.type", type, error_msg) ||
            !gguf_read_required_u32(ctx_gguf, "moe_cov.version", version, error_msg) ||
            !gguf_read_required_str(ctx_gguf, "moe_cov.convention", convention, error_msg) ||
            !gguf_read_required_str(ctx_gguf, "moe_cov.precision", precision_str, error_msg) ||
            !gguf_read_required_str(ctx_gguf, "moe_cov.model_fingerprint", fingerprint, error_msg) ||
            !gguf_read_required_u32(ctx_gguf, "moe_cov.target_count", target_count, error_msg)) {
            break;
        }

        if (type != "moe_covariance") {
            error_msg = string_format("covariance file '%s' has incompatible general.type='%s'", file_path.string().c_str(), type.c_str());
            break;
        }
        if (version != 1) {
            error_msg = string_format("covariance file '%s' has unsupported moe_cov.version=%u", file_path.string().c_str(), version);
            break;
        }
        if (convention != "population") {
            error_msg = string_format("covariance file '%s' has incompatible moe_cov.convention='%s'", file_path.string().c_str(), convention.c_str());
            break;
        }

        const std::string expected_precision = moe_cov_precision_to_str(precision);
        if (precision_str != expected_precision) {
            error_msg = string_format(
                    "covariance file '%s' precision mismatch: existing=%s requested=%s",
                    file_path.string().c_str(),
                    precision_str.c_str(),
                    expected_precision.c_str());
            break;
        }

        if (!expected_fingerprint.empty() && expected_fingerprint != "unknown" &&
            !fingerprint.empty() && fingerprint != "unknown" &&
            fingerprint != expected_fingerprint) {
            error_msg = string_format(
                    "covariance file '%s' model fingerprint mismatch: existing='%s' current='%s'",
                    file_path.string().c_str(),
                    fingerprint.c_str(),
                    expected_fingerprint.c_str());
            break;
        }

        if (!gguf_read_optional_str(ctx_gguf, "moe_cov.created_at", out_meta.created_at, error_msg)) {
            break;
        }
        if (!gguf_read_optional_sources(ctx_gguf, out_meta.sources, error_msg)) {
            break;
        }

        out_targets.reserve(target_count);
        for (uint32_t i = 0; i < target_count; ++i) {
            moe_cov_target_data target;
            if (!read_target_from_file(
                        ctx_gguf,
                        ctx_data,
                        string_format("t%u", i),
                        precision,
                        target,
                        error_msg)) {
                break;
            }
            out_targets.push_back(std::move(target));
        }

        if (out_targets.size() != target_count) {
            if (error_msg.empty()) {
                error_msg = string_format("failed to parse all covariance targets from '%s'", file_path.string().c_str());
            }
            break;
        }

        ok = true;
    } while (false);

    gguf_free(ctx_gguf);
    ggml_free(ctx_data);

    return ok;
}

static bool merge_target_data(
        moe_cov_target_data & dst,
        const moe_cov_target_data & src,
        enum imatrix_cov_precision precision,
        std::string & error_msg) {
    if (dst.dim != src.dim) {
        error_msg = string_format(
                "covariance target dimension mismatch for layer=%u expert=%u role=%s tensor=%s: %u vs %u",
                dst.layer,
                dst.expert,
                dst.role.c_str(),
                dst.tensor_name.c_str(),
                dst.dim,
                src.dim);
        return false;
    }

    dst.n += src.n;

    if (precision == IMATRIX_COV_F8) {
        if (dst.sum_f8.size() != src.sum_f8.size() || dst.outer_f8.size() != src.outer_f8.size()) {
            error_msg = "f8 merge size mismatch";
            return false;
        }

        for (size_t i = 0; i < dst.sum_f8.size(); ++i) {
            int v = (int) dst.sum_f8[i] + (int) src.sum_f8[i];
            v = std::max(-127, std::min(127, v));
            dst.sum_f8[i] = (int8_t) v;
        }
        for (size_t i = 0; i < dst.outer_f8.size(); ++i) {
            int v = (int) dst.outer_f8[i] + (int) src.outer_f8[i];
            v = std::max(-127, std::min(127, v));
            dst.outer_f8[i] = (int8_t) v;
        }
    } else if (precision == IMATRIX_COV_F16) {
        if (dst.sum_f16.size() != src.sum_f16.size() || dst.outer_f16.size() != src.outer_f16.size()) {
            error_msg = "f16 merge size mismatch";
            return false;
        }

        for (size_t i = 0; i < dst.sum_f16.size(); ++i) {
            const float v = ggml_fp16_to_fp32(dst.sum_f16[i]) + ggml_fp16_to_fp32(src.sum_f16[i]);
            dst.sum_f16[i] = ggml_fp32_to_fp16(v);
        }
        for (size_t i = 0; i < dst.outer_f16.size(); ++i) {
            const float v = ggml_fp16_to_fp32(dst.outer_f16[i]) + ggml_fp16_to_fp32(src.outer_f16[i]);
            dst.outer_f16[i] = ggml_fp32_to_fp16(v);
        }
    } else if (precision == IMATRIX_COV_F32) {
        if (dst.sum_f32.size() != src.sum_f32.size() || dst.outer_f32.size() != src.outer_f32.size()) {
            error_msg = "f32 merge size mismatch";
            return false;
        }

        for (size_t i = 0; i < dst.sum_f32.size(); ++i) {
            dst.sum_f32[i] += src.sum_f32[i];
        }
        for (size_t i = 0; i < dst.outer_f32.size(); ++i) {
            dst.outer_f32[i] += src.outer_f32[i];
        }
    } else {
        if (dst.sum_f64.size() != src.sum_f64.size() || dst.outer_f64.size() != src.outer_f64.size()) {
            error_msg = "f64 merge size mismatch";
            return false;
        }

        for (size_t i = 0; i < dst.sum_f64.size(); ++i) {
            dst.sum_f64[i] += src.sum_f64[i];
        }
        for (size_t i = 0; i < dst.outer_f64.size(); ++i) {
            dst.outer_f64[i] += src.outer_f64[i];
        }
    }

    return true;
}

static bool merge_targets(
        const std::vector<moe_cov_target_data> & existing_targets,
        const std::vector<moe_cov_target_data> & new_targets,
        enum imatrix_cov_precision precision,
        std::vector<moe_cov_target_data> & out_targets,
        std::string & error_msg) {
    std::map<moe_cov_merge_key, moe_cov_target_data> merged;

    for (const auto & target : existing_targets) {
        moe_cov_merge_key key{target.layer, target.expert, target.role, target.role_variant, target.tensor_name};
        auto it = merged.find(key);
        if (it == merged.end()) {
            merged.emplace(std::move(key), target);
            continue;
        }

        if (!merge_target_data(it->second, target, precision, error_msg)) {
            return false;
        }
    }

    for (const auto & target : new_targets) {
        moe_cov_merge_key key{target.layer, target.expert, target.role, target.role_variant, target.tensor_name};
        auto it = merged.find(key);
        if (it == merged.end()) {
            merged.emplace(std::move(key), target);
            continue;
        }

        if (!merge_target_data(it->second, target, precision, error_msg)) {
            return false;
        }
    }

    out_targets.clear();
    out_targets.reserve(merged.size());
    for (auto & kv : merged) {
        out_targets.push_back(std::move(kv.second));
    }

    return true;
}

static bool write_covariance_file(
        const std::filesystem::path & out_path,
        const std::filesystem::path & tmp_path,
        enum imatrix_cov_precision precision,
        const std::string & fingerprint,
        const moe_cov_write_metadata & metadata,
        const common_params & params,
        const std::vector<moe_cov_target_data> & targets,
        std::string & error_msg) {
    size_t ctx_size = 0;
    for (const auto & target : targets) {
        const int64_t dim = target.dim;
        ctx_size += GGML_PAD(ggml_tensor_overhead() + ggml_row_size(GGML_TYPE_I64, 1), GGML_MEM_ALIGN);

        const enum ggml_type tensor_type = moe_cov_precision_to_ggml_type(precision);
        ctx_size += GGML_PAD(ggml_tensor_overhead() + ggml_row_size(tensor_type, dim), GGML_MEM_ALIGN);
        ctx_size += GGML_PAD(ggml_tensor_overhead() + ggml_row_size(tensor_type, dim*dim), GGML_MEM_ALIGN);
        ctx_size += GGML_PAD(ggml_tensor_overhead() + ggml_row_size(tensor_type, dim*dim), GGML_MEM_ALIGN);
    }
    if (ctx_size == 0) {
        ctx_size = GGML_MEM_ALIGN;
    }

    struct ggml_init_params ggml_params = {
        /* .mem_size   = */ ctx_size,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ false,
    };

    struct ggml_context * ctx = ggml_init(ggml_params);
    if (!ctx) {
        error_msg = "failed to initialize GGML context for covariance output";
        return false;
    }

    struct gguf_context * ctx_gguf = gguf_init_empty();
    if (!ctx_gguf) {
        ggml_free(ctx);
        error_msg = "failed to initialize GGUF context for covariance output";
        return false;
    }

    const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    const std::string updated_at = std::to_string((int64_t) now);
    const std::string created_at = metadata.created_at.empty() ? updated_at : metadata.created_at;

    gguf_set_val_str(ctx_gguf, "general.type", "moe_covariance");
    gguf_set_val_u32(ctx_gguf, "moe_cov.version", 1);
    gguf_set_val_str(ctx_gguf, "moe_cov.convention", "population");
    gguf_set_val_str(ctx_gguf, "moe_cov.precision", moe_cov_precision_to_str(precision));
    gguf_set_val_str(ctx_gguf, "moe_cov.model_fingerprint", fingerprint.c_str());
    gguf_set_val_str(ctx_gguf, "moe_cov.created_at", created_at.c_str());
    gguf_set_val_str(ctx_gguf, "moe_cov.updated_at", updated_at.c_str());
    gguf_set_val_u32(ctx_gguf, "moe_cov.target_count", (uint32_t) targets.size());

    std::vector<const char *> source_ptrs;
    source_ptrs.reserve(metadata.sources.size());
    for (const auto & source : metadata.sources) {
        source_ptrs.push_back(source.c_str());
    }
    gguf_set_arr_str(ctx_gguf, "moe_cov.sources", source_ptrs.data(), source_ptrs.size());

    gguf_set_val_str(ctx_gguf, "moe_cov.filters.layers", params.cov_layers.c_str());
    gguf_set_val_str(ctx_gguf, "moe_cov.filters.experts", params.cov_experts.c_str());
    gguf_set_val_str(ctx_gguf, "moe_cov.filters.targets", params.cov_targets.c_str());

    const enum ggml_type tensor_type = moe_cov_precision_to_ggml_type(precision);

    for (size_t i = 0; i < targets.size(); ++i) {
        const auto & target = targets[i];
        const std::string tid = string_format("t%zu", i);

        gguf_set_val_u32(ctx_gguf, string_format("moe_cov.target.%s.layer", tid.c_str()).c_str(), target.layer);
        gguf_set_val_u32(ctx_gguf, string_format("moe_cov.target.%s.expert", tid.c_str()).c_str(), target.expert);
        gguf_set_val_str(ctx_gguf, string_format("moe_cov.target.%s.role", tid.c_str()).c_str(), target.role.c_str());
        gguf_set_val_str(ctx_gguf, string_format("moe_cov.target.%s.tensor_name", tid.c_str()).c_str(), target.tensor_name.c_str());
        gguf_set_val_u32(ctx_gguf, string_format("moe_cov.target.%s.dim", tid.c_str()).c_str(), target.dim);
        if (!target.role_variant.empty()) {
            gguf_set_val_str(ctx_gguf, string_format("moe_cov.target.%s.role_variant", tid.c_str()).c_str(), target.role_variant.c_str());
        }

        {
            struct ggml_tensor * t_n = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
            ggml_set_name(t_n, string_format("moe_cov.%s.n", tid.c_str()).c_str());
            ((int64_t *) t_n->data)[0] = (int64_t) target.n;
            gguf_add_tensor(ctx_gguf, t_n);
        }

        const enum ggml_type cov_tensor_type =
                precision == IMATRIX_COV_F8 ? GGML_TYPE_F32 : tensor_type;

        struct ggml_tensor * t_sum = ggml_new_tensor_1d(ctx, tensor_type, target.dim);
        struct ggml_tensor * t_outer = ggml_new_tensor_2d(ctx, tensor_type, target.dim, target.dim);
        struct ggml_tensor * t_cov = ggml_new_tensor_2d(ctx, cov_tensor_type, target.dim, target.dim);

        ggml_set_name(t_sum, string_format("moe_cov.%s.sum", tid.c_str()).c_str());
        ggml_set_name(t_outer, string_format("moe_cov.%s.outer", tid.c_str()).c_str());
        ggml_set_name(t_cov, string_format("moe_cov.%s.cov_pop", tid.c_str()).c_str());

        if (precision == IMATRIX_COV_F8) {
            if (target.sum_f8.size() != (size_t) target.dim ||
                target.outer_f8.size() != (size_t) target.dim * target.dim) {
                gguf_free(ctx_gguf);
                ggml_free(ctx);
                error_msg = string_format("f8 tensor size mismatch for target %s", target.tensor_name.c_str());
                return false;
            }

            std::memcpy(t_sum->data, target.sum_f8.data(), target.sum_f8.size() * sizeof(int8_t));
            std::memcpy(t_outer->data, target.outer_f8.data(), target.outer_f8.size() * sizeof(int8_t));

            std::vector<float> cov(target.outer_f8.size(), 0.0f);
            if (target.n > 1) {
                std::vector<float> mean(target.dim, 0.0f);
                for (uint32_t j = 0; j < target.dim; ++j) {
                    mean[j] = (float) target.sum_f8[j] / (float) target.n;
                }
                for (uint32_t r = 0; r < target.dim; ++r) {
                    for (uint32_t c = 0; c < target.dim; ++c) {
                        const size_t idx = (size_t) r * target.dim + c;
                        cov[idx] = (float) target.outer_f8[idx] / (float) target.n - mean[r] * mean[c];
                    }
                }
            }
            std::memcpy(t_cov->data, cov.data(), cov.size() * sizeof(float));
        } else if (precision == IMATRIX_COV_F16) {
            if (target.sum_f16.size() != (size_t) target.dim ||
                target.outer_f16.size() != (size_t) target.dim * target.dim) {
                gguf_free(ctx_gguf);
                ggml_free(ctx);
                error_msg = string_format("f16 tensor size mismatch for target %s", target.tensor_name.c_str());
                return false;
            }

            std::memcpy(t_sum->data, target.sum_f16.data(), target.sum_f16.size() * sizeof(ggml_fp16_t));
            std::memcpy(t_outer->data, target.outer_f16.data(), target.outer_f16.size() * sizeof(ggml_fp16_t));

            std::vector<ggml_fp16_t> cov(target.outer_f16.size(), ggml_fp32_to_fp16(0.0f));
            if (target.n > 1) {
                std::vector<float> mean(target.dim, 0.0f);
                for (uint32_t j = 0; j < target.dim; ++j) {
                    mean[j] = ggml_fp16_to_fp32(target.sum_f16[j]) / (float) target.n;
                }
                for (uint32_t r = 0; r < target.dim; ++r) {
                    for (uint32_t c = 0; c < target.dim; ++c) {
                        const size_t idx = (size_t) r * target.dim + c;
                        const float pop = ggml_fp16_to_fp32(target.outer_f16[idx]) / (float) target.n - mean[r] * mean[c];
                        cov[idx] = ggml_fp32_to_fp16(pop);
                    }
                }
            }
            std::memcpy(t_cov->data, cov.data(), cov.size() * sizeof(ggml_fp16_t));
        } else if (precision == IMATRIX_COV_F32) {
            if (target.sum_f32.size() != (size_t) target.dim ||
                target.outer_f32.size() != (size_t) target.dim * target.dim) {
                gguf_free(ctx_gguf);
                ggml_free(ctx);
                error_msg = string_format("f32 tensor size mismatch for target %s", target.tensor_name.c_str());
                return false;
            }

            std::memcpy(t_sum->data, target.sum_f32.data(), target.sum_f32.size() * sizeof(float));
            std::memcpy(t_outer->data, target.outer_f32.data(), target.outer_f32.size() * sizeof(float));

            std::vector<float> cov(target.outer_f32.size(), 0.0f);
            if (target.n > 1) {
                std::vector<float> mean(target.dim, 0.0f);
                for (uint32_t j = 0; j < target.dim; ++j) {
                    mean[j] = target.sum_f32[j] / (float) target.n;
                }
                for (uint32_t r = 0; r < target.dim; ++r) {
                    for (uint32_t c = 0; c < target.dim; ++c) {
                        const size_t idx = (size_t) r * target.dim + c;
                        cov[idx] = target.outer_f32[idx] / (float) target.n - mean[r] * mean[c];
                    }
                }
            }
            std::memcpy(t_cov->data, cov.data(), cov.size() * sizeof(float));
        } else {
            if (target.sum_f64.size() != (size_t) target.dim ||
                target.outer_f64.size() != (size_t) target.dim * target.dim) {
                gguf_free(ctx_gguf);
                ggml_free(ctx);
                error_msg = string_format("f64 tensor size mismatch for target %s", target.tensor_name.c_str());
                return false;
            }

            std::memcpy(t_sum->data, target.sum_f64.data(), target.sum_f64.size() * sizeof(double));
            std::memcpy(t_outer->data, target.outer_f64.data(), target.outer_f64.size() * sizeof(double));

            std::vector<double> cov(target.outer_f64.size(), 0.0);
            if (target.n > 1) {
                std::vector<double> mean(target.dim, 0.0);
                for (uint32_t j = 0; j < target.dim; ++j) {
                    mean[j] = target.sum_f64[j] / (double) target.n;
                }
                for (uint32_t r = 0; r < target.dim; ++r) {
                    for (uint32_t c = 0; c < target.dim; ++c) {
                        const size_t idx = (size_t) r * target.dim + c;
                        cov[idx] = target.outer_f64[idx] / (double) target.n - mean[r] * mean[c];
                    }
                }
            }
            std::memcpy(t_cov->data, cov.data(), cov.size() * sizeof(double));
        }

        gguf_add_tensor(ctx_gguf, t_sum);
        gguf_add_tensor(ctx_gguf, t_outer);
        gguf_add_tensor(ctx_gguf, t_cov);
    }

    const bool write_ok = gguf_write_to_file(ctx_gguf, tmp_path.string().c_str(), false);

    gguf_free(ctx_gguf);
    ggml_free(ctx);

    if (!write_ok) {
        error_msg = string_format("failed to write covariance temp file '%s'", tmp_path.string().c_str());
        return false;
    }

    std::error_code ec;
    std::filesystem::rename(tmp_path, out_path, ec);
    if (ec) {
        std::error_code ec_copy;
        std::filesystem::copy_file(tmp_path, out_path, std::filesystem::copy_options::overwrite_existing, ec_copy);
        if (ec_copy) {
            std::filesystem::remove(tmp_path, ec);
            error_msg = string_format("failed to replace covariance file '%s'", out_path.string().c_str());
            return false;
        }

        std::filesystem::remove(tmp_path, ec);
    }

    return true;
}

bool moe_cov_write_file(
        const common_params & params,
        const llama_model * model,
        const std::vector<moe_cov_target_data> & targets,
        std::string & error_msg) {
    namespace fs = std::filesystem;

    const fs::path out_path(params.cov_out_file);
    if (out_path.empty()) {
        error_msg = "covariance output path is empty";
        return false;
    }

    std::error_code ec;

    if (params.cov_file_mode == IMATRIX_COV_CREATE && fs::exists(out_path, ec)) {
        error_msg = string_format("covariance file '%s' already exists", params.cov_out_file.c_str());
        return false;
    }

    const fs::path out_dir = out_path.has_parent_path() ? out_path.parent_path() : fs::path(".");
    if (!fs::exists(out_dir, ec)) {
        error_msg = string_format("covariance output directory '%s' does not exist", out_dir.string().c_str());
        return false;
    }

    char model_desc[256] = {0};
    if (model != nullptr) {
        llama_model_desc(model, model_desc, sizeof(model_desc));
    }
    const std::string fingerprint = model_desc[0] != '\0' ? std::string(model_desc) : std::string("unknown");

    std::vector<moe_cov_target_data> final_targets = targets;
    moe_cov_write_metadata metadata;
    if (!params.prompt_file.empty()) {
        metadata.sources.push_back(params.prompt_file);
    }

    if (params.cov_file_mode == IMATRIX_COV_APPEND_MERGE) {
        if (!fs::exists(out_path, ec)) {
            error_msg = string_format("covariance file '%s' does not exist for append mode", params.cov_out_file.c_str());
            return false;
        }

        std::vector<moe_cov_target_data> existing_targets;
        if (!load_existing_covariance(out_path, params.cov_precision, fingerprint, existing_targets, metadata, error_msg)) {
            return false;
        }

        if (!params.prompt_file.empty()) {
            std::set<std::string> seen(metadata.sources.begin(), metadata.sources.end());
            if (!seen.count(params.prompt_file)) {
                metadata.sources.push_back(params.prompt_file);
            }
        }

        if (!merge_targets(existing_targets, targets, params.cov_precision, final_targets, error_msg)) {
            return false;
        }
    }

    const auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const fs::path tmp_path = fs::path(string_format("%s.tmp.%lld", out_path.string().c_str(), (long long) now));

    return write_covariance_file(out_path, tmp_path, params.cov_precision, fingerprint, metadata, params, final_targets, error_msg);
}
