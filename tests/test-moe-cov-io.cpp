#include "common.h"
#include "ggml.h"
#include "gguf.h"

#include "../tools/imatrix/moe-cov-io.h"

#include <algorithm>
#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

static common_params make_cov_params(
        const std::string & out_file,
        enum imatrix_cov_file_mode mode,
        enum imatrix_cov_precision precision,
        const std::string & prompt_file = "") {
    common_params params;
    params.moe_trace_cov = true;
    params.cov_out_file = out_file;
    params.cov_file_mode = mode;
    params.cov_precision = precision;
    params.prompt_file = prompt_file;
    params.cov_layers = "";
    params.cov_experts = "";
    params.cov_targets = "all";
    return params;
}

static moe_cov_target_data make_target_f32(
        uint32_t layer,
        uint32_t expert,
        const std::string & role,
        const std::string & tensor_name,
        uint32_t dim,
        uint64_t n,
        std::vector<float> sum,
        std::vector<float> outer,
        const std::string & role_variant = "") {
    moe_cov_target_data t;
    t.layer = layer;
    t.expert = expert;
    t.role = role;
    t.role_variant = role_variant;
    t.tensor_name = tensor_name;
    t.dim = dim;
    t.n = n;
    t.sum_f32 = std::move(sum);
    t.outer_f32 = std::move(outer);
    return t;
}

static std::string require_val_str(const gguf_context * ctx, const std::string & key) {
    const int64_t kid = gguf_find_key(ctx, key.c_str());
    assert(kid >= 0);
    assert(gguf_get_kv_type(ctx, kid) == GGUF_TYPE_STRING);
    return gguf_get_val_str(ctx, kid);
}

static uint32_t require_val_u32(const gguf_context * ctx, const std::string & key) {
    const int64_t kid = gguf_find_key(ctx, key.c_str());
    assert(kid >= 0);
    assert(gguf_get_kv_type(ctx, kid) == GGUF_TYPE_UINT32);
    return gguf_get_val_u32(ctx, kid);
}

static std::vector<std::string> require_val_arr_str(const gguf_context * ctx, const std::string & key) {
    const int64_t kid = gguf_find_key(ctx, key.c_str());
    assert(kid >= 0);
    assert(gguf_get_kv_type(ctx, kid) == GGUF_TYPE_ARRAY);
    assert(gguf_get_arr_type(ctx, kid) == GGUF_TYPE_STRING);

    const size_t n = gguf_get_arr_n(ctx, kid);
    std::vector<std::string> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        out.emplace_back(gguf_get_arr_str(ctx, kid, i));
    }
    return out;
}

static gguf_context * load_gguf_with_data(const std::string & path, ggml_context ** ctx_data) {
    gguf_init_params params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ ctx_data,
    };
    return gguf_init_from_file(path.c_str(), params);
}

static uint64_t tensor_n_value(const ggml_context * data_ctx, const std::string & name) {
    const ggml_tensor * t = ggml_get_tensor(const_cast<ggml_context *>(data_ctx), name.c_str());
    assert(t != nullptr);
    assert(t->type == GGML_TYPE_I64);
    assert(ggml_nelements(t) == 1);
    const int64_t v = ((const int64_t *) t->data)[0];
    assert(v >= 0);
    return (uint64_t) v;
}

static std::vector<float> tensor_f32_data(const ggml_context * data_ctx, const std::string & name, size_t n) {
    const ggml_tensor * t = ggml_get_tensor(const_cast<ggml_context *>(data_ctx), name.c_str());
    assert(t != nullptr);
    assert(t->type == GGML_TYPE_F32);
    assert((size_t) ggml_nelements(t) == n);
    const float * p = (const float *) t->data;
    return std::vector<float>(p, p + n);
}

static bool any_abs_gt(const std::vector<float> & v, float eps) {
    for (const float x : v) {
        if (std::fabs(x) > eps) {
            return true;
        }
    }
    return false;
}

static std::string find_target_id(
        const gguf_context * ctx,
        uint32_t layer,
        uint32_t expert,
        const std::string & role,
        const std::string & tensor_name) {
    const uint32_t n_targets = require_val_u32(ctx, "moe_cov.target_count");
    for (uint32_t i = 0; i < n_targets; ++i) {
        const std::string tid = string_format("t%u", i);
        const uint32_t cur_layer = require_val_u32(ctx, string_format("moe_cov.target.%s.layer", tid.c_str()));
        const uint32_t cur_expert = require_val_u32(ctx, string_format("moe_cov.target.%s.expert", tid.c_str()));
        const std::string cur_role = require_val_str(ctx, string_format("moe_cov.target.%s.role", tid.c_str()));
        const std::string cur_tensor_name = require_val_str(ctx, string_format("moe_cov.target.%s.tensor_name", tid.c_str()));
        if (cur_layer == layer && cur_expert == expert && cur_role == role && cur_tensor_name == tensor_name) {
            return tid;
        }
    }
    return "";
}

int main() {
    namespace fs = std::filesystem;

    const fs::path out_path = fs::temp_directory_path() / "llama-test-moe-cov-io.gguf";
    {
        std::error_code ec;
        fs::remove(out_path, ec);
    }

    {
        std::string error_msg;
        const common_params create_params = make_cov_params(out_path.string(), IMATRIX_COV_CREATE, IMATRIX_COV_F32, "source-a.txt");

        std::vector<moe_cov_target_data> targets;
        targets.push_back(make_target_f32(
            /*layer=*/ 0,
            /*expert=*/ 0,
            /*role=*/ "up",
            /*tensor_name=*/ "blk.0.ffn_up_exps.weight",
            /*dim=*/ 2,
            /*n=*/ 0,
            /*sum=*/ {0.0f, 0.0f},
            /*outer=*/ {0.0f, 0.0f, 0.0f, 0.0f}));

        targets.push_back(make_target_f32(
            /*layer=*/ 1,
            /*expert=*/ 2,
            /*role=*/ "gate",
            /*tensor_name=*/ "blk.1.ffn_gate_exps.weight",
            /*dim=*/ 2,
            /*n=*/ 1,
            /*sum=*/ {2.0f, -3.0f},
            /*outer=*/ {4.0f, -6.0f, -6.0f, 9.0f}));

        assert(moe_cov_write_file(create_params, nullptr, targets, error_msg));
    }

    // Validate n=0 and n=1 covariance output behavior and metadata.
    {
        ggml_context * data_ctx = nullptr;
        gguf_context * gguf_ctx = load_gguf_with_data(out_path.string(), &data_ctx);
        assert(gguf_ctx != nullptr);
        assert(data_ctx != nullptr);

        assert(require_val_str(gguf_ctx, "general.type") == "moe_covariance");
        assert(require_val_u32(gguf_ctx, "moe_cov.version") == 1);
        assert(require_val_str(gguf_ctx, "moe_cov.convention") == "population");
        assert(require_val_str(gguf_ctx, "moe_cov.precision") == "f32");
        assert(require_val_u32(gguf_ctx, "moe_cov.target_count") == 2);

        const std::vector<std::string> sources = require_val_arr_str(gguf_ctx, "moe_cov.sources");
        assert(sources.size() == 1);
        assert(sources[0] == "source-a.txt");

        const std::string tid_n0 = find_target_id(gguf_ctx, 0, 0, "up", "blk.0.ffn_up_exps.weight");
        const std::string tid_n1 = find_target_id(gguf_ctx, 1, 2, "gate", "blk.1.ffn_gate_exps.weight");
        assert(!tid_n0.empty());
        assert(!tid_n1.empty());

        assert(tensor_n_value(data_ctx, string_format("moe_cov.%s.n", tid_n0.c_str())) == 0);
        assert(tensor_n_value(data_ctx, string_format("moe_cov.%s.n", tid_n1.c_str())) == 1);

        const std::vector<float> cov_n0 = tensor_f32_data(data_ctx, string_format("moe_cov.%s.cov_pop", tid_n0.c_str()), 4);
        const std::vector<float> cov_n1 = tensor_f32_data(data_ctx, string_format("moe_cov.%s.cov_pop", tid_n1.c_str()), 4);
        const bool all_zero_n0 = std::all_of(cov_n0.begin(), cov_n0.end(), [](float value) { return value == 0.0f; });
        const bool all_zero_n1 = std::all_of(cov_n1.begin(), cov_n1.end(), [](float value) { return value == 0.0f; });
        if (!all_zero_n0 || !all_zero_n1) {
            return 1;
        }

        gguf_free(gguf_ctx);
        ggml_free(data_ctx);
    }

    std::string created_at_before;
    {
        ggml_context * data_ctx = nullptr;
        gguf_context * gguf_ctx = load_gguf_with_data(out_path.string(), &data_ctx);
        assert(gguf_ctx != nullptr);
        created_at_before = require_val_str(gguf_ctx, "moe_cov.created_at");
        gguf_free(gguf_ctx);
        ggml_free(data_ctx);
    }

    // Append merge: overlap one target + add one new target.
    {
        std::string error_msg;
        const common_params append_params = make_cov_params(out_path.string(), IMATRIX_COV_APPEND_MERGE, IMATRIX_COV_F32, "source-b.txt");

        std::vector<moe_cov_target_data> targets;
        targets.push_back(make_target_f32(
            /*layer=*/ 1,
            /*expert=*/ 2,
            /*role=*/ "gate",
            /*tensor_name=*/ "blk.1.ffn_gate_exps.weight",
            /*dim=*/ 2,
            /*n=*/ 1,
            /*sum=*/ {1.0f, 1.0f},
            /*outer=*/ {1.0f, 1.0f, 1.0f, 1.0f}));

        targets.push_back(make_target_f32(
            /*layer=*/ 3,
            /*expert=*/ 1,
            /*role=*/ "down",
            /*tensor_name=*/ "blk.3.ffn_down_exps.weight",
            /*dim=*/ 2,
            /*n=*/ 2,
            /*sum=*/ {3.0f, 5.0f},
            /*outer=*/ {9.0f, 15.0f, 15.0f, 25.0f}));

        assert(moe_cov_write_file(append_params, nullptr, targets, error_msg));
    }

    {
        ggml_context * data_ctx = nullptr;
        gguf_context * gguf_ctx = load_gguf_with_data(out_path.string(), &data_ctx);
        assert(gguf_ctx != nullptr);
        assert(data_ctx != nullptr);

        const std::string created_at_after = require_val_str(gguf_ctx, "moe_cov.created_at");
        assert(created_at_after == created_at_before);

        const std::vector<std::string> sources = require_val_arr_str(gguf_ctx, "moe_cov.sources");
        assert(sources.size() == 2);
        assert(std::find(sources.begin(), sources.end(), "source-a.txt") != sources.end());
        assert(std::find(sources.begin(), sources.end(), "source-b.txt") != sources.end());

        assert(require_val_u32(gguf_ctx, "moe_cov.target_count") == 3);

        const std::string tid_merged = find_target_id(gguf_ctx, 1, 2, "gate", "blk.1.ffn_gate_exps.weight");
        assert(!tid_merged.empty());
        assert(tensor_n_value(data_ctx, string_format("moe_cov.%s.n", tid_merged.c_str())) == 2);

        const std::vector<float> sum_merged = tensor_f32_data(data_ctx, string_format("moe_cov.%s.sum", tid_merged.c_str()), 2);
        assert(sum_merged[0] == 3.0f);
        assert(sum_merged[1] == -2.0f);

        const std::string tid_new = find_target_id(gguf_ctx, 3, 1, "down", "blk.3.ffn_down_exps.weight");
        assert(!tid_new.empty());
        assert(tensor_n_value(data_ctx, string_format("moe_cov.%s.n", tid_new.c_str())) == 2);

        gguf_free(gguf_ctx);
        ggml_free(data_ctx);
    }

    // Precision mismatch in append mode must fail.
    {
        std::string error_msg;
        const common_params bad_append_params = make_cov_params(out_path.string(), IMATRIX_COV_APPEND_MERGE, IMATRIX_COV_F16, "source-c.txt");
        const std::vector<moe_cov_target_data> no_targets;
        assert(!moe_cov_write_file(bad_append_params, nullptr, no_targets, error_msg));
        assert(error_msg.find("precision mismatch") != std::string::npos);
    }

    // f8 accumulation should still preserve non-trivial covariance signal in cov_pop (written as f32).
    {
        const fs::path out_f8_path = fs::temp_directory_path() / "llama-test-moe-cov-io-f8.gguf";
        {
            std::error_code ec;
            fs::remove(out_f8_path, ec);
        }

        std::string error_msg;
        const common_params f8_params = make_cov_params(out_f8_path.string(), IMATRIX_COV_CREATE, IMATRIX_COV_F8, "source-f8.txt");

        moe_cov_target_data down;
        down.layer = 0;
        down.expert = 0;
        down.role = "down";
        down.tensor_name = "blk.0.ffn_down_exps.weight";
        down.dim = 2;
        down.n = 136;
        down.sum_f8 = {4, -7};
        down.outer_f8 = {
            4, 0,
            0, 8,
        };

        assert(moe_cov_write_file(f8_params, nullptr, {down}, error_msg));

        ggml_context * data_ctx = nullptr;
        gguf_context * gguf_ctx = load_gguf_with_data(out_f8_path.string(), &data_ctx);
        assert(gguf_ctx != nullptr);
        assert(data_ctx != nullptr);

        const std::string tid = find_target_id(gguf_ctx, 0, 0, "down", "blk.0.ffn_down_exps.weight");
        assert(!tid.empty());

        const std::vector<float> cov = tensor_f32_data(data_ctx, string_format("moe_cov.%s.cov_pop", tid.c_str()), 4);
        assert(any_abs_gt(cov, 1e-6f));

        gguf_free(gguf_ctx);
        ggml_free(data_ctx);

        {
            std::error_code ec;
            fs::remove(out_f8_path, ec);
        }
    }

    {
        std::error_code ec;
        fs::remove(out_path, ec);
    }

    return 0;
}
