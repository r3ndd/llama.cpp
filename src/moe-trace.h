#pragma once

#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

class llama_model;

inline bool llama_moe_trace_validate_topk_consistency(
        const int32_t * topk_ids,
        const float   * topk_weights,
        int n_topk,
        int n_tokens,
        int n_expert,
        std::string * err_msg = nullptr) {
    if (topk_ids == nullptr || topk_weights == nullptr) {
        if (err_msg) {
            *err_msg = "top-k buffers are null";
        }
        return false;
    }
    if (n_topk <= 0 || n_tokens < 0) {
        if (err_msg) {
            *err_msg = "invalid top-k/tokens dimensions";
        }
        return false;
    }
    if (n_expert <= 0) {
        if (err_msg) {
            *err_msg = "invalid expert count";
        }
        return false;
    }

    for (int tok = 0; tok < n_tokens; ++tok) {
        const int row = tok * n_topk;
        for (int k = 0; k < n_topk; ++k) {
            const int32_t id = topk_ids[row + k];
            const float w = topk_weights[row + k];

            if (id < 0 || id >= n_expert) {
                if (err_msg) {
                    *err_msg = "top-k expert id out of range";
                }
                return false;
            }
            if (!std::isfinite(w)) {
                if (err_msg) {
                    *err_msg = "top-k weight is not finite";
                }
                return false;
            }

            for (int j = k + 1; j < n_topk; ++j) {
                if (id == topk_ids[row + j]) {
                    if (err_msg) {
                        *err_msg = "top-k expert ids contain duplicates";
                    }
                    return false;
                }
            }
        }
    }

    return true;
}

inline bool llama_moe_trace_validate_topk_parity(
        const int32_t * topk_ids,
        int n_topk,
        const int32_t * argsort_ids,
        int n_argsort,
        int n_tokens,
        std::string * err_msg = nullptr) {
    if (topk_ids == nullptr || argsort_ids == nullptr) {
        if (err_msg) {
            *err_msg = "parity buffers are null";
        }
        return false;
    }
    if (n_topk <= 0 || n_argsort <= 0 || n_tokens < 0) {
        if (err_msg) {
            *err_msg = "invalid parity dimensions";
        }
        return false;
    }
    if (n_topk > n_argsort) {
        if (err_msg) {
            *err_msg = "top-k width exceeds argsort width";
        }
        return false;
    }

    for (int tok = 0; tok < n_tokens; ++tok) {
        const int topk_row = tok * n_topk;
        const int arg_row = tok * n_argsort;
        for (int k = 0; k < n_topk; ++k) {
            if (topk_ids[topk_row + k] != argsort_ids[arg_row + k]) {
                if (err_msg) {
                    *err_msg = "top-k ids mismatch vs argsort prefix";
                }
                return false;
            }
        }
    }

    return true;
}

inline bool llama_moe_trace_validate_topk_expert_outputs(
        const float * topk_expert_outputs,
        int n_topk,
        int n_tokens,
        int n_embd,
        std::string * err_msg = nullptr) {
    if (topk_expert_outputs == nullptr) {
        if (err_msg) {
            *err_msg = "top-k expert output buffer is null";
        }
        return false;
    }
    if (n_topk <= 0 || n_embd <= 0 || n_tokens < 0) {
        if (err_msg) {
            *err_msg = "invalid top-k expert output dimensions";
        }
        return false;
    }

    const size_t n = (size_t) n_topk * (size_t) n_tokens * (size_t) n_embd;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(topk_expert_outputs[i])) {
            if (err_msg) {
                *err_msg = "top-k expert output is not finite";
            }
            return false;
        }
    }

    return true;
}

class llama_moe_trace_writer {
public:
    llama_moe_trace_writer(const llama_model & model, const std::string & output_path);
    ~llama_moe_trace_writer();

    bool valid() const;

    void begin_graph();
    void reset_registry();

    void register_tensor(const ggml_tensor * t, const char * name, int il);

    bool wants_tensor(const ggml_tensor * t) const;
    bool observe_tensor(const ggml_tensor * t);

private:
    enum class tensor_kind {
        H_PRE,
        TOPK,
        ARGSORT,
        WEIGHTS,
        TOPK_EXPERT_OUTPUTS,
        Y_FULL,
    };

    struct tensor_meta {
        tensor_kind kind;
        int layer = -1;
    };

    struct layer_pending {
        int n_tokens = -1;
        int n_embd = -1;
        int n_topk = -1;

        std::vector<float> h_pre;
        std::vector<int32_t> topk_ids;
        std::vector<int32_t> argsort_ids;
        std::vector<float> topk_weights;
        std::vector<float> topk_expert_outputs;
        std::vector<float> y_full;

        bool has_h_pre = false;
        bool has_topk = false;
        bool has_argsort = false;
        bool has_weights = false;
        bool has_topk_expert_outputs = false;
        bool has_y_full = false;
    };

    bool read_tensor_f32(const ggml_tensor * t, std::vector<float> & out) const;
    bool read_tensor_i32(const ggml_tensor * t, std::vector<int32_t> & out) const;
    bool ingest(const tensor_meta & meta, const ggml_tensor * t);
    bool try_finalize_layer(int layer);
    bool flush_npz();

private:
    std::string output_path;
    std::string model_id;
    int32_t n_layer = 0;
    int32_t n_expert = 0;
    int32_t n_expert_used = 0;

    bool is_valid = false;
    bool flushed = false;
    bool warn_once_bad_tensor = false;
    bool warn_once_bad_parity = false;

    int32_t n_embd = -1;
    int32_t n_topk = -1;

    std::unordered_map<const ggml_tensor *, tensor_meta> tensor_registry;
    std::unordered_map<int, layer_pending> pending_by_layer;

    std::vector<int32_t> layer_ids;
    std::vector<int32_t> token_ids;
    std::vector<ggml_fp16_t> h_pre_moe;
    std::vector<int32_t> topk_ids;
    std::vector<ggml_fp16_t> topk_weights;
    std::vector<ggml_fp16_t> topk_expert_outputs;
    std::vector<ggml_fp16_t> y_full;
};
