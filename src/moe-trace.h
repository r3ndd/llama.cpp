#pragma once

#include "ggml.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

class llama_model;

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
        WEIGHTS,
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
        std::vector<float> topk_weights;
        std::vector<float> y_full;

        bool has_h_pre = false;
        bool has_topk = false;
        bool has_weights = false;
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

    int32_t n_embd = -1;
    int32_t n_topk = -1;

    std::unordered_map<const ggml_tensor *, tensor_meta> tensor_registry;
    std::unordered_map<int, layer_pending> pending_by_layer;

    std::vector<int32_t> layer_ids;
    std::vector<int32_t> token_ids;
    std::vector<ggml_fp16_t> h_pre_moe;
    std::vector<int32_t> topk_ids;
    std::vector<ggml_fp16_t> topk_weights;
    std::vector<ggml_fp16_t> y_full;
};
