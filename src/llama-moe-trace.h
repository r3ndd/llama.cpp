#pragma once

#include "llama-batch.h"
#include "llama-hparams.h"

#include "ggml.h"

#include <cstdint>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

struct llama_moe_trace_config {
    bool        enabled              = false;
    std::string path                 = "";
    std::string format               = "jsonl";
    std::string precision            = "f16";
    float       sample_rate          = 1.0f;
    int32_t     max_rows_total       = 200000;
    int32_t     max_rows_per_layer   = 0;
    int32_t     max_rows_per_expert  = 0;
    int32_t     buffer_rows          = 2048;
    int32_t     flush_interval_ms    = 1000;
    bool        strict               = false;
};

class llama_moe_trace {
public:
    static std::unique_ptr<llama_moe_trace> init(
            const llama_moe_trace_config & config,
            const llama_hparams          & hparams,
            const std::string            & model_name,
            uint64_t                       seed);

    ~llama_moe_trace();

    bool enabled() const;
    bool wants_tensor(const char * tensor_name) const;
    void set_ubatch(const llama_ubatch & ubatch);
    bool capture_eval(ggml_tensor * t);
    void flush();

private:
    struct row {
        int32_t layer = 0;
        int32_t expert = 0;
        std::vector<float> inputs;
        int32_t seq_id = -1;
        int32_t token_pos = -1;
        int32_t ubatch_index = -1;
        int32_t expert_rank = -1;
    };

    struct token_meta {
        int32_t seq_id = -1;
        int32_t token_pos = -1;
        int32_t ubatch_index = -1;
    };

    struct layer_state {
        std::vector<int32_t> topk;
        bool has_topk = false;
    };

    llama_moe_trace(
            const llama_moe_trace_config & config,
            const llama_hparams          & hparams,
            const std::string            & model_name,
            uint64_t                       seed);

    bool init_writer();
    bool on_topk(ggml_tensor * t, int il);
    bool on_expert_in(ggml_tensor * t, int il);
    void build_token_meta(const llama_ubatch & ubatch, std::vector<token_meta> & out) const;
    bool should_keep(int32_t layer, int32_t expert);
    void append_row(row && out_row);
    void flush_if_needed();
    bool flush_impl();

private:
    llama_moe_trace_config config;
    const llama_hparams & hparams;
    std::string model_name;

    bool active = false;
    bool degraded = false;
    bool has_ubatch_meta = false;

    std::ofstream writer;
    std::vector<row> row_buffer;
    std::unordered_map<int32_t, layer_state> states;
    std::vector<token_meta> ubatch_meta;

    std::unordered_map<int32_t, int64_t> count_by_layer;
    std::unordered_map<uint64_t, int64_t> count_by_layer_expert;

    int64_t rows_emitted = 0;
    int64_t rows_dropped_total = 0;
    int64_t rows_dropped_cap_total = 0;
    int64_t rows_dropped_cap_layer = 0;
    int64_t rows_dropped_cap_expert = 0;
    int64_t rows_dropped_sample = 0;
    int64_t rows_dropped_shape = 0;
    int64_t rows_dropped_io = 0;
    int64_t flush_count = 0;
    int64_t flush_error_count = 0;

    std::mt19937 rng;
    std::uniform_real_distribution<float> bernoulli;
    int64_t flush_last_ms = 0;
};
