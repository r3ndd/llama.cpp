#pragma once

#include "ggml.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct llama_cparams;
struct llama_model;

enum class llama_moe_lookup_vector_dtype : uint32_t {
    FP16 = 1,
};

enum class llama_moe_lookup_scaling_mode : uint32_t {
    S_MISSING = 1,
};

// Sidecar binary schema v1 (little-endian):
// [header][model_id_bytes][layer_0][layer_1]...
// layer_i := [layer_header][centroids_f16][contributions_f16][replaced_ids_u32]
struct llama_moe_lookup_header_v1 {
    uint32_t magic;            // "ELT1"
    uint32_t format_version;   // 1
    uint32_t model_id_len;
    uint32_t n_layer;
    uint32_t n_embd;
    uint32_t n_expert;
    uint32_t n_expert_used;
    uint32_t vector_dtype;     // llama_moe_lookup_vector_dtype
    uint32_t scaling_mode;     // llama_moe_lookup_scaling_mode
    uint32_t n_layers_payload;
};

struct llama_moe_lookup_layer_header_v1 {
    uint32_t layer_id;
    uint32_t n_keys;
    uint32_t replaced_count;
};

struct llama_moe_lookup_layer {
    uint32_t layer_id = 0;
    uint32_t n_keys = 0;

    // [n_embd, n_keys]
    std::vector<ggml_fp16_t> centroids;
    // [n_embd, n_keys]
    std::vector<ggml_fp16_t> contributions;
    // [n_keys]
    std::vector<float> centroid_l2_sq;
    // [n_expert]
    std::vector<uint8_t> replaced_mask;

    bool valid() const;
};

class llama_moe_lookup_table {
public:
    static std::unique_ptr<llama_moe_lookup_table> load(
            const llama_model & model,
            const llama_cparams & cparams,
            std::string & warning_out);

    bool valid() const;

    uint32_t format_version() const;
    llama_moe_lookup_vector_dtype vector_dtype() const;
    llama_moe_lookup_scaling_mode scaling_mode() const;

    const llama_moe_lookup_layer * layer(uint32_t layer_id) const;
    bool has_any_active_layers() const;

private:
    bool is_valid = false;

    uint32_t fmt_version = 0;
    llama_moe_lookup_vector_dtype dtype = llama_moe_lookup_vector_dtype::FP16;
    llama_moe_lookup_scaling_mode scaling = llama_moe_lookup_scaling_mode::S_MISSING;

    uint32_t n_layer = 0;
    uint32_t n_embd = 0;
    uint32_t n_expert = 0;
    uint32_t n_expert_used = 0;

    std::string model_id;

    std::unordered_map<uint32_t, llama_moe_lookup_layer> layers;
};
