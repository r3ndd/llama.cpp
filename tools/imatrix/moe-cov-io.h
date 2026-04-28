#pragma once

#include "common.h"
#include "ggml.h"

#include <string>
#include <vector>

struct llama_model;

struct moe_cov_target_data {
    uint32_t layer = 0;
    uint32_t expert = 0;
    uint32_t dim = 0;
    uint64_t n = 0;

    std::string role;
    std::string role_variant;
    std::string tensor_name;

    std::vector<int8_t> sum_f8;
    std::vector<int8_t> outer_f8;

    std::vector<ggml_fp16_t> sum_f16;
    std::vector<ggml_fp16_t> outer_f16;

    std::vector<float> sum_f32;
    std::vector<float> outer_f32;

    std::vector<double> sum_f64;
    std::vector<double> outer_f64;
};

bool moe_cov_write_file(
        const common_params & params,
        const llama_model * model,
        const std::vector<moe_cov_target_data> & targets,
        std::string & error_msg);
