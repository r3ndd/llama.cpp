#include "testing.h"

#include "../src/moe-trace.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

static void test_topk_consistency_ok(testing & t) {
    const int n_tokens = 2;
    const int n_topk = 3;
    const int n_expert = 8;

    const std::vector<int32_t> ids = {
        1, 4, 6,
        0, 3, 7,
    };
    const std::vector<float> weights = {
        0.50f, 0.30f, 0.20f,
        0.70f, 0.20f, 0.10f,
    };

    std::string err;
    const bool ok = llama_moe_trace_validate_topk_consistency(
        ids.data(),
        weights.data(),
        n_topk,
        n_tokens,
        n_expert,
        &err);

    t.assert_true("valid top-k IDs/weights should pass", ok);
    t.assert_true("error should be empty on success", err.empty());
}

static void test_topk_consistency_rejects_duplicate_ids(testing & t) {
    const std::vector<int32_t> ids = { 2, 2, 1 };
    const std::vector<float> weights = { 0.6f, 0.3f, 0.1f };

    std::string err;
    const bool ok = llama_moe_trace_validate_topk_consistency(
        ids.data(),
        weights.data(),
        3,
        1,
        16,
        &err);

    t.assert_true("duplicate expert IDs must fail", !ok);
    t.assert_equal("duplicate expert IDs error", std::string("top-k expert ids contain duplicates"), err);
}

static void test_topk_consistency_rejects_non_finite_weights(testing & t) {
    const std::vector<int32_t> ids = { 0, 1 };
    const std::vector<float> weights = { 0.5f, std::numeric_limits<float>::infinity() };

    std::string err;
    const bool ok = llama_moe_trace_validate_topk_consistency(
        ids.data(),
        weights.data(),
        2,
        1,
        4,
        &err);

    t.assert_true("non-finite weights must fail", !ok);
    t.assert_equal("non-finite weight error", std::string("top-k weight is not finite"), err);
}

static void test_topk_consistency_rejects_out_of_range_expert(testing & t) {
    const std::vector<int32_t> ids = { 0, 8 };
    const std::vector<float> weights = { 0.5f, 0.5f };

    std::string err;
    const bool ok = llama_moe_trace_validate_topk_consistency(
        ids.data(),
        weights.data(),
        2,
        1,
        8,
        &err);

    t.assert_true("out-of-range expert id must fail", !ok);
    t.assert_equal("out-of-range expert id error", std::string("top-k expert id out of range"), err);
}

static void test_topk_parity_ok(testing & t) {
    const int n_tokens = 2;
    const int n_topk = 2;
    const int n_argsort = 5;

    const std::vector<int32_t> topk = {
        9, 3,
        7, 1,
    };
    const std::vector<int32_t> argsort = {
        9, 3, 5, 8, 1,
        7, 1, 0, 4, 2,
    };

    std::string err;
    const bool ok = llama_moe_trace_validate_topk_parity(
        topk.data(),
        n_topk,
        argsort.data(),
        n_argsort,
        n_tokens,
        &err);

    t.assert_true("top-k should match argsort prefix", ok);
    t.assert_true("error should be empty on success", err.empty());
}

static void test_topk_parity_rejects_prefix_mismatch(testing & t) {
    const std::vector<int32_t> topk = { 9, 2 };
    const std::vector<int32_t> argsort = { 9, 3, 1 };

    std::string err;
    const bool ok = llama_moe_trace_validate_topk_parity(
        topk.data(),
        2,
        argsort.data(),
        3,
        1,
        &err);

    t.assert_true("mismatched prefix should fail parity", !ok);
    t.assert_equal("prefix mismatch error", std::string("top-k ids mismatch vs argsort prefix"), err);
}

static void test_topk_parity_rejects_invalid_dimensions(testing & t) {
    const std::vector<int32_t> topk = { 0, 1, 2 };
    const std::vector<int32_t> argsort = { 0, 1 };

    std::string err;
    const bool ok = llama_moe_trace_validate_topk_parity(
        topk.data(),
        3,
        argsort.data(),
        2,
        1,
        &err);

    t.assert_true("n_topk cannot exceed n_argsort", !ok);
    t.assert_equal("dimension check error", std::string("top-k width exceeds argsort width"), err);
}

static void test_topk_expert_outputs_ok(testing & t) {
    const int n_tokens = 2;
    const int n_topk = 2;
    const int n_embd = 3;

    const std::vector<float> outputs = {
        0.1f, 0.2f, 0.3f,
        0.4f, 0.5f, 0.6f,
        0.7f, 0.8f, 0.9f,
        1.0f, 1.1f, 1.2f,
    };

    std::string err;
    const bool ok = llama_moe_trace_validate_topk_expert_outputs(
        outputs.data(),
        n_topk,
        n_tokens,
        n_embd,
        &err);

    t.assert_true("finite top-k expert outputs should pass", ok);
    t.assert_true("error should be empty on success", err.empty());
}

static void test_topk_expert_outputs_rejects_non_finite(testing & t) {
    const std::vector<float> outputs = {
        0.1f, 0.2f,
        std::numeric_limits<float>::quiet_NaN(), 0.4f,
    };

    std::string err;
    const bool ok = llama_moe_trace_validate_topk_expert_outputs(
        outputs.data(),
        2,
        1,
        2,
        &err);

    t.assert_true("non-finite expert output must fail", !ok);
    t.assert_equal("non-finite expert output error", std::string("top-k expert output is not finite"), err);
}

static void test_expected_topk_accepts_matching_width(testing & t) {
    std::string err;
    const bool ok = llama_moe_trace_validate_expected_topk(8, 8, &err);

    t.assert_true("matching traced top-k width should pass", ok);
    t.assert_true("error should be empty on success", err.empty());
}

static void test_expected_topk_rejects_mismatch(testing & t) {
    std::string err;
    const bool ok = llama_moe_trace_validate_expected_topk(256, 8, &err);

    t.assert_true("top-k width mismatch must fail", !ok);
    t.assert_equal(
        "top-k width mismatch error",
        std::string("top-k width does not match model n_expert_used"),
        err);
}

int main() {
    testing t(std::cout);
    t.test("topk consistency accepts valid IDs/weights", test_topk_consistency_ok);
    t.test("topk consistency rejects duplicate IDs", test_topk_consistency_rejects_duplicate_ids);
    t.test("topk consistency rejects non-finite weights", test_topk_consistency_rejects_non_finite_weights);
    t.test("topk consistency rejects out-of-range IDs", test_topk_consistency_rejects_out_of_range_expert);
    t.test("topk parity accepts argsort prefix match", test_topk_parity_ok);
    t.test("topk parity rejects prefix mismatch", test_topk_parity_rejects_prefix_mismatch);
    t.test("topk parity rejects invalid dimensions", test_topk_parity_rejects_invalid_dimensions);
    t.test("topk expert outputs accepts finite values", test_topk_expert_outputs_ok);
    t.test("topk expert outputs rejects non-finite values", test_topk_expert_outputs_rejects_non_finite);
    t.test("expected top-k accepts matching width", test_expected_topk_accepts_matching_width);
    t.test("expected top-k rejects mismatched width", test_expected_topk_rejects_mismatch);
    return t.summary();
}
