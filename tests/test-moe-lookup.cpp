#include "testing.h"

#include "../src/moe-lookup.h"

static void test_layer_valid_true_on_consistent_payload(testing & t) {
    llama_moe_lookup_layer layer;
    layer.layer_id = 0;
    layer.n_keys = 2;
    layer.centroids.resize(8);
    layer.contributions.resize(8);
    layer.centroid_l2_sq.resize(2);
    layer.replaced_mask.resize(4, 0);

    t.assert_true("layer payload should be valid", layer.valid());
}

static void test_layer_valid_false_on_shape_mismatch(testing & t) {
    llama_moe_lookup_layer layer;
    layer.layer_id = 0;
    layer.n_keys = 3;
    layer.centroids.resize(8);
    layer.contributions.resize(7);
    layer.centroid_l2_sq.resize(3);
    layer.replaced_mask.resize(4, 0);

    t.assert_true("shape mismatch should invalidate layer", !layer.valid());
}

static void test_layer_valid_false_on_invalid_key_partition(testing & t) {
    llama_moe_lookup_layer layer;
    layer.layer_id = 0;
    layer.n_keys = 3;
    layer.centroids.resize(10);
    layer.contributions.resize(10);
    layer.centroid_l2_sq.resize(3);
    layer.replaced_mask.resize(4, 0);

    t.assert_true("non-divisible key partition should invalidate layer", !layer.valid());
}

int main() {
    testing t(std::cout);
    t.test("moe lookup layer valid accepts consistent payload", test_layer_valid_true_on_consistent_payload);
    t.test("moe lookup layer valid rejects shape mismatch", test_layer_valid_false_on_shape_mismatch);
    t.test("moe lookup layer valid rejects invalid key partition", test_layer_valid_false_on_invalid_key_partition);
    return t.summary();
}
