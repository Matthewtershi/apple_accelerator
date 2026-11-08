#include <cstdlib>
#include <filesystem>

#include <catch2/catch_test_macros.hpp>
#include <mlx/mlx.h>

#include "bonsai/config.hpp"
#include "bonsai/model.hpp"
#include "bonsai/weights.hpp"

namespace fs = std::filesystem;
namespace mx = mlx::core;

namespace {

// Empty if unset: real-model tests skip instead of failing on machines without the weights
fs::path model_dir() {
    const char* dir = std::getenv("BONSAI_MODEL_DIR");
    return dir ? fs::path(dir) : fs::path();
}

fs::path reference_file() {
    return fs::path(BONSAI_REFERENCE_DIR) / "ref_cpu.safetensors";
}

// Upcast to fp32 so the subtraction doesn't add its own fp16 rounding
float max_abs_diff(const mx::array& a, const mx::array& b) {
    return mx::max(mx::abs(mx::astype(a, mx::float32) - mx::astype(b, mx::float32)))
        .item<float>();
}

}  // namespace

// 1. Does MLX's fused kernel compute the formula from Part 1?
TEST_CASE("fast::rms_norm matches the formula", "[layers]") {
    const auto x = mx::random::normal({2, 8});
    const auto w = mx::random::normal({8});
    const float eps = 1e-6f;
    // keepdims=true: mean stays [2, 1] so it broadcasts against x [2, 8]
    const auto naive = x * mx::rsqrt(mx::mean(mx::square(x), -1, true) + eps) * w;
    REQUIRE(mx::allclose(mx::fast::rms_norm(x, w, eps), naive, 1e-5, 1e-5).item<bool>());
}

// 2. Is the quantized matmul really "x times dequantized W, transposed"?
TEST_CASE("QuantLinear::forward equals x · dequantize(W)ᵀ", "[layers]") {
    const auto w = mx::random::normal({8, 64});
    // quantize returns {packed codes, scales, biases}
    const auto q = mx::quantize(w, 32, 2);
    const bonsai::QuantLinear lin{q[0], q[1], q[2], 32, 2};

    const auto x = mx::random::normal({3, 64});
    const auto expected = mx::matmul(x, mx::transpose(mx::dequantize(q[0], q[1], q[2], 32, 2)));
    REQUIRE(mx::allclose(lin.forward(x), expected, 1e-4, 1e-4).item<bool>());
}

// 3. Do we match mlx-lm on the real model?
TEST_CASE("embedding and layer-0 RMSNorm match mlx-lm", "[layers]") {
    if (model_dir().empty() || !fs::exists(reference_file())) {
        SKIP("set BONSAI_MODEL_DIR and run scripts/ref/dump_reference.py");
    }
    const auto cfg = bonsai::ModelConfig::from_json(model_dir() / "config.json");
    const auto weights = bonsai::Weights::load(model_dir(), cfg);
    auto [ref, meta] = mx::load_safetensors(reference_file().string());

    const auto table = bonsai::QuantLinear::from_weights(weights, "model.embed_tokens", cfg);
    const auto h = bonsai::embed(table, ref.at("input_ids"));
    const auto normed = mx::fast::rms_norm(
        h, weights.get("model.layers.0.input_layernorm.weight"), cfg.rms_norm_eps);

    // INFO prints only when a REQUIRE fails: shows how far off we were
    INFO("embed max abs diff = " << max_abs_diff(h, ref.at("embed_out")));
    REQUIRE(mx::allclose(h, ref.at("embed_out"), 1e-2, 1e-2).item<bool>());

    INFO("norm max abs diff = " << max_abs_diff(normed, ref.at("layer0_attn_norm")));
    REQUIRE(mx::allclose(normed, ref.at("layer0_attn_norm"), 1e-2, 1e-2).item<bool>());
}
