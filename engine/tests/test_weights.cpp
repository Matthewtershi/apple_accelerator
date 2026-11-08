#include <filesystem>
#include <string>
#include <unordered_map>

#include <catch2/catch_test_macros.hpp>
#include <mlx/mlx.h>

#include "bonsai/config.hpp"
#include "bonsai/weights.hpp"

namespace fs = std::filesystem;
namespace mx = mlx::core;

namespace {

// C++20 designated initializers: name each field, in declaration order
const bonsai::ModelConfig kBonsai8B{
    .vocab_size = 151669,
    .hidden_size = 4096,
    .intermediate_size = 12288,
    .num_hidden_layers = 36,
    .num_attention_heads = 32,
    .num_key_value_heads = 8,
    .head_dim = 128,
    .rope_theta = 1e6f,
    .rms_norm_eps = 1e-6f,
    .tie_word_embeddings = false,
    .rope_scaling = {"yarn", 4.0f, 16384},
    .group_size = 128,
    .bits = 2,
};

// Same structure at toy size, so the test files are kilobytes, not 2.3 GB
const bonsai::ModelConfig kTiny{
    .vocab_size = 16,
    .hidden_size = 64,
    .intermediate_size = 128,
    .num_hidden_layers = 2,
    .num_attention_heads = 4,
    .num_key_value_heads = 2,
    .head_dim = 16,
    .rope_theta = 1e6f,
    .rms_norm_eps = 1e-6f,
    .tie_word_embeddings = false,
    .rope_scaling = {"yarn", 4.0f, 16384},
    .group_size = 32,
    .bits = 2,
};

// A zero-filled tensor for every expected name, matching shape and dtype
std::unordered_map<std::string, mx::array> fake_tensors(const bonsai::ModelConfig& cfg) {
    std::unordered_map<std::string, mx::array> out;
    for (const auto& [name, spec] : bonsai::expected_tensors(cfg)) {
        out.emplace(name, mx::zeros(spec.shape, spec.dtype));
    }
    return out;
}

// Save tensors as dir/model.safetensors in a fresh temp dir and return the dir
fs::path save_model(const std::unordered_map<std::string, mx::array>& tensors,
                    const std::string& name) {
    const auto dir = fs::temp_directory_path() / ("bonsai_" + name);
    fs::remove_all(dir);
    fs::create_directories(dir);
    mx::save_safetensors((dir / "model.safetensors").string(), tensors);
    return dir;
}

}  // namespace

TEST_CASE("expected_tensors matches the real Bonsai 8B layout", "[weights]") {
    const auto specs = bonsai::expected_tensors(kBonsai8B);

    // 6 for embed_tokens + lm_head, 1 final norm, 25 per layer
    REQUIRE(specs.size() == 907);

    const auto& q = specs.at("model.layers.0.self_attn.q_proj.weight");
    REQUIRE(q.shape == mx::Shape{4096, 256});
    REQUIRE(q.dtype == mx::uint32);

    const auto& down = specs.at("model.layers.35.mlp.down_proj.scales");
    REQUIRE(down.shape == mx::Shape{4096, 96});

    REQUIRE(specs.at("model.layers.0.self_attn.k_norm.weight").shape == mx::Shape{128});
}

TEST_CASE("load accepts a complete model", "[weights]") {
    const auto dir = save_model(fake_tensors(kTiny), "complete");
    const auto weights = bonsai::Weights::load(dir, kTiny);

    REQUIRE(weights.all().size() == bonsai::expected_tensors(kTiny).size());
    REQUIRE(weights.get("lm_head.weight").shape() == mx::Shape{16, 4});
    REQUIRE_THROWS(weights.get("no.such.tensor"));
}

TEST_CASE("load throws on a missing tensor", "[weights]") {
    auto tensors = fake_tensors(kTiny);
    tensors.erase("model.layers.1.mlp.up_proj.scales");
    REQUIRE_THROWS(bonsai::Weights::load(save_model(tensors, "missing"), kTiny));
}

TEST_CASE("load throws on a wrong shape", "[weights]") {
    auto tensors = fake_tensors(kTiny);
    tensors.insert_or_assign("model.norm.weight", mx::zeros({32}, mx::float16));
    REQUIRE_THROWS(bonsai::Weights::load(save_model(tensors, "wrong_shape"), kTiny));
}

TEST_CASE("load throws on an unexpected tensor", "[weights]") {
    auto tensors = fake_tensors(kTiny);
    tensors.emplace("model.extra.weight", mx::zeros({4}, mx::float16));
    REQUIRE_THROWS(bonsai::Weights::load(save_model(tensors, "extra"), kTiny));
}

TEST_CASE("load throws when the file is missing", "[weights]") {
    REQUIRE_THROWS(bonsai::Weights::load("/nonexistent", kTiny));
}
