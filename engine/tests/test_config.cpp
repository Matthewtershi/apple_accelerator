#include <filesystem>
#include <fstream>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include "bonsai/config.hpp"

namespace fs = std::filesystem;
using nlohmann::json;

namespace {

// The fields from_json reads, with Bonsai 8B's real values
json valid_config() {
    return {
        {"model_type", "qwen3"},
        {"vocab_size", 151669},
        {"hidden_size", 4096},
        {"intermediate_size", 12288},
        {"num_hidden_layers", 36},
        {"num_attention_heads", 32},
        {"num_key_value_heads", 8},
        {"head_dim", 128},
        {"rope_theta", 1000000.0},
        {"rms_norm_eps", 1e-06},
        {"tie_word_embeddings", false},
        {"rope_scaling",
         {{"rope_type", "yarn"}, {"factor", 4.0}, {"original_max_position_embeddings", 16384}}},
        {"quantization", {{"group_size", 128}, {"bits", 2}}},
    };
}

// Write j to a temp file and return its path
fs::path write_config(const json& j, const std::string& name) {
    const auto path = fs::temp_directory_path() / ("bonsai_" + name + ".json");
    std::ofstream(path) << j.dump();
    return path;
}

}  // namespace

TEST_CASE("from_json reads a valid Bonsai 8B config", "[config]") {
    const auto cfg = bonsai::ModelConfig::from_json(write_config(valid_config(), "valid"));

    REQUIRE(cfg.hidden_size == 4096);
    REQUIRE(cfg.num_hidden_layers == 36);
    REQUIRE(cfg.num_key_value_heads == 8);
    REQUIRE(cfg.rope_scaling.type == "yarn");
    REQUIRE(cfg.rope_scaling.original_max_position_embeddings == 16384);
    REQUIRE(cfg.bits == 2);
}

TEST_CASE("from_json throws on a missing file", "[config]") {
    REQUIRE_THROWS(bonsai::ModelConfig::from_json("/nonexistent/config.json"));
}

TEST_CASE("from_json throws on a missing key", "[config]") {
    auto j = valid_config();
    j.erase("head_dim");
    REQUIRE_THROWS(bonsai::ModelConfig::from_json(write_config(j, "missing_key")));
}

TEST_CASE("from_json throws on a value that isn't Bonsai 8B", "[config]") {
    auto j = valid_config();
    j["hidden_size"] = 2048;
    REQUIRE_THROWS(bonsai::ModelConfig::from_json(write_config(j, "wrong_value")));
}

TEST_CASE("from_json throws on a different model family", "[config]") {
    auto j = valid_config();
    j["model_type"] = "llama";
    REQUIRE_THROWS(bonsai::ModelConfig::from_json(write_config(j, "wrong_family")));
}
