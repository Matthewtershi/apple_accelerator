#include "bonsai/config.hpp"

#include <fstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace bonsai {

// unnamed namespace: helpers visible only inside this file
namespace {

// template: the compiler stamps out one copy per type used (int, float, bool)
template <typename T>
void expect(const char* field, const T& actual, const T& expected) {
    if (actual != expected) {
        throw std::runtime_error(std::string("config.json: ") + field + " is " +
                                 std::to_string(actual) + ", expected " +
                                 std::to_string(expected));
    }
}

}  // namespace

ModelConfig ModelConfig::from_json(const std::filesystem::path& path) {
    // RAII: the file closes itself when `file` goes out of scope
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("cannot open " + path.string());
    }
    const nlohmann::json j = nlohmann::json::parse(file);

    if (j.at("model_type").get<std::string>() != "qwen3") {
        throw std::runtime_error("config.json: model_type is not qwen3");
    }

    // .at() throws on a missing key; operator[] would silently return null
    ModelConfig cfg;
    cfg.vocab_size = j.at("vocab_size").get<int>();
    cfg.hidden_size = j.at("hidden_size").get<int>();
    cfg.intermediate_size = j.at("intermediate_size").get<int>();
    cfg.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    cfg.num_attention_heads = j.at("num_attention_heads").get<int>();
    cfg.num_key_value_heads = j.at("num_key_value_heads").get<int>();
    cfg.head_dim = j.at("head_dim").get<int>();
    cfg.rope_theta = j.at("rope_theta").get<float>();
    cfg.rms_norm_eps = j.at("rms_norm_eps").get<float>();
    cfg.tie_word_embeddings = j.at("tie_word_embeddings").get<bool>();

    const auto& rope = j.at("rope_scaling");
    cfg.rope_scaling.type = rope.at("rope_type").get<std::string>();
    cfg.rope_scaling.factor = rope.at("factor").get<float>();
    cfg.rope_scaling.original_max_position_embeddings =
        rope.at("original_max_position_embeddings").get<int>();

    const auto& quant = j.at("quantization");
    cfg.group_size = quant.at("group_size").get<int>();
    cfg.bits = quant.at("bits").get<int>();

    // One model only: any other value means the wrong files were loaded
    expect("vocab_size", cfg.vocab_size, 151669);
    expect("hidden_size", cfg.hidden_size, 4096);
    expect("intermediate_size", cfg.intermediate_size, 12288);
    expect("num_hidden_layers", cfg.num_hidden_layers, 36);
    expect("num_attention_heads", cfg.num_attention_heads, 32);
    expect("num_key_value_heads", cfg.num_key_value_heads, 8);
    expect("head_dim", cfg.head_dim, 128);
    expect("rope_theta", cfg.rope_theta, 1e6f);
    expect("rms_norm_eps", cfg.rms_norm_eps, 1e-6f);
    expect("tie_word_embeddings", cfg.tie_word_embeddings, false);
    expect("rope_scaling.factor", cfg.rope_scaling.factor, 4.0f);
    expect("rope_scaling.original_max_position_embeddings",
           cfg.rope_scaling.original_max_position_embeddings, 16384);
    expect("group_size", cfg.group_size, 128);
    expect("bits", cfg.bits, 2);
    if (cfg.rope_scaling.type != "yarn") {
        throw std::runtime_error("config.json: rope_scaling.rope_type is not yarn");
    }

    return cfg;
}

}  // namespace bonsai
