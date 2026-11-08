#pragma once

#include <filesystem>
#include <string>

namespace bonsai {

// YaRN RoPE settings: stretch the 16K training context to 64K (used in M4)
struct RopeScaling {
    std::string type;
    float factor = 0.0f;
    int original_max_position_embeddings = 0;
};

// Model hyperparameters from config.json. See engine/README.md §3.2.
// "= 0" defaults: an uninitialized int would hold garbage
struct ModelConfig {
    int vocab_size = 0;
    int hidden_size = 0;
    int intermediate_size = 0;
    int num_hidden_layers = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int head_dim = 0;
    float rope_theta = 0.0f;
    float rms_norm_eps = 0.0f;
    bool tie_word_embeddings = false;
    RopeScaling rope_scaling;
    int group_size = 0;
    int bits = 0;

    // Parse config.json; throws if a field is missing or differs from Bonsai 8B.
    // static: a named constructor, called as ModelConfig::from_json(path)
    static ModelConfig from_json(const std::filesystem::path& path);
};

}  // namespace bonsai
