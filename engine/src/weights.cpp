#include "bonsai/weights.hpp"

#include <sstream>
#include <stdexcept>
#include <utility>

namespace bonsai {

namespace {

// A quantized linear layer is three tensors. Codes are packed 32/bits per uint32,
// with one fp16 scale and bias per group of group_size inputs.
void add_quantized(std::map<std::string, TensorSpec>& specs, const std::string& prefix,
                   int out_features, int in_features, const ModelConfig& cfg) {
    const int packed = in_features * cfg.bits / 32;
    const int groups = in_features / cfg.group_size;
    specs[prefix + ".weight"] = {{out_features, packed}, mx::uint32};
    specs[prefix + ".scales"] = {{out_features, groups}, mx::float16};
    specs[prefix + ".biases"] = {{out_features, groups}, mx::float16};
}

}  // namespace

std::map<std::string, TensorSpec> expected_tensors(const ModelConfig& cfg) {
    std::map<std::string, TensorSpec> specs;
    const int hidden = cfg.hidden_size;
    const int q_out = cfg.num_attention_heads * cfg.head_dim;
    const int kv_out = cfg.num_key_value_heads * cfg.head_dim;

    // The embedding is quantized too: a [vocab, hidden] matrix like lm_head
    add_quantized(specs, "model.embed_tokens", cfg.vocab_size, hidden, cfg);
    add_quantized(specs, "lm_head", cfg.vocab_size, hidden, cfg);
    specs["model.norm.weight"] = {{hidden}, mx::float16};

    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        const std::string layer = "model.layers." + std::to_string(i);
        specs[layer + ".input_layernorm.weight"] = {{hidden}, mx::float16};
        specs[layer + ".post_attention_layernorm.weight"] = {{hidden}, mx::float16};

        // QK-norm weights are per head_dim, shared across heads
        specs[layer + ".self_attn.q_norm.weight"] = {{cfg.head_dim}, mx::float16};
        specs[layer + ".self_attn.k_norm.weight"] = {{cfg.head_dim}, mx::float16};
        add_quantized(specs, layer + ".self_attn.q_proj", q_out, hidden, cfg);
        add_quantized(specs, layer + ".self_attn.k_proj", kv_out, hidden, cfg);
        add_quantized(specs, layer + ".self_attn.v_proj", kv_out, hidden, cfg);
        add_quantized(specs, layer + ".self_attn.o_proj", hidden, q_out, cfg);

        add_quantized(specs, layer + ".mlp.gate_proj", cfg.intermediate_size, hidden, cfg);
        add_quantized(specs, layer + ".mlp.up_proj", cfg.intermediate_size, hidden, cfg);
        add_quantized(specs, layer + ".mlp.down_proj", hidden, cfg.intermediate_size, cfg);
    }
    return specs;
}

std::string format_shape(const mx::Shape& shape) {
    std::string out = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            out += ", ";
        }
        out += std::to_string(shape[i]);
    }
    return out + "]";
}

std::string format_dtype(mx::Dtype dtype) {
    // MLX only provides operator<< for Dtype, so print into a string stream
    std::ostringstream os;
    os << dtype;
    return os.str();
}

Weights::Weights(std::unordered_map<std::string, mx::array> tensors)
    : tensors_(std::move(tensors)) {}

Weights Weights::load(const std::filesystem::path& dir, const ModelConfig& cfg) {
    const auto file = dir / "model.safetensors";
    if (!std::filesystem::exists(file)) {
        throw std::runtime_error("weights: " + file.string() + " not found");
    }
    // structured binding: unpack the returned pair into two names
    auto [tensors, metadata] = mx::load_safetensors(file.string());

    const auto expected = expected_tensors(cfg);
    for (const auto& [name, spec] : expected) {
        const auto it = tensors.find(name);
        if (it == tensors.end()) {
            throw std::runtime_error("weights: missing tensor " + name);
        }
        const mx::array& arr = it->second;
        if (arr.shape() != spec.shape || arr.dtype() != spec.dtype) {
            throw std::runtime_error("weights: " + name + " is " + format_shape(arr.shape()) +
                                     " " + format_dtype(arr.dtype()) + ", expected " +
                                     format_shape(spec.shape) + " " + format_dtype(spec.dtype));
        }
    }
    // Every expected tensor exists, so a size mismatch means extras
    if (tensors.size() != expected.size()) {
        for (const auto& [name, arr] : tensors) {
            if (!expected.contains(name)) {
                throw std::runtime_error("weights: unexpected tensor " + name);
            }
        }
    }

    // std::move: hand the map's storage to Weights instead of copying it
    return Weights(std::move(tensors));
}

const mx::array& Weights::get(const std::string& name) const {
    const auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("weights: no tensor named " + name);
    }
    return it->second;
}

}  // namespace bonsai
