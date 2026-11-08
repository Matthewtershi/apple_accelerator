#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <unordered_map>

#include <mlx/mlx.h>

#include "bonsai/config.hpp"

namespace bonsai {

namespace mx = mlx::core;

// The shape and dtype one tensor must have
struct TensorSpec {
    mx::Shape shape;
    // default needed: std::map::operator[] default-constructs before assigning
    mx::Dtype dtype = mx::float16;
};

// Every tensor the model needs, derived from the config (907 for Bonsai 8B).
// std::map, not unordered_map: sorted by name, so iteration order is stable
std::map<std::string, TensorSpec> expected_tensors(const ModelConfig& cfg);

// "[4096, 256]": same format as scripts/inspect_reference.py
std::string format_shape(const mx::Shape& shape);

// "float16": same format as scripts/inspect_reference.py
std::string format_dtype(mx::Dtype dtype);

// Owns every tensor of the model, checked against expected_tensors() at load
class Weights {
public:
    // Load dir/model.safetensors; throws on a missing, extra or wrong-shape tensor.
    // lazy: tensor data is read from disk only when first evaluated
    static Weights load(const std::filesystem::path& dir, const ModelConfig& cfg);

    // Look up one tensor by name; throws if it doesn't exist
    const mx::array& get(const std::string& name) const;

    // All tensors, for inspection
    const std::unordered_map<std::string, mx::array>& all() const { return tensors_; }

private:
    // private constructor: the only way to get a Weights is load(), which validates
    explicit Weights(std::unordered_map<std::string, mx::array> tensors);

    std::unordered_map<std::string, mx::array> tensors_;
};

}  // namespace bonsai
