// Print every tensor as "name shape dtype", sorted, to stdout so it can be diffed
// against scripts/reference_tensors.txt. The config summary goes to stderr.
#include <algorithm>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "bonsai/config.hpp"
#include "bonsai/weights.hpp"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: inspect <model_dir>\n";
        return 1;
    }

    try {
        const std::filesystem::path dir = argv[1];
        const auto cfg = bonsai::ModelConfig::from_json(dir / "config.json");
        const auto weights = bonsai::Weights::load(dir, cfg);

        // unordered_map has no order; sort the names so the output is diffable
        std::vector<std::string> names;
        for (const auto& [name, arr] : weights.all()) {
            names.push_back(name);
        }
        std::sort(names.begin(), names.end());

        for (const auto& name : names) {
            const auto& arr = weights.get(name);
            std::cout << name << " " << bonsai::format_shape(arr.shape()) << " "
                      << bonsai::format_dtype(arr.dtype()) << "\n";
        }

        std::cerr << "tensors: " << names.size() << "\n"
                  << "layers: " << cfg.num_hidden_layers << ", hidden: " << cfg.hidden_size
                  << ", heads: " << cfg.num_attention_heads << "/" << cfg.num_key_value_heads
                  << " x " << cfg.head_dim << "\n"
                  << "quantization: bits=" << cfg.bits << ", group_size=" << cfg.group_size
                  << "\n"
                  << "rope: theta=" << cfg.rope_theta << ", " << cfg.rope_scaling.type
                  << " x" << cfg.rope_scaling.factor << "\n";
    } catch (const std::exception& e) {
        // Both our runtime_errors and nlohmann's exceptions derive from std::exception
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
