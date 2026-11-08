#pragma once

#include <string>
#include <mlx/mlx.h>
#include "bonsai/config.hpp"
#include "bonsai/weights.hpp"

namespace bonsai {

struct QuantLinear {
    mx::array weight;
    mx::array scales;
    mx::array biases;
    int group_size = 0;
    int bits = 0;

    static QuantLinear from_weights(const Weights& w, const std::string& prefix, const ModelConfig& cfg);

    mx::array forward(const mx::array& x) const;
};

mx::array embed(const QuantLinear& table, const mx::array& ids);

}