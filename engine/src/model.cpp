#include "bonsai/model.hpp"

namespace bonsai {

QuantLinear QuantLinear::from_weights(const Weights& w, const std::string& prefix, const ModelConfig& cfg) {
    return {w.get(prefix + ".weight"), w.get(prefix + ".scales"), w.get(prefix + ".biases"), 
            cfg.group_size, cfg.bits};
}

mx::array QuantLinear::forward(const mx::array& x) const {
    // transpose=true: weight is stored [out, in], and we want x · Wᵀ
    return mx::quantized_matmul(x, weight, scales, biases, true, group_size, bits);
}

mx::array embed(const QuantLinear& table, const mx::array& ids) {
    // Gather only the needed rows, still packed: [B, L, 256], not the whole 151,669-row table
    const auto w = mx::take(table.weight, ids, 0);
    const auto s = mx::take(table.scales, ids, 0);
    const auto b = mx::take(table.biases, ids, 0);
    // lazy: this builds the graph; nothing is unpacked until eval
    return mx::dequantize(w, s, b, table.group_size, table.bits);
}

}