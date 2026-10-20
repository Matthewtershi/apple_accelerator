#include <catch2/catch_test_macros.hpp>
#include <mlx/mlx.h>

namespace mx = mlx::core;

TEST_CASE("MLX adds two scalars", "[smoke]") {
    auto c = mx::array({1.0f}) + mx::array({2.0f});
    mx::eval(c);

    REQUIRE(c.item<float>() == 3.0f);
}

TEST_CASE("MLX adds two vectors elementwise", "[smoke]") {
    auto c = mx::array({1.0f, 2.0f, 3.0f}) + mx::array({10.0f, 20.0f, 30.0f});
    REQUIRE(c.shape() == mx::Shape{3});
    REQUIRE(mx::allclose(c, mx::array({11.0f, 22.0f, 33.0f})).item<bool>());
}

