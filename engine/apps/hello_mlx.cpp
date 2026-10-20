#include <iostream>
#include <mlx/mlx.h>

namespace mx = mlx::core;

int main() {
    mx::array a ({1.0f, 2.0f, 3.0f}); // lazy evaluation -> no memory allocated yet
    mx::array b ({10.0f, 20.0f, 30.0f}); 
    
    auto c = a + b;
    std::cout << "shape before eval: " << c.shape().size() << "-D, " << c.size() << std::endl;

    // eval runs the graph and allocates memory for the output array
    mx::eval(c); 
    std::cout << "device: " << mx::default_device() << std::endl;
    std::cout << "a + b = " << c << std::endl;

    return 0;
}