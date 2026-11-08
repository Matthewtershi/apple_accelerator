cmake_minimum_required(VERSION 3.24)
project(bonsai_engine LANGUAGES CXX)

# C++20 exactly; EXTENSIONS OFF disables GNU-only dialect (-std=gnu++20) so code stays portable to the Mac's clang
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# writes compile_commands.json so VS Code / clangd can resolve MLX headers
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# MLX's CMake files ship inside the pip package; ask the venv's Python where it is
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_VARIABLE MLX_ROOT            # <Pkg>_ROOT = where find_package(<Pkg>) searches first
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
find_package(MLX CONFIG REQUIRED)     # CONFIG: use MLXConfig.cmake shipped by MLX itself

# Download Catch2 at configure time and build it with the project; no system install needed
include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.8.1                      # pinned: same test framework on every machine
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(Catch2)

# Header-only JSON parser for config.json; the release tarball is smaller than a clone
FetchContent_Declare(
  json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

# INTERFACE library: holds no code, just carries flags to whatever links it.
# Applied only to our targets so warnings inside Catch2 can't break our build.
add_library(bonsai_warnings INTERFACE)
target_compile_options(bonsai_warnings INTERFACE -Wall -Wextra -Wpedantic -Werror)

# The engine as one library; apps and tests link it instead of recompiling sources.
# json is PRIVATE: no public header includes it, so users of bonsai don't need it.
add_library(bonsai src/config.cpp src/weights.cpp src/model.cpp)
target_include_directories(bonsai PUBLIC include)
target_link_libraries(bonsai PUBLIC mlx PRIVATE nlohmann_json::nlohmann_json bonsai_warnings)

add_executable(hello_mlx apps/hello_mlx.cpp)
target_link_libraries(hello_mlx PRIVATE mlx bonsai_warnings)

add_executable(inspect apps/inspect.cpp)
target_link_libraries(inspect PRIVATE bonsai bonsai_warnings)

enable_testing()
# catch_discover_tests registers each TEST_CASE with ctest individually
list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
include(Catch)

# One executable per test file; Catch2WithMain supplies main()
foreach(test test_smoke test_config test_weights test_layers)
  add_executable(${test} tests/${test}.cpp)
  target_link_libraries(${test} PRIVATE bonsai Catch2::Catch2WithMain bonsai_warnings)
  catch_discover_tests(${test})
endforeach()

target_link_libraries(test_config PRIVATE nlohmann_json::nlohmann_json)

# Compile-time constant: the test finds references without depending on the working directory
target_compile_definitions(test_layers PRIVATE
  BONSAI_REFERENCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/reference")
