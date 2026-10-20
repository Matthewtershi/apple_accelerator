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

# INTERFACE library: holds no code, just carries flags to whatever links it.
# Applied only to our targets so warnings inside Catch2 can't break our build.
add_library(bonsai_warnings INTERFACE)
target_compile_options(bonsai_warnings INTERFACE -Wall -Wextra -Wpedantic -Werror)

add_executable(hello_mlx apps/hello_mlx.cpp)
target_link_libraries(hello_mlx PRIVATE mlx bonsai_warnings)

enable_testing()
add_executable(test_smoke tests/test_smoke.cpp)
# Catch2WithMain supplies main(), so test files contain only TEST_CASEs
target_link_libraries(test_smoke PRIVATE mlx Catch2::Catch2WithMain bonsai_warnings)

# catch_discover_tests registers each TEST_CASE with ctest individually
list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
include(Catch)
catch_discover_tests(test_smoke)
