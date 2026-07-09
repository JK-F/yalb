cmake_minimum_required(VERSION 3.21)
project(GemvBenchmarks LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# Eigen: use an installed version if available, otherwise download it
find_package(Eigen3 5.0.1 QUIET)
if(NOT TARGET Eigen3::Eigen)
    FetchContent_Declare(
        eigen3
        URL https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz
    )
    set(BUILD_TESTING OFF CACHE BOOL "Disable Eigen tests" FORCE)
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "Disable Eigen docs" FORCE)
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "Disable Eigen tests" FORCE)
    FetchContent_MakeAvailable(eigen3)
    if(NOT TARGET Eigen3::Eigen)
        add_library(Eigen3::Eigen ALIAS eigen)
    endif()
endif()

# google/benchmark
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark self-tests" FORCE)
FetchContent_Declare(
    benchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.9.1.tar.gz
)
FetchContent_MakeAvailable(benchmark)

add_executable(bench gemv_benchmarks.cpp)
target_link_libraries(bench PRIVATE Eigen3::Eigen benchmark::benchmark)
