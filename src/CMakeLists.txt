set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(boltzman boltzman.cpp)
target_include_directories(boltzman PUBLIC .)
target_link_libraries(boltzman PUBLIC MPI::MPI_CXX Eigen3::Eigen Kokkos::kokkos)
