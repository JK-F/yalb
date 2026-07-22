set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 1. Define the executable targets
add_executable(milestone02 milestone02.cpp)
add_executable(milestone03 milestone03.cpp)
add_executable(milestone04 milestone04.cpp)
add_executable(milestone05 milestone05.cpp)
add_executable(milestone06 milestone06.cpp)

# 2. Apply include directories to both targets
target_include_directories(milestone02 PRIVATE ../src)
target_include_directories(milestone03 PRIVATE ../src)
target_include_directories(milestone04 PRIVATE ../src)
target_include_directories(milestone05 PRIVATE ../src)
target_include_directories(milestone06 PRIVATE ../src)

# 3. Link the necessary dependencies to both targets
target_link_libraries(milestone02 PRIVATE boltzman MPI::MPI_CXX Eigen3::Eigen Kokkos::kokkos)
target_link_libraries(milestone03 PRIVATE boltzman MPI::MPI_CXX Eigen3::Eigen Kokkos::kokkos)
target_link_libraries(milestone04 PRIVATE boltzman MPI::MPI_CXX Eigen3::Eigen Kokkos::kokkos)
target_link_libraries(milestone05 PRIVATE boltzman MPI::MPI_CXX Eigen3::Eigen Kokkos::kokkos)
target_link_libraries(milestone06 PRIVATE boltzman MPI::MPI_CXX Eigen3::Eigen Kokkos::kokkos)
