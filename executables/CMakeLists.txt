set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# GTest dependency
find_package(GTest)
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# 1. Define the executable targets
add_executable(milestone02 milestone02.cpp)
add_executable(milestone03 milestone03.cpp)
add_executable(milestone04 milestone04.cpp)
add_executable(milestone05 milestone05.cpp)

# 2. Apply include directories to both targets
target_include_directories(milestone02 PRIVATE ../src)
target_include_directories(milestone03 PRIVATE ../src)
target_include_directories(milestone04 PRIVATE ../src)
target_include_directories(milestone05 PRIVATE ../src)

# 3. Link the necessary dependencies to both targets
target_link_libraries(milestone02 PRIVATE boltzman MPI::MPI_CXX Eigen3::Eigen Kokkos::kokkos)
target_link_libraries(milestone03 PRIVATE boltzman MPI::MPI_CXX Eigen3::Eigen Kokkos::kokkos)
target_link_libraries(milestone04 PRIVATE boltzman MPI::MPI_CXX Eigen3::Eigen Kokkos::kokkos)
target_link_libraries(milestone05 PRIVATE boltzman MPI::MPI_CXX Eigen3::Eigen Kokkos::kokkos)

# 4. Test executables
add_executable(test_milestone02 test_milestone02.cpp)
add_executable(test_milestone03 test_milestone03.cpp)

target_include_directories(test_milestone02 PRIVATE ../src)
target_include_directories(test_milestone03 PRIVATE ../src)

target_link_libraries(test_milestone02 PRIVATE boltzman MPI::MPI_CXX Eigen3::Eigen Kokkos::kokkos GTest::gtest)
target_link_libraries(test_milestone03 PRIVATE boltzman MPI::MPI_CXX Eigen3::Eigen Kokkos::kokkos GTest::gtest)

# Enable CTest framework for this subdirectory scope
include(CTest)

# 5. Register tests with CTest
include(GoogleTest)
gtest_discover_tests(test_milestone02)
gtest_discover_tests(test_milestone03)
