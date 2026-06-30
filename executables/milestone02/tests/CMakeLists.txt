find_package(GTest)
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# Define your source files here
set(TEST_SOURCES milestone02_test.cpp ../boltzman.hpp)

# Renamed target to avoid collision with root tests
add_executable(milestone02_tests ${TEST_SOURCES})

# Target linking and includes (adjust 'lib' if your milestone uses a different target name)
target_link_libraries(milestone02_tests PRIVATE lib GTest::gtest)
target_include_directories(milestone02_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(milestone02_tests PRIVATE lib GTest::gtest MPI::MPI_CXX)

# Automatically discover the individual GTest macros
include(GoogleTest)
gtest_discover_tests(milestone02_tests)
