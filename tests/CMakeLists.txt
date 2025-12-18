cmake_minimum_required(VERSION 3.14)

# GoogleTest requires at least C++14
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Include GoogleTest helpers
include(GoogleTest)

# Create test executable
add_executable(mx_tests
  mx.test.cpp
)

# Link against gtest and gtest_main
target_link_libraries(mx_tests
  gtest
  gtest_main
)

# Discover tests
gtest_discover_tests(mx_tests)
