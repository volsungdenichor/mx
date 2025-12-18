cmake_minimum_required(VERSION 3.14)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

add_executable(mx_tests
  vector.test.cpp
)

target_link_libraries(mx_tests
  gtest
  gtest_main
  gmock
  gmock_main
)

gtest_discover_tests(mx_tests)
