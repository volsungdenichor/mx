cmake_minimum_required(VERSION 3.14)

set(UNIT_TEST_BINARY mx_tests)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)

FetchContent_Declare(
  trx
  GIT_REPOSITORY https://github.com/volsungdenichor/trx.git
  GIT_TAG main
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest trx)

include(GoogleTest)

set(UNIT_TEST_FILES
  vector.test.cpp
  dcel.test.cpp
  dyn_matrix.test.cpp
)

add_executable(${UNIT_TEST_BINARY}
  ${UNIT_TEST_FILES}
)

target_include_directories(${UNIT_TEST_BINARY} PRIVATE
  ${trx_SOURCE_DIR}/include
)

target_link_libraries(${UNIT_TEST_BINARY}
  gtest
  gtest_main
  gmock
  gmock_main
)

gtest_discover_tests(${UNIT_TEST_BINARY})
