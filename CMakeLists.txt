cmake_minimum_required(VERSION 3.14)
project(mx VERSION 0.1.0 LANGUAGES CXX)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add the include directory
include_directories(include)

# Enable testing
enable_testing()

# Add tests subdirectory
add_subdirectory(tests)
