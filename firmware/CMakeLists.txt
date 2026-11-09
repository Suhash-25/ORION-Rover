# Host build of the portable firmware core and its unit tests (the verified path).
#
#   cmake -S firmware -B /tmp/orion_fw_build && cmake --build /tmp/orion_fw_build -j2
#   ctest --test-dir /tmp/orion_fw_build --output-on-failure
#
# Needs only a C++17 compiler and GoogleTest. The board build is PlatformIO (platformio.ini).
# orion_hardware's colcon test also compiles and runs these tests, so CI covers them.
cmake_minimum_required(VERSION 3.16)
project(orion_firmware_host CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wpedantic -Wconversion -Wno-sign-conversion)

option(ORION_FW_SANITIZE "Build host tests with AddressSanitizer + UBSan" ON)
if(ORION_FW_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

find_package(GTest REQUIRED)
enable_testing()

add_library(orion_fw_core STATIC lib/orion_core/control_core.cpp)
target_include_directories(orion_fw_core PUBLIC lib/orion_proto lib/orion_core)

add_executable(test_proto test/test_proto/test_main.cpp test/test_proto/test_proto.cpp)
target_include_directories(test_proto PRIVATE lib/orion_proto)
target_compile_definitions(test_proto PRIVATE
  ORION_PROTO_VECTORS="${CMAKE_CURRENT_SOURCE_DIR}/lib/orion_proto/test_vectors.json")
target_link_libraries(test_proto GTest::gtest)
add_test(NAME test_proto COMMAND test_proto)

add_executable(test_core test/test_core/test_main.cpp test/test_core/test_control_core.cpp)
target_link_libraries(test_core orion_fw_core GTest::gtest)
add_test(NAME test_core COMMAND test_core)
