cmake_minimum_required(VERSION 3.16)
project(orion_hardware)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------------------------------------------------
# Shared protocol header. It lives with the firmware so the microcontroller and the host compile
# the very same file. The default path works from a checkout of this repository (colcon in
# robot_ws, CI); override ORION_FIRMWARE_DIR if the package is built elsewhere.
# ----------------------------------------------------------
set(ORION_FIRMWARE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../firmware"
  CACHE PATH "Path to the ORION firmware tree (contains lib/orion_proto/orion_proto.hpp)")
get_filename_component(ORION_FIRMWARE_DIR "${ORION_FIRMWARE_DIR}" ABSOLUTE)
set(ORION_PROTO_DIR "${ORION_FIRMWARE_DIR}/lib/orion_proto")
if(NOT EXISTS "${ORION_PROTO_DIR}/orion_proto.hpp")
  message(FATAL_ERROR
    "orion_proto.hpp not found under ${ORION_PROTO_DIR}. Build from the ORION repository "
    "checkout or pass -DORION_FIRMWARE_DIR=<repo>/firmware.")
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(orion_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(std_msgs REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/conversions.cpp
  src/firmware_link.cpp
  src/orion_system.cpp
  src/serial_transport.cpp
)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${ORION_PROTO_DIR}>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)
target_link_libraries(${PROJECT_NAME} PUBLIC
  hardware_interface::hardware_interface
  pluginlib::pluginlib
  rclcpp::rclcpp
  rclcpp_lifecycle::rclcpp_lifecycle
  ${orion_msgs_TARGETS}
  ${std_msgs_TARGETS}
)
pluginlib_export_plugin_description_file(hardware_interface orion_hardware_plugin.xml)

install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include/${PROJECT_NAME})
install(FILES ${ORION_PROTO_DIR}/orion_proto.hpp DESTINATION include/${PROJECT_NAME})

# Python reference implementation + fake firmware.
ament_python_install_package(orion_hardware_py)
install(PROGRAMS scripts/fake_firmware DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # No per-file license headers in this repo (LICENSE covers it); C++ style is uncrustify's job.
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  set(ament_cmake_flake8_CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../../flake8.ini")
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)

  ament_add_gtest(test_conversions test/test_conversions.cpp)
  target_link_libraries(test_conversions ${PROJECT_NAME})

  ament_add_gtest(test_firmware_link test/test_firmware_link.cpp)
  target_link_libraries(test_firmware_link ${PROJECT_NAME})

  ament_add_gtest(test_safety_gates test/test_safety_gates.cpp)
  target_link_libraries(test_safety_gates ${PROJECT_NAME})

  ament_add_gtest(test_orion_system test/test_orion_system.cpp ENV ROS_DOMAIN_ID=72)
  target_link_libraries(test_orion_system ${PROJECT_NAME})

  # The firmware's own host tests, so CI covers the microcontroller code without PlatformIO.
  set(FW_TEST "${ORION_FIRMWARE_DIR}/test")
  ament_add_gtest(test_firmware_proto
    ${FW_TEST}/test_proto/test_main.cpp ${FW_TEST}/test_proto/test_proto.cpp
    SKIP_LINKING_MAIN_LIBRARIES)
  target_include_directories(test_firmware_proto PRIVATE ${ORION_PROTO_DIR})
  target_compile_definitions(test_firmware_proto PRIVATE
    ORION_PROTO_VECTORS="${ORION_PROTO_DIR}/test_vectors.json")

  ament_add_gtest(test_firmware_core
    ${FW_TEST}/test_core/test_main.cpp ${FW_TEST}/test_core/test_control_core.cpp
    ${ORION_FIRMWARE_DIR}/lib/orion_core/control_core.cpp
    SKIP_LINKING_MAIN_LIBRARIES)
  target_include_directories(test_firmware_core PRIVATE
    ${ORION_PROTO_DIR} ${ORION_FIRMWARE_DIR}/lib/orion_core)

  ament_add_pytest_test(test_protocol_py test/test_protocol.py TIMEOUT 60)
  ament_add_pytest_test(test_fake_firmware test/test_fake_firmware.py TIMEOUT 60)
  ament_add_pytest_test(test_xacro_serial test/test_xacro_serial.py TIMEOUT 60)
  # Full stack on a socat PTY pair: controller_manager + OrionSystem + diff_drive_controller
  # against the fake firmware. Own ROS domain so parallel tests cannot interfere.
  ament_add_pytest_test(test_integration test/test_integration.py TIMEOUT 240
    ENV ROS_DOMAIN_ID=72)
endif()

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(hardware_interface pluginlib rclcpp rclcpp_lifecycle orion_msgs std_msgs)
ament_package()
