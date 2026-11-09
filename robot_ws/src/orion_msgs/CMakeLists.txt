cmake_minimum_required(VERSION 3.8)
project(orion_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TerrainState.msg"
  "msg/SafetyStatus.msg"
  "msg/PolicyStatus.msg"
  "msg/RoverHealth.msg"
  "msg/WheelPerturbation.msg"
  "srv/SetMode.srv"
  "srv/StartTrial.srv"
  "srv/StopTrial.srv"
  "srv/EstopReset.srv"
  DEPENDENCIES std_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_dependencies(rosidl_default_runtime)
ament_package()
