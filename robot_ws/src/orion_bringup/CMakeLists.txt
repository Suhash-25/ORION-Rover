cmake_minimum_required(VERSION 3.8)
project(orion_bringup)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

###########################################################
# Install Launch Files
###########################################################

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

###########################################################
# Install RViz Configuration
###########################################################

install(
  DIRECTORY rviz
  DESTINATION share/${PROJECT_NAME}
)

###########################################################
# Install Configuration Files
###########################################################

install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

###########################################################
# Install Gazebo Worlds
###########################################################

install(
  DIRECTORY worlds
  DESTINATION share/${PROJECT_NAME}
)

###########################################################
# Install Maps
###########################################################

install(
  DIRECTORY maps
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # No per-file license headers in this repo (LICENSE covers it); C++ style is uncrustify's job.
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  set(ament_cmake_flake8_CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../../flake8.ini")
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
