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

ament_package()
