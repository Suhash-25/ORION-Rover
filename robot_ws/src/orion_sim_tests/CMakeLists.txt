cmake_minimum_required(VERSION 3.8)
project(orion_sim_tests)

find_package(ament_cmake REQUIRED)

if(BUILD_TESTING)
  find_package(launch_testing_ament_cmake REQUIRED)
  # Each test boots Gazebo headless. TIMEOUT is the hard ceiling per test file.
  # Separate ROS domain + Gazebo partition per test so a leftover simulator from
  # one test can never answer the next test's service calls.
  add_launch_test(test/test_drive.py TIMEOUT 240
    ENV ROS_DOMAIN_ID=61 GZ_PARTITION=orion_test_drive)
  add_launch_test(test/test_nav.py TIMEOUT 360
    ENV ROS_DOMAIN_ID=62 GZ_PARTITION=orion_test_nav)
  add_launch_test(test/test_policy.py TIMEOUT 240
    ENV ROS_DOMAIN_ID=75 GZ_PARTITION=orion_test_policy)
endif()

ament_package()
