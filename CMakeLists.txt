cmake_minimum_required(VERSION 3.0.2)
project(observability_enhancement)

find_package(catkin REQUIRED COMPONENTS
  dji_osdk_ros
  roscpp
  rospy
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS dji_osdk_ros roscpp rospy std_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${dji_osdk_ros_INCLUDE_DIRS}
)

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

add_library(constants SHARED shared/constants.cpp)
target_link_libraries(constants
  ${catkin_LIBRARIES}
  ${dji_osdk_ros_LIBRARIES}
  djiosdk-core.a
  dji-linker.a
)
