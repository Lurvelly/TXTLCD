cmake_minimum_required(VERSION 3.15)
project(text_lcd)

set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  rosbag
  std_msgs
  image_transport
  cv_bridge
  tf
  message_generation
)

#find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(
  include
	${catkin_INCLUDE_DIRS} 
	${PCL_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  src/clipper/include
)

add_service_files(
  DIRECTORY srv
  FILES
  save_map.srv
  save_pose.srv
)

generate_messages(
 DEPENDENCIES
 geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs message_runtime
  DEPENDS EIGEN3 PCL 
  INCLUDE_DIRS include
)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/clipper)

add_executable(text_lcd 
  src/laserPosegraphOptimization.cpp
)
target_compile_options(text_lcd 
  PRIVATE ${OpenMP_CXX_FLAGS}
)

message(STATUS "=========================clipper: ${clipper}")
target_link_libraries(text_lcd 
  ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES}
  ${OpenMP_CXX_FLAGS}
  clipper
  yaml-cpp
)




