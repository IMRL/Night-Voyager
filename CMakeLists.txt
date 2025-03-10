cmake_minimum_required(VERSION 3.0.2)
project(night_voyager)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty")

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")
# Enable compile optimizations
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")

# # Enable debug flags (use if you want to debug in gdb)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -Wall -Wuninitialized -fno-omit-frame-pointer")
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 -Wall -Wextra -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -Wextra -fomit-frame-pointer -DNDEBUG")

# if(NOT CMAKE_BUILD_TYPE)
#     set(CMAKE_BUILD_TYPE Debug)
# endif()


## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS 
  cv_bridge
  geometry_msgs
  nav_msgs
  pcl_ros
  roscpp
  tf
  rospy
  sensor_msgs
  std_msgs
  image_transport
  visualization_msgs
  message_generation
)

find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED)
find_package(OpenCV 3.2.0 REQUIRED)
find_package(TBB REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem thread date_time)
find_package(catkin QUIET COMPONENTS roscpp)
message(STATUS "OPENCV: " ${OpenCV_VERSION} " | BOOST: " ${Boost_VERSION})

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  BoundingBox.msg
  BoundingBoxes.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES Night_Voyager
  CATKIN_DEPENDS message_runtime roscpp rospy sensor_msgs std_msgs nav_msgs geometry_msgs image_transport visualization_msgs tf cv_bridge
  DEPENDS EIGEN3 PCL
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
file(GLOB_RECURSE INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include")
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${INCLUDE_DIRS}
  ${TBB_INCLUDE_DIRS}
  ${tf_INCLUDE_DIRS}
)

# list(APPEND thirdparty_libraries
#   ${catkin_LIBRARIES}
#   ${OpenCV_LIBRARIES}
#   ${Boost_LIBRARIES}
#   ${PCL_LIBRARIES} 
# )

file(GLOB_RECURSE NIGHT_VOYAGER_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

list(FILTER NIGHT_VOYAGER_SRCS EXCLUDE REGEX ".*main\\.cpp$")

# add_library(${PROJECT_NAME} SHARED ${NIGHT_VOYAGER_LIBRARIES})
# target_link_libraries(
#   ${PROJECT_NAME}
#   ${catkin_LIBRARIES}
#   ${OpenCV_LIBRARIES}
#   ${Boost_LIBRARIES}
#   ${PCL_LIBRARIES}
#   ${cv_bridge_LIBRARIES}
#   ${image_transport_LIBRARIES}
#   ${TBB_LIBRARIES}
#   ${tf_LIBRARIES}
# )
# target_include_directories(${PROJECT_NAME} PUBLIC include)

add_executable(night_voyager_node src/main.cpp ${NIGHT_VOYAGER_SRCS})
add_dependencies(night_voyager_node ${PROJECT_NAME}_generate_messages_cpp ${catkin_EXPORTED_TARGETS})
target_link_libraries(night_voyager_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${Boost_LIBRARIES} ${PCL_LIBRARIES}
                                         ${cv_bridge_LIBRARIES} ${image_transport_LIBRARIES} ${TBB_LIBRARIES} ${tf_LIBRARIES})

