cmake_minimum_required(VERSION 3.10.0)
project(bag_processor VERSION 0.1.0 LANGUAGES C CXX)

include(FetchContent)
FetchContent_Declare(rerun_sdk URL https://github.com/rerun-io/rerun/releases/latest/download/rerun_cpp_sdk.zip)
FetchContent_MakeAvailable(rerun_sdk)


find_package(fmt REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(rosbag2_transport REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(OpenCV REQUIRED core imgproc imgcodecs videoio)
find_package(GeographicLib REQUIRED)
find_package(GTSAM REQUIRED)

# get_property(importTargets DIRECTORY "${CMAKE_SOURCE_DIR}" PROPERTY IMPORTED_TARGETS)
# message("${importTargets}")

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    fmt::fmt
    rcl::rcl
    rosbag2_cpp::rosbag2_cpp
    rosbag2_transport::rosbag2_transport
    sensor_msgs::sensor_msgs_library
    ${GeographicLib_LIBRARIES}
    gtsam
    rerun_sdk
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
)
