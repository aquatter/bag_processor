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
find_package(OpenCV REQUIRED core imgproc imgcodecs videoio calib3d)
find_package(GeographicLib REQUIRED)
find_package(GTSAM REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(range-v3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(ng-log REQUIRED)

# get_property(importTargets DIRECTORY "${CMAKE_SOURCE_DIR}" PROPERTY IMPORTED_TARGETS)
# message("${importTargets}")

file(GLOB ALGLIB_SOURCES alglib-cpp/*.cpp)

# message(STATUS ${ALGLIB_SOURCES})

add_executable(${PROJECT_NAME}
    main.cpp
    sort-cpp/src/kalman_filter.cpp
    sort-cpp/src/munkres.cpp
    sort-cpp/src/track.cpp
    sort-cpp/src/tracker.cpp
    source/csv_parser.cpp
    source/bag_processor.cpp
    ${ALGLIB_SOURCES}
)

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
    nlohmann_json::nlohmann_json
    range-v3::range-v3 range-v3::meta range-v3::concepts
    yaml-cpp::yaml-cpp
    ng-log::ng-log
)

target_include_directories(${PROJECT_NAME} PRIVATE
    include
    sort-cpp/include
    csvworker
    alglib-cpp
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
)
