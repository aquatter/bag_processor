cmake_minimum_required(VERSION 3.10.0)
project(bag_processor VERSION 0.1.0 LANGUAGES C CXX)

# include(FetchContent)
# FetchContent_Declare(rerun_sdk URL https://github.com/rerun-io/rerun/releases/latest/download/rerun_cpp_sdk.zip)
# FetchContent_MakeAvailable(rerun_sdk)


find_package(fmt REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(rosbag2_transport REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(OpenCV REQUIRED core imgproc imgcodecs videoio calib3d features2d)
find_package(GeographicLib REQUIRED)
find_package(GTSAM REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(range-v3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(ng-log REQUIRED)
find_package(flann REQUIRED)
find_package(OpenMP COMPONENTS CXX REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)
find_package(Eigen3 REQUIRED)

pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat
    libavcodec
    libavutil
    libswscale
)

add_subdirectory(open_vins/ov_core)
add_subdirectory(bag_converter)

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
    source/feature_tracker.cpp
    source/utils.cpp
    source/types.cpp
    source/bag_loader.cpp
    source/triangulation.cpp
    source/tracks_collection.cpp
    source/cartesian_converter.cpp
    source/mp4_image_loader.cpp
    source/extract_frames.cpp
    source/feature_matcher.cpp
    source/track_merger.cpp
    source/load_parquet.cpp
    ${ALGLIB_SOURCES}
)

target_link_libraries(${PROJECT_NAME}
    Arrow::arrow_shared
    Parquet::parquet_shared
    PkgConfig::FFMPEG
    ${OpenCV_LIBS}
    fmt::fmt
    rcl::rcl
    rosbag2_cpp::rosbag2_cpp
    rosbag2_transport::rosbag2_transport
    sensor_msgs::sensor_msgs_library
    ${GeographicLib_LIBRARIES}
    gtsam
    nlohmann_json::nlohmann_json
    range-v3::range-v3 range-v3::meta range-v3::concepts
    yaml-cpp::yaml-cpp
    ng-log::ng-log
    ov_core_lib
    flann::flann
    OpenMP::OpenMP_CXX
    gpmf-extractor
)

target_include_directories(${PROJECT_NAME} PRIVATE
    include
    sort-cpp/include
    csvworker
    alglib-cpp
    open_vins/ov_core/src
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
)

add_executable(combine_bags
    combine_bags.cpp
    source/csv_parser.cpp
    source/bag_processor.cpp
    source/feature_tracker.cpp
    source/utils.cpp
    source/types.cpp
    source/bag_loader.cpp
    source/triangulation.cpp
    source/tracks_collection.cpp
    source/cartesian_converter.cpp
    source/feature_matcher.cpp
    source/mp4_image_loader.cpp
    source/extract_frames.cpp
    source/track_merger.cpp
    source/load_parquet.cpp
    ${ALGLIB_SOURCES}
)

target_link_libraries(combine_bags
    PkgConfig::FFMPEG
    ${OpenCV_LIBS}
    fmt::fmt
    rcl::rcl
    rosbag2_cpp::rosbag2_cpp
    rosbag2_transport::rosbag2_transport
    sensor_msgs::sensor_msgs_library
    ${GeographicLib_LIBRARIES}
    gtsam
    nlohmann_json::nlohmann_json
    range-v3::range-v3 range-v3::meta range-v3::concepts
    yaml-cpp::yaml-cpp
    ng-log::ng-log
    ov_core_lib
    flann::flann
    OpenMP::OpenMP_CXX
    gpmf-extractor
    Arrow::arrow_shared
    Parquet::parquet_shared
)

target_include_directories(combine_bags PRIVATE
    include
    sort-cpp/include
    csvworker
    alglib-cpp
    open_vins/ov_core/src
)

set_target_properties(combine_bags PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
)

# find_path(FFMPEG_INCLUDE_DIR
#     NAMES libavformat/avformat.h
#     HINTS
#     ${FFMPEG_ROOT}
#     ${FFMPEG_ROOT}/include
#     PATH_SUFFIXES include
# )

# find_library(AVFORMAT_LIB NAMES avformat HINTS ${FFMPEG_ROOT} ${FFMPEG_ROOT}/lib PATH_SUFFIXES lib)
# find_library(AVCODEC_LIB NAMES avcodec HINTS ${FFMPEG_ROOT} ${FFMPEG_ROOT}/lib PATH_SUFFIXES lib)
# find_library(AVUTIL_LIB NAMES avutil HINTS ${FFMPEG_ROOT} ${FFMPEG_ROOT}/lib PATH_SUFFIXES lib)
# find_library(SWSCALE_LIB NAMES swscale HINTS ${FFMPEG_ROOT} ${FFMPEG_ROOT}/lib PATH_SUFFIXES lib)

add_executable(test_matching
    test_match.cpp
    source/cartesian_converter.cpp

)


set_target_properties(test_matching PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)

target_link_options(test_matching PRIVATE -Wl,--no-as-needed)
target_link_libraries(test_matching PRIVATE
    PkgConfig::FFMPEG
    ${OpenCV_LIBS}
    fmt::fmt
    range-v3::range-v3 range-v3::meta range-v3::concepts
    Eigen3::Eigen
    # yaml-cpp::yaml-cpp
    # ng-log::ng-log
    # OpenMP::OpenMP_CXX
    # gpmf-extractor
    ${GeographicLib_LIBRARIES}
)

target_include_directories(test_matching PRIVATE
    include
    sort-cpp/include
)

set_target_properties(test_matching PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
)

