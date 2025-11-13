#include <bag_processor.hpp>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fmt/color.h>
#include <fmt/format.h>
#include <memory>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgcodecs.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/utilities.hpp>
#include <rerun.hpp>
#include <sensor_msgs/msg/image.hpp>

void LogFormatter(std::ostream &s, const nglog::LogMessage &m, void *) {

  std::string prefix_str{};

  switch (m.severity()) {
  case nglog::NGLOG_INFO:
    prefix_str = fmt::format(fmt::fg(fmt::color::spring_green), "INFO");
    break;
  case nglog::NGLOG_WARNING:
    prefix_str =
        fmt::format(fmt::fg(fmt::color::light_golden_rod_yellow), "WARNING");
    break;
  case nglog::NGLOG_ERROR:
    prefix_str = fmt::format(fmt::fg(fmt::color::indian_red), "ERROR");
    break;
  case nglog::NGLOG_FATAL:
    prefix_str = fmt::format(fmt::fg(fmt::color::medium_violet_red), "FATAL");
    break;
  }

  s << fmt::format("[{} {}.{}.{} {}:{}:{} {}:{}]", prefix_str, m.time().day(),
                   1 + m.time().month(), 1900 + m.time().year(),
                   m.time().hour(), m.time().min(), m.time().sec(),
                   m.basename(), m.line());
}

int main(const int argc, const char *const *argv) {

  nglog::InitializeLogging(argv[0]);
  nglog::InstallPrefixFormatter(&LogFormatter);
  FLAGS_stderrthreshold = 0;

  auto rec{std::make_shared<rerun::RecordingStream>("bag_converter")};
  rec->connect_grpc().exit_on_failure();

  constexpr static std::string_view file_name{"GX010004"};
  constexpr static std::string_view folder_name{"domodedovo/gopro_01_11"};

  try {
    BagProcessor bag_proc{
        {.bag_path_ = fmt::format("/root/data/{}/{}", folder_name.data(),
                                  file_name.data()),
         .annotations_path_ = fmt::format("/root/data/{}/detections_{}.json",
                                          folder_name.data(), file_name.data()),
         .calibration_path_ =
             "/root/data/domodedovo/gopro_01_11/GX010005-camchain.yaml",
         //  "/root/data/calibration_gopro_15_10/results/"
         // "wide_stab_on-camchain.yaml",
         //  "/root/data/calib_bmi160_1204x768_29_09/"
         //  "rosbag2_2025_09_26-22_19_00_converted-camchain.yaml",
         //  "/root/data/calibration_20_08/imu_camera_20_08-camchain.yaml",
         .ground_truth_path_ =
             fmt::format("/root/data/{}/gt.geojson", folder_name.data()),
         .use_klt_ = false,
         .use_logger_ = true,
         .rec_ = rec}};

    LOG(INFO) << "Bag processor initialized";

    const int64_t from_timestamp{1755767829610259789 - 2 * 1'000'000'000l};
    const int64_t to_timestamp{from_timestamp + 2 * 1'000'000'000l};

    const auto found_landmarks{bag_proc.triangulate_tracks()};

    bag_proc.log_gps_path_map()
        .log_landmarks_map(found_landmarks)
        .save_geojson(found_landmarks,
                      fmt::format("/root/data/{}/{}.geojson",
                                  folder_name.data(), file_name.data()));

    // .log_track(338);
    // .log_landmarks_map(found_landmarks);

    // .log_gps_path()
    // .log_axis()
    // .log_camera(168215094333)
    // .log_poly(168215094333)
    // .log_images(168215094333, 168215094333);
    // .log_landmarks_map(found_landmarks)
    // .save_geojson(found_landmarks,
    //               fmt::format("/root/data/{}/{}.geojson",
    //                           folder_name.data(), file_name.data()));

    // .log_ground_truth_landmarks(bag_proc.most_frequent_landmark())
    // .log_track("3.1_148")
    // .log_images(from_timestamp, to_timestamp);

    // for (auto &&landmark : found_landmarks) {
    //   if (landmark.code_ == bag_proc.most_frequent_landmark()) {
    //     Landmark l{landmark};
    //     l.code_ += "__detected";
    //     bag_proc.log_landmark_map(l, {0, 255, 0});
    //   }
    // }

    // .log_ground_truth_landmarks()
    // .log_track("3.27_43")
    // .log_gps_path()
    // .log_axis()
    // .log_track_directions("3.27_43", 150.0f);

    // .log_landmarks_map(found_landmarks)

    // bag_proc.log_axis().log_gps_path_map();
    // bag_proc.log_axis().log_gps_path().log_poly(1756128239494864977);

    // bag_proc.log_axis().log_gps_path().log_images(from_timestamp,
    // to_timestamp); .log_track("3.27_116");

    // .log_direction("8.1.1_37", 1755768830363105351, 30.0f);

    // .log_track("8.1.1_37");
    // .log_poly(1755768834763552351)

    // .log_poly(1755768834405726351);
    // .log_camera(1755768834405726351);
    // bag_proc.log_images(from_timestamp, to_timestamp);
    /// world/camera_1755768834405726351
  } catch (const std::exception &ex) {
    LOG(ERROR) << ex.what();
  }

  return EXIT_SUCCESS;
}