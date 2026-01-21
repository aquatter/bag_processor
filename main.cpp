#include <CLI/CLI.hpp>
#include <Eigen/Core>
#include <bag_processor.hpp>
#include <cstdlib>
#include <fmt/color.h>
#include <fmt/format.h>
#include <iostream>
#include <memory>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/utilities.hpp>
#if USE_RERUN
#include <rerun.hpp>
#endif
#include <sensor_msgs/msg/image.hpp>
#include <serialization.hpp>
#include <string>
#include <tracks_collecton.hpp>

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

  s << fmt::format("[{}]", prefix_str);

  // s << fmt::format("[{} {}.{}.{} {}:{}:{} {}:{}]", prefix_str,
  // m.time().day(),
  //                  1 + m.time().month(), 1900 + m.time().year(),
  //                  m.time().hour(), m.time().min(), m.time().sec(),
  //                  m.basename(), m.line());
}

int main(const int argc, const char *const *argv) {

  nglog::InitializeLogging(argv[0]);
  nglog::InstallPrefixFormatter(&LogFormatter);
  FLAGS_stderrthreshold = 0;

  try {
    CLI::App app{"Process ROS2 bag or GoPro video"};

    BagProcessorSettings set{};

    app.add_option("-i, --input", set.bag_path_,
                   "Specify input path to ROS2 bag or GoPro mp4 file")
        ->required()
        ->check(CLI::ExistingFile | CLI::ExistingDirectory);

    app.add_option("-a, --annotations", set.annotations_path_,
                   "Specify path to detections file")
        ->required()
        ->check(CLI::ExistingFile);

    app.add_option("-c, --calibration", set.calibration_path_,
                   "Specify path to camera calibration yaml file")
        ->required()
        ->check(CLI::ExistingFile);

    app.add_option("-g, --ground-truth", set.ground_truth_path_,
                   "Specify path to ground truth geojson file");

    app.add_option("--correction-angle", set.correction_angle_,
                   "Specify correction angle in degrees")
        ->default_val(0.0);

    app.add_option("--camera-gps-delta", set.camera_gps_delta_,
                   "Specify camera to GPS timestamp delta in nanoseconds")
        ->default_val(0);

    app.add_option("--session-name", set.session_name_,
                   "Specify session name for logging")
        ->required();

    app.add_flag("--gopro-mode", set.gopro_mode_,
                 "Enable GoPro video processing mode")
        ->default_val(false);

    app.add_flag("--use-logger", set.use_logger_, "Enable rerun logger")
        ->default_val(false);

    app.add_option("--image-topic", set.compressed_image_topic_,
                   "Specify compressed image topic name")
        ->default_val("/camera/image_raw/compressed");

    app.add_option("--gps-topic", set.gps_topic_, "Specify GPS topic name")
        ->default_val("/fix");

    bool eval_metrics{false};

    app.add_flag("--calculate-metrics", eval_metrics,
                 "Calculate metrics after processing")
        ->default_val(false);

    std::string output_path{};

    app.add_option("-o, --output", output_path,
                   "Specify output path to save processed bag and metrics")
        ->required()
        ->check(CLI::ExistingDirectory);

    app.add_option(
        "--exclude-gps", set.gps_exclusion_intervals_,
        "Intervals in seconds where GPS measurements should be excluded");

    CLI11_PARSE(app, argc, argv);

    set.use_klt_ = false;
    auto bag_proc{std::make_shared<BagProcessor>(set)};
#if USE_RERUN
    if (set.use_logger_) {
      auto rec{std::make_shared<rerun::RecordingStream>("bag_converter")};
      rec->connect_grpc().exit_on_failure();
      bag_proc->set_rerun(rec);
    }
#endif

    bag_proc->calculate();

    if (eval_metrics) {
      bag_proc->calculate_metrics(
          fmt::format("{}/prec_recall.json", output_path));
    }

    bag_proc->save(fmt::format("{}/archive.bin", output_path));

  } catch (const std::exception &ex) {
    LOG(ERROR) << ex.what();
    return EXIT_FAILURE;
  }

  LOG(INFO) << "Processing completed successfully";
  return EXIT_SUCCESS;
}