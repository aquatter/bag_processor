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
#include <rerun.hpp>
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
                   "Specify path to detections json file")
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

    if (set.use_logger_) {
      auto rec{std::make_shared<rerun::RecordingStream>("bag_converter")};
      rec->connect_grpc().exit_on_failure();
      bag_proc->set_rerun(rec);
    }

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

#if 1
  constexpr static std::string_view file_name{"GX010004"};
  constexpr static std::string_view folder_name{"domodedovo/gopro_01_11"};

  BagProcessorSettings set{
      .bag_path_ = fmt::format("/root/data/{}/{}.MP4", folder_name.data(),
                               file_name.data()),
      .annotations_path_ = fmt::format("/root/data/{}/detections_{}.json",
                                       folder_name.data(), file_name.data()),
      .calibration_path_ =
          "/root/data/calibration_gopro13_05_11/GX010005-camchain.yaml",
      .ground_truth_path_ =
          fmt::format("/root/data/{}/gt.geojson", folder_name.data()),
      .correction_angle_ = 4.0,
      .camera_gps_delta_ = 0l,
      .use_klt_ = false,
      .use_logger_ = true,
      .session_name_ = "gopro_01_11",
      .compressed_image_topic_ = "/camera/image_raw/compressed",
      .gps_topic_ = "/fix",
      .gopro_mode_ = true};
#else
  constexpr static std::string_view file_name{"video20250831-125017"};
  constexpr static std::string_view folder_name{"domodedovo/video_domodedovo"};

  BagProcessorSettings set{
      .bag_path_ =
          fmt::format("/root/data/{}/{}", folder_name.data(), file_name.data()),
      .annotations_path_ = fmt::format("/root/data/{}/detections_{}.json",
                                       folder_name.data(), file_name.data()),

      .calibration_path_ = "/root/data/domodedovo/video_domodedovo/calib/"
                           "VID_20250929_134020-camchain.yaml",
      .ground_truth_path_ =
          fmt::format("/root/data/{}/gt.geojson", folder_name.data()),
      .correction_angle_ = 0.0,
      .camera_gps_delta_ = 1'300'000'000l,
      .use_klt_ = false,
      .use_logger_ = true,
      .session_name_ = "video_gps",
      .compressed_image_topic_ = "/camera/image_raw/compressed",
      .gps_topic_ = "/fix"};
#endif

  auto rec{std::make_shared<rerun::RecordingStream>("bag_converter")};
  rec->connect_grpc().exit_on_failure();

  try {
#if 1
    {

      auto bag_proc1{
          BagProcessor::load("/root/data/domodedovo/gopro_01_11/archive.bin")};

      auto bag_proc2{BagProcessor::load(
          "/root/data/domodedovo/video_domodedovo/archive.bin")};

#if 0
      BagLoader loader1{BagLoader::Settings{
          .compressed_image_topic_ = "/camera/image_raw/compressed",
          .path_to_bag_ = bag_proc1.settings().bag_path_,
          .rec_ = rec}};


      BagLoader loader2{BagLoader::Settings{
          .compressed_image_topic_ = "/camera/image_raw/compressed",
          .path_to_bag_ = bag_proc2.settings().bag_path_,
          .rec_ = rec}};

      const auto det1{bag_proc1.get_tracks().at(205).dets_[103]};
      const auto det2{bag_proc2.get_tracks().at(214).dets_[0]};

      auto img1 = loader1.load_image(det1.timestamp_ -
                                     bag_proc1.settings().camera_gps_delta_);

      auto img2 = loader2.load_image(det2.timestamp_ -
                                     bag_proc2.settings().camera_gps_delta_);

      cv::rectangle(img1, det1.box_, {0.0, 255.0, 0.0}, 2);
      cv::rectangle(img2, det2.box_, {0.0, 255.0, 0.0}, 2);

      cv::imwrite("/root/data/images/image3.png", img1);
      cv::imwrite("/root/data/images/image4.png", img2);
#endif

      TracksCollection track_collection{};
      track_collection.set_rerun(rec);
      track_collection.merge(bag_proc1);
      track_collection.merge(bag_proc2);

      return EXIT_SUCCESS;
    }
#endif

    BagProcessor bag_proc{set};

    bag_proc.set_rerun(rec);
    bag_proc.calculate();
    bag_proc.calculate_metrics(
        fmt::format("/root/data/{}/prec_recall.json", folder_name));

    bag_proc.save(fmt::format("/root/data/{}/archive.bin", folder_name));
  } catch (const std::exception &ex) {
    LOG(ERROR) << ex.what();
  }

  return EXIT_SUCCESS;
}