#include <Eigen/Geometry>
#include <GeographicLib/LocalCartesian.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <bag_loader.hpp>
#include <bag_processor.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/math/constants/constants.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <csv_parser.hpp>
#include <exception>
#include <extract_frames.hpp>
#include <feature_tracker.hpp>
#include <filesystem>
#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fstream>
#include <gpmf_frame.hpp>
#include <gpmf_parser.hpp>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Quaternion.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/triangulation.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <interpolation.h>
#include <iterator>
#include <limits>
#include <memory>
#include <mp4_image_loader.hpp>
#include <mutex>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <numbers>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <progress_bar.hpp>
#include <random>
#include <range/v3/action/push_back.hpp>
#include <range/v3/algorithm/copy.hpp>
#include <range/v3/algorithm/count_if.hpp>
#include <range/v3/algorithm/find_if.hpp>
#include <range/v3/algorithm/max.hpp>
#include <range/v3/algorithm/sort.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/drop.hpp>
#include <range/v3/view/enumerate.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/linear_distribute.hpp>
#include <range/v3/view/take.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/view/zip.hpp>
#include <rerun.hpp>
#include <rerun/archetypes/geo_line_strings.hpp>
#include <rerun/archetypes/geo_points.hpp>
#include <rerun/archetypes/points3d.hpp>
#include <rerun/components/class_id.hpp>
#include <rerun/components/geo_line_string.hpp>
#include <rerun/components/image_plane_distance.hpp>
#include <rerun_logging.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <stdexcept>
#include <string>
#include <thread>
#include <tracker.h>
#include <triangulation.hpp>
#include <types.hpp>
#include <unordered_map>
#include <unordered_set>
#include <utils.hpp>
#include <vector>
#include <yaml-cpp/node/node.h>
#include <yaml-cpp/node/parse.h>
#include <yaml-cpp/yaml.h>

using ranges::to;
using ranges::views::drop;
using ranges::views::enumerate;
using ranges::views::ints;
using ranges::views::linear_distribute;
using ranges::views::take;
using ranges::views::transform;
using ranges::views::zip;

BagProcessor::BagProcessor() = default;

BagProcessor::BagProcessor(const BagProcessorSettings &set) : set_{set} {
  set_.print();
}

void BagProcessor::calculate() {

  load_measurements(set_.bag_path_);

  LOG(INFO) << "Measurements loaded";

  load_calibration(set_.calibration_path_);
  calib_.print();

  load_ground_truth_landmarks(set_.ground_truth_path_);

  load_tracks();

  LOG(INFO) << "Tracks loaded";

  collect_detections();
  link_detections(image_detections_);
  link_tracks(image_tracks_, image_detections_);

  LOG(INFO) << "Tracks linked";
  LOG(INFO) << "# detections: " << image_detections_.size();
  LOG(INFO) << "# valid tracks: " << select_valid_tracks();
  LOG(INFO) << "# triangulated landmarks: " << triangulate_tracks();

  extract_images();
  calculate_metrics();

  LOG(INFO) << "# combined landmarks: "
            << combine_landmarks(image_tracks_, local_converter_);

  calculate_metrics();
}

void BagProcessor::calculate_metrics(
    std::optional<std::filesystem::path> path) {

  if (not std::filesystem::exists(set_.ground_truth_path_)) {
    return;
  }

  const Metrics metrics{Metrics::Settings{}, gps_,
                        load_ground_truth_landmarks(set_.ground_truth_path_)};
  auto res{metrics.eval(image_tracks_)};

  if (path.has_value()) {
    res.save(path->string(), set_.session_name_);
  }
}

void BagProcessor::optimize_angle(double from, double to, ptrdiff_t num) {

  float max_auc{0.0f};
  Metrics::Result best_result{};
  float best_angle{0.0f};
  std::vector<Landmark> best_landmarks{};

  const Metrics metrics{Metrics::Settings{}, gps_,
                        load_ground_truth_landmarks(set_.ground_truth_path_)};

  for (auto &&angle : linear_distribute(from, to, num)) {
    change_angle(angle);

    triangulate_tracks();
    combine_landmarks(image_tracks_, local_converter_);

    auto res{metrics.eval(image_tracks_)};

    if (max_auc < res.precision_auc_) {
      max_auc = res.precision_auc_;
      best_result = std::move(res);
      best_angle = angle;
    }
  }

  LOG(INFO) << "Best angle: " << best_angle;
  LOG(INFO) << "Best auc: " << max_auc;
}

void BagProcessor::collect_detections() {

  for (auto &&[track_id, track] : image_tracks_) {
    for (auto &&det : track.dets_) {
      image_detections_[det.timestamp_].det_id_to_detection_[det.det_id_] =
          &det;
      image_detections_[det.timestamp_].track_id_to_detection_[det.track_id_] =
          &det;
      image_detections_[det.timestamp_].dets_.push_back(&det);
      image_detections_[det.timestamp_].timestamp_ = det.timestamp_;
    }
  }
}

void BagProcessor::load_calibration(const std::string_view path) {
  YAML::Node calib{YAML::LoadFile(path.data())};

  const auto intrinsics{calib["cam0"]["intrinsics"].as<std::vector<double>>()};
  const auto distortion{
      calib["cam0"]["distortion_coeffs"].as<std::vector<double>>()};

  calib_.camera_resolution_.x() = calib["cam0"]["resolution"][0].as<int>();
  calib_.camera_resolution_.y() = calib["cam0"]["resolution"][1].as<int>();

  calib_.camera_matrix_ = cv::Mat_<double>::eye(3, 3);
  calib_.camera_matrix_(0, 0) = intrinsics[0];
  calib_.camera_matrix_(1, 1) = intrinsics[1];
  calib_.camera_matrix_(0, 2) = intrinsics[2];
  calib_.camera_matrix_(1, 2) = intrinsics[3];

  calib_.dist_coeffs_ = cv::Mat_<double>(distortion, true);

  calib_.cal3_s2_ = gtsam::Cal3_S2{intrinsics[0], intrinsics[1], 0.0,
                                   intrinsics[2], intrinsics[3]};
}

void BagProcessor::load_detections(const std::string_view path) {
#if 0
  std::ifstream f{path.data()};

  if (f.is_open() and f.good()) {

    std::string str_line{};
    while (std::getline(f, str_line)) {
      const nlohmann::json json_root = nlohmann::json::parse(str_line);

      if (json_root["status"].get<std::string>() != "processed") {
        continue;
      }

      const auto num_objects{json_root["total_signs_found"].get<size_t>()};

      if (num_objects == 0) {
        continue;
      }

      ImageDetections image_detections;

      image_detections.timestamp_ =
          json_root["stamp_sec"].get<int64_t>() * 1'000'000'000l +
          json_root["stamp_nanosec"].get<int64_t>() + camera_gps_delta_;

      if (json_root.contains("frame_idx")) {
        image_detections.id_ = json_root["frame_idx"].get<uint64_t>();
      }

      image_detections.dets_.reserve(num_objects);

      for (auto &&det : json_root["detections"]) {

        if (det["label"].get<std::string>() != "traffic_sign") {
          continue;
        }

        // if (det["attributes"]["code"] == "UNK") {
        //   continue;
        // }

        Detection im_det;

        im_det.id_ = 0;
        im_det.code_ = det["attributes"]["code"].get<std::string>();
        im_det.class_ = det["attributes"]["class"].get<std::string>();
        im_det.box_.x = static_cast<int>(det["bbox"][0].get<float>() + 0.5f);
        im_det.box_.y = static_cast<int>(det["bbox"][1].get<float>() + 0.5f);
        im_det.box_.width =
            static_cast<float>(det["bbox"][2].get<int>() + 0.5f) -
            im_det.box_.x + 1;
        im_det.box_.height =
            static_cast<float>(det["bbox"][3].get<int>() + 0.5f) -
            im_det.box_.y + 1;
        im_det.confidence_ = det["confidence"].get<float>();
        im_det.timestamp_ = image_detections.timestamp_;

        image_detections.dets_.emplace_back(im_det);
      }

      image_detections_.emplace_back(image_detections);
    }

  } else {
    throw std::runtime_error{
        fmt::format("unable to open file: {}", path.data())};
  }

  std::sort(
      image_detections_.begin(), image_detections_.end(),
      [](const auto &a, const auto &b) { return a.timestamp_ < b.timestamp_; });

#endif
}

void BagProcessor::load_tracks() {

  std::ifstream f{set_.annotations_path_};

  if (f.is_open() and f.good()) {

    std::unordered_map<size_t, size_t> image_id_to_ind{};

    for (auto &&[i, cam] : enumerate(camera_)) {
      image_id_to_ind[cam.image_id_] = i;
    }

    nlohmann::json json_root = nlohmann::json::parse(f);
    const auto num_tracks{json_root["tracks"].size()};

    image_tracks_.reserve(num_tracks);

    size_t track_id{0};
    size_t detection_id{0};

    for (auto &&json_track : json_root["tracks"]) {

      if (json_track["label"].get<std::string>() != "traffic_sign") {
        continue;
      }

      const auto sign_code{json_track["attributes"]["code"].get<std::string>()};

      if (sign_code == "UNK") {
        continue;
      }

      const auto sign_class{
          json_track["attributes"]["class"].get<std::string>()};

      const auto num_dets{json_track["occurrences"].size()};

      ImageTrack track{};
      track.name_ = set_.session_name_;
      track.id_ = track_id;
      track.code_ = sign_code;
      track.length_ = 0.0;
      track.calib_ = calib_;
      track.geodetic_origin_ = local_converter_.origin();

      for (auto &&json_det : json_track["occurrences"]) {
        Detection det{};

        det.image_id_ = json_det["frame"].get<int>();

        if (not image_id_to_ind.contains(det.image_id_)) {
          continue;
        }

        det.timestamp_ = camera_[image_id_to_ind.at(det.image_id_)].timestamp_;
        // json_det["timestamp"].get<int64_t>() + set_.camera_gps_delta_;
        det.box_.x = json_det["bbox"][0].get<int>();
        det.box_.y = json_det["bbox"][1].get<int>();
        det.box_.width = json_det["bbox"][2].get<int>() - det.box_.x + 1;
        det.box_.height = json_det["bbox"][3].get<int>() - det.box_.y + 1;
        det.code_ = sign_code;
        det.class_ = sign_class;
        det.center_ = {det.box_.x + (det.box_.width >> 1),
                       det.box_.y + (det.box_.height >> 1)};
        det.det_id_ = detection_id;
        det.track_id_ = track_id;
        det.cumulative_length_ = 0.0;
        det.center_undistorted_ = calib_.undistort_point(det.center_);
        det.enu_ = camera_[image_id_to_ind.at(det.image_id_)].enu_;
        det.gps_ind_ = std::distance(
            &camera_[0], &camera_[image_id_to_ind.at(det.image_id_)]);

        track.dets_.emplace_back(det);
        ++detection_id;
      }

      image_tracks_[track_id] = std::move(track);

      for (auto &&d : image_tracks_.at(track_id).dets_) {
        image_tracks_.at(track_id).stamp_to_detection_[d.timestamp_] = &d;
      }

      ++track_id;
    }
  }

  for (auto &&[track_id, track] : image_tracks_) {

    bool first_pose{true};
    Eigen::Vector2d prev_pose{};
    double length{0.0};

    for (auto &&d : track.dets_) {

      if (d.enu_.has_value()) {
        if (first_pose) {
          first_pose = false;
          prev_pose = d.enu_.value();
        } else {
          length += (d.enu_.value() - prev_pose).norm();
          prev_pose = d.enu_.value();
        }
      }

      d.cumulative_length_ = length;
    }
    track.length_ = length;
  }

  LOG(INFO) << "num loaded tracks: " << image_tracks_.size();
}

void BagProcessor::load_measurements(const std::string_view path) {

  std::vector<CameraMeasurement> camera{};
  Eigen::Vector2d prev_enu{std::numeric_limits<double>::max(),
                           std::numeric_limits<double>::max()};

  if (set_.gopro_mode_) {
    GPMFParserSettings parser_set{};
    parser_set.paths_to_mp4_.push_back(set_.bag_path_);
    parser_set.save_bag_ = false;
    parser_set.save_geojson_ = false;
    parser_set.no_imu_ = true;
    parser_set.callback_ = [this, &camera](const GPMFChunkBase *chunk) {
      switch (chunk->whoami()) {

      case ChunkType::GPS: {
        const auto &m{static_cast<const GPSChunk *>(chunk)->measurements_};

        gps_ = std::move(gps_) |
               ranges::actions::push_back(
                   m | transform([](const GPSChunk::Measurement &val) {
                     return GpsMeasurement{.timestamp_ = val.timestamp_,
                                           .enu_ = Eigen::Vector2d::Zero(),
                                           .latlon_ = val.lla_.head<2>()};
                   }));

        break;
      }

      case ChunkType::Camera: {
        const auto &m{static_cast<const SHUTChunk *>(chunk)->measurements_};

        camera = std::move(camera) |
                 ranges::actions::push_back(
                     m | transform([this](const int64_t &val) {
                       return CameraMeasurement{
                           .timestamp_ = val + set_.camera_gps_delta_,
                           .enu_ = Eigen::Vector2d::Zero()};
                     }));

        break;
      }

      default:
        break;
      };
    };

    GPMFParser parser{parser_set};
    parser.parse();

    local_converter_.set_origin(gps_[gps_.size() >> 1].latlon_);
    // local_converter_.set_origin(gps_.front().latlon_);

    for (auto &&gps : gps_) {
      gps.enu_ = local_converter_.enu(gps.latlon_);

      if ((gps.enu_ - prev_enu).norm() > 2.0) {
        prev_enu = gps.enu_;
        stable_gps_.emplace_back(gps.timestamp_, gps.enu_, gps.latlon_);
      }
    }
  } else {

    rclcpp::Serialization<sensor_msgs::msg::CompressedImage>
        serialization_image;
    rclcpp::Serialization<sensor_msgs::msg::NavSatFix> serialization_gps;
    rosbag2_cpp::Reader reader{};

    reader.open(path.data());

    while (reader.has_next()) {
      auto msg{reader.read_next()};
      const rclcpp::SerializedMessage serialized_msg{*msg->serialized_data};

      if (msg->topic_name == set_.compressed_image_topic_) {
        sensor_msgs::msg::CompressedImage image_msg;
        serialization_image.deserialize_message(&serialized_msg, &image_msg);

        const auto timestamp{
            static_cast<int64_t>(image_msg.header.stamp.sec) * 1'000'000'000 +
            static_cast<int64_t>(image_msg.header.stamp.nanosec)};

        camera.emplace_back(CameraMeasurement{
            .timestamp_ = timestamp + set_.camera_gps_delta_});

      } else if (msg->topic_name == set_.gps_topic_) {
        sensor_msgs::msg::NavSatFix gps_msg;
        serialization_gps.deserialize_message(&serialized_msg, &gps_msg);

        const auto timestamp{
            static_cast<int64_t>(gps_msg.header.stamp.sec) * 1'000'000'000 +
            static_cast<int64_t>(gps_msg.header.stamp.nanosec)};

        if (not local_converter_.origin_set()) {
          local_converter_.set_origin({gps_msg.latitude, gps_msg.longitude});
        }

        const Eigen::Vector2d current_enu{
            local_converter_.enu({gps_msg.latitude, gps_msg.longitude})};

        if ((current_enu - prev_enu).norm() > 2.0) {
          prev_enu = current_enu;
          stable_gps_.emplace_back(
              timestamp, current_enu,
              Eigen::Vector2d{gps_msg.latitude, gps_msg.longitude});
        }

        gps_.emplace_back(timestamp, current_enu,
                          Eigen::Vector2d{gps_msg.latitude, gps_msg.longitude});
      }
    }
  }

  std::sort(gps_.begin(), gps_.end(), [](const auto &a, const auto &b) {
    return a.timestamp_ < b.timestamp_;
  });

  for (auto &&[i, cam] : enumerate(camera)) {
    const auto it{
        std::upper_bound(stable_gps_.begin(), stable_gps_.end(), cam.timestamp_,
                         [](const int64_t val, const GpsMeasurement &m) {
                           return val < m.timestamp_;
                         })};

    if (it != stable_gps_.end()) {
      const auto ind{std::distance(stable_gps_.begin(), it)};

      if (ind > 0) {

        const auto t{static_cast<double>(cam.timestamp_ -
                                         stable_gps_[ind - 1].timestamp_) /
                     static_cast<double>(stable_gps_[ind].timestamp_ -
                                         stable_gps_[ind - 1].timestamp_)};

        const Eigen::Vector2d enu{stable_gps_[ind - 1].enu_ * (1.0 - t) +
                                  stable_gps_[ind].enu_ * t};

        camera_.emplace_back(i, cam.timestamp_, enu);
      }
    }
  }

  std::sort(camera_.begin(), camera_.end(), [](const auto &a, const auto &b) {
    return a.timestamp_ < b.timestamp_;
  });
}

std::vector<Landmark>
BagProcessor::load_ground_truth_landmarks(const std::string_view path) {

  std::vector<Landmark> res{};

  std::ifstream f{path.data()};

  if (f.is_open() and f.good()) {
    nlohmann::json j{};
    f >> j;

    size_t landmark_id{0};

    for (auto &&feature : j["features"]) {

      Landmark landmark{};

      landmark.id_ = landmark_id;
      landmark.code_ = feature["properties"]["sign_id"].get<std::string>();

      double longitude{0.0};
      double latitude{0.0};

      if (feature["geometry"]["coordinates"].is_array() and
          feature["geometry"]["coordinates"].size() == 2) {

        longitude = feature["geometry"]["coordinates"][0].get<double>();
        latitude = feature["geometry"]["coordinates"][1].get<double>();

      } else {
        continue;
      }

      landmark.latlon_.x() = latitude;
      landmark.latlon_.y() = longitude;

      landmark.enu_ = local_converter_.enu({latitude, longitude});

      res.push_back(landmark);
      ++landmark_id;

      if (not feature["properties"]["plate_id"].is_null()) {

        const std::string plate_id_str{
            feature["properties"]["plate_id"].get<std::string>()};

        std::stringstream string_stream{plate_id_str};
        std::string token{};

        while (std::getline(string_stream, token, ';')) {
          landmark.id_ = landmark_id;
          landmark.code_ = token;
          res.push_back(landmark);
          ++landmark_id;
        }
      }
    }
  }

  return res;
}

cv::Mat_<cv::Vec3b> BagProcessor::load_image(int64_t timestamp) const {
  rclcpp::Serialization<sensor_msgs::msg::CompressedImage> serialization_image;
  rosbag2_cpp::Reader reader{};
  reader.open(set_.bag_path_);
  reader.seek(timestamp - set_.camera_gps_delta_);

  while (reader.has_next()) {
    auto msg{reader.read_next()};

    if (msg->topic_name == "/camera/image_raw/compressed") {
      const rclcpp::SerializedMessage serialized_msg{*msg->serialized_data};

      sensor_msgs::msg::CompressedImage image_msg;
      serialization_image.deserialize_message(&serialized_msg, &image_msg);

      const auto bag_timestamp{
          static_cast<int64_t>(image_msg.header.stamp.sec) * 1'000'000'000l +
          static_cast<int64_t>(image_msg.header.stamp.nanosec) +
          set_.camera_gps_delta_};

      if (bag_timestamp == timestamp) {
        return cv::imdecode(image_msg.data, cv::IMREAD_UNCHANGED);
      }
    }
  }

  return {};
}

std::optional<Eigen::Isometry3d>
BagProcessor::estimate_camera_pos(Detection &d) {
  const auto [points_in_the_radius, direction] = get_points_in_the_radius(
      camera_, search_radius_, d.enu_.value(), d.gps_ind_);

  if (points_in_the_radius.size() < 5) {
    LOG(WARNING) << "unable to interpolate path, too little points: "
                 << points_in_the_radius.size() << ", " << d.timestamp_;

    d.enu_ = std::nullopt;
    return std::nullopt;
  }

  Eigen::Vector2d estimated_direction{};

  const auto res{
      estimate_direction<poly_degree_>(points_in_the_radius, d.enu_.value())};

  // if (log_poly_) {
  //   ::log_poly("map/poly", rec_, res, {0, 0, 255}, local_converter_);
  // }

  estimated_direction = res.direction_;

  if (estimated_direction.dot(direction) < 0.0) {
    estimated_direction *= -1.0;
  }

  d.direction_ = estimated_direction;
  d.enu_ = res.point_;

  return Eigen::Isometry3d{
      // Eigen::Translation3d{d.pose_.value().x(), d.pose_.value().y(), 0.0f}
      // *
      Eigen::Translation3d{res.point_.x(), res.point_.y(), 0.0f} *
      Eigen::AngleAxisd{set_.correction_angle_ *
                            boost::math::double_constants::degree,
                        Eigen::Vector3d::UnitZ()} *
      Eigen::AngleAxisd{
          -std::atan2(estimated_direction.x(), estimated_direction.y()),
          Eigen::Vector3d::UnitZ()} *
      Eigen::AngleAxisd{-0.5 * std::numbers::pi, Eigen::Vector3d::UnitX()}};

  return std::nullopt;
}

std::optional<Eigen::Isometry3d>
BagProcessor::estimate_camera_pos(int64_t timestamp) const {

  const auto it{
      std::upper_bound(camera_.begin(), camera_.end(), timestamp,
                       [](const int64_t val, const CameraMeasurement &m) {
                         return val < m.timestamp_;
                       })};

  if (it != camera_.end()) {
    const auto ind{std::distance(camera_.begin(), it)};

    if (ind > 0) {

      const auto t{
          static_cast<double>(timestamp - gps_[ind - 1].timestamp_) /
          static_cast<double>(gps_[ind].timestamp_ - gps_[ind - 1].timestamp_)};

      const Eigen::Vector2d cam_pose{gps_[ind - 1].enu_ * (1.0 - t) +
                                     gps_[ind].enu_ * t};

      const auto [points_in_the_radius, direction] =
          get_points_in_the_radius(camera_, search_radius_, cam_pose, ind);

      // auto [estimated_direction, poly] = estimate_direction<poly_degree_>(
      // points_in_the_radius, cam_pose.head(2));

      // auto estimated_direction{
      //     estimate_direction_spline(points_in_the_radius,
      //     cam_pose.head(2))};

      const auto res{
          estimate_direction<poly_degree_>(points_in_the_radius, cam_pose)};

      auto estimated_direction{res.direction_};

      // if (estimated_direction.has_value())
      {

        if (estimated_direction.dot(direction) < 0.0) {
          estimated_direction *= -1.0;
        }

        // return Eigen::Isometry3d::Identity();

        return Eigen::Isometry3d{
            Eigen::Translation3d{cam_pose.x(), cam_pose.y(), 0.0f} *
            Eigen::AngleAxisd{set_.correction_angle_ * std::numbers::pi / 180.0,
                              Eigen::Vector3d::UnitZ()} *
            Eigen::AngleAxisd{
                -std::atan2(estimated_direction.x(), estimated_direction.y()),
                Eigen::Vector3d::UnitZ()} *
            Eigen::AngleAxisd{-0.5 * std::numbers::pi,
                              Eigen::Vector3d::UnitX()}};
      }
    }
  }

  LOG(WARNING) << "unable to estimate camera pose at " << timestamp;
  return std::nullopt;
}

BagProcessor &
BagProcessor::log_ground_truth_landmarks(const std::string_view landmark_code) {
  if (rec_) {
    const auto ground_truth_landmarks{
        load_ground_truth_landmarks(set_.ground_truth_path_)};

    for (auto &&l : ground_truth_landmarks) {
      if (l.code_ == landmark_code) {
        log_landmark_map(l, {255, 0, 0});
      }
    }
  }

  return *this;
}

BagProcessor &BagProcessor::log_ground_truth_landmarks() {
  if (rec_) {

    std::unordered_map<std::string, rerun::Color> color_map{};
    std::mt19937 gen{};
    std::uniform_int_distribution<uint8_t> distrib{0, 255};

    std::vector<rerun::Color> colors{};
    std::vector<rerun::LatLon> coords{};
    std::vector<rerun::components::ClassId> class_ids{};

    const auto ground_truth_landmarks{
        load_ground_truth_landmarks(set_.ground_truth_path_)};

    for (auto &&l : ground_truth_landmarks) {
      if (not color_map.contains(l.code_)) {
        color_map[l.code_] =
            rerun::Color{distrib(gen), distrib(gen), distrib(gen)};
      }

      colors.emplace_back(color_map[l.code_]);
      coords.emplace_back(l.latlon_.x(), l.latlon_.y());

      // log_landmark_map(l, color_map[l.code_]);
    }

    rec_->log("map/gt", rerun::GeoPoints{coords}.with_colors(colors).with_radii(
                            rerun::Radius::ui_points(5.0f)));
  }

  return *this;
}

BagProcessor &
BagProcessor::log_landmarks_map(std::span<const Landmark> landmarks) {

  if (rec_) {
    std::unordered_map<std::string, rerun::Color> color_map{};
    std::mt19937 gen{};
    std::uniform_int_distribution<uint8_t> distrib{0, 255};

    for (auto &&l : landmarks) {
      if (not color_map.contains(l.code_)) {
        color_map[l.code_] =
            rerun::Color{distrib(gen), distrib(gen), distrib(gen)};
      }

      log_landmark_map(l, color_map[l.code_]);
    }
  }

  return *this;
}

BagProcessor &BagProcessor::log_landmark_map(Landmark landmark,
                                             rerun::Color color) {
  if (rec_) {

    const Eigen::Isometry3d landmark_transf{
        Eigen::Translation3d{
            Eigen::Vector3d{landmark.enu_.x(), landmark.enu_.y(), 0.0}} *
        Eigen::AngleAxisd{-landmark.azimuth_, Eigen::Vector3d::UnitZ()}};

    const std::array<Eigen::Vector3d, 4> arrow{
        {{0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {-0.2, 0.7, 0.0}, {0.2, 0.7, 0.0}}};

    const std::vector<rerun::DVec2D> lla_arrow{
        arrow | transform([&landmark_transf, this](auto &&p) {
          const Eigen::Vector3d p0{landmark_transf * p};

          const auto latlon{local_converter_.latlon(p0.head<2>())};
          return rerun::DVec2D{latlon.x(), latlon.y()};
        }) |
        to<std::vector>()};

    const std::string entity_path{
        fmt::format("map/{}_{}", landmark.code_, landmark.id_)};

    rec_->log(
        entity_path,
        rerun::GeoLineStrings{{rerun::components::GeoLineString::from_lat_lon(
                                   {lla_arrow[0], lla_arrow[1]}),
                               rerun::components::GeoLineString::from_lat_lon(
                                   {lla_arrow[2], lla_arrow[1], lla_arrow[3]})}}
            .with_colors(color));

    rec_->log(entity_path,
              rerun::GeoPoints{
                  {rerun::LatLon{landmark.latlon_.x(), landmark.latlon_.y()}}}
                  .with_colors(color)
                  .with_radii(rerun::Radius::ui_points(5.0f)));
  }

  return *this;
}

BagProcessor &BagProcessor::log_gps_path_map() {
  if (rec_) {

    const std::vector<rerun::DVec2D> gps_path{
        gps_ | transform([](auto &&val) {
          return rerun::DVec2D{static_cast<float>(val.latlon_.x()),
                               static_cast<float>(val.latlon_.y())};
        }) |
        to<std::vector>()};

    rec_->log("map/path",
              rerun::GeoLineStrings{
                  rerun::components::GeoLineString::from_lat_lon(gps_path)}
                  .with_colors(rerun::Color{0, 255, 0})
                  .with_radii(rerun::Radius::ui_points(2.0f)));
  }
  return *this;
}

BagProcessor &BagProcessor::log_gps_path() {
  if (rec_) {

    const std::vector<rerun::Vec3D> gps_path{
        gps_ | transform([](auto &&val) {
          return rerun::Vec3D{static_cast<float>(val.enu_.x()),
                              static_cast<float>(val.enu_.y()), 0.0f};
        }) |
        to<std::vector>()};

    rec_->log("world/path",
              rerun::LineStrips3D{rerun::LineStrip3D{gps_path}}.with_colors(
                  rerun::Color{0, 255, 0}));
  }
  return *this;
}

BagProcessor &BagProcessor::log_axis() {
  if (rec_) {

    rec_->log("world/X",
              rerun::LineStrips3D{
                  rerun::LineStrip3D{{rerun::Vec3D{0.0f, 0.0f, 0.0f},
                                      rerun::Vec3D{30.0f, 0.0f, 0.0f}}}}
                  .with_colors(rerun::Color{255, 0, 0})
                  .with_radii(rerun::Radius::ui_points(10.0f)));

    rec_->log("world/Y",
              rerun::LineStrips3D{
                  rerun::LineStrip3D{{rerun::Vec3D{0.0f, 0.0f, 0.0f},
                                      rerun::Vec3D{0.0f, 30.0f, 0.0f}}}}
                  .with_colors(rerun::Color{0, 255, 0})
                  .with_radii(rerun::Radius::ui_points(10.0f)));

    rec_->log("world/Z",
              rerun::LineStrips3D{
                  rerun::LineStrip3D{{rerun::Vec3D{0.0f, 0.0f, 0.0f},
                                      rerun::Vec3D{0.0f, 0.0f, 30.0f}}}}
                  .with_colors(rerun::Color{0, 0, 255})
                  .with_radii(rerun::Radius::ui_points(10.0f)));
  }

  return *this;
}

BagProcessor &BagProcessor::log_camera(int64_t timestamp) {

  if (rec_) {

    const cv::Mat_<cv::Vec3b> img = load_image(timestamp);
    cv::Mat_<cv::Vec3b> img_undist = calib_.undistort_image(img);
    cv::cvtColor(img_undist, img_undist, cv::COLOR_BGR2RGB);

    if (auto cam_pose{estimate_camera_pos(timestamp)}; cam_pose.has_value()) {

      const Eigen::Matrix3f r{cam_pose->linear().cast<float>()};

      rec_->log(fmt::format("world/camera_{}", timestamp),
                rerun::Transform3D{
                    rerun::Vec3D{
                        static_cast<float>(cam_pose->translation().x()),
                        static_cast<float>(cam_pose->translation().y()), 0.0f},
                    rerun::Mat3x3{r.data()}});

      rerun::Mat3x3 rerun_camera{rerun::Mat3x3::IDENTITY};
      rerun_camera.flat_columns[0] = 0.1 * calib_.cal3_s2_.fx();
      rerun_camera.flat_columns[4] = 0.1 * calib_.cal3_s2_.fy();
      rerun_camera.flat_columns[6] = 0.1 * calib_.cal3_s2_.px();
      rerun_camera.flat_columns[7] = 0.1 * calib_.cal3_s2_.py();

      rec_->log(
          fmt::format("world/camera_{}/image", timestamp),
          rerun::Pinhole{rerun::components::PinholeProjection{rerun_camera}});

      rec_->log(
          fmt::format("world/camera_{}/image", timestamp),
          rerun::Image::from_rgb24(
              rerun::Collection<uint8_t>::borrow(
                  img_undist.data,
                  img_undist.cols * img_undist.rows * img_undist.channels()),
              rerun::WidthHeight{static_cast<uint32_t>(img_undist.cols),
                                 static_cast<uint32_t>(img_undist.rows)}));
    }
  }

  return *this;
}

BagProcessor &BagProcessor::log_poly(int64_t timestamp) {

  if (rec_) {

    const auto it{
        std::upper_bound(camera_.begin(), camera_.end(), timestamp,
                         [](const double val, const CameraMeasurement &m) {
                           return val < m.timestamp_;
                         })};

    if (it != camera_.end()) {
      const auto ind{std::distance(camera_.begin(), it)};

      if (ind > 0) {

        const auto t{static_cast<double>(timestamp - gps_[ind - 1].timestamp_) /
                     static_cast<double>(gps_[ind].timestamp_ -
                                         gps_[ind - 1].timestamp_)};

        const Eigen::Vector2d cam_pose{gps_[ind - 1].enu_ * (1.0 - t) +
                                       gps_[ind].enu_ * t};

        const auto [points_in_the_radius, direction] =
            get_points_in_the_radius(camera_, search_radius_, cam_pose, ind);

        std::vector<rerun::Position3D> selected_points{};

        for (auto &&p : points_in_the_radius) {
          selected_points.emplace_back(static_cast<float>(p.x()),
                                       static_cast<float>(p.y()), 0.0f);
        }

        rec_->log(fmt::format("world/selected_points_{}", timestamp),
                  rerun::Points3D{selected_points}
                      .with_colors(rerun::Color{255, 0, 0})
                      .with_radii(rerun::Radius::ui_points(4.0f)));

        if (points_in_the_radius.size() > 5) {

          std::vector<rerun::Vec3D> poly_points{};
          // = interpolate_spline(points_in_the_radius, cam_pose.head(2));

          if (poly_points.empty()) {

            const auto res{estimate_direction<poly_degree_>(
                points_in_the_radius, cam_pose.head(2))};

            if (res.horizontal_) {
              auto [min_it, max_it] = std::minmax_element(
                  points_in_the_radius.begin(), points_in_the_radius.end(),
                  [](const auto &a, const auto &b) { return a.x() < b.x(); });

              for (auto &&x :
                   linear_distribute(min_it->x(), max_it->x(), 300)) {

                double y{0.0};
                double x_val{1.0};

                for (auto &&p : res.point_) {
                  y += p * x_val;
                  x_val *= x;
                }

                poly_points.emplace_back(x, y, 0.0);
              }
            } else {

              auto [min_it, max_it] = std::minmax_element(
                  points_in_the_radius.begin(), points_in_the_radius.end(),
                  [](const auto &a, const auto &b) { return a.y() < b.y(); });

              for (auto &&y :
                   linear_distribute(min_it->y(), max_it->y(), 300)) {

                double x{0.0};
                double y_val{1.0};

                for (auto &&p : res.poly_) {
                  x += p * y_val;
                  y_val *= y;
                }

                poly_points.emplace_back(x, y, 0.0);
              }
            }
          }

          if (not poly_points.empty()) {
            rec_->log(fmt::format("world/poly_{}", timestamp),
                      rerun::LineStrips3D{rerun::LineStrip3D{poly_points}}
                          .with_colors(rerun::Color{255, 105, 40})
                          .with_radii(rerun::Radius::ui_points(4.0f)));
          }
        }
      }
    }
  }

  return *this;
}

BagProcessor &BagProcessor::log_track(size_t track_id) {

  if (rec_) {
    if (image_tracks_.contains(track_id)) {

      bool first_image{false};
      Eigen::Vector2d prev_pose{Eigen::Vector2d::Zero()};

      std::vector<rerun::LatLon> landmark_pos_map{};

      for (auto &&det : image_tracks_.at(track_id).dets_) {

        cv::Mat_<cv::Vec3b> img = load_image(det.timestamp_);

        if (not img.empty()) {

          cv::rectangle(img, det.box_, {255.0, 0.0, 0.0}, 5);

          cv::imwrite(fmt::format("/root/data/images/{}_{}.png", det.timestamp_,
                                  track_id),
                      img);

          cv::resize(img, img, {img.cols >> 2, img.rows >> 2});
          cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

          // cv::Mat_<cv::Vec3b> img_undist;
          // cv::undistort(img, img_undist, camera_matrix_, dist_coeffs_);

          // cv::imwrite(fmt::format("/root/data/images/{}.png",
          // det.timestamp_),
          //             img);

          // estimate_camera_pos(det);

          if (auto cam_pose{estimate_camera_pos(det)}; cam_pose.has_value()) {

            const Eigen::Matrix3f r{cam_pose->linear().cast<float>()};

            if (first_image) {
              prev_pose = cam_pose->translation().head<2>();
              first_image = false;
            } else {
              const Eigen::Vector2d curr_pose{
                  cam_pose->translation().head<2>()};

              if ((curr_pose - prev_pose).squaredNorm() >
                  dist_threshold_squared_) {

                prev_pose = curr_pose;
              } else {
                continue;
              }
            }

            const auto pund{calib_.undistort_point(cv::Point2f{
                static_cast<float>(det.box_.x + (det.box_.width >> 1)),
                static_cast<float>(det.box_.y + (det.box_.height >> 1))})};

            const double z0{150.0};
            const double x{(pund.x - calib_.cal3_s2_.px()) * z0 /
                           calib_.cal3_s2_.fx()};

            const double y{(pund.y - calib_.cal3_s2_.py()) * z0 /
                           calib_.cal3_s2_.fy()};

            const Eigen::Vector3d p0{cam_pose->translation()};
            const Eigen::Vector3d p1{(*cam_pose) * Eigen::Vector3d{x, y, z0}};

            auto p{local_converter_.latlon(p0.head<2>())};
            const rerun::DVec2D lla0{p.x(), p.y()};

            p = local_converter_.latlon(p1.head<2>());
            const rerun::DVec2D lla1{p.x(), p.y()};

            rec_->log(fmt::format("map/dir_{}", det.timestamp_),
                      rerun::GeoLineStrings{
                          rerun::components::GeoLineString::from_lat_lon(
                              {lla0, lla1})}
                          .with_colors(rerun::Color{0xfca40bff})
                          .with_radii(rerun::Radius::ui_points(0.5f)));

            rec_->log(
                fmt::format("world/camera_{}", det.timestamp_),
                rerun::Transform3D{
                    rerun::Vec3D{
                        static_cast<float>(cam_pose->translation().x()),
                        static_cast<float>(cam_pose->translation().y()), 0.0f},
                    rerun::Mat3x3{r.data()}});

            rerun::Mat3x3 rerun_camera{rerun::Mat3x3::IDENTITY};
            rerun_camera.flat_columns[0] = 0.25 * calib_.cal3_s2_.fx();
            rerun_camera.flat_columns[4] = 0.25 * calib_.cal3_s2_.fy();
            rerun_camera.flat_columns[6] = 0.25 * calib_.cal3_s2_.px();
            rerun_camera.flat_columns[7] = 0.25 * calib_.cal3_s2_.py();

            rec_->log(fmt::format("world/camera_{}/image", det.timestamp_),
                      rerun::Pinhole{
                          rerun::components::PinholeProjection{rerun_camera}}
                          .with_image_plane_distance(
                              rerun::components::ImagePlaneDistance{3.0f}));

            rec_->log(fmt::format("world/camera_{}/image", det.timestamp_),
                      rerun::Image::from_rgb24(
                          rerun::Collection<uint8_t>::borrow(
                              img.data, img.cols * img.rows * img.channels()),
                          rerun::WidthHeight{static_cast<uint32_t>(img.cols),
                                             static_cast<uint32_t>(img.rows)}));

            p = local_converter_.latlon(det.enu_.value());
            const rerun::LatLon lla{p.x(), p.y()};

            landmark_pos_map.push_back(lla);
          }
        } else {
          LOG(ERROR) << fmt::format("image {} not found\n", det.timestamp_);
        }
      }

      rec_->log(fmt::format("map/track_{}", track_id),
                rerun::GeoPoints{landmark_pos_map}
                    .with_colors(rerun::Color{0, 0, 255})
                    .with_radii(rerun::Radius::ui_points(5.0f)));
    }
  }

  return *this;
}

BagProcessor &BagProcessor::log_images(int64_t from, int64_t to) {
#if 0
  if (rec_) {

    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<> distrib{0, 255};

    for (auto &&camera_measurement : camera_) {
      const auto camera_timestamp{camera_measurement.timestamp_};

      if (camera_timestamp >= from and camera_timestamp <= to) {

        cv::Mat_<cv::Vec3b> img = load_image(camera_timestamp);

        if (not img.empty()) {
          if (image_detections_.contains(camera_timestamp)) {
            for (auto &&[det_id, d] :
                 image_detections_[camera_timestamp].det_id_to_detection_) {

              if (not color_map_.contains(d->code_)) {

                color_map_[d->code_] =
                    cv::Scalar{static_cast<double>(distrib(gen)),
                               static_cast<double>(distrib(gen)),
                               static_cast<double>(distrib(gen))};
              }

              // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
              cv::rectangle(img, d->box_, color_map_[d->code_], 3);
              cv::putText(img,
                          fmt::format("{} {} ({:.2f})", d->class_, d->code_,
                                      d->confidence_),
                          {d->box_.x - 25, d->box_.y - 25},
                          cv::FONT_HERSHEY_COMPLEX, 1.0, color_map_[d->code_],
                          2, cv::LINE_AA);
            }
          }

          cv::imwrite(
              fmt::format("/root/data/images/image_{}.png", camera_timestamp),
              img);
          // rec_->log("image",
          //           rerun::Image::from_rgb24(
          //               rerun::Collection<uint8_t>::borrow(
          //                   img.data, img.cols * img.rows *
          //                   img.channels()),
          //               rerun::WidthHeight{static_cast<uint32_t>(img.cols),
          //                                  static_cast<uint32_t>(img.rows)}));
        }
      }
    }
  }

#endif
  return *this;
}

size_t BagProcessor::triangulate_tracks() {

  size_t num_triangulated{0};

  for (auto &&[track_id, track] : image_tracks_) {

    if (not track.valid_) {
      continue;
    }

    triangulate(track);

    if (not track.landmark_.has_value()) {
      LOG(WARNING) << "unable to triangulate track " << track_id;
    } else {
      ++num_triangulated;
    }
  }

  LOG(INFO) << "num triangulated tracks: " << num_triangulated;
  return num_triangulated;
}

float BagProcessor::estimate_azimuth(const Eigen::Isometry3d pose,
                                     const Eigen::Vector2f p2d) const {
  const double x{(p2d.x() - calib_.cal3_s2_.px()) / calib_.cal3_s2_.fx()};
  const double y{(p2d.y() - calib_.cal3_s2_.py()) / calib_.cal3_s2_.fy()};

  const Eigen::Vector3d p0{pose.translation()};
  const Eigen::Vector3d p1{pose * Eigen::Vector3d{x, y, 1.0}};

  const Eigen::Vector2d dir{(p1 - p0).head<2>().normalized()};

  const float angle{static_cast<float>(std::atan2(dir.y(), dir.x())) *
                    boost::math::float_constants::radian};

  return angle < 0.0f ? 360.0f + angle : angle;
}

size_t BagProcessor::select_valid_tracks() {

  size_t num_valid_tracks{0};

  for (auto &&[track_id, track] : image_tracks_) {
    Eigen::Vector2d prev_camera_pose{Eigen::Vector2d::Zero()};
    bool first_pose{true};

    float min_angle{360.0f};
    float max_angle{0.0f};

    // log_poly_ = false;

    // if (track_id == 101) {
    //   log_track_map(rec_, track, {255, 0, 0});
    //   log_poly_ = true;
    // }

    for (auto &&d : track.dets_) {

      if (not d.enu_.has_value()) {
        continue;
      }

      const auto camera_pos{estimate_camera_pos(d)};

      if (camera_pos.has_value()) {

        const float angle{
            estimate_azimuth(camera_pos.value(), {d.center_undistorted_.x,
                                                  d.center_undistorted_.y})};

        min_angle = std::min(min_angle, angle);
        max_angle = std::max(max_angle, angle);

        d.cam_to_world_ = camera_pos;
        d.angle_ = angle;
      }
    }

    // if (track_id == 101) {
    //   log_track_map(rec_, track, {255, 0, 0},
    //                 fmt::format("{}_{}_after", track.code_, track_id));
    // }

    const float delta_angle{max_angle - min_angle > 180.0f
                                ? 360.0f - (max_angle - min_angle)
                                : max_angle - min_angle};

    track.delta_angle_ = delta_angle;

    if (delta_angle >= angle_threshold_deg_) {
      track.valid_ = true;
      ++num_valid_tracks;
    } else {
      track.valid_ = false;
    }
  }

  return num_valid_tracks;
}

void BagProcessor::triangulate(ImageTrack &track) {

  if (not track.valid_) {
    return;
  }

  std::optional<Eigen::Vector2d> prev_camera_pose{};
  std::vector<TrackPoint> track_points{};
  std::vector<size_t> selected_detections{};

  for (auto &&[i, d] : enumerate(track.dets_)) {

    if (not d.cam_to_world_.has_value()) {
      continue;
    }

    if (not prev_camera_pose.has_value()) {
      prev_camera_pose = d.cam_to_world_.value().translation().head<2>();
    } else {
      const Eigen::Vector2d current_camera_pose{
          d.cam_to_world_.value().translation().head<2>()};

      if ((current_camera_pose - prev_camera_pose.value()).squaredNorm() >=
          dist_threshold_squared_) {
        prev_camera_pose = current_camera_pose;
      } else {
        continue;
      }
    }

    if (track.roi().contains(d.center_)) {
      track_points.push_back(track.track_point(i));
      selected_detections.push_back(i);
    }
  }

  track.selected_detections_ = std::move(selected_detections);

  try {
    auto landmark{triangulate_on_boxes(track_points)};

    if (not landmark.has_value()) {
      track.valid_ = false;
      LOG(WARNING) << "unable to triangulate landmark: " << track.id_;
      return;
    }

    Landmark l{landmark.value()};
    correct_orientation(l, camera_);

    l.latlon_ = local_converter_.latlon(l.enu_);
    l.code_ = track.code_;
    l.id_ = track.id_;

    track.landmark_ = l;

  } catch (std::exception &ex) {
    LOG(ERROR) << ex.what() << "track id: " << track.id_;
  }
}

BagProcessor &BagProcessor::log_track_directions(size_t track_id,
                                                 float ray_length) {
  if (rec_) {
    if (image_tracks_.contains(track_id)) {

      for (auto &&d : image_tracks_.at(track_id).dets_) {
        log_direction(track_id, d.timestamp_, ray_length);
      }
    }
  }

  return *this;
}

BagProcessor &BagProcessor::log_direction(size_t track_id, int64_t timestamp,
                                          float ray_length) {
  if (rec_) {

    if (image_tracks_.contains(track_id)) {
      for (auto &&d : image_tracks_.at(track_id).dets_) {
        if (d.timestamp_ == timestamp) {
          auto camera_pos{estimate_camera_pos(d)};

          if (camera_pos.has_value()) {

            const float z0{ray_length};
            const float x{static_cast<float>(
                (d.center_undistorted_.x - calib_.cal3_s2_.px()) * z0 /
                calib_.cal3_s2_.fx())};

            const float y{static_cast<float>(
                (d.center_undistorted_.y - calib_.cal3_s2_.py()) * z0 /
                calib_.cal3_s2_.fy())};

            const Eigen::Vector3f p0{camera_pos->translation().cast<float>()};
            const Eigen::Vector3f p1{camera_pos->cast<float>() *
                                     Eigen::Vector3f{x, y, z0}};

            rec_->log(
                fmt::format("world/dir_{}", timestamp),
                rerun::LineStrips3D{
                    rerun::LineStrip3D{{rerun::Vec3D{p0.x(), p0.y(), p0.z()},
                                        rerun::Vec3D{p1.x(), p1.y(), p1.z()}}}}
                    .with_colors(rerun::Color{255, 255, 40})
                    .with_radii(rerun::Radius::ui_points(1.0f)));
          }
        }
      }
    }
  }
  return *this;
}

void BagProcessor::save_geojson(std::span<const Landmark> landmarks,
                                const std::string_view path) const {

  nlohmann::json j{};
  j["type"] = "FeatureCollection";
  j["name"] = set_.session_name_;

  for (auto &&landmark : landmarks) {
    nlohmann::json feature{};
    feature["type"] = "Feature";
    feature["geometry"]["type"] = "Point";
    feature["geometry"]["coordinates"] = {landmark.latlon_.y(),
                                          landmark.latlon_.x()};
    feature["properties"]["sign_id"] = landmark.code_;
    feature["properties"]["description"] = landmark.code_;
    feature["properties"]["marker-color"] = "#1e98ff";

    j["features"].push_back(feature);
  }

  nlohmann::json gps_track{};
  gps_track["type"] = "Feature";
  gps_track["geometry"]["type"] = "LineString";
  gps_track["properties"]["description"] = "GPS track";
  gps_track["properties"]["marker-color"] = "#ed4543";

  for (auto &&gps : gps_) {
    gps_track["geometry"]["coordinates"].push_back(
        {gps.latlon_.y(), gps.latlon_.x()});
  }

  j["features"].push_back(gps_track);

  std::ofstream f{path.data()};
  f << j.dump(4);
}

#if 0
void BagProcessor::track_features() {

  FeatureTracker::Settings feature_tracker_set{};

  feature_tracker_set.fast_threshold_ = 20;
  feature_tracker_set.gridx_ = calib_.camera_resolution_.x() >> 3;
  feature_tracker_set.gridy_ = calib_.camera_resolution_.y() >> 3;
  feature_tracker_set.num_feats_ =
      feature_tracker_set.gridx_ * feature_tracker_set.gridy_ * 30;
  feature_tracker_set.minpxdist_ = 1;
  feature_tracker_set.angle_threshold_deg_ = angle_threshold_deg_;
  feature_tracker_set.use_klt_ = set_.use_klt_;
  feature_tracker_set.save_debug_images_ = false;

  if (rec_) {
    feature_tracker_set.rec_ = rec_.get();
    feature_tracker_set.local_converter_ = local_converter_.get();
  }

  feature_tracker_ = std::make_unique<FeatureTracker>(feature_tracker_set);

  if (not set_.use_klt_) {
    return;
  }

  rclcpp::Serialization<sensor_msgs::msg::CompressedImage> serialization_image;
  rclcpp::Serialization<sensor_msgs::msg::NavSatFix> serialization_gps;
  rosbag2_cpp::Reader reader{};

  reader.open(set_.bag_path_);

  std::vector<ProgressBar::ProgressInfo> topics{};

  for (auto &&topic_info : reader.get_metadata().topics_with_message_count) {
    if (topic_info.topic_metadata.name == "/camera/image_raw/compressed") {
      topics.emplace_back(topic_info.message_count, 0,
                          "/camera/image_raw/compressed", 0);
      break;
    }
  }

  LOG(INFO) << "Tracking features...";

  int num_found{0};

  ProgressBar progress{topics};
  progress.draw();

  while (reader.has_next()) {
    auto msg{reader.read_next()};
    const rclcpp::SerializedMessage serialized_msg{*msg->serialized_data};

    if (msg->topic_name == "/camera/image_raw/compressed") {
      sensor_msgs::msg::CompressedImage image_msg;
      serialization_image.deserialize_message(&serialized_msg, &image_msg);

      const auto timestamp{
          static_cast<int64_t>(image_msg.header.stamp.sec) * 1'000'000'000 +
          static_cast<int64_t>(image_msg.header.stamp.nanosec) +
          set_.camera_gps_delta_};

      {
        cv::Mat_<cv::Vec3b> img =
            cv::imdecode(image_msg.data, cv::IMREAD_UNCHANGED);

        cv::Mat_<uint8_t> img_grey{};
        cv::cvtColor(img, img_grey, cv::COLOR_BGR2GRAY);

        if (image_detections_.contains(timestamp)) {
          feature_tracker_->add(img_grey, image_detections_[timestamp]);
        } else {
          feature_tracker_->add(img_grey,
                                ImageDetections{.timestamp_ = timestamp});
        }

        progress.advance(msg->topic_name);
      }
    }
  }

  progress.done();
  LOG(INFO) << "Finalizing...";
  feature_tracker_->finalize();
}
#endif

void BagProcessor::change_angle(double angle_deg) {
  for (auto &&[track_id, track] : image_tracks_) {

    if (not track.valid_) {
      continue;
    }

    for (auto &&d : track.dets_) {

      if (not d.direction_.has_value()) {
        continue;
      }

      d.cam_to_world_ = Eigen::Isometry3d{
          Eigen::Translation3d{d.enu_.value().x(), d.enu_.value().y(), 0.0f} *
          Eigen::AngleAxisd{angle_deg * boost::math::double_constants::degree,
                            Eigen::Vector3d::UnitZ()} *
          Eigen::AngleAxisd{
              -std::atan2(d.direction_.value().x(), d.direction_.value().y()),
              Eigen::Vector3d::UnitZ()} *
          Eigen::AngleAxisd{-boost::math::double_constants::half_pi,
                            Eigen::Vector3d::UnitX()}};
    }
  }
}

void BagProcessor::save(std::filesystem::path path) const {
  std::ofstream f{path, std::ios_base::binary};

  if (f.is_open() and f.good()) {
    boost::archive::binary_oarchive oa{f};
    oa << *this;
    return;
  }

  throw std::runtime_error{"unable to store object"};
}

BagProcessor::ptr BagProcessor::load(std::filesystem::path path) {
  auto obj{std::make_shared<BagProcessor>()};

  std::ifstream f{path, std::ios_base::binary};

  if (f.is_open() and f.good()) {
    boost::archive::binary_iarchive ia{f};
    ia >> *obj;
    return obj;
  }

  throw std::runtime_error{"unable to load object"};
  return {};
}

size_t BagProcessor::num_valid_tracks() const {
  return ranges::count_if(image_tracks_, [](auto &&val) {
    return val.second.landmark_.has_value();
  });
}

void BagProcessor::extract_images() {

  LOG(INFO) << "Extracting images...";

  std::unordered_set<size_t> frames_to_extract{};

  for (auto &&[track_id, track] : image_tracks_) {

    if (not track.valid_) {
      continue;
    }

    for (auto &&det_ind : track.selected_detections_) {

      if (frames_to_extract.contains(track.dets_[det_ind].image_id_)) {
        continue;
      }

      std::unordered_map<std::string, int> class_count{};

      for (auto &&det :
           image_detections_[track.dets_[det_ind].timestamp_].dets_) {

        ++class_count[det->class_];
      }

      for (auto &&[class_id, count] : class_count) {
        if (count > 1) {
          frames_to_extract.insert(track.dets_[det_ind].image_id_);
          break;
        }
      }
    }
  }

#if 1
  std::unique_ptr<LoaderBase> image_loader{};

  if (set_.gopro_mode_) {
    image_loader = std::make_unique<Mp4ImageLoader>(set_.bag_path_);
  } else {
    image_loader = std::make_unique<BagLoader>(BagLoader::Settings{
        .compressed_image_topic_ = set_.compressed_image_topic_,
        .path_to_bag_ = set_.bag_path_,
        .timestamp_delta_ = set_.camera_gps_delta_,
        .rec_ = {}});
  }

#endif
  // auto detector{cv::AKAZE::create()};

#if 1
  std::vector<ProgressBar::ProgressInfo> topics{
      ProgressBar::ProgressInfo{.message_count_ = frames_to_extract.size(),
                                .processed_count_ = 0,
                                .topic_name_ = "frames: ",
                                .ind_ = 0}};

  ProgressBar progress{topics};
  progress.draw();
#endif

#if 0
  {
    std::vector<std::thread> threads{};

    std::mutex progress_protector{};
    std::atomic_size_t ind{0};
    std::vector<size_t> image_ids{};
    image_ids.reserve(descriptors_.size());

    for (auto &&[image_id, descriprtor] : descriptors_) {
      image_ids.push_back(image_id);
    }

    std::sort(image_ids.begin(), image_ids.end());

    for (auto &&i : ints(0u, std::thread::hardware_concurrency())) {
      threads.emplace_back([&, this]() {
        auto detector{cv::AKAZE::create()};

        std::unique_ptr<LoaderBase> image_loader{};

        if (set_.gopro_mode_) {
          image_loader = std::make_unique<Mp4ImageLoader>(set_.bag_path_);
        } else {
          image_loader = std::make_unique<BagLoader>(BagLoader::Settings{
              .compressed_image_topic_ = set_.compressed_image_topic_,
              .path_to_bag_ = set_.bag_path_,
              .timestamp_delta_ = set_.camera_gps_delta_,
              .rec_ = {}});
        }

        while (true) {
          const auto image_ind{ind.fetch_add(1)};

          if (image_ind >= image_ids.size()) {
            break;
          }

          auto &descriptor{descriptors_[image_ids[image_ind]]};

          if (auto img = image_loader->load_image(image_ids[image_ind]);
              not img.empty()) {
            cv::Mat_<uint8_t> gray_img;
            cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);

            std::vector<cv::KeyPoint> keypoints{};
            cv::Mat_<uint8_t> desc;

            detector->detectAndCompute(gray_img, cv::noArray(), keypoints,
                                       desc);

            descriptor.keypoints_ =
                keypoints |
                transform([](const cv::KeyPoint &kp) { return kp.pt; }) |
                to<std::vector>();

            descriptor.descriptors_ = desc;
            calib_.undistort_points(descriptor.keypoints_);
          }

          {
            std::lock_guard<std::mutex> lock{progress_protector};
            progress.advance("descriptors: ");
          }
        }
      });
    }

    for (auto &t : threads) {
      if (t.joinable()) {
        t.join();
      }
    }
  }
#endif

  std::vector<size_t> image_ids{};
  image_ids.reserve(frames_to_extract.size());

  for (auto &&image_id : frames_to_extract) {
    image_ids.push_back(image_id);
  }

  std::sort(image_ids.begin(), image_ids.end());
  image_loader->set_progress([&progress]() { progress.advance("frames: "); });

  const auto bufs{image_loader->extract(image_ids)};

  for (auto &&[image_id, buf] : zip(image_ids, bufs)) {
    selected_frames_[image_id] = std::move(buf);
  }

#if 0
  for (auto &&image_id : image_ids) {

    if (auto img = image_loader->load_image(image_id); not img.empty()) {

      cv::Mat_<uint8_t> gray_img;
      cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);

      std::vector<cv::KeyPoint> keypoints{};
      cv::Mat_<uint8_t> desc;

      detector->detectAndCompute(gray_img, cv::noArray(), keypoints, desc);

      descriptors_.at(image_id).keypoints_ =
          keypoints | transform([](const cv::KeyPoint &kp) { return kp.pt; }) |
          to<std::vector>();

      descriptors_.at(image_id).descriptors_ = desc;
      calib_.undistort_points(descriptors_.at(image_id).keypoints_);
    }

    progress.advance("descriptors: ");
  }

#endif
  progress.done();
}

void BagProcessorSettings::print() const noexcept {
  LOG(INFO) << fmt::format(fmt::fg(fmt::color::green_yellow), "Input path: ")
            << fmt::format(fmt::fg(fmt::color::orange_red), "{}", bag_path_);

  LOG(INFO) << fmt::format(fmt::fg(fmt::color::green_yellow),
                           "Path to detections: ")
            << fmt::format(fmt::fg(fmt::color::orange_red), "{}",
                           annotations_path_);

  LOG(INFO) << fmt::format(fmt::fg(fmt::color::green_yellow),
                           "Path to calibration: ")
            << fmt::format(fmt::fg(fmt::color::orange_red), "{}",
                           calibration_path_);

  if (not ground_truth_path_.empty()) {
    LOG(INFO) << fmt::format(fmt::fg(fmt::color::green_yellow),
                             "Path to ground truth labels: ")
              << fmt::format(fmt::fg(fmt::color::orange_red), "{}",
                             ground_truth_path_);
  }

  LOG(INFO) << fmt::format(fmt::fg(fmt::color::green_yellow),
                           "Camera correction angle: ")
            << fmt::format(fmt::fg(fmt::color::orange_red), "{}",
                           correction_angle_);

  LOG(INFO) << fmt::format(fmt::fg(fmt::color::green_yellow),
                           "Difference between camera and GPS in nanoseconds: ")
            << fmt::format(fmt::fg(fmt::color::orange_red), "{}",
                           camera_gps_delta_);

  LOG(INFO) << fmt::format(fmt::fg(fmt::color::green_yellow), "Session name: ")
            << fmt::format(fmt::fg(fmt::color::orange_red), "{}",
                           session_name_);
  if (not gopro_mode_) {
    LOG(INFO) << fmt::format(fmt::fg(fmt::color::green_yellow), "GoPro mode: ")
              << fmt::format(fmt::fg(fmt::color::orange_red), "False");

    LOG(INFO) << fmt::format(fmt::fg(fmt::color::green_yellow), "Image topic: ")
              << fmt::format(fmt::fg(fmt::color::orange_red), "{}",
                             compressed_image_topic_);

    LOG(INFO) << fmt::format(fmt::fg(fmt::color::green_yellow), "GPS topic: ")
              << fmt::format(fmt::fg(fmt::color::orange_red), "{}", gps_topic_);
  } else {
    LOG(INFO) << fmt::format(fmt::fg(fmt::color::green_yellow), "GoPro mode: ")
              << fmt::format(fmt::fg(fmt::color::orange_red), "True");
  }
}