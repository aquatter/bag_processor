#include <Eigen/Geometry>
#include <Eigen/src/Core/Matrix.h>
#include <GeographicLib/LocalCartesian.hpp>
#include <algorithm>
#include <bag_processor.hpp>
#include <cstddef>
#include <cstdint>
#include <csv_parser.hpp>
#include <deque>
#include <exception>
#include <filesystem>
#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fstream>
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
#include <memory>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <numbers>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <random>
#include <range/v3/algorithm/max.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/enumerate.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/linear_distribute.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/view/zip.hpp>
#include <rerun.hpp>
#include <rerun/archetypes/geo_points.hpp>
#include <rerun/archetypes/points3d.hpp>
#include <rerun/components/image_plane_distance.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <stdexcept>
#include <string>
#include <tracker.h>
#include <unordered_map>
#include <vector>
#include <yaml-cpp/node/node.h>
#include <yaml-cpp/node/parse.h>
#include <yaml-cpp/yaml.h>

using ranges::to;
using ranges::views::enumerate;
using ranges::views::ints;
using ranges::views::linear_distribute;
using ranges::views::transform;
using ranges::views::zip;

std::tuple<std::vector<Eigen::Vector2d>, Eigen::Vector2d>
get_points_in_the_radius(std::span<const GpsMeasurement> points, double rad,
                         Eigen::Vector3d query_point, ptrdiff_t ind) {

  const double rad_squared{rad * rad};

  Eigen::Vector2d first_point{};
  Eigen::Vector2d last_point{};
  std::deque<Eigen::Vector2d> points_queue;

  int num_added{0};

  for (ptrdiff_t i{ind - 1}; i >= 0; --i) {
    if ((points[i].position_ - query_point).squaredNorm() < rad_squared or
        num_added < 5) {
      points_queue.emplace_front(points[i].position_.x(),
                                 points[i].position_.y());
      ++num_added;
      first_point = points[i].position_.head(2);
    } else {
      break;
    }
  }

  num_added = 0;
  for (ptrdiff_t i{ind}; i < points.size(); ++i) {
    if ((points[i].position_ - query_point).squaredNorm() < rad_squared or
        num_added < 5) {
      points_queue.emplace_back(points[i].position_.x(),
                                points[i].position_.y());

      ++num_added;
      last_point = points[i].position_.head(2);
    } else {
      break;
    }
  }

  last_point = (last_point - first_point).normalized();

  return {
      std::vector<Eigen::Vector2d>{points_queue.begin(), points_queue.end()},
      last_point};
}

std::vector<rerun::Vec3D>
interpolate_spline(std::span<const Eigen::Vector2d> points,
                   Eigen::Vector2d query_point) {

  alglib::spline1dinterpolant c;
  alglib::real_1d_array x;
  alglib::real_1d_array y;

  std::vector<Eigen::Vector2d> sorted_points{points.begin(), points.end()};

  x.setlength(points.size());
  y.setlength(points.size());

  std::vector<rerun::Vec3D> poly_points;

  if (std::is_sorted(
          points.begin(), points.end(),
          [](const auto &a, const auto &b) { return a.x() < b.x(); }) or
      std::is_sorted(points.begin(), points.end(),
                     [](const auto &a, const auto &b) { return a.x() > b.x(); })

  ) {

    std::sort(sorted_points.begin(), sorted_points.end(),
              [](const auto &a, const auto &b) { return a.x() < b.x(); });

    for (auto &&[i, p] : enumerate(sorted_points)) {
      x(i) = p.x();
      y(i) = p.y();
    }

    alglib::spline1dbuildcubic(x, y, c);

    for (auto &&x_val : linear_distribute(sorted_points.front().x(),
                                          sorted_points.back().x(), 30)) {

      poly_points.emplace_back(x_val, alglib::spline1dcalc(c, x_val), 0.0);
    }
  } else if (std::is_sorted(
                 points.begin(), points.end(),
                 [](const auto &a, const auto &b) { return a.y() < b.y(); }) or
             std::is_sorted(
                 points.begin(), points.end(),
                 [](const auto &a, const auto &b) { return a.y() > b.y(); })) {

    std::sort(sorted_points.begin(), sorted_points.end(),
              [](const auto &a, const auto &b) { return a.y() < b.y(); });

    for (auto &&[i, p] : enumerate(sorted_points)) {
      x(i) = p.y();
      y(i) = p.x();
    }

    alglib::spline1dbuildakima(x, y, c);

    for (auto &&y_val : linear_distribute(sorted_points.front().y(),
                                          sorted_points.back().y(), 30)) {

      poly_points.emplace_back(alglib::spline1dcalc(c, y_val), y_val, 0.0);
    }
  }

  return poly_points;
}

std::optional<Eigen::Vector2d>
estimate_direction_spline(std::span<const Eigen::Vector2d> points,
                          Eigen::Vector2d query_point) {
  alglib::spline1dinterpolant c;
  alglib::real_1d_array x;
  alglib::real_1d_array y;

  x.setlength(points.size());
  y.setlength(points.size());

  const size_t n{points.size() - 1};

  if (std::is_sorted(
          points.begin(), points.end(),
          [](const auto &a, const auto &b) { return a.x() < b.x(); })) {

    for (auto &&[i, p] : enumerate(points)) {
      x(i) = p.x();
      y(i) = p.y();
    }

    alglib::spline1dbuildakima(x, y, c);

    double y_val{0.0};
    double d_y_val{0.0};
    double d2_y_val{0.0};

    alglib::spline1ddiff(c, query_point.x(), y_val, d_y_val, d2_y_val);
    return Eigen::Vector2d{1.0, d_y_val}.normalized();

  } else if (std::is_sorted(
                 points.begin(), points.end(),
                 [](const auto &a, const auto &b) { return a.x() > b.x(); })) {

    for (auto &&[i, p] : enumerate(points)) {
      x(n - i) = p.x();
      y(n - i) = p.y();
    }

    alglib::spline1dbuildakima(x, y, c);

    double y_val{0.0};
    double d_y_val{0.0};
    double d2_y_val{0.0};

    alglib::spline1ddiff(c, query_point.x(), y_val, d_y_val, d2_y_val);
    return Eigen::Vector2d{1.0, d_y_val}.normalized();

  } else if (std::is_sorted(
                 points.begin(), points.end(),
                 [](const auto &a, const auto &b) { return a.y() < b.y(); })) {

    for (auto &&[i, p] : enumerate(points)) {
      x(i) = p.y();
      y(i) = p.x();
    }

    alglib::spline1dbuildakima(x, y, c);

    double y_val{0.0};
    double d_y_val{0.0};
    double d2_y_val{0.0};

    alglib::spline1ddiff(c, query_point.y(), y_val, d_y_val, d2_y_val);
    return Eigen::Vector2d{d_y_val, 1.0}.normalized();

  } else if (std::is_sorted(
                 points.begin(), points.end(),
                 [](const auto &a, const auto &b) { return a.y() > b.y(); })) {
    for (auto &&[i, p] : enumerate(points)) {
      x(n - i) = p.y();
      y(n - i) = p.x();
    }

    alglib::spline1dbuildakima(x, y, c);

    double y_val{0.0};
    double d_y_val{0.0};
    double d2_y_val{0.0};

    alglib::spline1ddiff(c, query_point.y(), y_val, d_y_val, d2_y_val);
    // return Eigen::Vector2d{1.0, -d_y_val}.normalized();
    return Eigen::Vector2d{d_y_val, 1.0}.normalized();
  }

  LOG(WARNING) << "unable to approximate spline";
  return std::nullopt;
}

template <int degree>
std::tuple<Eigen::Vector2d, std::array<double, degree + 1>, bool>
estimate_direction(std::span<const Eigen::Vector2d> points,
                   Eigen::Vector2d query_point) {

  bool horizontal_dir{true};

  {
    auto [min_x, max_x] = std::minmax_element(
        points.begin(), points.end(),
        [](const auto &a, const auto &b) { return a.x() < b.x(); });

    auto [min_y, max_y] = std::minmax_element(
        points.begin(), points.end(),
        [](const auto &a, const auto &b) { return a.y() < b.y(); });

    horizontal_dir =
        (max_x->x() - min_x->x()) > (max_y->y() - min_y->y()) ? true : false;
  }

  const int n{static_cast<int>(points.size())};

  Eigen::MatrixXd A;
  Eigen::MatrixXd b;
  A.resize(n, degree + 1);
  b.resize(n, 1);

  if (horizontal_dir) {
    for (auto &&[i, p] : enumerate(points)) {
      double x_val{1.0};

      for (int &&j : ints(0, degree + 1)) {
        A(i, j) = x_val;
        x_val *= static_cast<double>(p.x());
      }

      b(i) = p.y();
    }
  } else {
    for (auto &&[i, p] : enumerate(points)) {
      double x_val{1.0};

      for (int &&j : ints(0, degree + 1)) {
        A(i, j) = x_val;
        x_val *= static_cast<double>(p.y());
      }

      b(i) = p.x();
    }
  }

  const Eigen::MatrixXd p{(A.transpose() * A).ldlt().solve(A.transpose() * b)};

  std::array<double, degree + 1> poly{};
  for (auto &&i : ints(0, degree + 1)) {
    poly[i] = p(i);
  }

  if (horizontal_dir) {
    double y_prime{0.0};
    double x_val{1.0};

    for (auto &&i : ints(1, degree + 1)) {
      y_prime += p(i) * x_val * static_cast<double>(i);
      x_val *= query_point.x();
    }

    return {Eigen::Vector2d{1.0, y_prime}.normalized(), poly, horizontal_dir};
  }

  double x_prime{0.0};
  double y_val{1.0};

  for (auto &&i : ints(1, degree + 1)) {
    x_prime += p(i) * y_val * static_cast<double>(i);
    y_val *= query_point.y();
  }

  return {Eigen::Vector2d{x_prime, 1.0}.normalized(), poly, horizontal_dir};
}

Eigen::Vector3d triangulate_gtsam(
    const gtsam::CameraSet<gtsam::PinholeCamera<gtsam::Cal3_S2>> &cameras,
    const gtsam::Point2Vector &measurements) {

  auto p3d{gtsam::triangulateDLT(gtsam::projectionMatricesFromCameras(cameras),
                                 measurements)};

  const auto measurement_noise{gtsam::noiseModel::Isotropic::Sigma(2, 1.0)};

  const auto [graph, values] = gtsam::triangulationGraph(
      cameras, measurements, gtsam::Symbol{'p', 0}, p3d, measurement_noise);

  gtsam::LevenbergMarquardtParams params;
  params.verbosityLM = gtsam::LevenbergMarquardtParams::TRYLAMBDA;
  params.verbosity = gtsam::NonlinearOptimizerParams::ERROR;
  params.lambdaInitial = 1;
  params.lambdaFactor = 10;
  params.maxIterations = 100;
  params.absoluteErrorTol = 1.0;
  params.verbosityLM = gtsam::LevenbergMarquardtParams::SILENT;
  params.verbosity = gtsam::NonlinearOptimizerParams::SILENT;
  params.linearSolverType =
      gtsam::NonlinearOptimizerParams::MULTIFRONTAL_CHOLESKY;

  gtsam::LevenbergMarquardtOptimizer optimizer{graph, values, params};
  gtsam::Values result{optimizer.optimize()};

  gtsam::Marginals marginals{graph, result};
  auto cov{marginals.marginalCovariance(gtsam::Symbol{'p', 0})};

  std::cout << cov << std::endl;
  std::cout << "determinant: " << cov.determinant() << std::endl;

  return result.at<gtsam::Point3>(gtsam::Symbol{'p', 0});
}

BagProcessor::BagProcessor(const BagProcessorSettings &set) : set_{set} {

  if (set.use_logger_) {
    rec_ = std::make_unique<rerun::RecordingStream>("bag_converter");
    rec_->connect_grpc().exit_on_failure();
  }

  load_measurements(set.bag_path_);
  load_calibration(set.calibration_path_);
  load_detections(set.annotations_path_);

  for (auto &&image_det : image_detections_) {
    image_det.timestamp_ = camera_[image_det.id_].timestamp_;
  }

  create_tracks();
  load_ground_truth_landmarks(set.ground_truth_path_);

  std::unordered_map<std::string, int> landmark_frequency{};

  int max_frequency{0};
  std::string most_frequent_landmark{};

  for (auto &&landmark : ground_truth_landmarks_) {

    if (landmark.code_ == "address" or landmark.code_ == "barrier") {
      continue;
    }

    ++landmark_frequency[landmark.code_];

    if (max_frequency < landmark_frequency[landmark.code_]) {
      max_frequency = landmark_frequency[landmark.code_];
      most_frequent_landmark = landmark.code_;
    }
  }

  LOG(INFO) << "mos frequent landmark: " << most_frequent_landmark;
  LOG(INFO) << "num detections: " << image_detections_.size();
}

void BagProcessor::load_calibration(const std::string_view path) {
  YAML::Node calib{YAML::LoadFile(path.data())};

  const auto intrinsics{calib["cam0"]["intrinsics"].as<std::vector<double>>()};
  const auto distortion{
      calib["cam0"]["distortion_coeffs"].as<std::vector<double>>()};

  camera_matrix_ = cv::Mat_<double>::eye(3, 3);
  camera_matrix_(0, 0) = intrinsics[0];
  camera_matrix_(1, 1) = intrinsics[1];
  camera_matrix_(0, 2) = intrinsics[2];
  camera_matrix_(1, 2) = intrinsics[3];

  dist_coeffs_ = cv::Mat_<double>(distortion, true);

  gtsam_cal3_s2 = gtsam::Cal3_S2{intrinsics[0], intrinsics[1], 0.0,
                                 intrinsics[2], intrinsics[3]};
}

void BagProcessor::load_detections(const std::string_view path) {

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

      image_detections.id_ = json_root["frame_idx"].get<uint64_t>();

      image_detections.dets_.reserve(num_objects);

      for (auto &&det : json_root["detections"]) {

        if (det["label"].get<std::string>() != "traffic_sign") {
          continue;
        }
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
}

void BagProcessor::create_tracks() {
  // image_detections_ = parse_csv(path);

  std::unordered_map<std::string, Tracker> trackers;

  std::vector<cv::Rect> boxes;

  for (auto &&camera_meausurement : camera_) {

    const auto camera_timestamp{camera_meausurement.timestamp_};

    const auto dets_it{std::find_if(image_detections_.begin(),
                                    image_detections_.end(),
                                    [camera_timestamp](auto &&val) {
                                      return camera_timestamp == val.timestamp_;
                                    })};

    std::unordered_map<std::string, std::vector<cv::Rect>> det_map;

    if (dets_it != image_detections_.end()) {

      for (auto &&d : dets_it->dets_) {
        det_map[d.code_].push_back(d.box_);

        if (not trackers.contains(d.code_)) {
          trackers[d.code_] = {};
        }
      }
    }

    for (auto &&[code, tracker] : trackers) {

      const auto &current_dets{det_map[code]};

      tracker.Run(current_dets);
      const auto tracks{tracker.GetTracks()};

      for (auto &&[id, track] : tracks) {

        // if (code == "5.19.1" and id == 71) {
        //   cv::Mat_<cv::Vec3b> img = load_image(camera_timestamp);

        //   for (auto &&d : current_dets) {
        //     cv::rectangle(img, d, {255.0, 0.0, 0.0}, 3);
        //     cv::rectangle(img, track.GetStateAsBbox(), {0.0, 255.0, 0.0}, 3);
        //   }

        //   cv::imwrite(
        //       fmt::format("/root/data/images1/{}.png", camera_timestamp),
        //       img);
        // }

        if (track.coast_cycles_ < kMaxCoastCycles and
            track.hit_streak_ >= kMinHits) {

          size_t det_ind{0};
          float max_iou{-1.0f};

          for (auto &&[i, d] : enumerate(current_dets)) {
            const float iou{Tracker::CalculateIou(d, track)};
            if (max_iou < iou) {
              det_ind = i;
              max_iou = iou;
            }
          }

          if (max_iou > 0.0f) {
            Detection d;
            d.timestamp_ = camera_timestamp;
            d.box_ = current_dets[det_ind];
            d.id_ = id;
            d.code_ = code;
            // d.class_ = dets.dets_[det_ind].class_;
            // d.confidence_ = dets.dets_[det_ind].confidence_;

            const auto track_id{fmt::format("{}_{}", code, id)};

            if (not image_tracks_.contains(track_id)) {
              image_tracks_[track_id].id_ = track_id;
            }

            image_tracks_[track_id].dets_.emplace_back(d);
          }
        }
      }
    }

#if 0
    if (dets.dets_.size() > 10) {

      cv::Mat_<cv::Vec3b> img =
          load_image("/root/data/rosbag2_2025_08_21-09_32_11", dets.timestamp_);

      if (not img.empty()) {

        for (auto &&d : dets.dets_) {
          cv::rectangle(img, d.box_, {255.0, 0.0, 200.0}, 5);
        }

        cv::imwrite("test.png", img);
      }
    }
#endif
  }

  for (auto &&[track_id, track] : image_tracks_) {

    bool first_pose{true};
    Eigen::Vector3d prev_pose{};
    double length{0.0};

    for (auto &&d : track.dets_) {

      const auto it{
          std::upper_bound(gps_.begin(), gps_.end(), d.timestamp_,
                           [](const double val, const GpsMeasurement &m) {
                             return val < m.timestamp_;
                           })};

      if (it != gps_.end()) {
        const auto ind{std::distance(gps_.begin(), it)};

        if (ind > 0) {

          const auto t{
              static_cast<double>(d.timestamp_ - gps_[ind - 1].timestamp_) /
              static_cast<double>(gps_[ind].timestamp_ -
                                  gps_[ind - 1].timestamp_)};

          d.pose_ = Eigen::Vector3d{gps_[ind - 1].position_ * (1.0 - t) +
                                    gps_[ind].position_ * t};

          d.gps_ind_ = ind;
        }
      }

      if (d.pose_.has_value()) {
        if (first_pose) {
          first_pose = false;
          prev_pose = d.pose_.value();
        } else {
          length += (d.pose_.value() - prev_pose).norm();
          prev_pose = d.pose_.value();
        }
      }
    }
    track.length_ = length;
  }

  LOG(INFO) << "num created tracks: " << image_tracks_.size();
}

void BagProcessor::load_measurements(const std::string_view path) {

  rclcpp::Serialization<sensor_msgs::msg::CompressedImage> serialization_image;
  rclcpp::Serialization<sensor_msgs::msg::NavSatFix> serialization_gps;
  rosbag2_cpp::Reader reader{};

  reader.open(path.data());

  Eigen::Vector3d prev_enu{Eigen::Vector3d::Zero()};

  uint64_t frame_idx{0};

  while (reader.has_next()) {
    auto msg{reader.read_next()};
    const rclcpp::SerializedMessage serialized_msg{*msg->serialized_data};

    if (msg->topic_name == "/camera/image_raw/compressed") {
      sensor_msgs::msg::CompressedImage image_msg;
      serialization_image.deserialize_message(&serialized_msg, &image_msg);

      const auto timestamp{
          static_cast<int64_t>(image_msg.header.stamp.sec) * 1'000'000'000 +
          static_cast<int64_t>(image_msg.header.stamp.nanosec)};

      camera_.emplace_back(timestamp + camera_gps_delta_, frame_idx);
      ++frame_idx;

    } else if (msg->topic_name == "/fix") {
      sensor_msgs::msg::NavSatFix gps_msg;
      serialization_gps.deserialize_message(&serialized_msg, &gps_msg);

      const auto timestamp{static_cast<int64_t>(gps_msg.header.stamp.sec) *
                               1'000'000'000 +
                           static_cast<int64_t>(gps_msg.header.stamp.nanosec)};

      if (not local_converter_) {
        local_converter_ = std::make_unique<GeographicLib::LocalCartesian>(
            gps_msg.latitude, gps_msg.longitude, gps_msg.altitude);

        gps_.emplace_back(timestamp, prev_enu,
                          Eigen::Vector3d{gps_msg.latitude, gps_msg.longitude,
                                          gps_msg.altitude});
      }

      Eigen::Vector3d enu{};
      local_converter_->Forward(gps_msg.latitude, gps_msg.longitude,
                                gps_msg.altitude, enu.x(), enu.y(), enu.z());

      if ((enu - prev_enu).squaredNorm() >= 1.0) {
        gps_.emplace_back(timestamp, enu,
                          Eigen::Vector3d{gps_msg.latitude, gps_msg.longitude,
                                          gps_msg.altitude});
        prev_enu = enu;
      }
    }
  }

  std::sort(camera_.begin(), camera_.end(), [](const auto &a, const auto &b) {
    return a.timestamp_ < b.timestamp_;
  });
  std::sort(gps_.begin(), gps_.end(), [](const auto &a, const auto &b) {
    return a.timestamp_ < b.timestamp_;
  });
}

void BagProcessor::load_ground_truth_landmarks(const std::string_view path) {

  std::ifstream f{path.data()};

  if (f.is_open() and f.good()) {
    nlohmann::json j{};
    f >> j;

    for (auto &&[i, feature] : enumerate(j["features"])) {

      Landmark landmark{};

      landmark.id_ = i;
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

      landmark.lla_.x() = latitude;
      landmark.lla_.y() = longitude;

      Eigen::Vector3d enu{};
      local_converter_->Forward(latitude, longitude, 0.0, enu.x(), enu.y(),
                                enu.z());

      landmark.position_ = enu;

      ground_truth_landmarks_.push_back(landmark);

      if (not feature["properties"]["plate_id"].is_null()) {

        const std::string plate_id_str{
            feature["properties"]["plate_id"].get<std::string>()};

        std::stringstream string_stream{plate_id_str};
        std::string token{};

        while (std::getline(string_stream, token, ';')) {
          landmark.code_ = token;
          ground_truth_landmarks_.push_back(landmark);
        }
      }
    }
  }
}

cv::Mat_<cv::Vec3b> BagProcessor::load_image(int64_t timestamp) const {
  rclcpp::Serialization<sensor_msgs::msg::CompressedImage> serialization_image;
  rosbag2_cpp::Reader reader{};
  reader.open(set_.bag_path_);

  while (reader.has_next()) {
    auto msg{reader.read_next()};

    if (msg->topic_name == "/camera/image_raw/compressed") {
      const rclcpp::SerializedMessage serialized_msg{*msg->serialized_data};

      sensor_msgs::msg::CompressedImage image_msg;
      serialization_image.deserialize_message(&serialized_msg, &image_msg);

      const auto bag_timestamp{
          static_cast<int64_t>(image_msg.header.stamp.sec) * 1'000'000'000l +
          static_cast<int64_t>(image_msg.header.stamp.nanosec) +
          camera_gps_delta_};

      if (bag_timestamp == timestamp) {
        return cv::imdecode(image_msg.data, cv::IMREAD_UNCHANGED);
      }
    }
  }

  return {};
}

std::optional<Eigen::Isometry3d>
BagProcessor::estimate_camera_pos(const Detection &d) const {
  const auto [points_in_the_radius, direction] = get_points_in_the_radius(
      gps_, search_radius_, d.pose_.value(), d.gps_ind_);

  if (points_in_the_radius.size() < 5) {
    LOG(WARNING) << "unable to interpolate spline, too little points: "
                 << points_in_the_radius.size() << ", " << d.timestamp_;
    return std::nullopt;
  }

  std::optional<Eigen::Vector2d> estimated_direction{};
  // estimate_direction_spline(points_in_the_radius,
  // d.pose_.value().head(2))};

  if (not estimated_direction.has_value()) {
    // LOG(WARNING) << "fallback to poly approximation";

    const auto [dir, poly_coeffs, hor_dir] = estimate_direction<poly_degree_>(
        points_in_the_radius, d.pose_.value().head(2));

    estimated_direction = dir;
  }

  if (estimated_direction.has_value()) {

    if (estimated_direction.value().dot(direction) < 0.0) {
      estimated_direction.value() *= -1.0;
    }

    return Eigen::Isometry3d{
        Eigen::Translation3d{d.pose_.value().x(), d.pose_.value().y(), 0.0f} *
        Eigen::AngleAxisd{correction_angle_ * std::numbers::pi / 180.0,
                          Eigen::Vector3d::UnitZ()} *
        Eigen::AngleAxisd{-std::atan2(estimated_direction.value().x(),
                                      estimated_direction.value().y()),
                          Eigen::Vector3d::UnitZ()} *
        Eigen::AngleAxisd{-0.5 * std::numbers::pi, Eigen::Vector3d::UnitX()}};
  }

  return std::nullopt;
}

std::optional<Eigen::Isometry3d>
BagProcessor::estimate_camera_pos(int64_t timestamp) const {

  const auto it{std::upper_bound(gps_.begin(), gps_.end(), timestamp,
                                 [](const double val, const GpsMeasurement &m) {
                                   return val < m.timestamp_;
                                 })};

  if (it != gps_.end()) {
    const auto ind{std::distance(gps_.begin(), it)};

    if (ind > 0) {

      const auto t{
          static_cast<double>(timestamp - gps_[ind - 1].timestamp_) /
          static_cast<double>(gps_[ind].timestamp_ - gps_[ind - 1].timestamp_)};

      const Eigen::Vector3d cam_pose{gps_[ind - 1].position_ * (1.0 - t) +
                                     gps_[ind].position_ * t};

      const auto [points_in_the_radius, direction] =
          get_points_in_the_radius(gps_, search_radius_, cam_pose, ind);

      // auto [estimated_direction, poly] = estimate_direction<poly_degree_>(
      // points_in_the_radius, cam_pose.head(2));

      auto estimated_direction{
          estimate_direction_spline(points_in_the_radius, cam_pose.head(2))};

      if (estimated_direction.has_value()) {

        if (estimated_direction.value().dot(direction) < 0.0) {
          estimated_direction.value() *= -1.0;
        }

        // return Eigen::Isometry3d::Identity();

        return Eigen::Isometry3d{
            Eigen::Translation3d{cam_pose.x(), cam_pose.y(), 0.0f} *
            Eigen::AngleAxisd{correction_angle_ * std::numbers::pi / 180.0,
                              Eigen::Vector3d::UnitZ()} *
            Eigen::AngleAxisd{-std::atan2(estimated_direction.value().x(),
                                          estimated_direction.value().y()),
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
    for (auto &&l : ground_truth_landmarks_) {
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

    for (auto &&l : ground_truth_landmarks_) {
      if (not color_map.contains(l.code_)) {
        color_map[l.code_] =
            rerun::Color{distrib(gen), distrib(gen), distrib(gen)};
      }

      log_landmark_map(l, color_map[l.code_]);
    }
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
    rec_->log(
        fmt::format("map/{}_{}", landmark.code_, landmark.id_),
        rerun::GeoPoints{{rerun::LatLon{landmark.lla_.x(), landmark.lla_.y()}}}
            .with_colors(color)
            .with_radii(rerun::Radius::ui_points(5.0f)));
  }

  return *this;
}

BagProcessor &BagProcessor::log_gps_path_map() {
  if (rec_) {

    const std::vector<rerun::DVec2D> gps_path{
        gps_ | transform([](auto &&val) {
          return rerun::DVec2D{static_cast<float>(val.lla_.x()),
                               static_cast<float>(val.lla_.y())};
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
          return rerun::Vec3D{static_cast<float>(val.position_.x()),
                              static_cast<float>(val.position_.y()), 0.0f};
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
    cv::Mat_<cv::Vec3b> img_undist;
    cv::undistort(img, img_undist, camera_matrix_, dist_coeffs_);
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
      rerun_camera.flat_columns[0] = gtsam_cal3_s2.fx();
      rerun_camera.flat_columns[4] = gtsam_cal3_s2.fy();
      rerun_camera.flat_columns[6] = gtsam_cal3_s2.px();
      rerun_camera.flat_columns[7] = gtsam_cal3_s2.py();

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
        std::upper_bound(gps_.begin(), gps_.end(), timestamp,
                         [](const double val, const GpsMeasurement &m) {
                           return val < m.timestamp_;
                         })};

    if (it != gps_.end()) {
      const auto ind{std::distance(gps_.begin(), it)};

      if (ind > 0) {

        const auto t{static_cast<double>(timestamp - gps_[ind - 1].timestamp_) /
                     static_cast<double>(gps_[ind].timestamp_ -
                                         gps_[ind - 1].timestamp_)};

        const Eigen::Vector3d cam_pose{gps_[ind - 1].position_ * (1.0 - t) +
                                       gps_[ind].position_ * t};

        const auto [points_in_the_radius, direction] =
            get_points_in_the_radius(gps_, search_radius_, cam_pose, ind);

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

            auto [estimated_direction, poly, hor_dir] =
                estimate_direction<poly_degree_>(points_in_the_radius,
                                                 cam_pose.head(2));

            if (hor_dir) {
              auto [min_it, max_it] = std::minmax_element(
                  points_in_the_radius.begin(), points_in_the_radius.end(),
                  [](const auto &a, const auto &b) { return a.x() < b.x(); });

              for (auto &&x :
                   linear_distribute(min_it->x(), max_it->x(), 300)) {

                double y{0.0};
                double x_val{1.0};

                for (auto &&p : poly) {
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

                for (auto &&p : poly) {
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

BagProcessor &BagProcessor::log_track(const std::string_view track_id) {

  if (rec_) {
    if (image_tracks_.contains(track_id.data())) {

      bool first_image{false};
      Eigen::Vector2d prev_pose{Eigen::Vector2d::Zero()};

      std::vector<rerun::LatLon> landmark_pos_map{};

      for (auto &&det : image_tracks_.at(track_id.data()).dets_) {

        cv::Mat_<cv::Vec3b> img = load_image(det.timestamp_);

        if (not img.empty()) {

          cv::rectangle(img, det.box_, {255.0, 0.0, 0.0}, 5);

          cv::resize(img, img, {img.cols >> 2, img.rows >> 2});
          cv::imwrite(fmt::format("/root/data/images/{}_{}.png", det.timestamp_,
                                  track_id),
                      img);

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

            rec_->log(
                fmt::format("world/camera_{}", det.timestamp_),
                rerun::Transform3D{
                    rerun::Vec3D{
                        static_cast<float>(cam_pose->translation().x()),
                        static_cast<float>(cam_pose->translation().y()), 0.0f},
                    rerun::Mat3x3{r.data()}});

            rerun::Mat3x3 rerun_camera{rerun::Mat3x3::IDENTITY};
            rerun_camera.flat_columns[0] = 0.25 * gtsam_cal3_s2.fx();
            rerun_camera.flat_columns[4] = 0.25 * gtsam_cal3_s2.fy();
            rerun_camera.flat_columns[6] = 0.25 * gtsam_cal3_s2.px();
            rerun_camera.flat_columns[7] = 0.25 * gtsam_cal3_s2.py();

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

            rerun::LatLon lla{};
            double alt{0.0};

            local_converter_->Reverse(det.pose_->x(), det.pose_->y(),
                                      det.pose_->z(), lla.lat_lon.xy[0],
                                      lla.lat_lon.xy[1], alt);

            landmark_pos_map.push_back(lla);
          }
        } else {
          LOG(ERROR) << fmt::format("image {} not found\n", det.timestamp_);
        }
      }

      rec_->log(fmt::format("map/track_{}", track_id.data()),
                rerun::GeoPoints{landmark_pos_map}
                    .with_colors(rerun::Color{0, 0, 255})
                    .with_radii(rerun::Radius::ui_points(5.0f)));
    }
  }

  return *this;
}

BagProcessor &BagProcessor::log_images(int64_t from, int64_t to) {

  if (rec_) {
    std::unordered_map<std::string, cv::Scalar> color_map{};

    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<> distrib{0, 255};

    for (auto &&camera_measurement : camera_) {
      const auto camera_timestamp{camera_measurement.timestamp_};

      if (camera_timestamp >= from and camera_timestamp <= to) {

        cv::Mat_<cv::Vec3b> img = load_image(camera_timestamp);

        if (not img.empty()) {

          const auto dets_it{
              std::find_if(image_detections_.begin(), image_detections_.end(),
                           [camera_timestamp](auto &&val) {
                             return camera_timestamp == val.timestamp_;
                           })};

          if (dets_it != image_detections_.end()) {
            for (auto &&d : dets_it->dets_) {

              if (not color_map.contains(d.code_)) {

                color_map[d.code_] =
                    cv::Scalar{static_cast<double>(distrib(gen)),
                               static_cast<double>(distrib(gen)),
                               static_cast<double>(distrib(gen))};
              }

              // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
              cv::rectangle(img, d.box_, color_map[d.code_], 5);
              cv::putText(
                  img, fmt::format("{} ({:.2f})", d.code_, d.confidence_),
                  {d.box_.x - 25, d.box_.y - 25}, cv::FONT_HERSHEY_COMPLEX, 1.0,
                  color_map[d.code_], 2, cv::LINE_AA);
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

  return *this;
}

Landmark BagProcessor::triangulate(std::string track_id) const {

  if (image_tracks_.contains(track_id)) {
    auto l{triangulate(image_tracks_.at(track_id))};

    if (l.has_value()) {
      return l.value();
    }

    LOG(WARNING) << "unable to triangulate track " << track_id;
    return {};
  }

  LOG(WARNING) << "unable to find track " << track_id;
  return {};
}

std::vector<Landmark>
BagProcessor::triangulate_tracks(double min_track_length) const {

  std::vector<Landmark> res{};
  int num_processed_tracks{0};
  int num_skipped_tracks{0};

  for (auto &&[track_id, track] : image_tracks_) {
    if (track.length_ >= min_track_length) {
      ++num_processed_tracks;

      auto l{triangulate(track)};

      if (l.has_value()) {
        res.emplace_back(l.value());
      } else {
        LOG(WARNING) << "unable to triangulate track " << track_id;
      }
    } else {
      ++num_skipped_tracks;
      LOG(WARNING) << "skipping track: " << track.id_
                   << ", length: " << track.length_;
    }
  }

  LOG(INFO) << "num processed tracks: " << num_processed_tracks;
  LOG(INFO) << "num triangulated tracks: " << res.size();
  LOG(INFO) << "num skipped tracks: " << num_skipped_tracks;

  return res;
}

std::optional<Landmark>
BagProcessor::triangulate(const ImageTrack &track) const {

  const auto measurement_noise{gtsam::noiseModel::Isotropic::Sigma(2, 1.0)};
  gtsam::CameraSet<gtsam::PinholeCamera<gtsam::Cal3_S2>> cameras{};
  gtsam::Point2Vector measurements{};
  std::vector<cv::Point2f> points{};

  Eigen::Vector2d prev_camera_pose{Eigen::Vector2d::Zero()};
  bool first_pose{true};

  for (auto &&d : track.dets_) {

    if (not d.pose_.has_value()) {

      LOG(WARNING) << "detection doesn't have a pose value " << d.timestamp_;
      continue;
    }

    const auto camera_pos{estimate_camera_pos(d)};

    if (camera_pos.has_value()) {

      if (first_pose) {
        first_pose = false;
        prev_camera_pose = camera_pos->translation().head<2>();
      } else {

        const Eigen::Vector2d current_camera_pose{
            camera_pos->translation().head<2>()};

        if ((current_camera_pose - prev_camera_pose).squaredNorm() >=
            dist_threshold_squared_) {
          prev_camera_pose = current_camera_pose;
        } else {
          continue;
        }
      }

      points.push_back(
          cv::Point2f{static_cast<float>(d.box_.x + (d.box_.width >> 1)),
                      static_cast<float>(d.box_.y + (d.box_.height >> 1))});

      const gtsam::Pose3 pose{gtsam::Rot3{camera_pos->linear()},
                              gtsam::Vector3{camera_pos->translation()}};

      cameras.emplace_back(pose, gtsam_cal3_s2);
    } else {
      LOG(WARNING) << "unable to estimate camera transform at " << d.timestamp_;
    }
  }

  if (points.size() < 5) {
    LOG(WARNING) << "too little track points: " << points.size()
                 << ", id: " << track.id_;
    return std::nullopt;
  }

  cv::undistortImagePoints(points, points, camera_matrix_, dist_coeffs_);

  for (auto &&p : points) {
    measurements.emplace_back(p.x, p.y);
  }

  // const Eigen::Vector3d p3d{triangulate_gtsam(cameras, measurements)};

  try {
    Landmark l{};
    l.position_ = gtsam::triangulatePoint3(cameras, measurements, 1.0e-9, true,
                                           measurement_noise, true);

    local_converter_->Reverse(l.position_.x(), l.position_.y(), l.position_.z(),
                              l.lla_.x(), l.lla_.y(), l.lla_.z());

    l.code_ = track.dets_.front().code_;
    l.id_ = track.dets_.front().id_;
    return l;
  } catch (std::exception &ex) {
    LOG(ERROR) << ex.what() << "track id: " << track.id_;
  }

  return std::nullopt;

  // if (rec_) {
  //   rec_->log(
  //       fmt::format("world/{}", track.id_),
  //       rerun::Points3D{{rerun::Position3D{static_cast<float>(p3d.x()),
  //                                          static_cast<float>(p3d.y()),
  //                                          0.0f}}}
  //           .with_labels(rerun::Text{track.dets_.front().code_})
  //           .with_colors(rerun::Color{255, 255, 0})
  //           .with_radii(rerun::Radius::ui_points(10.0f)));
  // }
}

BagProcessor &
BagProcessor::log_track_directions(const std::string_view track_id,
                                   float ray_length) {
  if (rec_) {
    if (image_tracks_.contains(track_id.data())) {

      for (auto &&d : image_tracks_.at(track_id.data()).dets_) {
        log_direction(track_id, d.timestamp_, ray_length);
      }
    }
  }

  return *this;
}

BagProcessor &BagProcessor::log_direction(const std::string_view track_id,
                                          int64_t timestamp, float ray_length) {
  if (rec_) {

    if (image_tracks_.contains(track_id.data())) {
      for (auto &&d : image_tracks_.at(track_id.data()).dets_) {
        if (d.timestamp_ == timestamp) {
          auto camera_pos{estimate_camera_pos(d)};

          if (camera_pos.has_value()) {

            std::vector<cv::Point2f> points{};

            points.push_back(cv::Point2f{
                static_cast<float>(d.box_.x + (d.box_.width >> 1)),
                static_cast<float>(d.box_.y + (d.box_.height >> 1))});

            cv::undistortImagePoints(points, points, camera_matrix_,
                                     dist_coeffs_);

            const float z0{ray_length};
            const float x{static_cast<float>(
                (points[0].x - gtsam_cal3_s2.px()) * z0 / gtsam_cal3_s2.fx())};

            const float y{static_cast<float>(
                (points[0].y - gtsam_cal3_s2.py()) * z0 / gtsam_cal3_s2.fy())};

            Eigen::Vector3f p0{camera_pos->translation().cast<float>()};
            Eigen::Vector3f p1{camera_pos->cast<float>() *
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
  j["name"] = std::filesystem::path{set_.bag_path_}
                  .filename()
                  .replace_extension("")
                  .string();

  for (auto &&landmark : landmarks) {
    nlohmann::json feature{};
    feature["type"] = "Feature";
    feature["geometry"]["type"] = "Point";
    feature["geometry"]["coordinates"] = {landmark.lla_.y(), landmark.lla_.x()};
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
        {gps.lla_.y(), gps.lla_.x()});
  }

  j["features"].push_back(gps_track);

  std::ofstream f{path.data()};
  f << j.dump(4);
}