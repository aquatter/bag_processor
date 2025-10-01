#pragma once
#include <Eigen/Geometry>
#include <GeographicLib/LocalCartesian.hpp>
#include <cstdint>
#include <gtsam/geometry/Cal3_S2.h>
#include <memory>
#include <ng-log/logging.h>
#include <opencv2/core.hpp>
#include <optional>
#include <rerun.hpp>
#include <string>
#include <string_view>
#include <types.hpp>
#include <unordered_map>

struct BagProcessorSettings {
  std::string bag_path_;
  std::string annotations_path_;
  std::string calibration_path_;
  std::string ground_truth_path_;
  bool use_logger_;
};

class BagProcessor {
public:
  BagProcessor(const BagProcessorSettings &set);

  BagProcessor &log_gps_path();
  BagProcessor &log_gps_path_map();
  BagProcessor &log_landmark_map(Landmark l, rerun::Color color);
  BagProcessor &log_landmarks_map(std::span<const Landmark> l);
  BagProcessor &log_ground_truth_landmarks();
  BagProcessor &
  log_ground_truth_landmarks(const std::string_view landmark_code);

  BagProcessor &log_axis();
  BagProcessor &log_camera(int64_t timestamp);
  BagProcessor &log_poly(int64_t timestamp);
  BagProcessor &log_track(const std::string_view track_id);

  BagProcessor &log_direction(const std::string_view track_id,
                              int64_t timestamp, float ray_length);

  BagProcessor &log_track_directions(const std::string_view track_id,
                                     float ray_length);

  BagProcessor &log_images(int64_t from, int64_t to);

  Landmark triangulate(std::string track_id) const;
  std::vector<Landmark> triangulate_tracks(double min_track_length = 5.0) const;

  void save_geojson(std::span<const Landmark> landmarks,
                    const std::string_view path) const;

private:
  void load_calibration(const std::string_view path);
  void create_tracks();
  void load_measurements(const std::string_view path);
  void load_ground_truth_landmarks(const std::string_view path);
  void load_detections(const std::string_view path);

  cv::Mat_<cv::Vec3b> load_image(int64_t timestamp) const;
  std::optional<Eigen::Isometry3d> estimate_camera_pos(int64_t timestamp) const;
  std::optional<Eigen::Isometry3d>
  estimate_camera_pos(const Detection &d) const;
  std::optional<Landmark> triangulate(const ImageTrack &track) const;

  BagProcessorSettings set_;
  std::unique_ptr<rerun::RecordingStream> rec_;
  gtsam::Cal3_S2 gtsam_cal3_s2;
  cv::Mat_<double> camera_matrix_;
  cv::Mat_<double> dist_coeffs_;
  std::unordered_map<std::string, ImageTrack> image_tracks_;
  std::vector<CameraMeasurement> camera_;
  std::vector<GpsMeasurement> gps_;
  std::vector<ImageDetections> image_detections_;
  std::unique_ptr<GeographicLib::LocalCartesian> local_converter_;
  std::vector<Landmark> ground_truth_landmarks_;

  static constexpr int poly_degree_{3};
  static constexpr double search_radius_{20.0};
  static constexpr int64_t camera_gps_delta_{1'000'000'000l};
  // static constexpr int64_t camera_gps_delta_{0};
  static constexpr double correction_angle_{2.0};
  static constexpr double dist_threshold_squared_{0.3 * 0.3};
};