#pragma once

#include <Eigen/Geometry>
#include <boost/math/constants/constants.hpp>
#include <cartesian_converter.hpp>
#include <cstddef>
#include <cstdint>
#include <feature_tracker.hpp>
#include <filesystem>
#include <gtsam/geometry/Cal3_S2.h>
#include <memory>
#include <ng-log/logging.h>
#include <opencv2/core.hpp>
#include <optional>
#include <rerun.hpp>
#include <serialization.hpp>
#include <span>
#include <string>
#include <string_view>
#include <types.hpp>
#include <unordered_map>
#include <vector>

struct BagProcessorSettings {
  std::string bag_path_;
  std::string annotations_path_;
  std::string calibration_path_;
  std::string ground_truth_path_;
  double correction_angle_;
  int64_t camera_gps_delta_;
  bool use_klt_;
  bool use_logger_;
  std::string session_name_;
  std::string compressed_image_topic_;
  std::string gps_topic_;

  template <typename Archive> void serialize(Archive &ar, const uint32_t) {
    ar & bag_path_;
    ar & annotations_path_;
    ar & calibration_path_;
    ar & ground_truth_path_;
    ar & correction_angle_;
    ar & camera_gps_delta_;
    ar & use_klt_;
    ar & use_logger_;
    ar & session_name_;
    ar & compressed_image_topic_;
    ar & gps_topic_;
  }
};

class TrackIndexer;
class TracksCollection;

class BagProcessor {
public:
  using ptr = std::shared_ptr<BagProcessor>;

  BagProcessor();
  BagProcessor(const BagProcessorSettings &set);

  void calculate();
  void
  calculate_metrics(std::optional<std::filesystem::path> path = std::nullopt);

  void optimize_angle(double from, double to, ptrdiff_t num);

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
  BagProcessor &log_track(size_t track_id);

  BagProcessor &log_direction(size_t track_id, int64_t timestamp,
                              float ray_length);

  BagProcessor &log_track_directions(size_t track_id, float ray_length);

  BagProcessor &log_images(int64_t from, int64_t to);

  size_t triangulate_tracks();

  void save_geojson(std::span<const Landmark> landmarks,
                    const std::string_view path) const;

  std::string most_frequent_landmark() const;

  void save(std::filesystem::path path) const;

  [[nodiscard]] static BagProcessor::ptr load(std::filesystem::path path);

  void set_rerun(std::shared_ptr<rerun::RecordingStream> rec) { rec_ = rec; }

  ImageTrack::map_type get_tracks() const;

  const BagProcessorSettings &settings() const { return set_; }

private:
  void load_calibration(const std::string_view path);
  void load_tracks();
  void load_measurements(const std::string_view path);
  void load_ground_truth_landmarks(const std::string_view path);
  void load_detections(const std::string_view path);
  void calculate_most_frequent_landmark();
  float estimate_azimuth(const Eigen::Isometry3d pose,
                         const Eigen::Vector2f p2d) const;

  cv::Mat_<cv::Vec3b> load_image(int64_t timestamp) const;
  std::optional<Eigen::Isometry3d> estimate_camera_pos(int64_t timestamp) const;
  std::optional<Eigen::Isometry3d> estimate_camera_pos(Detection &d);
  void triangulate(ImageTrack &track);
  void collect_detections();
  void track_features();
  void change_angle(double angle_deg);

  std::vector<size_t> get_valid_tracks();

  BagProcessorSettings set_;
  std::shared_ptr<rerun::RecordingStream> rec_;
  std::unordered_map<size_t, ImageTrack> image_tracks_;
  std::vector<CameraMeasurement> camera_;
  std::vector<GpsMeasurement> gps_;
  std::unordered_map<int64_t, ImageDetections> image_detections_;
  CartesianConverter local_converter_;
  std::vector<Landmark> ground_truth_landmarks_;
  std::string most_frequent_landmark_;
  std::unordered_map<std::string, cv::Scalar> color_map_;

  std::vector<size_t> valid_tracks_;
  size_t track_num_;
  CalibrationDesc calib_;

  friend class TrackIndexer;
  friend class TracksCollection;
  friend class boost::serialization::access;

  template <typename Archive> void serialize(Archive &ar, const unsigned int) {
    ar & set_;
    ar & image_tracks_;
    ar & camera_;
    ar & gps_;
    ar & ground_truth_landmarks_;
    ar & most_frequent_landmark_;
    ar & valid_tracks_;
    ar & track_num_;
    ar & calib_;
    ar & local_converter_;

    if (Archive::is_loading::value) {
      collect_detections();
    }
  }

public:
  static constexpr int poly_degree_{3};
  static constexpr double search_radius_{20.0};
  static constexpr double dist_threshold_squared_{0.3 * 0.3};
  static constexpr float angle_threshold_deg_{20.0f};
  static constexpr double max_dist_to_track_sqr_{15.0 * 15.0};
  static constexpr double azimuth_correction_threshold_{
      45.0 * boost::math::double_constants::degree};
};