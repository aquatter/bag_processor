#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cstddef>
#include <cstdint>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct Detection {

  size_t det_id_;
  size_t track_id_;
  int64_t timestamp_;
  std::string class_;
  std::string code_;
  float confidence_;
  cv::Rect box_;
  cv::Point center_;
  cv::Point2f center_undistorted_;
  std::optional<Eigen::Vector3d> pose_;
  std::optional<Eigen::Isometry3d> cam_to_world_;
  float angle_;
  ptrdiff_t gps_ind_;
  std::optional<Eigen::Vector2d> direction_;
  std::unordered_set<size_t> linked_detections_;
  double cumulative_length_;

  [[nodiscard]] bool should_be_linked(const Detection &d) const;
  void link(Detection &d);
};

struct ImageDetections {
  using map_type = std::unordered_map<int64_t, ImageDetections>;

  int64_t timestamp_;
  std::vector<Detection *> dets_;
  std::unordered_map<size_t, Detection *> det_id_to_detection_;
  std::unordered_map<size_t, Detection *> track_id_to_detection_;
};

struct ImageTrack {
  using map_type = std::unordered_map<size_t, ImageTrack>;

  size_t id_;
  std::string code_;
  std::vector<Detection> dets_;
  std::unordered_map<int64_t, Detection *> stamp_to_detection_;

  float delta_angle_;
  bool valid_;
  double length_;

  std::unordered_set<size_t> linked_tracks_;

  [[nodiscard]] bool should_be_linked(const ImageTrack &track) const;
  void link(ImageTrack &d);
};

struct GpsMeasurement {
  int64_t timestamp_;
  Eigen::Vector3d position_;
  Eigen::Vector3d lla_;
};

struct CameraMeasurement {
  int64_t timestamp_;
  uint64_t id_;
};

struct Landmark {
  size_t id_;
  std::string code_;
  Eigen::Vector3d position_;
  Eigen::Vector3d lla_;
  double azimuth_;
};