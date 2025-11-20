#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cstddef>
#include <cstdint>
#include <opencv2/core.hpp>
#include <optional>
#include <serialization.hpp>
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

  template <typename Archive> void serialize(Archive &ar, const unsigned int) {
    ar & det_id_;
    ar & track_id_;
    ar & timestamp_;
    ar & class_;
    ar & code_;
    ar & confidence_;
    ar & box_;
    ar & center_;
    ar & center_undistorted_;
    ar & pose_;
    ar & cam_to_world_;
    ar & angle_;
    ar & gps_ind_;
    ar & direction_;
    ar & linked_detections_;
    ar & cumulative_length_;
  }
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

  template <typename Archive> void serialize(Archive &ar, const unsigned int) {
    ar & id_;
    ar & code_;
    ar & dets_;
    ar & delta_angle_;
    ar & valid_;
    ar & length_;
    ar & linked_tracks_;

    if (Archive::is_loading::value) {
      for (auto &&d : dets_) {
        stamp_to_detection_[d.timestamp_] = &d;
      }
    }
  }
};

struct GpsMeasurement {
  int64_t timestamp_;
  Eigen::Vector3d position_;
  Eigen::Vector3d lla_;

  template <typename Archive> void serialize(Archive &ar, const unsigned int) {
    ar & timestamp_;
    ar & position_;
    ar & lla_;
  }
};

struct CameraMeasurement {
  int64_t timestamp_;
  uint64_t id_;

  template <typename Archive> void serialize(Archive &ar, const unsigned int) {
    ar & timestamp_;
    ar & id_;
  }
};

struct Landmark {
  size_t id_;
  std::string code_;
  Eigen::Vector3d position_;
  Eigen::Vector3d lla_;
  double azimuth_;
  double dist_std_dev_;

  template <typename Archive> void serialize(Archive &ar, const unsigned int) {
    ar & id_;
    ar & code_;
    ar & position_;
    ar & lla_;
    ar & azimuth_;
    ar & dist_std_dev_;
  }
};