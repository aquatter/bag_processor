#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/Pose3.h>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <optional>
#include <serialization.hpp>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct Plane3d {
  Eigen::Vector3d centroid_;
  Eigen::Vector3d normal_;

  template <typename Archive> void serialize(Archive &ar, const unsigned int) {
    ar & centroid_;
    ar & normal_;
  }
};

struct CalibrationDesc {
  gtsam::Cal3_S2 cal3_s2_;
  cv::Mat_<double> camera_matrix_;
  cv::Mat_<double> dist_coeffs_;
  Eigen::Vector2i camera_resolution_;

  cv::Point2f undistort_point(const cv::Point2f &p) const;

  template <typename Scalar>
  Eigen::Matrix<Scalar, 2, 1>
  undistort_point(const Eigen::Matrix<Scalar, 2, 1> &p) const {

    const auto p_und{undistort_point(
        cv::Point2f{static_cast<float>(p.x()), static_cast<float>(p.y())})};

    return {static_cast<Scalar>(p_und.x), static_cast<Scalar>(p_und.y)};
  }

  void undistort_points(std::vector<cv::Point2f> &points) const;

  cv::Mat_<cv::Vec3b> undistort_image(cv::Mat_<cv::Vec3b> image) const;

  void print() const noexcept;

  template <typename Archive> void serialize(Archive &ar, const unsigned int) {
    ar & cal3_s2_;
    ar & camera_matrix_;
    ar & dist_coeffs_;
    ar & camera_resolution_;
  }
};

struct Detection {

  size_t det_id_;
  size_t track_id_;
  size_t image_id_;
  int64_t timestamp_;
  std::string class_;
  std::string code_;
  float confidence_;
  cv::Rect box_;
  cv::Point center_;
  cv::Point2f center_undistorted_;
  std::optional<Eigen::Vector2d> enu_;
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
    ar & image_id_;
    ar & timestamp_;
    ar & class_;
    ar & code_;
    ar & confidence_;
    ar & box_;
    ar & center_;
    ar & center_undistorted_;
    ar & enu_;
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

struct TrackPoint {
  size_t id_;
  int64_t timestamp_;
  cv::Rect box_;
  cv::Point center_;
  cv::Point2f center_undistorted_;
  Eigen::Isometry3d pose_;
  float angle_;
  Eigen::Vector2d direction_;
  CalibrationDesc calib_;

  operator gtsam::PinholeCamera<gtsam::Cal3_S2>() const {
    return gtsam::PinholeCamera<gtsam::Cal3_S2>{
        gtsam::Pose3{gtsam::Rot3{pose_.linear()},
                     gtsam::Vector3{pose_.translation()}},
        calib_.cal3_s2_};
  }
};

struct Landmark {
  size_t id_;
  std::string code_;
  Eigen::Vector2d enu_;
  Eigen::Vector2d latlon_;
  double azimuth_;
  double dist_variance_;

  template <typename Archive> void serialize(Archive &ar, const unsigned int) {
    ar & id_;
    ar & code_;
    ar & enu_;
    ar & latlon_;
    ar & azimuth_;
    ar & dist_variance_;
  }
};

struct ImageTrack {
  using map_type = std::unordered_map<size_t, ImageTrack>;
  using vec_type = std::vector<ImageTrack>;
  using span_type = std::span<const ImageTrack>;

  std::string name_;
  size_t id_;
  std::string code_;
  std::vector<Detection> dets_;
  std::unordered_map<int64_t, Detection *> stamp_to_detection_;

  float delta_angle_;
  bool valid_;
  double length_;

  std::unordered_set<size_t> linked_tracks_;
  CalibrationDesc calib_;

  Eigen::Vector2d geodetic_origin_;

  std::optional<Landmark> landmark_;
  std::vector<size_t> selected_detections_;
  std::vector<size_t> composed_from_;

  [[nodiscard]] bool should_be_linked(const ImageTrack &track) const;
  void link(ImageTrack &d);

  TrackPoint track_point(size_t i) const;
  std::vector<TrackPoint> selected_track_points() const;

  [[nodiscard]] cv::Rect roi() const noexcept {
    return {roi_border_, roi_border_,
            calib_.camera_resolution_.x() - 2 * roi_border_,
            calib_.camera_resolution_.y() - 2 * roi_border_};
  }

  template <typename Archive> void serialize(Archive &ar, const unsigned int) {
    ar & name_;
    ar & id_;
    ar & code_;
    ar & dets_;
    ar & delta_angle_;
    ar & valid_;
    ar & length_;
    ar & linked_tracks_;
    ar & calib_;
    ar & geodetic_origin_;
    ar & landmark_;
    ar & selected_detections_;
    ar & composed_from_;

    if (Archive::is_loading::value) {
      for (auto &&d : dets_) {
        stamp_to_detection_[d.timestamp_] = &d;
      }
    }
  }

  static constexpr int roi_border_{170};
};

struct GpsMeasurement {
  int64_t timestamp_;
  Eigen::Vector2d enu_;
  Eigen::Vector2d latlon_;

  template <typename Archive> void serialize(Archive &ar, const unsigned int) {
    ar & timestamp_;
    ar & enu_;
    ar & latlon_;
  }
};

struct CameraMeasurement {
  size_t image_id_;
  int64_t timestamp_;
  Eigen::Vector2d enu_;

  template <typename Archive> void serialize(Archive &ar, const unsigned int) {
    ar & image_id_;
    ar & timestamp_;
    ar & enu_;
  }
};

struct Descriptors {
  size_t image_id_;
  cv::Mat_<uint8_t> descriptors_;
  std::vector<cv::KeyPoint> keypoints_;
};

struct LoaderBase {
  virtual cv::Mat_<cv::Vec3b> load_image(size_t image_id) = 0;
  virtual std::vector<std::vector<uint8_t>>
  extract(std::span<const size_t> frame_list) = 0;

  void set_progress(std::function<void()> f) { prog_ = std::move(f); }

  void progress() {
    if (prog_) {
      prog_();
    }
  }

  virtual ~LoaderBase() = default;
  std::function<void()> prog_;
};

inline constexpr double constexpr_cos(double x) {
  double cos{1.0};
  double pow{x};

  for (auto fac{1ull}, n{1ull}; n != 19; fac *= ++n, pow *= x) {
    if ((n & 1) == 0)
      cos += (n & 2 ? -pow : pow) / fac;
  }

  return cos;
}