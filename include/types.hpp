#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <opencv2/core.hpp>
#include <optional>
#include <string>
#include <vector>

struct Detection {
  int id_;
  int64_t timestamp_;
  std::string class_;
  std::string code_;
  float confidence_;
  cv::Rect box_;
  Eigen::Vector2i center_;
  std::optional<Eigen::Vector3d> pose_;
  ptrdiff_t gps_ind_;
};

struct ImageDetections {
  int64_t timestamp_;
  uint64_t id_;
  std::vector<Detection> dets_;
};

struct ImageTrack {
  std::string id_;
  std::vector<Detection> dets_;
  double length_;
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
  int id_;
  std::string code_;
  Eigen::Vector3d position_;
  Eigen::Vector3d lla_;
};