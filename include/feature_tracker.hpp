#pragma once
// clang-format off
#include <Eigen/Geometry>
#include <GeographicLib/LocalCartesian.hpp>
#include <cstdint>
#include <memory>
#include <opencv2/core.hpp>
#include <optional>
#include <types.hpp>
#if USE_RERUN
#include <rerun.hpp>
#endif
#include <span>
// clang-format on
class FeatureTracker {
public:
  struct Settings {
    int num_feats_;
    int fast_threshold_;
    int gridx_;
    int gridy_;
    int minpxdist_;
    float angle_threshold_deg_;
    bool use_klt_;
    bool save_debug_images_;
    CalibrationDesc calib_;
#if USE_RERUN
    rerun::RecordingStream *rec_{nullptr};
#endif
    GeographicLib::LocalCartesian *local_converter_{nullptr};
  };

  FeatureTracker(const Settings &set);

  void add(cv::Mat_<uint8_t> image, const ImageDetections &dets);
  void finalize();

  [[nodiscard]] std::optional<Landmark>
  triangulate(std::span<const TrackPoint> track);

  ~FeatureTracker();

private:
  struct impl;
  std::unique_ptr<impl> pimpl_;
};