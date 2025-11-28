#pragma once
#include <cstdint>
#include <memory>
#include <opencv2/core.hpp>
#include <rerun.hpp>
#include <string>
#include <string_view>
#include <types.hpp>

class BagLoader {
public:
  struct Settings {
    std::string compressed_image_topic_;
    std::string path_to_bag_;
    int64_t timestamp_delta_;
    std::shared_ptr<rerun::RecordingStream> rec_;
  };

  BagLoader(const Settings &set);
  cv::Mat_<cv::Vec3b> load_image(int64_t timestamp);

  void dump_tracks(const ImageTrack::map_type &tracks,
                   const ImageDetections::map_type &detections,
                   size_t track_id);

  void dump_detection(const std::string_view path, const Detection &det);

  ~BagLoader();

private:
  struct impl;
  std::unique_ptr<impl> pimpl_;
};