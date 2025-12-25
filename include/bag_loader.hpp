#pragma once
#include <cstdint>
#include <memory>
#include <opencv2/core.hpp>
#include <rerun.hpp>
#include <string>
#include <string_view>
#include <types.hpp>

class BagLoader : public LoaderBase {
public:
  struct Settings {
    std::string compressed_image_topic_;
    std::string path_to_bag_;
    int64_t timestamp_delta_;
    std::shared_ptr<rerun::RecordingStream> rec_;
  };

  BagLoader(const Settings &set);

  void dump_tracks(const ImageTrack::map_type &tracks,
                   const ImageDetections::map_type &detections,
                   size_t track_id);

  void dump_detection(const std::string_view path, const Detection &det);

  cv::Mat_<cv::Vec3b> load_image(int64_t timestamp);
  cv::Mat_<cv::Vec3b> load_image(size_t image_id) override;

  std::vector<std::vector<uint8_t>>
  extract(std::span<const size_t> frame_list) override;

  ~BagLoader();

private:
  struct impl;
  std::unique_ptr<impl> pimpl_;
};