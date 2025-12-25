#pragma once
#include <cstddef>
#include <cstdint>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <span>
#include <string>
#include <string_view>
#include <types.hpp>
#include <vector>

class Mp4ImageLoader : public LoaderBase {
public:
  Mp4ImageLoader(const std::string_view path);

  cv::Mat_<cv::Vec3b> load_image(size_t image_id) override;

  std::vector<std::vector<uint8_t>>
  extract(std::span<const size_t> frame_list) override;

private:
  std::vector<std::string> mp4_paths_;
  std::vector<size_t> frame_counts_;
  size_t frame_ind_;
  int64_t file_ind_;
  cv::VideoCapture cap_;
};