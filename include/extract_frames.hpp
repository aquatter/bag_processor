#pragma once
#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <opencv2/core.hpp>
#include <span>
#include <vector>

class ExtractFrames {
public:
  ExtractFrames(std::filesystem::path video_path);
  ~ExtractFrames();

  std::vector<std::vector<uint8_t>> extract(std::span<const size_t> frame_list);
  void set_progress(std::function<void()> f);

private:
  struct impl;
  std::unique_ptr<impl> pimpl_;
};