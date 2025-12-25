#include <boost/algorithm/string.hpp>
#include <cstddef>
#include <extract_frames.hpp>
#include <filesystem>
#include <fmt/format.h>
#include <mp4_image_loader.hpp>
#include <ng-log/logging.h>
#include <opencv2/videoio.hpp>
#include <range/v3/view/zip.hpp>
#include <vector>

using ranges::views::zip;

Mp4ImageLoader::Mp4ImageLoader(const std::string_view path)
    : frame_ind_{0}, file_ind_{-1} {

  std::vector<std::string> mp4_paths;

  if (std::filesystem::file_type::directory ==
      std::filesystem::status(path).type()) {

    for (auto &&entry : std::filesystem::directory_iterator{path}) {

      if (entry.is_regular_file() and
          boost::algorithm::to_lower_copy(entry.path().extension().string()) ==
              ".mp4") {

        mp4_paths.emplace_back(entry.path().string());
      }
    }
  } else {

    if (std::filesystem::file_type::regular !=
        std::filesystem::status(path).type()) {
      throw std::runtime_error{
          fmt::format("Bag path {} is not a file or directory", path)};
    }

    if (boost::algorithm::to_lower_copy(
            std::filesystem::path(path).extension().string()) != ".mp4") {
      throw std::runtime_error{
          fmt::format("Bag path {} is not a mp4 file", path)};
    }

    mp4_paths.emplace_back(path);
  }

  std::sort(mp4_paths.begin(), mp4_paths.end());

  for (const auto &mp4_path : mp4_paths) {
    cv::VideoCapture cap{mp4_path};

    if (not cap.isOpened()) {
      throw std::runtime_error{
          fmt::format("Cannot open mp4 file: {}", mp4_path)};
    }

    frame_counts_.push_back(
        static_cast<size_t>(cap.get(cv::CAP_PROP_FRAME_COUNT)));

    mp4_paths_.push_back(mp4_path);
  }
}

cv::Mat_<cv::Vec3b> Mp4ImageLoader::load_image(size_t image_id) {
  size_t accumulated_frames{0};
  size_t mp4_index{0};

  for (const auto &frame_count : frame_counts_) {
    if (image_id < accumulated_frames + frame_count) {
      break;
    }
    accumulated_frames += frame_count;
    ++mp4_index;
  }

  if (mp4_index >= mp4_paths_.size()) {
    throw std::runtime_error{
        fmt::format("Image id {} is out of range", image_id)};
  }
  const size_t local_image_id{image_id - accumulated_frames};

  if (mp4_index != file_ind_) {
    file_ind_ = mp4_index;
    cap_ = cv::VideoCapture{mp4_paths_[mp4_index]};

    if (not cap_.isOpened()) {
      throw std::runtime_error{
          fmt::format("Cannot open mp4 file: {}", mp4_paths_[mp4_index])};
    }
  }

  cv::Mat_<cv::Vec3b> img{};
  cap_.set(cv::CAP_PROP_POS_FRAMES, static_cast<double>(local_image_id));
  cap_ >> img;

  if (img.empty()) {
    throw std::runtime_error{
        fmt::format("Cannot load image id {} from mp4 file: {}", local_image_id,
                    mp4_paths_[mp4_index])};
  }

  return img;
}

std::vector<std::vector<uint8_t>>
Mp4ImageLoader::extract(std::span<const size_t> frame_list) {

  std::vector<std::vector<uint8_t>> res{};
  res.reserve(frame_list.size());

  size_t accumulated_frames{0};
  size_t frame_list_ind{0};

  for (auto &&[num_frames, mp4_path] : zip(frame_counts_, mp4_paths_)) {
    std::vector<size_t> frames_to_extract{};

    while (frame_list_ind < frame_list.size() and
           frame_list[frame_list_ind] < accumulated_frames + num_frames) {

      frames_to_extract.push_back(frame_list[frame_list_ind] -
                                  accumulated_frames);

      ++frame_list_ind;
    }

    if (not frames_to_extract.empty()) {

      ExtractFrames extractor{mp4_path};
      extractor.set_progress([this]() { progress(); });

      auto bufs{extractor.extract(frames_to_extract)};

      res.insert(res.end(), bufs.begin(), bufs.end());
    }

    accumulated_frames += num_frames;
  }

  return res;
}
