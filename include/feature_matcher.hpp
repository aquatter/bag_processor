#pragma once
#include <cstdint>
#include <opencv2/core.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/features2d.hpp>
#include <string_view>
#include <types.hpp>
#include <unordered_map>

class FeatureMatcher {
public:
  FeatureMatcher();

  bool estimate_homography(cv::Mat_<cv::Vec3b> image1,
                           const CalibrationDesc &calib1,
                           const std::string_view tag1,
                           cv::Mat_<cv::Vec3b> image2,
                           const CalibrationDesc &calib2,
                           const std::string_view tag2);

  bool estimate_homography(std::span<const uint8_t> buf1,
                           const CalibrationDesc &calib1,
                           const std::string_view tag1,
                           std::span<const uint8_t> buf2,
                           const CalibrationDesc &calib2,
                           const std::string_view tag2);

  bool estimate_homography(const Descriptors &desc1,
                           const std::string_view tag1,
                           const CalibrationDesc &calib1,
                           const Descriptors &desc2,
                           const std::string_view tag2,
                           const CalibrationDesc &calib2);

  std::vector<cv::Point2f> warp_points(const std::vector<cv::Point2f> &points,
                                       const std::string_view tag1,
                                       const std::string_view tag2);

  void clear();

private:
  bool contains(const std::string_view tag) const;
  Descriptors extract_descriptors(std::span<const uint8_t> buf) const;
  Descriptors extract_descriptors(cv::Mat_<cv::Vec3b> image) const;

  std::unordered_map<std::string, cv::Mat_<double>> homographies_;
  std::unordered_map<std::string, Descriptors> descriptors_;

  cv::Ptr<cv::AKAZE> detector_;
  std::vector<cv::DMatch> inlier_matches_;
};