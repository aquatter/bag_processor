#pragma once
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <string_view>
#include <types.hpp>
#include <unordered_map>

class FeatureMatcher {
public:
  bool estimate_homography(const Descriptors &desc1,
                           const std::string_view tag1,
                           const Descriptors &desc2,
                           const std::string_view tag2);

  std::vector<cv::Point2f> warp_points(const std::vector<cv::Point2f> &points,
                                       const std::string_view tag1,
                                       const std::string_view tag2);

private:
  bool contains(const std::string_view tag) const;
  std::unordered_map<std::string, cv::Mat_<double>> homographies_;
};