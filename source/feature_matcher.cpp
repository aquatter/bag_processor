#include <feature_matcher.hpp>
#include <fmt/core.h>
#include <fmt/format.h>
#include <ng-log/logging.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>

bool FeatureMatcher::estimate_homography(const Descriptors &desc1,
                                         const std::string_view tag1,
                                         const Descriptors &desc2,
                                         const std::string_view tag2) {

  const std::string key1{fmt::format("{}_{}", tag1, tag2)};
  const std::string key2{fmt::format("{}_{}", tag2, tag1)};

  if (homographies_.contains(key1)) {
    return not homographies_.at(key1).empty();
  }

  if (homographies_.contains(key2)) {
    return not homographies_.at(key2).empty();
  }

  cv::BFMatcher matcher{cv::NORM_HAMMING};
  std::vector<std::vector<cv::DMatch>> matches;

  matcher.knnMatch(desc1.descriptors_, desc2.descriptors_, matches, 2);

  std::vector<cv::DMatch> good_matches;

  for (const auto &match : matches) {

    if (match.size() < 2) {
      continue;
    }

    if (match[0].distance < 0.75f * match[1].distance) {
      good_matches.push_back(match[0]);
    }
  }

  if (good_matches.size() < 20) {
    LOG(WARNING) << "Not enough good matches to estimate homography between "
                 << tag1 << " and " << tag2 << ": " << good_matches.size();

    return false;
  }

  std::vector<cv::Point2f> pointsA;
  std::vector<cv::Point2f> pointsB;

  pointsA.reserve(good_matches.size());
  pointsB.reserve(good_matches.size());

  for (const auto &match : good_matches) {
    pointsA.push_back(desc1.keypoints_[match.queryIdx]);
    pointsB.push_back(desc2.keypoints_[match.trainIdx]);
  }

  const cv::Mat_<double> H =
      cv::findHomography(pointsA, pointsB, cv::RANSAC, 3.0);

  homographies_[key1] = H;
  return H.empty() == false;
}

std::vector<cv::Point2f>
FeatureMatcher::warp_points(const std::vector<cv::Point2f> &points,
                            const std::string_view tag1,
                            const std::string_view tag2) {
  const std::string key1{fmt::format("{}_{}", tag1, tag2)};
  const std::string key2{fmt::format("{}_{}", tag2, tag1)};

  std::vector<cv::Point2f> points_projected{};

  if (contains(key1)) {
    cv::perspectiveTransform(points, points_projected, homographies_.at(key1));
  } else if (contains(key2)) {
    cv::perspectiveTransform(points, points_projected,
                             homographies_.at(key2).inv());
  }

  return points_projected;
}

bool FeatureMatcher::contains(const std::string_view tag) const {
  return homographies_.contains(tag.data()) and
         not homographies_.at(tag.data()).empty();
}
