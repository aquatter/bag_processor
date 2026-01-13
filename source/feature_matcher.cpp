#include <cstdint>
#include <feature_matcher.hpp>
#include <fmt/core.h>
#include <fmt/format.h>
#include <ng-log/logging.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/transform.hpp>

using ranges::to;
using ranges::views::transform;

FeatureMatcher::FeatureMatcher() : detector_{cv::AKAZE::create()} {}

bool FeatureMatcher::estimate_homography(cv::Mat_<cv::Vec3b> image1,
                                         const CalibrationDesc &calib1,
                                         const std::string_view tag1,
                                         cv::Mat_<cv::Vec3b> image2,
                                         const CalibrationDesc &calib2,
                                         const std::string_view tag2) {

  const std::string key1{fmt::format("{}_{}", tag1, tag2)};
  const std::string key2{fmt::format("{}_{}", tag2, tag1)};

  if (homographies_.contains(key1)) {
    return not homographies_.at(key1).empty();
  }

  if (homographies_.contains(key2)) {
    return not homographies_.at(key2).empty();
  }

  if (not descriptors_.contains(tag1.data())) {
    descriptors_[tag1.data()] = extract_descriptors(image1);
  }

  if (not descriptors_.contains(tag2.data())) {
    descriptors_[tag2.data()] = extract_descriptors(image2);
  }

  auto res{estimate_homography(descriptors_.at(tag1.data()), tag1, calib1,
                               descriptors_.at(tag2.data()), tag2, calib2)};

  cv::Mat_<cv::Vec3b> img_matches{};
  cv::drawMatches(image1, descriptors_[tag1.data()].keypoints_, image2,
                  descriptors_[tag2.data()].keypoints_, inlier_matches_,
                  img_matches);

  cv::imwrite("matches.png", img_matches);
  return res;
}

bool FeatureMatcher::estimate_homography(std::span<const uint8_t> buf1,
                                         const CalibrationDesc &calib1,
                                         const std::string_view tag1,
                                         std::span<const uint8_t> buf2,
                                         const CalibrationDesc &calib2,
                                         const std::string_view tag2) {

  const std::string key1{fmt::format("{}_{}", tag1, tag2)};
  const std::string key2{fmt::format("{}_{}", tag2, tag1)};

  if (homographies_.contains(key1)) {
    return not homographies_.at(key1).empty();
  }

  if (homographies_.contains(key2)) {
    return not homographies_.at(key2).empty();
  }

  if (not descriptors_.contains(tag1.data())) {
    descriptors_[tag1.data()] = extract_descriptors(buf1);
  }

  if (not descriptors_.contains(tag2.data())) {
    descriptors_[tag2.data()] = extract_descriptors(buf2);
  }

  auto res{estimate_homography(descriptors_.at(tag1.data()), tag1, calib1,
                               descriptors_.at(tag2.data()), tag2, calib2)};

#if 0
  if (res) {
    cv::Mat_<uint8_t> img1 = cv::imdecode(
        {buf1.data(), static_cast<int>(buf1.size())}, cv::IMREAD_GRAYSCALE);

    cv::imwrite("/root/data/comparison/image1.png", img1);

    cv::Mat_<uint8_t> img2 = cv::imdecode(
        {buf2.data(), static_cast<int>(buf2.size())}, cv::IMREAD_GRAYSCALE);

    cv::imwrite("/root/data/comparison/image2.png", img2);

    cv::Mat_<cv::Vec3b> img_matches{};
    cv::drawMatches(img1, descriptors_[tag1.data()].keypoints_, img2,
                    descriptors_[tag2.data()].keypoints_, inlier_matches_,
                    img_matches);

    cv::imwrite("/root/data/comparison/matches.png", img_matches);
  }
#endif

  return res;
}

bool FeatureMatcher::estimate_homography(const Descriptors &desc1,
                                         const std::string_view tag1,
                                         const CalibrationDesc &calib1,
                                         const Descriptors &desc2,
                                         const std::string_view tag2,
                                         const CalibrationDesc &calib2) {

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
    pointsA.push_back(desc1.keypoints_[match.queryIdx].pt);
    pointsB.push_back(desc2.keypoints_[match.trainIdx].pt);
  }

  calib1.undistort_points(pointsA);
  calib2.undistort_points(pointsB);

  std::vector<uint8_t> inlier_mask{};
  const cv::Mat_<double> H =
      cv::findHomography(pointsA, pointsB, cv::RANSAC, 3.0, inlier_mask);

  homographies_[key1] = H;
  inlier_matches_.clear();

  for (size_t i = 0; i < good_matches.size(); ++i) {
    if (inlier_mask[i] == 1) {
      inlier_matches_.push_back(good_matches[i]);
    }
  }

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

Descriptors
FeatureMatcher::extract_descriptors(cv::Mat_<cv::Vec3b> image) const {
  cv::Mat_<uint8_t> imgGray{};
  cv::cvtColor(image, imgGray, cv::COLOR_BGR2GRAY);

  Descriptors d{};
  detector_->detectAndCompute(imgGray, cv::noArray(), d.keypoints_,
                              d.descriptors_);
  return d;
}

Descriptors
FeatureMatcher::extract_descriptors(std::span<const uint8_t> buf) const {
  cv::Mat_<uint8_t> img = cv::imdecode(
      {buf.data(), static_cast<int>(buf.size())}, cv::IMREAD_GRAYSCALE);

  Descriptors d{};
  detector_->detectAndCompute(img, cv::noArray(), d.keypoints_, d.descriptors_);
  return d;
}

void FeatureMatcher::clear() {
  homographies_.clear();
  descriptors_.clear();
}