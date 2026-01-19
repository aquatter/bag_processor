#include <cstdint>
#include <feature_matcher.hpp>
#include <fmt/core.h>
#include <fmt/format.h>
#include <mutex>
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

  {
    std::lock_guard<std::mutex> lock{homoghaphy_protector_};
    if (homographies_.contains(key1)) {
      return not homographies_.at(key1).empty();
    }

    if (homographies_.contains(key2)) {
      return not homographies_.at(key2).empty();
    }
  }

  Descriptors desc1;
  Descriptors desc2;

  {
    std::lock_guard<std::mutex> lock{descriptor_protector_};

    if (not descriptors_.contains(tag1.data())) {
      descriptors_[tag1.data()] = extract_descriptors_impl(image1);
    }

    if (not descriptors_.contains(tag2.data())) {
      descriptors_[tag2.data()] = extract_descriptors_impl(image2);
    }

    desc1 = descriptors_[tag1.data()];
    desc2 = descriptors_[tag2.data()];
  }

  auto H{estimate_homography_impl(desc1, tag1, calib1, desc2, tag2, calib2)};

  {
    std::lock_guard<std::mutex> lock{homoghaphy_protector_};
    homographies_[key1] = H;
  }

#if 0
  cv::Mat_<cv::Vec3b> img_matches{};
  cv::drawMatches(image1, descriptors_[tag1.data()].keypoints_, image2,
                  descriptors_[tag2.data()].keypoints_, inlier_matches_,
                  img_matches);

  cv::imwrite("matches.png", img_matches);
#endif

  return not H.empty();
}

bool FeatureMatcher::estimate_homography(std::span<const uint8_t> buf1,
                                         const CalibrationDesc &calib1,
                                         const std::string_view tag1,
                                         std::span<const uint8_t> buf2,
                                         const CalibrationDesc &calib2,
                                         const std::string_view tag2) {

  const std::string key1{fmt::format("{}_{}", tag1, tag2)};
  const std::string key2{fmt::format("{}_{}", tag2, tag1)};

  {
    std::lock_guard<std::mutex> lock{homoghaphy_protector_};
    if (homographies_.contains(key1)) {
      return not homographies_.at(key1).empty();
    }

    if (homographies_.contains(key2)) {
      return not homographies_.at(key2).empty();
    }
  }

  Descriptors desc1;
  Descriptors desc2;

  {
    std::lock_guard<std::mutex> lock{descriptor_protector_};

    if (not descriptors_.contains(tag1.data())) {
      LOG(INFO) << fmt::format("desciptors not found for {}", tag1.data());
      return false;
      // descriptors_[tag1.data()] = extract_descriptors_impl(buf1);
    }

    if (not descriptors_.contains(tag2.data())) {
      LOG(INFO) << fmt::format("desciptors not found for {}", tag2.data());
      return false;
      // descriptors_[tag2.data()] = extract_descriptors_impl(buf2);
    }

    desc1 = descriptors_.at(tag1.data());
    desc2 = descriptors_.at(tag2.data());
  }

  const auto H{
      estimate_homography_impl(desc1, tag1, calib1, desc2, tag2, calib2)};

  {
    std::lock_guard<std::mutex> lock{homoghaphy_protector_};
    homographies_[key1] = H;
  }

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

  return not H.empty();
}

cv::Mat_<double> FeatureMatcher::estimate_homography_impl(
    const Descriptors &desc1, const std::string_view tag1,
    const CalibrationDesc &calib1, const Descriptors &desc2,
    const std::string_view tag2, const CalibrationDesc &calib2) const {

  const std::string key1{fmt::format("{}_{}", tag1, tag2)};
  const std::string key2{fmt::format("{}_{}", tag2, tag1)};

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

    return {};
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

  return H;

#if 0
  homographies_[key1] = H;
  inlier_matches_.clear();

  for (size_t i = 0; i < good_matches.size(); ++i) {
    if (inlier_mask[i] == 1) {
      inlier_matches_.push_back(good_matches[i]);
    }
  }

  return H.empty() == false;
#endif
}

std::vector<cv::Point2f>
FeatureMatcher::warp_points(const std::vector<cv::Point2f> &points,
                            const std::string_view tag1,
                            const std::string_view tag2) {
  const std::string key1{fmt::format("{}_{}", tag1, tag2)};
  const std::string key2{fmt::format("{}_{}", tag2, tag1)};

  std::vector<cv::Point2f> points_projected{};
  cv::Mat_<double> H{};

  {
    std::lock_guard<std::mutex> lock{homoghaphy_protector_};
    if (contains(key1)) {
      H = homographies_.at(key1);
    } else if (contains(key2)) {
      H = homographies_.at(key2).inv();
    }
  }

  cv::perspectiveTransform(points, points_projected, H);

  return points_projected;
}

bool FeatureMatcher::contains(const std::string_view tag) const {
  return homographies_.contains(tag.data()) and
         not homographies_.at(tag.data()).empty();
}

Descriptors
FeatureMatcher::extract_descriptors_impl(cv::Mat_<cv::Vec3b> image) const {
  cv::Mat_<uint8_t> imgGray{};
  cv::cvtColor(image, imgGray, cv::COLOR_BGR2GRAY);

  Descriptors d{};
  detector_->detectAndCompute(imgGray, cv::noArray(), d.keypoints_,
                              d.descriptors_);
  return d;
}

Descriptors
FeatureMatcher::extract_descriptors_impl(std::span<const uint8_t> buf) const {
  cv::Mat_<uint8_t> img = cv::imdecode(
      {buf.data(), static_cast<int>(buf.size())}, cv::IMREAD_GRAYSCALE);

  Descriptors d{};
  detector_->detectAndCompute(img, cv::noArray(), d.keypoints_, d.descriptors_);
  return d;
}

void FeatureMatcher::extract_descriptors(std::span<const uint8_t> buf,
                                         const std::string_view tag) {
  {
    std::lock_guard<std::mutex> lock{descriptor_protector_};

    if (descriptors_.contains(tag.data())) {
      return;
    }
  }

  const auto desc{extract_descriptors_impl(buf)};

  std::lock_guard<std::mutex> lock{descriptor_protector_};
  descriptors_[tag.data()] = desc;
}

void FeatureMatcher::clear() {
  homographies_.clear();
  descriptors_.clear();
}