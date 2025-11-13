#include <Eigen/Core>
#include <algorithm>
#include <boost/math/constants/constants.hpp>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <types.hpp>
#include <unordered_set>

using boost::math::float_constants::degree;
using boost::math::float_constants::pi;
using boost::math::float_constants::two_pi;

Eigen::Vector2f to_eigen(const cv::Point2f &p) {
  return {static_cast<float>(p.x), static_cast<float>(p.y)};
}

bool Detection::should_be_linked(const Detection &d) const {
  if (det_id_ == d.det_id_) {
    return false;
  }

  return cv::Rect{box_.x - box_.width, box_.y, 3 * box_.width, box_.height}
             .contains(d.center_) or
         (d.center_.x >= box_.x and d.center_.x <= (box_.x + box_.width));
}

void Detection::link(Detection &d) {
  linked_detections_.insert(d.det_id_);
  d.linked_detections_.insert(det_id_);
}

bool ImageTrack::should_be_linked(const ImageTrack &track) const {

  std::pair<float, float> min_max_angle{std::numeric_limits<float>::max(),
                                        std::numeric_limits<float>::min()};

  std::pair<float, float> min_max_length{std::numeric_limits<float>::max(),
                                         std::numeric_limits<float>::min()};

  for (auto &&det : dets_) {
    if (track.stamp_to_detection_.contains(det.timestamp_)) {

      const Eigen::Vector2f dir{
          (to_eigen(det.center_undistorted_) -
           to_eigen(track.stamp_to_detection_.at(det.timestamp_)
                        ->center_undistorted_))};

      const float angle{[&dir]() {
        const float angle{std::atan2(dir.y(), dir.x())};
        return angle < 0.0f ? two_pi - angle : angle;
      }()};

      min_max_angle = {std::min(min_max_angle.first, angle),
                       std::max(min_max_angle.second, angle)};

      min_max_length = {std::min(min_max_length.first,
                                 static_cast<float>(det.cumulative_length_)),
                        std::max(min_max_length.second,
                                 static_cast<float>(det.cumulative_length_))};
    }
  }

  // max angle deviation
  const float delta_angle{[&min_max_angle]() {
    const float delta_angle{
        std::abs(min_max_angle.first - min_max_angle.second)};
    return delta_angle > pi ? two_pi - delta_angle : delta_angle;
  }()};

  // length of tracks intersection
  const float delta_length{
      std::abs(min_max_length.first - min_max_length.second)};

  if ((length_ / delta_length > 0.7 or track.length_ / delta_angle > 0.7) and
      (delta_angle < 5.0f * degree)) {
    return true;
  }

  return false;
}

void ImageTrack::link(ImageTrack &d) {
  linked_tracks_.insert(d.id_);
  d.linked_tracks_.insert(id_);
}