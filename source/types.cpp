#include <Eigen/Core>
#include <algorithm>
#include <boost/math/constants/constants.hpp>
#include <cmath>
#include <fmt/core.h>
#include <fmt/format.h>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <types.hpp>
#include <unordered_set>

using boost::math::float_constants::degree;
using boost::math::float_constants::pi;
using boost::math::float_constants::radian;
using boost::math::float_constants::two_pi;

template <typename F> class Accumulator {
public:
  void add(float val) {
    if (started_) {
      last_val_ = val;
    } else {
      first_val_ = val;
      last_val_ = val;
      started_ = true;
    }
  }

  void pause() {
    if (not started_) {
      return;
    }

    total_val_ += f(last_val_, first_val_);
    started_ = false;
  }

  float get() {
    pause();
    return total_val_;
  }

private:
  bool started_{false};
  float first_val_{0.0f};
  float last_val_{0.0f};
  float total_val_{0.0f};
  F f{};
};

using length_accumulator =
    Accumulator<decltype([](const float &a, const float &b) {
      return std::abs(a - b);
    })>;

using angle_accumulator =
    Accumulator<decltype([](const float &a, const float &b) {
      const auto delta_angle{std::abs(a - b)};
      return delta_angle > pi ? two_pi - delta_angle : delta_angle;
    })>;

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

  cv::Mat_<cv::Vec3b> dbg_img{};
  bool dbg_case{false};

  if ((id_ == 198 and track.id_ == 199) or (id_ == 199 and track.id_ == 198)) {
    dbg_img = cv::Mat_<cv::Vec3b>::zeros(2160, 3840);
    dbg_case = true;
  }

  const cv::Rect roi{170, 170, 3500, 1820};

  dbg_case = false;

  angle_accumulator angle_accum{};
  length_accumulator length_accum{};

  for (auto &&det : dets_) {
    if (not roi.contains(det.center_)) {
      continue;
    }

    if (track.stamp_to_detection_.contains(det.timestamp_)) {

      if (not det.linked_detections_.contains(
              track.stamp_to_detection_.at(det.timestamp_)->det_id_)) {
        angle_accum.pause();
        length_accum.pause();
        continue;
      }

      if (not roi.contains(
              track.stamp_to_detection_.at(det.timestamp_)->center_)) {
        continue;
      }

      const Eigen::Vector2f dir{
          (to_eigen(det.center_undistorted_) -
           to_eigen(track.stamp_to_detection_.at(det.timestamp_)
                        ->center_undistorted_))};

      if (dbg_case) {
        cv::line(
            dbg_img, det.center_undistorted_,
            track.stamp_to_detection_.at(det.timestamp_)->center_undistorted_,
            {0.0, 0.0, 255.0}, 1, cv::LINE_AA);
      }

      const float angle{[&dir]() {
        const float angle{std::atan2(dir.y(), dir.x())};
        return angle < 0.0f ? two_pi + angle : angle;
      }()};

      angle_accum.add(angle);
      length_accum.add(static_cast<float>(det.cumulative_length_));
    }
  }

  const float delta_angle{angle_accum.get()};
  const float delta_length{length_accum.get()};

  if (dbg_case) {
    cv::imwrite("/root/data/images/dbg_image_directions.png", dbg_img);
    fmt::print("delta angle: {}\n", delta_angle * radian);
  }

  // max angle deviation

  // length of tracks intersection

  const double length_ratio_1{delta_length / length_};
  const double length_ratio_2{delta_length / track.length_};

  const bool length_criteria{length_ratio_1 > 0.7 or length_ratio_2 > 0.7};
  const bool angle_criteria{delta_angle < 15.0f * degree};

  const bool res{length_criteria and angle_criteria};

  if (res) {
    fmt::print(
        "\e[38;2;154;205;50mlinked track\e[0m \e[38;2;255;127;80m{}\e[0m "
        "\e[38;2;154;205;50mwith\e[0m \e[38;2;255;127;80m{}\e[0m\n",
        id_, track.id_);
  }

  return res;
}

void ImageTrack::link(ImageTrack &d) {
  linked_tracks_.insert(d.id_);
  d.linked_tracks_.insert(id_);
}