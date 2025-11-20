#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/unordered_set.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>
#include <gtsam/geometry/Cal3_S2.h>
#include <opencv2/core.hpp>
#include <optional>
#include <range/v3/view/iota.hpp>

namespace boost {
namespace serialization {

template <typename Archive, typename Scalar, int Rows, int Cols>
void serialize(Archive &ar, Eigen::Matrix<Scalar, Rows, Cols> &m,
               const unsigned int) {

  for (auto &&i : ranges::views::ints(0, Rows * Cols)) {
    ar & m.data()[i];
  }
}

template <typename Archive, typename T>
void serialize(Archive &ar, std::optional<T> &m, const unsigned int) {

  bool has_value{m.has_value()};
  ar & has_value;

  if (Archive::is_loading::value) {
    if (has_value) {
      T obj{};
      ar & obj;
      m = obj;
    }
  } else {
    if (has_value) {
      ar & m.value();
    }
  }
}

template <typename Archive, typename Scalar, int Dim>
void serialize(Archive &ar, Eigen::Transform<Scalar, Dim, Eigen::Isometry> &m,
               const unsigned int) {
  if (Archive::is_loading::value) {
    Eigen::Matrix<Scalar, Dim + 1, Dim + 1> matrix{};
    ar & matrix;
    m = Eigen::Transform<Scalar, Dim, Eigen::Isometry>{matrix};
  } else {
    Eigen::Matrix<Scalar, Dim + 1, Dim + 1> matrix{m.matrix()};
    ar & matrix;
  }
}

template <typename Archive>
void serialize(Archive &ar, gtsam::Cal3_S2 &gtsam_cal3_s2, const unsigned int) {

  double fx{gtsam_cal3_s2.fx()};
  double fy{gtsam_cal3_s2.fy()};
  double s{gtsam_cal3_s2.skew()};
  double u0{gtsam_cal3_s2.px()};
  double v0{gtsam_cal3_s2.py()};

  ar & fx;
  ar & fy;
  ar & u0;
  ar & v0;
  ar & s;

  if (Archive::is_loading::value) {
    gtsam_cal3_s2 = {fx, fy, s, u0, v0};
  }
}

template <typename Archive, typename Scalar>
void serialize(Archive &ar, cv::Mat_<Scalar> &m, const unsigned int) {
  if (Archive::is_loading::value) {
    int cols{0};
    int rows{0};

    ar & cols;
    ar & rows;
    m.create(rows, cols);
  } else {
    ar & m.cols;
    ar & m.rows;
  }

  // clang-format off
  for (auto &&i : ranges::views::ints(0ul, m.total())) {
    ar & m(i);
  }
  // clang-format on
}

template <typename Archive, typename Scalar>
void serialize(Archive &ar, cv::Point_<Scalar> &p, const unsigned int) {
  ar & p.x;
  ar & p.y;
}

template <typename Archive, typename Scalar>
void serialize(Archive &ar, cv::Rect_<Scalar> &r, const unsigned int) {
  ar & r.x;
  ar & r.y;
  ar & r.width;
  ar & r.height;
}

} // namespace serialization
} // namespace boost