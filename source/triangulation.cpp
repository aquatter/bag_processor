#include "types.hpp"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <boost/math/constants/constants.hpp>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Quaternion.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/triangulation.h>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/enumerate.hpp>
#include <range/v3/view/linear_distribute.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/view/zip.hpp>
#include <triangulation.hpp>

using ranges::to;
using ranges::views::enumerate;
using ranges::views::linear_distribute;
using ranges::views::transform;
using ranges::views::zip;

std::vector<double> calculate_distances(
    gtsam::Point3 p3d,
    const gtsam::CameraSet<gtsam::PinholeCamera<gtsam::Cal3_S2>> &cameras,
    const gtsam::Point2Vector &measurements) {

  const auto lines{zip(cameras, measurements) | transform([](const auto &val) {
                     const auto &[cam, uv] = val;

                     const auto &calib{cam.calibration()};

                     const gtsam::Point3 p0{cam.pose().translation()};
                     const gtsam::Point3 p1{
                         cam.pose() *
                         gtsam::Point3{(uv.x() - calib.px()) / calib.fx(),
                                       (uv.y() - calib.py()) / calib.fy(),
                                       1.0}};

                     return Eigen::Hyperplane<double, 2>::Through(p0.head<2>(),
                                                                  p1.head<2>());
                   }) |
                   to<std::vector>()};

  const auto n{lines.size()};

  std::vector<double> dists{};
  dists.reserve(n);

  for (auto &&l : lines) {
    dists.push_back(l.absDistance(p3d.head<2>()));
  }

  return dists;
}

Plane3d fit_a_plane(std::span<const Eigen::Vector3d> points) {

  Eigen::Matrix3Xd m{};
  m.resize(3, points.size());

  for (int i{0}; auto &&p : points) {
    m.col(i) = p;
    ++i;
  }

  const Eigen::Vector3d center{m.rowwise().mean()};
  m.colwise() -= center;

  const Eigen::Vector3d normal{
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>{m * m.transpose()}
          .eigenvectors()
          .col(0)};

  return Plane3d{.centroid_ = center, .normal_ = normal};
}

double get_azimmuth(Eigen::Vector3d normal, std::span<const TrackPoint> track) {

  int num_positives{0};
  int num_negatives{0};

  for (auto &&track_point : track) {

    const Eigen::Vector3d n{
        track_point.pose_.inverse(Eigen::Isometry).linear() * normal};

    if (n.z() < 0.0) {
      ++num_negatives;
    } else {
      ++num_positives;
    }
  }

  if (num_positives >= num_negatives) {
    normal *= -1.0;
  }

  auto azimuth{std::atan2(normal.x(), normal.y())};

  if (azimuth < 0.0) {
    azimuth += boost::math::double_constants::two_pi;
  }

  return azimuth;
}

double calculate_variance(std::span<const double> distances) {

  Eigen::ArrayXd d{};
  d.resize(distances.size());

  for (auto &&[i, dist] : enumerate(distances)) {
    d(i) = dist;
  }

  const auto var{(d - d.mean()).square().sum() /
                 static_cast<double>(distances.size() - 1)};
  return var;
}

std::optional<Landmark>
triangulate_on_boxes(std::span<const TrackPoint> track) {

  gtsam::CameraSet<gtsam::PinholeCamera<gtsam::Cal3_S2>> cameras{};
  const auto measurement_noise{gtsam::noiseModel::Isotropic::Sigma(2, 5.0)};

  for (auto &&track_point : track) {
    cameras.emplace_back(track_point);
  }

  std::vector<Eigen::Vector3d> p3d{};
  std::vector<double> total_distances{};

  for (auto &&u : linear_distribute(0.0, 1.0, 5)) {
    for (auto &&v : linear_distribute(0.0, 1.0, 5)) {

      gtsam::Point2Vector measurements{};

      for (auto &&track_point : track) {

        const auto x{
            static_cast<double>(track_point.box_.x) * (1.0 - v) +
            static_cast<double>(track_point.box_.x + track_point.box_.width) *
                v};

        const auto y{
            static_cast<double>(track_point.box_.y) * (1.0 - u) +
            static_cast<double>(track_point.box_.y + track_point.box_.height) *
                u};

        measurements.push_back(
            track_point.calib_.undistort_point(Eigen::Vector2d{x, y}));
      }

      try {
        const auto triangulated_point{gtsam::triangulatePoint3(
            cameras, measurements, 1.0e-9, true, measurement_noise, true)};

        const auto distances{
            calculate_distances(triangulated_point, cameras, measurements)};

        total_distances.insert(total_distances.end(), distances.begin(),
                               distances.end());

        p3d.push_back(triangulated_point);
      } catch (...) {
      }
    }
  }

  if (p3d.size() >= 3) {
    const auto plane{fit_a_plane(p3d)};

    Landmark landmark{};
    landmark.enu_ = plane.centroid_.head<2>();
    landmark.azimuth_ = get_azimmuth(plane.normal_, track);
    landmark.dist_variance_ = calculate_variance(total_distances);

    return landmark;
  }

  return std::nullopt;
}
