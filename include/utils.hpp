#pragma once

#include <cartesian_converter.hpp>
#include <cstddef>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/CameraSet.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <memory>
#include <optional>
#include <range/v3/view/enumerate.hpp>
#include <range/v3/view/iota.hpp>
#include <rerun.hpp>
#include <span>
#include <string_view>
#include <types.hpp>
#include <vector>

Eigen::Vector3d triangulate_gtsam(
    const gtsam::CameraSet<gtsam::PinholeCamera<gtsam::Cal3_S2>> &cameras,
    const gtsam::Point2Vector &measurements);

std::tuple<std::vector<Eigen::Vector2d>, Eigen::Vector2d>
get_points_in_the_radius(std::span<const CameraMeasurement> points, double rad,
                         Eigen::Vector2d query_point, ptrdiff_t ind);

std::vector<rerun::Vec3D>
interpolate_spline(std::span<const Eigen::Vector2d> points,
                   Eigen::Vector2d query_point);

std::optional<Eigen::Vector2d>
estimate_direction_spline(std::span<const Eigen::Vector2d> points,
                          Eigen::Vector2d query_point);

void correct_orientation(Landmark &landmark,
                         std::span<const CameraMeasurement> gps_track);

void link_detections(ImageDetections::map_type &detections);

void link_tracks(ImageTrack::map_type &tracks,
                 ImageDetections::map_type &detections);

size_t combine_landmarks(ImageTrack::map_type &tracks,
                         const CartesianConverter &converter);

template <int degree> struct PolyResult {
  Eigen::Vector2d point_;
  Eigen::Vector2d direction_;
  std::array<double, degree + 1> poly_;
  bool horizontal_;
};

template <int degree>
PolyResult<degree> estimate_direction(std::span<const Eigen::Vector2d> points,
                                      Eigen::Vector2d query_point) {

  using ranges::views::enumerate;
  using ranges::views::ints;

  bool horizontal_dir{true};

  {
    auto [min_x, max_x] = std::minmax_element(
        points.begin(), points.end(),
        [](const auto &a, const auto &b) { return a.x() < b.x(); });

    auto [min_y, max_y] = std::minmax_element(
        points.begin(), points.end(),
        [](const auto &a, const auto &b) { return a.y() < b.y(); });

    horizontal_dir =
        (max_x->x() - min_x->x()) > (max_y->y() - min_y->y()) ? true : false;
  }

  const int n{static_cast<int>(points.size())};

  Eigen::MatrixXd A;
  Eigen::MatrixXd b;
  A.resize(n, degree + 1);
  b.resize(n, 1);

  if (horizontal_dir) {
    for (auto &&[i, p] : enumerate(points)) {
      double x_val{1.0};

      for (int &&j : ints(0, degree + 1)) {
        A(i, j) = x_val;
        x_val *= static_cast<double>(p.x());
      }

      b(i) = p.y();
    }
  } else {
    for (auto &&[i, p] : enumerate(points)) {
      double x_val{1.0};

      for (int &&j : ints(0, degree + 1)) {
        A(i, j) = x_val;
        x_val *= static_cast<double>(p.y());
      }

      b(i) = p.x();
    }
  }

  const Eigen::MatrixXd p{(A.transpose() * A).ldlt().solve(A.transpose() * b)};

  std::array<double, degree + 1> poly{};
  for (auto &&i : ints(0, degree + 1)) {
    poly[i] = p(i);
  }

  if (horizontal_dir) {
    double y_prime{0.0};
    double x_val{1.0};

    for (auto &&i : ints(1, degree + 1)) {
      y_prime += p(i) * x_val * static_cast<double>(i);
      x_val *= query_point.x();
    }

    double y{0.0};
    x_val = 1.0;

    for (auto &&i : ints(0, degree + 1)) {
      y += p(i) * x_val;
      x_val *= query_point.x();
    }

    return PolyResult<degree>{.point_ = Eigen::Vector2d{query_point.x(), y},
                              .direction_ =
                                  Eigen::Vector2d{1.0, y_prime}.normalized(),
                              .poly_ = poly,
                              .horizontal_ = horizontal_dir};
  }

  double x_prime{0.0};
  double y_val{1.0};

  for (auto &&i : ints(1, degree + 1)) {
    x_prime += p(i) * y_val * static_cast<double>(i);
    y_val *= query_point.y();
  }

  double x{0.0};
  y_val = 1.0;

  for (auto &&i : ints(0, degree + 1)) {
    x += p(i) * y_val;
    y_val *= query_point.y();
  }

  return PolyResult<degree>{.point_ = Eigen::Vector2d{x, query_point.y()},
                            .direction_ =
                                Eigen::Vector2d{x_prime, 1.0}.normalized(),
                            .poly_ = poly,
                            .horizontal_ = horizontal_dir};
}

class Metrics {
public:
  struct Settings {
    float max_distance_from_path_{15.0f};
    float distance_min_{1.0f};
    float distance_max_{30.0f};
    ptrdiff_t num_steps_{30};
  };

  struct Result {
    std::vector<float> precision_;
    std::vector<float> recall_;
    std::vector<float> distances_;
    float precision_auc_;
    float recall_auc_;

    bool save(const std::string_view path,
              const std::string_view title = {}) const;
  };

  Metrics(const Settings &set, std::span<const GpsMeasurement> gps_measurements,
          std::span<const Landmark> gt_landmarks);

  [[nodiscard]] Result eval(const ImageTrack::map_type &tracks) const;

  ~Metrics();

private:
  struct impl;
  std::unique_ptr<impl> pimpl_;
};