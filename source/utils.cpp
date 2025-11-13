#include <bag_processor.hpp>
#include <boost/container_hash/hash.hpp>
#include <boost/math/constants/constants.hpp>
#include <cmath>
#include <cstddef>
#include <flann/algorithms/dist.h>
#include <flann/flann.hpp>
#include <flann/util/matrix.h>
#include <functional>
#include <gtsam/geometry/triangulation.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <interpolation.h>
#include <iterator>
#include <memory>
#include <ng-log/logging.h>
#include <nlohmann/json.hpp>
#include <queue>
#include <range/v3/algorithm/copy.hpp>
#include <range/v3/algorithm/find_if.hpp>
#include <range/v3/algorithm/max.hpp>
#include <range/v3/algorithm/min_element.hpp>
#include <range/v3/algorithm/sort.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/enumerate.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/linear_distribute.hpp>
#include <range/v3/view/take.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/view/zip.hpp>
#include <rerun.hpp>
#include <span>
#include <types.hpp>
#include <unordered_set>
#include <utility>
#include <utils.hpp>
#include <vector>

using ranges::to;
using ranges::views::enumerate;
using ranges::views::ints;
using ranges::views::linear_distribute;
using ranges::views::transform;
using ranges::views::zip;

Eigen::Vector3d triangulate_gtsam(
    const gtsam::CameraSet<gtsam::PinholeCamera<gtsam::Cal3_S2>> &cameras,
    const gtsam::Point2Vector &measurements) {

  auto p3d{gtsam::triangulateDLT(gtsam::projectionMatricesFromCameras(cameras),
                                 measurements)};

  const auto measurement_noise{gtsam::noiseModel::Isotropic::Sigma(2, 1.0)};

  const auto [graph, values] = gtsam::triangulationGraph(
      cameras, measurements, gtsam::Symbol{'p', 0}, p3d, measurement_noise);

  gtsam::LevenbergMarquardtParams params;
  params.verbosityLM = gtsam::LevenbergMarquardtParams::TRYLAMBDA;
  params.verbosity = gtsam::NonlinearOptimizerParams::ERROR;
  params.lambdaInitial = 1;
  params.lambdaFactor = 10;
  params.maxIterations = 100;
  params.absoluteErrorTol = 1.0;
  params.verbosityLM = gtsam::LevenbergMarquardtParams::SILENT;
  params.verbosity = gtsam::NonlinearOptimizerParams::SILENT;
  params.linearSolverType =
      gtsam::NonlinearOptimizerParams::MULTIFRONTAL_CHOLESKY;

  gtsam::LevenbergMarquardtOptimizer optimizer{graph, values, params};
  gtsam::Values result{optimizer.optimize()};

  gtsam::Marginals marginals{graph, result};
  auto cov{marginals.marginalCovariance(gtsam::Symbol{'p', 0})};

  std::cout << cov << std::endl;
  std::cout << "determinant: " << cov.determinant() << std::endl;

  return result.at<gtsam::Point3>(gtsam::Symbol{'p', 0});
}

std::tuple<std::vector<Eigen::Vector2d>, Eigen::Vector2d>
get_points_in_the_radius(std::span<const GpsMeasurement> points, double rad,
                         Eigen::Vector3d query_point, ptrdiff_t ind) {

  const double rad_squared{rad * rad};

  Eigen::Vector2d first_point{};
  Eigen::Vector2d last_point{};
  std::deque<Eigen::Vector2d> points_queue;

  int num_added{0};

  for (ptrdiff_t i{ind - 1}; i >= 0; --i) {
    if ((points[i].position_ - query_point).squaredNorm() < rad_squared or
        num_added < 5) {
      points_queue.emplace_front(points[i].position_.x(),
                                 points[i].position_.y());
      ++num_added;
      first_point = points[i].position_.head(2);
    } else {
      break;
    }
  }

  num_added = 0;
  for (ptrdiff_t i{ind}; i < points.size(); ++i) {
    if ((points[i].position_ - query_point).squaredNorm() < rad_squared or
        num_added < 5) {
      points_queue.emplace_back(points[i].position_.x(),
                                points[i].position_.y());

      ++num_added;
      last_point = points[i].position_.head(2);
    } else {
      break;
    }
  }

  last_point = (last_point - first_point).normalized();

  return {
      std::vector<Eigen::Vector2d>{points_queue.begin(), points_queue.end()},
      last_point};
}

std::vector<rerun::Vec3D>
interpolate_spline(std::span<const Eigen::Vector2d> points,
                   Eigen::Vector2d query_point) {

  alglib::spline1dinterpolant c;
  alglib::real_1d_array x;
  alglib::real_1d_array y;

  std::vector<Eigen::Vector2d> sorted_points{points.begin(), points.end()};

  x.setlength(points.size());
  y.setlength(points.size());

  std::vector<rerun::Vec3D> poly_points;

  if (std::is_sorted(
          points.begin(), points.end(),
          [](const auto &a, const auto &b) { return a.x() < b.x(); }) or
      std::is_sorted(points.begin(), points.end(),
                     [](const auto &a, const auto &b) { return a.x() > b.x(); })

  ) {

    std::sort(sorted_points.begin(), sorted_points.end(),
              [](const auto &a, const auto &b) { return a.x() < b.x(); });

    for (auto &&[i, p] : enumerate(sorted_points)) {
      x(i) = p.x();
      y(i) = p.y();
    }

    alglib::spline1dbuildcubic(x, y, c);

    for (auto &&x_val : linear_distribute(sorted_points.front().x(),
                                          sorted_points.back().x(), 30)) {

      poly_points.emplace_back(x_val, alglib::spline1dcalc(c, x_val), 0.0);
    }
  } else if (std::is_sorted(
                 points.begin(), points.end(),
                 [](const auto &a, const auto &b) { return a.y() < b.y(); }) or
             std::is_sorted(
                 points.begin(), points.end(),
                 [](const auto &a, const auto &b) { return a.y() > b.y(); })) {

    std::sort(sorted_points.begin(), sorted_points.end(),
              [](const auto &a, const auto &b) { return a.y() < b.y(); });

    for (auto &&[i, p] : enumerate(sorted_points)) {
      x(i) = p.y();
      y(i) = p.x();
    }

    alglib::spline1dbuildakima(x, y, c);

    for (auto &&y_val : linear_distribute(sorted_points.front().y(),
                                          sorted_points.back().y(), 30)) {

      poly_points.emplace_back(alglib::spline1dcalc(c, y_val), y_val, 0.0);
    }
  }

  return poly_points;
}

std::optional<Eigen::Vector2d>
estimate_direction_spline(std::span<const Eigen::Vector2d> points,
                          Eigen::Vector2d query_point) {
  alglib::spline1dinterpolant c;
  alglib::real_1d_array x;
  alglib::real_1d_array y;

  x.setlength(points.size());
  y.setlength(points.size());

  const size_t n{points.size() - 1};

  if (std::is_sorted(
          points.begin(), points.end(),
          [](const auto &a, const auto &b) { return a.x() < b.x(); })) {

    for (auto &&[i, p] : enumerate(points)) {
      x(i) = p.x();
      y(i) = p.y();
    }

    alglib::spline1dbuildakima(x, y, c);

    double y_val{0.0};
    double d_y_val{0.0};
    double d2_y_val{0.0};

    alglib::spline1ddiff(c, query_point.x(), y_val, d_y_val, d2_y_val);
    return Eigen::Vector2d{1.0, d_y_val}.normalized();

  } else if (std::is_sorted(
                 points.begin(), points.end(),
                 [](const auto &a, const auto &b) { return a.x() > b.x(); })) {

    for (auto &&[i, p] : enumerate(points)) {
      x(n - i) = p.x();
      y(n - i) = p.y();
    }

    alglib::spline1dbuildakima(x, y, c);

    double y_val{0.0};
    double d_y_val{0.0};
    double d2_y_val{0.0};

    alglib::spline1ddiff(c, query_point.x(), y_val, d_y_val, d2_y_val);
    return Eigen::Vector2d{1.0, d_y_val}.normalized();

  } else if (std::is_sorted(
                 points.begin(), points.end(),
                 [](const auto &a, const auto &b) { return a.y() < b.y(); })) {

    for (auto &&[i, p] : enumerate(points)) {
      x(i) = p.y();
      y(i) = p.x();
    }

    alglib::spline1dbuildakima(x, y, c);

    double y_val{0.0};
    double d_y_val{0.0};
    double d2_y_val{0.0};

    alglib::spline1ddiff(c, query_point.y(), y_val, d_y_val, d2_y_val);
    return Eigen::Vector2d{d_y_val, 1.0}.normalized();

  } else if (std::is_sorted(
                 points.begin(), points.end(),
                 [](const auto &a, const auto &b) { return a.y() > b.y(); })) {
    for (auto &&[i, p] : enumerate(points)) {
      x(n - i) = p.y();
      y(n - i) = p.x();
    }

    alglib::spline1dbuildakima(x, y, c);

    double y_val{0.0};
    double d_y_val{0.0};
    double d2_y_val{0.0};

    alglib::spline1ddiff(c, query_point.y(), y_val, d_y_val, d2_y_val);
    // return Eigen::Vector2d{1.0, -d_y_val}.normalized();
    return Eigen::Vector2d{d_y_val, 1.0}.normalized();
  }

  LOG(WARNING) << "unable to approximate spline";
  return std::nullopt;
}

void correct_orientation(Landmark &landmark,
                         std::span<const GpsMeasurement> gps_track) {

  auto it{ranges::min_element(
      gps_track, [&landmark](const GpsMeasurement &a, const GpsMeasurement &b) {
        return (landmark.position_ - a.position_).squaredNorm() <
               (landmark.position_ - b.position_).squaredNorm();
      })};

  const auto ind{std::distance(gps_track.begin(), it)};

  const auto [points, direction] = get_points_in_the_radius(
      gps_track, BagProcessor::search_radius_, gps_track[ind].position_, ind);

  if ((gps_track[ind].position_ - landmark.position_).squaredNorm() >
      BagProcessor::max_dist_to_track_sqr_) {
    return;
  }

  auto [poly_direction, poly_coeffs, hor_dir] =
      estimate_direction<BagProcessor::poly_degree_>(
          points, gps_track[ind].position_.head<2>());

  if (poly_direction.dot(direction) > 0.0) {
    poly_direction *= -1.0;
  }

  if (Eigen::Vector2d{std::sin(landmark.azimuth_), std::cos(landmark.azimuth_)}
          .dot(poly_direction) >
      std::cos(BagProcessor::azimuth_correction_threshold_)) {

    auto azimuth{std::atan2(poly_direction.x(), poly_direction.y())};

    if (azimuth < 0.0) {
      azimuth += boost::math::double_constants::two_pi;
    }

    landmark.azimuth_ = azimuth;
  }
}

void link_detections(ImageDetections::map_type &detections) {
  for (auto &&[timestamp, dets] : detections) {
    for (auto &&i : ints(0ul, dets.dets_.size() - 1)) {
      for (auto &&j : ints(i + 1, dets.dets_.size())) {
        if (dets.dets_[i]->should_be_linked(*dets.dets_[j])) {
          dets.dets_[i]->link(*dets.dets_[j]);
        }
      }
    }
  }
}

void link_tracks(ImageTrack::map_type &tracks,
                 ImageDetections::map_type &detections) {

  std::unordered_set<std::pair<size_t, size_t>,
                     decltype([](const std::pair<size_t, size_t> &p) {
                       return boost::hash_value(p);
                     })>
      processed_tracks{};

  for (auto &&[track_id, track] : tracks) {

    std::unordered_set<size_t> link_candidates{};

    for (auto &&d : track.dets_) {
      for (auto &&det_id : d.linked_detections_) {
        link_candidates.insert(detections.at(d.timestamp_)
                                   .det_id_to_detection_.at(det_id)
                                   ->track_id_);
      }
    }

    for (auto &&other_track_id : link_candidates) {

      if (processed_tracks.contains(std::pair{track_id, other_track_id})) {
        continue;
      }

      if (track.should_be_linked(tracks.at(other_track_id))) {
        track.link(tracks.at(other_track_id));
      }

      processed_tracks.insert(std::pair{track_id, other_track_id});
    }
  }
}

struct Metrics::impl {
  impl(const Settings &set, std::span<const GpsMeasurement> gps_measurements,
       std::span<const Landmark> gt_landmarks)
      : set_{set} {

    auto gt_landmarks_vec{gt_landmarks | transform([](const Landmark &val) {
                            return std::pair{
                                static_cast<float>(val.position_.x()),
                                static_cast<float>(val.position_.y())};
                          }) |
                          to<std::vector>()};

    auto gps_vec{gps_measurements | transform([](const GpsMeasurement &val) {
                   return std::pair{static_cast<float>(val.position_.x()),
                                    static_cast<float>(val.position_.y())};
                 }) |
                 to<std::vector>()};

    const flann::Matrix<float> dataset{&gps_vec.front().first, gps_vec.size(),
                                       2ul};

    const flann::Matrix<float> query_dataset{&gt_landmarks_vec.front().first,
                                             gt_landmarks_vec.size(), 2ul};

    flann::Index<flann::L2_Simple<float>> index{
        dataset,
        flann::KDTreeSingleIndexParams{},
    };

    index.buildIndex();

    std::vector<std::vector<int>> indices{};
    std::vector<std::vector<float>> distances{};

    index.radiusSearch(query_dataset, indices, distances,
                       set_.max_distance_from_path_ *
                           set_.max_distance_from_path_,
                       flann::SearchParams{});

    for (auto &&[i, vec] : enumerate(indices)) {
      if (vec.empty()) {
        continue;
      }

      selected_gt_landmarks_.push_back(gt_landmarks[i]);
    }

    selected_gt_landmarks_vec_ =
        selected_gt_landmarks_ | transform([](const Landmark &val) {
          return std::pair{static_cast<float>(val.position_.x()),
                           static_cast<float>(val.position_.y())};
        }) |
        to<std::vector>();

    gt_dataset_ =
        flann::Matrix<float>{&selected_gt_landmarks_vec_.front().first,
                             selected_gt_landmarks_vec_.size(), 2ul};

    index_ = std::make_unique<flann::Index<flann::L2_Simple<float>>>(
        gt_dataset_, flann::KDTreeSingleIndexParams{});
    index_->buildIndex();
  }

  Metrics::Result eval(std::span<const Landmark> landmarks) const {

    auto landmarks_vec{landmarks | transform([](const Landmark &val) {
                         return std::pair{
                             static_cast<float>(val.position_.x()),
                             static_cast<float>(val.position_.y())};
                       }) |
                       to<std::vector>()};

    const flann::Matrix<float> query_dataset{&landmarks_vec.front().first,
                                             landmarks_vec.size(), 2ul};

    Metrics::Result res{};

    for (auto radius : linear_distribute(set_.distance_min_, set_.distance_max_,
                                         set_.num_steps_)) {

      std::vector<std::vector<int>> indices{};
      std::vector<std::vector<float>> distances{};

      float tp{0.0f};
      float fp{0.0f};
      std::unordered_set<size_t> taken{};

      index_->radiusSearch(query_dataset, indices, distances, radius * radius,
                           flann::SearchParams{});

      for (auto &&[landmark_ind, val] : enumerate(zip(indices, distances))) {
        const auto &[ind, dist] = val;

        bool is_match{false};

        for (auto &&[i, d] : zip(ind, dist)) {

          if ((not taken.contains(selected_gt_landmarks_[i].id_)) and
              landmarks[landmark_ind].code_ ==
                  selected_gt_landmarks_[i].code_) {

            taken.insert(selected_gt_landmarks_[i].id_);

            is_match = true;
            break;
          }
        }

        if (is_match) {
          ++tp;
        } else {
          ++fp;
        }
      }

      res.precision_.push_back(tp / (tp + fp));
      res.recall_.push_back(tp /
                            static_cast<float>(selected_gt_landmarks_.size()));
      res.distances_.push_back(radius);
    }

    res.auc_ = 0.0f;
    for (auto &&p : res.precision_) {
      res.auc_ += p;
    }

    LOG(INFO) << "AUC: " << res.auc_;
    return res;
  }

  Settings set_;
  std::vector<Landmark> selected_gt_landmarks_{};
  std::vector<std::pair<float, float>> selected_gt_landmarks_vec_{};
  flann::Matrix<float> gt_dataset_{};
  std::unique_ptr<flann::Index<flann::L2_Simple<float>>> index_;
};

Metrics::Metrics(const Settings &set,
                 std::span<const GpsMeasurement> gps_measurements,
                 std::span<const Landmark> gt_landmarks)
    : pimpl_{std::make_unique<impl>(set, gps_measurements, gt_landmarks)} {}

Metrics::Result Metrics::eval(std::span<const Landmark> landmarks) const {
  return pimpl_->eval(landmarks);
}

Metrics::~Metrics() = default;

bool Metrics::Result::save(const std::string_view path,
                           const std::string_view title) const {
  nlohmann::json j{};

  if (not title.empty()) {
    j["title"] = title.data();
  }

  j["precision"] = precision_;
  j["recall"] = recall_;
  j["distances"] = distances_;

  std::ofstream f{path.data()};

  if (f.is_open() and f.good()) {
    f << j.dump(4);
    return true;
  }

  return false;
}