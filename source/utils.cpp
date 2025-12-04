#include <bag_processor.hpp>
#include <boost/container_hash/hash.hpp>
#include <boost/math/constants/constants.hpp>
#include <cmath>
#include <cstddef>
#include <deque>
#include <flann/algorithms/dist.h>
#include <flann/flann.hpp>
#include <flann/util/matrix.h>
#include <fmt/color.h>
#include <fmt/core.h>
#include <gtsam/geometry/triangulation.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <interpolation.h>
#include <iterator>
#include <limits>
#include <memory>
#include <ng-log/logging.h>
#include <nlohmann/json.hpp>
#include <range/v3/algorithm/copy.hpp>
#include <range/v3/algorithm/count_if.hpp>
#include <range/v3/algorithm/find_if.hpp>
#include <range/v3/algorithm/max.hpp>
#include <range/v3/algorithm/min_element.hpp>
#include <range/v3/algorithm/sort.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/enumerate.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/linear_distribute.hpp>
#include <range/v3/view/take.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/view/zip.hpp>
#include <rerun.hpp>
#include <span>
#include <types.hpp>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <utils.hpp>
#include <vector>

using ranges::to;
using ranges::views::enumerate;
using ranges::views::filter;
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
get_points_in_the_radius(std::span<const CameraMeasurement> points, double rad,
                         Eigen::Vector2d query_point, ptrdiff_t ind) {

  const double rad_squared{rad * rad};

  Eigen::Vector2d first_point{};
  Eigen::Vector2d last_point{};
  std::deque<Eigen::Vector2d> points_queue;

  int num_added{0};

  for (ptrdiff_t i{ind - 1}; i >= 0; --i) {

    if ((points[i].enu_ - query_point).squaredNorm() < rad_squared
        //  or num_added < 5
    ) {
      points_queue.emplace_front(points[i].enu_);
      ++num_added;
      first_point = points[i].enu_;
    } else {
      break;
    }
  }

  num_added = 0;
  for (ptrdiff_t i{ind}; i < points.size(); ++i) {

    if ((points[i].enu_ - query_point).squaredNorm() < rad_squared
        // or num_added < 5
    ) {
      points_queue.emplace_back(points[i].enu_);

      ++num_added;
      last_point = points[i].enu_;
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
                         std::span<const CameraMeasurement> gps_track) {

  auto it{
      ranges::min_element(gps_track, [&landmark](const CameraMeasurement &a,
                                                 const CameraMeasurement &b) {
        return (landmark.enu_ - a.enu_).squaredNorm() <
               (landmark.enu_ - b.enu_).squaredNorm();
      })};

  const auto ind{std::distance(gps_track.begin(), it)};

  const auto [points, direction] = get_points_in_the_radius(
      gps_track, BagProcessor::search_radius_, gps_track[ind].enu_, ind);

  if ((gps_track[ind].enu_ - landmark.enu_).squaredNorm() >
      BagProcessor::max_dist_to_track_sqr_) {
    return;
  }

  const auto res{estimate_direction<BagProcessor::poly_degree_>(
      points, gps_track[ind].enu_)};

  auto poly_direction{res.direction_};

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

      processed_tracks.insert(std::pair{other_track_id, track_id});
    }
  }
}

struct Metrics::impl {
  impl(const Settings &set, std::span<const GpsMeasurement> gps_measurements,
       std::span<const Landmark> gt_landmarks)
      : set_{set} {

    auto gt_landmarks_vec{gt_landmarks | transform([](const Landmark &val) {
                            return std::pair{static_cast<float>(val.enu_.x()),
                                             static_cast<float>(val.enu_.y())};
                          }) |
                          to<std::vector>()};

    auto gps_vec{gps_measurements | transform([](const GpsMeasurement &val) {
                   return std::pair{static_cast<float>(val.enu_.x()),
                                    static_cast<float>(val.enu_.y())};
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
          return std::pair{static_cast<float>(val.enu_.x()),
                           static_cast<float>(val.enu_.y())};
        }) |
        to<std::vector>();

    gt_dataset_ =
        flann::Matrix<float>{&selected_gt_landmarks_vec_.front().first,
                             selected_gt_landmarks_vec_.size(), 2ul};

    index_ = std::make_unique<flann::Index<flann::L2_Simple<float>>>(
        gt_dataset_, flann::KDTreeSingleIndexParams{});
    index_->buildIndex();
  }

  Metrics::Result eval(const ImageTrack::map_type &tracks) const {

    std::vector<size_t> filtered_ids{};

    auto landmarks_vec{
        tracks |
        filter([](auto &&val) { return val.second.landmark_.has_value(); }) |
        transform([&filtered_ids](const auto &val) {
          filtered_ids.push_back(val.first);
          return std::pair{static_cast<float>(val.second.landmark_->enu_.x()),
                           static_cast<float>(val.second.landmark_->enu_.y())};
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
              tracks.at(filtered_ids[landmark_ind]).landmark_->code_ ==
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

    const float delta{1.0f / static_cast<float>(set_.num_steps_ - 1)};

    res.precision_auc_ = 0.0f;
    for (auto &&p : res.precision_) {
      res.precision_auc_ += p;
    }

    res.precision_auc_ -=
        0.5f * (res.precision_.front() + res.precision_.back());
    res.precision_auc_ *= delta;

    res.recall_auc_ = 0.0f;
    for (auto &&p : res.recall_) {
      res.recall_auc_ += p;
    }

    res.recall_auc_ -= 0.5f * (res.recall_.front() + res.recall_.back());
    res.recall_auc_ *= delta;

    LOG(INFO) << fmt::format(fmt::fg(fmt::color::yellow_green),
                             "Precision AUC: ")
              << fmt::format(fmt::fg(fmt::color::coral), "{}",
                             res.precision_auc_);

    LOG(INFO) << fmt::format(fmt::fg(fmt::color::yellow_green), "Recall AUC: ")
              << fmt::format(fmt::fg(fmt::color::coral), "{}", res.recall_auc_);

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

Metrics::Result Metrics::eval(const ImageTrack::map_type &tracks) const {
  return pimpl_->eval(tracks);
}

Metrics::~Metrics() = default;

bool Metrics::Result::save(const std::string_view path,
                           const std::string_view title) const {
  nlohmann::json j{};

  if (not title.empty()) {
    j["title"] = title.data();
  }

  j["precision_auc"] = precision_auc_;
  j["recall_auc"] = recall_auc_;
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

size_t combine_landmarks(ImageTrack::map_type &tracks,
                         const CartesianConverter &converter) {

  std::unordered_set<size_t> processed_tracks{};

  for (auto &&[track_id, track] : tracks) {

    if (not track.landmark_.has_value()) {
      continue;
    }

    if (processed_tracks.contains(track_id)) {
      continue;
    }

    std::deque<size_t> q{};
    q.push_back(track_id);
    processed_tracks.insert(track_id);

    std::unordered_set<size_t> defined_landmarks{};
    std::unordered_set<size_t> undefined_landmarks{};

    while (not q.empty()) {
      const auto id{q.front()};
      q.pop_front();

      if (not tracks.at(id).landmark_.has_value()) {
        undefined_landmarks.insert(id);
      } else {
        defined_landmarks.insert(id);
      }

      for (auto &&linked_id : tracks.at(id).linked_tracks_) {
        if (processed_tracks.contains(linked_id)) {
          continue;
        }

        q.push_back(linked_id);
        processed_tracks.insert(linked_id);
      }
    }

    double best_variance{std::numeric_limits<double>::max()};
    size_t best_tack_id{0};

    for (auto &&id : defined_landmarks) {
      if (best_variance > tracks.at(id).landmark_->dist_variance_) {
        best_variance = tracks.at(id).landmark_->dist_variance_;
        best_tack_id = id;
      }
    }

    const auto &best_landmark{tracks.at(best_tack_id).landmark_.value()};

    for (auto &&id : defined_landmarks) {
      if (id == best_tack_id) {
        continue;
      }

      tracks.at(id).landmark_->enu_ = best_landmark.enu_;
      tracks.at(id).landmark_->latlon_ = best_landmark.latlon_;
      tracks.at(id).landmark_->azimuth_ = best_landmark.azimuth_;
      tracks.at(id).landmark_->dist_variance_ = best_landmark.dist_variance_;
    }

    for (auto &&id : undefined_landmarks) {
      tracks.at(id).landmark_->enu_ = best_landmark.enu_;
      tracks.at(id).landmark_->latlon_ = best_landmark.latlon_;
      tracks.at(id).landmark_->azimuth_ = best_landmark.azimuth_;
      tracks.at(id).landmark_->dist_variance_ = best_landmark.dist_variance_;
    }

#if 0 
    Eigen::Vector2d mean_enu{Eigen::Vector2d::Zero()};
    double mean_azimuth{0.0};
    double norm{0.0};

    for (auto &&id : defined_landmarks) {
      const auto var{tracks.at(id).landmark_->dist_variance_};
      mean_enu += tracks.at(id).landmark_->enu_ / var;
      mean_azimuth += tracks.at(id).landmark_->azimuth_ / var;
      norm += 1.0 / var;
    }

    const double dist_variance{1.0 / norm};

    mean_enu *= dist_variance;
    mean_azimuth *= dist_variance;

    const auto mean_lla{converter.latlon(mean_enu)};

    for (auto &&id : defined_landmarks) {
      tracks.at(id).landmark_->enu_ = mean_enu;
      tracks.at(id).landmark_->latlon_ = mean_lla;
      tracks.at(id).landmark_->azimuth_ = mean_azimuth;
      tracks.at(id).landmark_->dist_variance_ = dist_variance;
    }

    for (auto &&id : undefined_landmarks) {
      tracks.at(id).landmark_->enu_ = mean_enu;
      tracks.at(id).landmark_->latlon_ = mean_lla;
      tracks.at(id).landmark_->azimuth_ = mean_azimuth;
      tracks.at(id).landmark_->dist_variance_ = dist_variance;
    }
#endif
  }

  return ranges::count_if(
      tracks, [](auto &&val) { return val.second.landmark_.has_value(); });
}
