#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/util/Constants.h>
#include <boost/math/constants/constants.hpp>
#include <cam/CamRadtan.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <feat/Feature.h>
#include <feat/FeatureDatabase.h>
#include <feature_tracker.hpp>
#include <fmt/core.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Quaternion.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/triangulation.h>
#include <gtsam/inference/Symbol.h>
#include <iterator>
#include <memory>
#include <ng-log/logging.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <range/v3/algorithm/find_if.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/concat.hpp>
#include <range/v3/view/enumerate.hpp>
#include <range/v3/view/linear_distribute.hpp>
#include <range/v3/view/take.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/view/zip.hpp>
#include <rerun.hpp>
#include <rerun/archetypes/geo_line_strings.hpp>
#include <rerun/archetypes/geo_points.hpp>
#include <rerun/components/class_id.hpp>
#include <rerun/components/geo_line_string.hpp>
#include <rerun/components/lat_lon.hpp>
#include <track/TrackBase.h>
#include <track/TrackKLT.h>
#include <types.hpp>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <utils/sensor_data.h>
#include <vector>

using ranges::to;
using ranges::views::concat;
using ranges::views::enumerate;
using ranges::views::linear_distribute;
using ranges::views::take;
using ranges::views::transform;
using ranges::views::zip;

struct FeatureTracker::impl {

  struct FeatureInTheImage {
    size_t id_;
    Eigen::Vector2f uv_;
  };

  impl(const FeatureTracker::Settings &set) : set_{set} {

    std::unordered_map<size_t, std::shared_ptr<ov_core::CamBase>> cameras{};
    cameras[0] =
        std::make_shared<ov_core::CamRadtan>(set_.width_, set_.height_);

    Eigen::MatrixXd calib;
    calib.resize(8, 1);

    for (auto &&[i, val] :
         enumerate(concat(set_.intrinsics_, set_.distortion_))) {
      calib(i) = val;
    }

    cameras[0]->set_value(calib);

    if (set_.use_klt_) {
      tracker_ = std::make_unique<ov_core::TrackKLT>(
          cameras, set_.num_feats_, 0, false, ov_core::TrackBase::HISTOGRAM,
          set_.fast_threshold_, set_.gridx_, set_.gridy_, set_.minpxdist_);

      mask_ = cv::Mat_<uint8_t>::zeros(set_.height_, set_.width_);
    }

    gtsam_cal3_s2_ =
        gtsam::Cal3_S2{set_.intrinsics_[0], set_.intrinsics_[1], 0.0,
                       set_.intrinsics_[2], set_.intrinsics_[3]};

    camera_matrix_ = cv::Mat_<double>::eye(3, 3);
    camera_matrix_(0, 0) = set_.intrinsics_[0];
    camera_matrix_(1, 1) = set_.intrinsics_[1];
    camera_matrix_(0, 2) = set_.intrinsics_[2];
    camera_matrix_(1, 2) = set_.intrinsics_[3];

    dist_coeffs_ = cv::Mat_<double>(set_.distortion_, true);
  }

  void add(cv::Mat_<uint8_t> image, const ImageDetections &dets) {

    if (not set_.use_klt_) {
      return;
    }

    const auto boxes{dets.dets_ |
                     transform([](const auto &val) { return val->box_; }) |
                     to<std::vector>()};

    tracker_->feed_new_camera(ov_core::CameraData{.timestamp = dets.timestamp_,
                                                  .sensor_ids = {0},
                                                  .images = {image},
                                                  .masks = {mask_},
                                                  .boxes = boxes});

    images_[dets.timestamp_] = image.clone();

    if (set_.save_debug_images_) {
      cv::Mat_<cv::Vec3b> img_tmp;
      cv::cvtColor(image, img_tmp, cv::COLOR_GRAY2BGR);

      tracker_->display_active(img_tmp, 255, 0, 0, 255, 255, 255);

      cv::imwrite(
          fmt::format("/root/data/images/img_active_{}.png", dets.timestamp_),
          img_tmp);

      cv::cvtColor(image, img_tmp, cv::COLOR_GRAY2BGR);
      tracker_->display_history(img_tmp, 255, 0, 0, 255, 255, 255);

      cv::imwrite(
          fmt::format("/root/data/images/img_history_{}.png", dets.timestamp_),
          img_tmp);
    }
  }

  void finalize() {
    if (not set_.use_klt_) {
      return;
    }

    const auto feature_db{
        tracker_->get_feature_database()->get_internal_data()};

    stamp_to_feature_id_.clear();

    for (auto &&[id, feature] : feature_db) {
      for (auto &&[stamp, uv] : zip(feature->timestamps[0], feature->uvs[0])) {
        stamp_to_feature_id_[stamp].emplace_back(id, uv);
      }
    }
  }

  std::optional<Landmark> triangulate(std::span<const TrackPoint> track) {

    std::unordered_map<int64_t, TrackPoint> timestamp_to_pose{};

    for (auto &&pose : track) {
      timestamp_to_pose[pose.timestamp_] = pose;
    }

    if (not set_.use_klt_) {
      return triangulate_on_boxes(timestamp_to_pose);
    }

    std::unordered_set<size_t> feaures_in_the_track{};

    const auto feature_db{
        tracker_->get_feature_database()->get_internal_data()};

    for (auto &&pose : track) {
      if (stamp_to_feature_id_.contains(pose.timestamp_)) {

        for (auto &&feature : stamp_to_feature_id_[pose.timestamp_]) {

          if (pose.box_.contains({static_cast<int>(feature.uv_.x() + 0.5f),
                                  static_cast<int>(feature.uv_.y() + 0.5f)})) {
            feaures_in_the_track.insert(feature.id_);
          }
        }
      }
    }

    const float track_delta_angle{
        std::abs(track.back().angle_ - track.front().angle_) > 180.0f
            ? 360.0f - std::abs(track.back().angle_ - track.front().angle_)
            : std::abs(track.back().angle_ - track.front().angle_)};

    const float angle_threshold{track_delta_angle * 3.0f / 4.0f};

    std::vector<size_t> features_filtered{};

    for (auto &&feature_id : feaures_in_the_track) {

      float min_angle{std::numeric_limits<float>::max()};
      float max_angle{std::numeric_limits<float>::min()};

      int num_measurements{0};

      for (auto &&[stamp, uv] : zip(feature_db.at(feature_id)->timestamps[0],
                                    feature_db.at(feature_id)->uvs[0])) {

        if (not timestamp_to_pose.contains(stamp)) {
          continue;
        }

        if (not timestamp_to_pose[stamp].box_.contains(
                {static_cast<int>(uv.x() + 0.5f),
                 static_cast<int>(uv.y() + 0.5f)})) {
          continue;
        }

        min_angle = std::min(min_angle, timestamp_to_pose[stamp].angle_);
        max_angle = std::max(max_angle, timestamp_to_pose[stamp].angle_);

        ++num_measurements;
      }

      if (num_measurements < 5) {
        continue;
      }

      const float delta_angle{max_angle - min_angle > 180.0f
                                  ? 360.0f - (max_angle - min_angle)
                                  : max_angle - min_angle};

      if (delta_angle < angle_threshold) {
        continue;
      }

      features_filtered.push_back(feature_id);
    }

    dump_features(timestamp_to_pose, features_filtered);

    if (features_filtered.size() >= 3) {
      auto res{triangulate_on_features(timestamp_to_pose, features_filtered)};

      if (res.has_value()) {
        return res;
      }
    }

    LOG(WARNING) << "triangulation based on features failed, fallback to "
                    "boxes, track_id: "
                 << track.begin()->id_;

    return triangulate_on_boxes(timestamp_to_pose);
  }

  void log_directions(
      const std::unordered_map<int64_t, TrackPoint> &timestamp_to_pose,
      size_t feature_id) {

    if (set_.rec_ == nullptr) {
      return;
    }

    const auto feature_db{
        tracker_->get_feature_database()->get_internal_data()};

    std::vector<rerun::components::GeoLineString> directions{};

    for (auto &&[stamp, track_point] : timestamp_to_pose) {

      auto it{
          ranges::find_if(feature_db.at(feature_id)->timestamps[0],
                          [&stamp](const int64_t &ts) { return ts == stamp; })};

      if (it != feature_db.at(feature_id)->timestamps[0].end()) {

        const auto feature_ind{std::distance(
            feature_db.at(feature_id)->timestamps[0].begin(), it)};

        const auto p{feature_db.at(feature_id)->uvs[0][feature_ind]};

        std::vector<cv::Point2f> cv_points{};
        cv_points.push_back(cv::Point2f{p.x(), p.y()});

        cv::undistortImagePoints(cv_points, cv_points, camera_matrix_,
                                 dist_coeffs_);

        const double z0{150.0};
        const double x{(cv_points[0].x - gtsam_cal3_s2_.px()) * z0 /
                       gtsam_cal3_s2_.fx()};

        const double y{(cv_points[0].y - gtsam_cal3_s2_.py()) * z0 /
                       gtsam_cal3_s2_.fy()};

        const Eigen::Vector3d p0{track_point.pose_.translation()};
        const Eigen::Vector3d p1{track_point.pose_ * Eigen::Vector3d{x, y, z0}};

        rerun::DVec2D lla0{};
        rerun::DVec2D lla1{};
        double h{0.0};

        set_.local_converter_->Reverse(p0.x(), p0.y(), p0.z(), lla0.xy[0],
                                       lla0.xy[1], h);
        set_.local_converter_->Reverse(p1.x(), p1.y(), p1.z(), lla1.xy[0],
                                       lla1.xy[1], h);

        directions.push_back(
            rerun::components::GeoLineString::from_lat_lon({lla0, lla1}));
      }
    }

    set_.rec_->log(fmt::format("map/feature_direction_{}", feature_id),
                   rerun::GeoLineStrings{directions}
                       .with_colors(rerun::Color{0xfca40bff})
                       .with_radii(rerun::Radius::ui_points(0.5f)));
  }

  void dump_features(
      const std::unordered_map<int64_t, TrackPoint> &timestamp_to_pose,
      std::span<const size_t> features_filtered) {

    const auto feature_db{
        tracker_->get_feature_database()->get_internal_data()};

    for (auto &&[stamp, track_point] : timestamp_to_pose) {

      if (not images_.contains(stamp)) {
        continue;
      }

      cv::Mat_<cv::Vec3b> img_tmp{};

      cv::cvtColor(images_[stamp], img_tmp, cv::COLOR_GRAY2BGR);
      cv::rectangle(img_tmp, track_point.box_, {255.0, 0.0, 0.0});

      for (auto &&feature_id : features_filtered) {

        cv::Scalar clr{255.0, 0.0, 0.0};

        if (feature_id % 2 == 0) {
          clr = cv::Scalar{0.0, 255.0, 0.0};
        }

        if (feature_id % 3 == 0) {
          clr = cv::Scalar{0.0, 0.0, 255.0};
        }

        if (feature_id % 5 == 0) {
          clr = cv::Scalar{0.0, 255.0, 255.0};
        }

        if (feature_id % 7 == 0) {
          clr = cv::Scalar{255.0, 0.0, 255.0};
        }

        if (feature_id % 11 == 0) {
          clr = cv::Scalar{255.0, 255.0, 0.0};
        }

        feature_id_to_color_[feature_id] = clr;

        auto it{ranges::find_if(
            feature_db.at(feature_id)->timestamps[0],
            [&stamp](const int64_t &ts) { return ts == stamp; })};

        if (it != feature_db.at(feature_id)->timestamps[0].end()) {

          const auto feature_ind{std::distance(
              feature_db.at(feature_id)->timestamps[0].begin(), it)};

          const auto p{feature_db.at(feature_id)->uvs[0][feature_ind]};

          cv::circle(
              img_tmp,
              {static_cast<int>(p.x() + 0.5f), static_cast<int>(p.y() + 0.5f)},
              1, clr, cv::FILLED, cv::LINE_AA);
        }
      }

      cv::Mat_<cv::Vec3b> img_und{};
      cv::undistort(img_tmp, img_und, camera_matrix_, dist_coeffs_);

      cv::imwrite(
          fmt::format("/root/data/images/image_with_features_{}.png", stamp),
          img_und);
    }
  }

  std::optional<Landmark> triangulate_on_features(
      const std::unordered_map<int64_t, TrackPoint> &timestamp_to_pose,
      std::span<const size_t> features_filtered) {

    const auto measurement_noise{gtsam::noiseModel::Isotropic::Sigma(2, 1.0)};

    const auto feature_db{
        tracker_->get_feature_database()->get_internal_data()};

    std::vector<Eigen::Vector3d> p3d{};
    std::vector<size_t> triangulated_features{};

    for (auto &&feature_id : features_filtered) {

      std::vector<cv::Point2f> points{};
      gtsam::CameraSet<gtsam::PinholeCamera<gtsam::Cal3_S2>> cameras{};

      for (auto &&[stamp, uv] : zip(feature_db.at(feature_id)->timestamps[0],
                                    feature_db.at(feature_id)->uvs[0])) {

        if (timestamp_to_pose.contains(stamp)) {
          if (timestamp_to_pose.at(stamp).box_.contains(
                  {static_cast<int>(uv.x() + 0.5f),
                   static_cast<int>(uv.y() + 0.5f)})) {

            points.emplace_back(uv.x(), uv.y());

            const gtsam::Pose3 pose{
                gtsam::Rot3{timestamp_to_pose.at(stamp).pose_.linear()},
                gtsam::Vector3{
                    timestamp_to_pose.at(stamp).pose_.translation()}};

            cameras.emplace_back(pose, gtsam_cal3_s2_);
          }
        }
      }

      cv::undistortImagePoints(points, points, camera_matrix_, dist_coeffs_);

      const gtsam::Point2Vector measurements{
          points |
          transform([](const auto &p) { return gtsam::Point2{p.x, p.y}; }) |
          to<std::vector<gtsam::Point2,
                         Eigen::aligned_allocator<gtsam::Point2>>>()};

      try {
        p3d.push_back(gtsam::triangulatePoint3(cameras, measurements, 1.0e-9,
                                               true, measurement_noise, true));

        triangulated_features.push_back(feature_id);
      } catch (...) {
      }
    }

    log_points(p3d, triangulated_features);
    // log_directions(timestamp_to_pose, 10195);

    if (p3d.size() >= 3) {
      return get_azimmuth(p3d, timestamp_to_pose);
    }

    return std::nullopt;
  }

  std::optional<Landmark> triangulate_on_boxes(
      const std::unordered_map<int64_t, TrackPoint> &timestamp_to_pose) {

    gtsam::CameraSet<gtsam::PinholeCamera<gtsam::Cal3_S2>> cameras{};
    const auto measurement_noise{gtsam::noiseModel::Isotropic::Sigma(2, 1.0)};

    float mean_box_ratio{0.0f};
    float rms_box_ratio{0.0f};

    for (auto &&[stamp, d] : timestamp_to_pose) {

      const float box_ratio{static_cast<float>(d.box_.width) /
                            static_cast<float>(d.box_.height)};

      mean_box_ratio += box_ratio;
      // LOG(INFO) << "box ratio:" << box_ratio;
    }

    mean_box_ratio /= static_cast<float>(timestamp_to_pose.size());

    // LOG(INFO) << "mean box ratio:" << mean_box_ratio;

    for (auto &&[stamp, d] : timestamp_to_pose) {

      const float box_ratio{static_cast<float>(d.box_.width) /
                            static_cast<float>(d.box_.height)};

      rms_box_ratio +=
          (box_ratio - mean_box_ratio) * (box_ratio - mean_box_ratio);
    }

    rms_box_ratio =
        std::sqrt(rms_box_ratio /
                  (static_cast<float>(timestamp_to_pose.size()) *
                   (static_cast<float>(timestamp_to_pose.size()) - 1.0f)));

    for (auto &&[stamp, track_point] : timestamp_to_pose) {

      const float box_ratio{static_cast<float>(track_point.box_.width) /
                            static_cast<float>(track_point.box_.height)};

      if (std::abs(box_ratio - mean_box_ratio) > 5.0f * rms_box_ratio) {
        continue;
      }

      const gtsam::Pose3 pose{gtsam::Rot3{track_point.pose_.linear()},
                              gtsam::Vector3{track_point.pose_.translation()}};

      cameras.emplace_back(pose, gtsam_cal3_s2_);
    }

    // LOG(INFO) << "rms box ratio: " << rms_box_ratio;
    std::vector<Eigen::Vector3d> p3d{};

    for (auto &&u : linear_distribute(0.0, 1.0, 5)) {
      for (auto &&v : linear_distribute(0.0, 1.0, 5)) {

        std::vector<cv::Point2f> points{};

        for (auto &&[stamp, track_point] : timestamp_to_pose) {

          const float box_ratio{static_cast<float>(track_point.box_.width) /
                                static_cast<float>(track_point.box_.height)};

          if (std::abs(box_ratio - mean_box_ratio) > 5.0f * rms_box_ratio) {
            continue;
          }

          const auto x{
              static_cast<double>(track_point.box_.x) * (1.0 - v) +
              static_cast<double>(track_point.box_.x + track_point.box_.width) *
                  v};
          const auto y{static_cast<double>(track_point.box_.y) * (1.0 - u) +
                       static_cast<double>(track_point.box_.y +
                                           track_point.box_.height) *
                           u};
          points.emplace_back(x, y);
        }

        cv::undistortImagePoints(points, points, camera_matrix_, dist_coeffs_);

        const gtsam::Point2Vector measurements{
            points |
            transform([](const auto &p) { return gtsam::Point2{p.x, p.y}; }) |
            to<std::vector<gtsam::Point2,
                           Eigen::aligned_allocator<gtsam::Point2>>>()};

        try {
          p3d.push_back(gtsam::triangulatePoint3(
              cameras, measurements, 1.0e-9, true, measurement_noise, true));
        } catch (...) {
        }
      }
    }

    log_points(p3d, {});

    if (p3d.size() >= 3) {
      return get_azimmuth(p3d, timestamp_to_pose);
    }

    return std::nullopt;
  }

  std::pair<Eigen::Vector3d, Eigen::Vector3d>
  fit_a_plane(std::span<const Eigen::Vector3d> points) {

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

    return {normal, center};
  }

  Landmark get_azimmuth(
      std::span<const Eigen::Vector3d> points,
      const std::unordered_map<int64_t, TrackPoint> &timestamp_to_pose) {

    auto [normal, center]{fit_a_plane(points)};

    int num_positives{0};
    int num_negatives{0};

    for (auto &&[stamp, track_point] : timestamp_to_pose) {

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

    return Landmark{.position_ = center, .azimuth_ = azimuth};
  }

  void log_points(std::span<const Eigen::Vector3d> points,
                  std::span<const size_t> feature_ids) {

    if (set_.rec_ == nullptr) {
      return;
    }

    const std::vector<rerun::components::LatLon> lla_vec{
        points | transform([this](auto &&p) {
          double lat{0.0};
          double lon{0.0};
          double h{0.0};

          set_.local_converter_->Reverse(p.x(), p.y(), p.z(), lat, lon, h);
          return rerun::components::LatLon{lat, lon};
        }) |
        to<std::vector>()};

    if (not feature_ids.empty()) {

      const std::vector<rerun::Color> colors{
          feature_ids | transform([this](size_t id) {
            if (feature_id_to_color_.contains(id)) {
              return rerun::Color{
                  static_cast<uint8_t>(feature_id_to_color_[id](2)),
                  static_cast<uint8_t>(feature_id_to_color_[id](1)),
                  static_cast<uint8_t>(feature_id_to_color_[id](0))};
            }

            return rerun::Color{0, 0, 0};
          }) |
          to<std::vector>()};

      const std::vector<rerun::components::ClassId> class_ids{
          feature_ids | transform([](size_t id) {
            return rerun::components::ClassId{static_cast<uint16_t>(id)};
          }) |
          to<std::vector>()};

      set_.rec_->log("map/points",
                     rerun::GeoPoints::from_lat_lon(lla_vec)
                         .with_colors(colors)
                         .with_radii(rerun::Radius::ui_points(4.0f))
                         .with_class_ids(class_ids));
    } else {
      set_.rec_->log("map/points",
                     rerun::GeoPoints::from_lat_lon(lla_vec)
                         .with_colors(rerun::Color{255, 0, 0})
                         .with_radii(rerun::Radius::ui_points(4.0f)));
    }
  }

  FeatureTracker::Settings set_;
  std::unique_ptr<ov_core::TrackKLT> tracker_;
  cv::Mat_<uint8_t> mask_;
  std::unordered_map<int64_t, std::vector<FeatureInTheImage>>
      stamp_to_feature_id_{};
  gtsam::Cal3_S2 gtsam_cal3_s2_;
  cv::Mat_<double> camera_matrix_;
  cv::Mat_<double> dist_coeffs_;

  std::unordered_map<int64_t, cv::Mat_<uint8_t>> images_;
  std::unordered_map<size_t, cv::Scalar> feature_id_to_color_;
};

FeatureTracker::FeatureTracker(const Settings &set)
    : pimpl_{std::make_unique<impl>(set)} {}

FeatureTracker::~FeatureTracker() = default;

void FeatureTracker::add(cv::Mat_<uint8_t> image, const ImageDetections &dets) {
  pimpl_->add(image, dets);
}

std::optional<Landmark>
FeatureTracker::triangulate(std::span<const TrackPoint> track) {
  return pimpl_->triangulate(track);
}

void FeatureTracker::finalize() { pimpl_->finalize(); }
