#include <Eigen/Core>
#include <GeographicLib/LocalCartesian.hpp>
#include <algorithm>
#include <bag_loader.hpp>
#include <boost/container_hash/hash.hpp>
#include <boost/math/constants/constants.hpp>
#include <cmath>
#include <cstddef>
#include <deque>
#include <feature_matcher.hpp>
#include <flann/flann.hpp>
#include <fmt/color.h>
#include <fmt/core.h>
#include <limits>
#include <memory>
#include <ng-log/logging.h>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <range/v3/algorithm/max.hpp>
#include <range/v3/algorithm/minmax.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/enumerate.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/reverse.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/view/zip.hpp>
#if USE_RERUN
#include <rerun.hpp>
#include <rerun_logging.hpp>
#endif
#include <span>
#include <tracks_collecton.hpp>
#include <triangulation.hpp>
#include <types.hpp>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using ranges::to;
using ranges::views::enumerate;
using ranges::views::filter;
using ranges::views::reverse;
using ranges::views::transform;
using ranges::views::zip;

struct Link {
  size_t bag_ind_;
  size_t track_id_;

  size_t operator()(const Link &link) const noexcept {
    size_t seed{0};
    boost::hash_combine(seed, link.bag_ind_);
    boost::hash_combine(seed, link.track_id_);
    return seed;
  }

  bool operator==(const Link &link) const noexcept {
    return link.bag_ind_ == bag_ind_ and link.track_id_ == track_id_;
  }
};

struct CombinedLandmarks {

  std::vector<Landmark> landmarks_;
  std::unordered_map<size_t, std::vector<Link>> landmark_to_bag_;
  std::unordered_map<Link, size_t, Link> bag_to_landmark_;

  void add(Link link, const Landmark &landmark) {
    landmark_to_bag_[landmarks_.size()].emplace_back(link);
    bag_to_landmark_[link] = landmarks_.size();
    landmarks_.push_back(std::move(landmark));
  }

  bool contain(Link link) const { return bag_to_landmark_.contains(link); }

  Landmark &at(Link link) { return landmarks_[bag_to_landmark_.at(link)]; }

  const Landmark &at(Link link) const {
    return landmarks_[bag_to_landmark_.at(link)];
  }

  std::span<const Link> linked_bags(Link link) const {
    return landmark_to_bag_.at(bag_to_landmark_.at(link));
  }

  std::span<const Link> linked_bags(size_t ind) const {
    return landmark_to_bag_.at(ind);
  }

  size_t landmark_index(Link link) const { return bag_to_landmark_.at(link); }

  void link(Link src, Link dst) {
    if (contain(dst)) {
      landmark_to_bag_.at(bag_to_landmark_.at(dst)).emplace_back(src);
      bag_to_landmark_[src] = bag_to_landmark_.at(dst);
    }
  }
};

struct MinMaxAccumulator {

  void add(double val) noexcept {
    min_ = std::min(min_, val);
    max_ = std::max(max_, val);
  }

  double delta() const noexcept { return max_ - min_; }

  double min_{std::numeric_limits<double>::max()};
  double max_{std::numeric_limits<double>::min()};
};

class TrackIndexer {
public:
  struct GeoPoint {
    size_t bag_index_;
    size_t track_id_;
    size_t detection_index_;
    Eigen::Vector2d enu_;
  };

  struct Result {
    struct DetPair {
      size_t dst_ind_;
      size_t src_ind_;
    };

    Link src_link_;
    Link dst_link_;
    std::vector<DetPair> det_inds_;
  };

  TrackIndexer(const std::vector<BagProcessor::ptr> &bags)
      : bags_{bags}, indexer_{flann::KDTreeSingleIndexParams{}} {

    for (auto &&[bag_ind, bag] : bags_ | enumerate) {
      for (auto &&[track_id, track] : bag->image_tracks_) {
        if (not track.valid_) {
          continue;
        }

        for (auto &&det_ind : track.selected_detections_) {
          const auto &d{track.dets_[det_ind]};

          if (not d.enu_.has_value()) {
            continue;
          }

          points_.emplace_back(bag_ind, track_id, det_ind, d.enu_.value());
        }
      }
    }

    auto data_vec{points_ | transform([](const auto &p) {
                    return std::pair{p.enu_.x(), p.enu_.y()};
                  }) |
                  to<std::vector>()};

    const flann::Matrix<double> dataset{&data_vec.front().first,
                                        data_vec.size(), 2ul};

    indexer_.buildIndex(dataset);
  }

  std::vector<Result> find(Link src_link) {

    const auto &track{
        bags_[src_link.bag_ind_]->image_tracks_.at(src_link.track_id_)};

    auto query_points{track.selected_detections_ |
                      filter([&track](const auto &det_ind) {
                        return track.dets_[det_ind].enu_.has_value();
                      }) |
                      transform([&track](const auto &det_ind) {
                        const auto &d{track.dets_[det_ind]};
                        return GeoPoint{.bag_index_ = 0,
                                        .track_id_ = track.id_,
                                        .detection_index_ = det_ind,
                                        .enu_ = d.enu_.value()};
                      }) |
                      to<std::vector>()};

    auto query_vec{query_points | transform([](const GeoPoint &p) {
                     return std::pair{p.enu_.x(), p.enu_.y()};
                   }) |
                   to<std::vector>()};

    const flann::Matrix<double> query_dataset{&query_vec.front().first,
                                              query_vec.size(), 2ul};

    std::vector<std::vector<int>> indices{};
    std::vector<std::vector<double>> distances{};

    indexer_.radiusSearch(query_dataset, indices, distances, search_rad_sqr_,
                          flann::SearchParams{});

    std::unordered_map<Link, std::vector<std::pair<size_t, size_t>>, Link>
        found_tracks{};

    for (auto &&[query_ind, data_ind] : enumerate(indices)) {

      std::unordered_map<Link, size_t, Link> found_indices{};

      for (auto &i : data_ind) {

        const auto &det{detection(i)};

        if (det.code_ != track.code_) {
          continue;
        }

        if (found_indices.contains(
                {points_[i].bag_index_, points_[i].track_id_})) {
          continue;
        }

        const Eigen::Vector2d dir1{det.direction_.value()};
        const Eigen::Vector2d dir2{
            track.dets_[query_points[query_ind].detection_index_]
                .direction_.value()};

        const auto dot_product{dir1.dot(dir2)};

        if (dot_product < cos_angle_threshold_) {
          continue;
        }

        found_indices[{points_[i].bag_index_, points_[i].track_id_}] =
            points_[i].detection_index_;
      }

      for (auto &&[dst_link, dst_det_ind] : found_indices) {
        found_tracks[dst_link].push_back(
            {dst_det_ind, query_points[query_ind].detection_index_});
      }
    }

    std::vector<Result> res{};

    for (auto &&[dst_link, ind_pairs] : found_tracks) {
      const auto &dst_track{
          bags_[dst_link.bag_ind_]->image_tracks_.at(dst_link.track_id_)};

      std::unordered_map<size_t, std::pair<size_t, double>> m{};

      for (auto &&[dst_ind, src_ind] : ind_pairs) {

        if (not m.contains(dst_ind)) {
          m[dst_ind] =
              std::pair{src_ind, (dst_track.dets_[dst_ind].enu_.value() -
                                  track.dets_[src_ind].enu_.value())
                                     .squaredNorm()};
        } else {
          const auto d{(dst_track.dets_[dst_ind].enu_.value() -
                        track.dets_[src_ind].enu_.value())
                           .squaredNorm()};

          if (m.at(dst_ind).second > d) {
            m.at(dst_ind).first = src_ind;
            m.at(dst_ind).second = d;
          }
        }
      }

      res.emplace_back(Result{.src_link_ = src_link,
                              .dst_link_ = dst_link,
                              .det_inds_ = m | transform([](const auto &p) {
                                             return Result::DetPair{
                                                 .dst_ind_ = p.first,
                                                 .src_ind_ = p.second.first};
                                           }) |
                                           to<std::vector>()});
    }

    return res;
  }

private:
  const Detection &detection(size_t i) const {
    return bags_[points_[i].bag_index_]
        ->image_tracks_.at(points_[i].track_id_)
        .dets_[points_[i].detection_index_];
  }

  const std::vector<BagProcessor::ptr> &bags_;
  flann::Index<flann::L2_Simple<double>> indexer_;
  std::vector<GeoPoint> points_;
  static constexpr float search_rad_sqr_{15.0f * 15.0f};
  static constexpr double cos_angle_threshold_{
      constexpr_cos(boost::math::double_constants::degree * 15.0)};
};

struct TracksCollection::impl {

  void init(BagProcessor::ptr bag) {
    converter_ = bag->local_converter_;
    bags_.push_back(bag);

    for (auto &&[track_id, track] : bag->image_tracks_) {
      if (not track.valid_) {
        continue;
      }

      if (not track.landmark_.has_value()) {
        continue;
      }

      landmarks_.add({0, track_id}, track.landmark_.value());
    }
  }

  void recalculate_coords(BagProcessor::ptr bag) {

    for (auto &&[track_id, track] : bag->image_tracks_) {
      for (auto &&d : track.dets_) {

        if (not d.enu_.has_value()) {
          continue;
        }

        const Eigen::Vector2d enu{
            converter_.enu(bag->local_converter_.latlon(d.enu_.value()))};

        d.enu_ = enu;

        if (d.cam_to_world_.has_value()) {
          d.cam_to_world_->translation() =
              Eigen::Vector3d{enu.x(), enu.y(), 0.0};
        }
      }

      if (track.landmark_.has_value()) {
        track.landmark_->enu_ = converter_.enu(track.landmark_->latlon_);
      }

      track.geodetic_origin_ = converter_.origin();
    }

    for (auto &&gps : bag->gps_) {
      gps.enu_ = converter_.enu(gps.latlon_);
    }
  }

  void merge(BagProcessor::ptr bag) {

    if (bags_.empty()) {
      init(bag);
      return;
    }

    recalculate_coords(bag);
    TrackIndexer indexer_{bags_};

    bags_.push_back(bag);

    std::unordered_map<size_t, std::vector<size_t>> should_be_added{};
    ImageTrack::map_type other_tracks_copy{};

    const auto num_valid_tracks{bag->num_valid_tracks()};
    size_t num_track{0};
    std::vector<size_t> affected_landmarks{};

    for (auto &&[track_id, track] : bag->image_tracks_) {

      if (not track.valid_) {
        continue;
      }

      ++num_track;
      LOG(INFO) << fmt::format("track {}/{}", num_track, num_valid_tracks);

      const auto found_tracks{indexer_.find({bags_.size() - 1, track_id})};

      if (found_tracks.empty()) {
        landmarks_.add({bags_.size() - 1, track_id}, track.landmark_.value());

        LOG(INFO) << fmt::format(fmt::fg(fmt::color::coral), "added ")
                  << fmt::format(fmt::fg(fmt::color::light_green), "{}:{}",
                                 track_id, track.code_);
        continue;
      }

      for (auto &&res : found_tracks) {

        if (not check_intersecton(res)) {
          continue;
        }

        if (not check_closest_box(res)) {
          continue;
        }

        auto landmark{try_link(res)};

        if (landmark.has_value()) {
          std::string msg{};

          for (auto &&link : landmarks_.linked_bags(res.dst_link_)) {
            msg = fmt::format("{} {}:{}:{}", msg, link.bag_ind_, link.track_id_,
                              get_track(link).code_);
          }

          landmarks_.at(res.dst_link_) = landmark.value();
          landmarks_.link(res.src_link_, res.dst_link_);

          LOG(INFO) << fmt::format(fmt::fg(fmt::color::coral), "linked ")
                    << fmt::format(fmt::fg(fmt::color::light_green), "{}:{}:{}",
                                   res.src_link_.bag_ind_,
                                   res.src_link_.track_id_,
                                   get_track(res.src_link_).code_)
                    << " ->"
                    << fmt::format(fmt::fg(fmt::color::light_green), "{}", msg);

          affected_landmarks.push_back(
              landmarks_.landmark_index(res.dst_link_));
        }
      }
    }

    combine_landmarks(affected_landmarks);
    log_current_state();
  }

  void combine_landmarks(std::span<const size_t> affected_landmarks) {

    std::unordered_set<size_t> processed_landmarks{};

    for (auto &&landmark_ind : affected_landmarks) {

      if (processed_landmarks.contains(landmark_ind)) {
        continue;
      }

      std::unordered_set<size_t> landmarks_to_combine{};

      std::deque<size_t> q{};
      q.push_back(landmark_ind);
      processed_landmarks.insert(landmark_ind);
      landmarks_to_combine.insert(landmark_ind);

      while (not q.empty()) {
        const auto id{q.front()};
        q.pop_front();

        const auto links{landmarks_.linked_bags(id)};

        for (auto &&[bag_ind, track_id] : links) {

          std::deque<size_t> q2{};
          std::unordered_set<size_t> processed_tracks{};

          q2.push_back(track_id);
          processed_tracks.insert(track_id);

          while (not q2.empty()) {

            const auto id{q2.front()};
            q2.pop_front();

            for (auto &&linked_id :
                 bags_[bag_ind]->image_tracks_.at(id).linked_tracks_) {

              if (processed_tracks.contains(linked_id)) {
                continue;
              }

              q2.push_back(linked_id);
              processed_tracks.insert(linked_id);

              if (landmarks_.contain({bag_ind, linked_id})) {
                const auto index{
                    landmarks_.landmark_index({bag_ind, linked_id})};

                if (processed_landmarks.contains(index)) {
                  continue;
                }

                landmarks_to_combine.insert(index);
                processed_landmarks.insert(index);
                q.push_back(index);
              }
            }
          }
        }
      }

      if (landmarks_to_combine.size() > 1) {

        Eigen::Vector2d mean_enu{Eigen::Vector2d::Zero()};
        double mean_azimuth{0.0};
        double norm{0.0};

        for (auto &&id : landmarks_to_combine) {
          const auto var{landmarks_.landmarks_[id].dist_variance_};
          mean_enu += landmarks_.landmarks_[id].enu_ / var;
          mean_azimuth += landmarks_.landmarks_[id].azimuth_ / var;
          norm += 1.0 / var;
        }

        const double dist_variance{1.0 / norm};
        mean_enu *= dist_variance;
        mean_azimuth *= dist_variance;

        const auto mean_lla{converter_.latlon(mean_enu)};

        for (auto &&id : landmarks_to_combine) {
          landmarks_.landmarks_[id].enu_ = mean_enu;
          landmarks_.landmarks_[id].latlon_ = mean_lla;
          landmarks_.landmarks_[id].azimuth_ = mean_azimuth;
          landmarks_.landmarks_[id].dist_variance_ = dist_variance;
        }
      }
    }
  }

  std::optional<Landmark> try_link(const TrackIndexer::Result &res) const {

    if (not landmarks_.contain(res.dst_link_)) {
      return std::nullopt;
    }

    std::vector<TrackPoint> track_points{};

    for (auto &&[bag_ind, track_id] : landmarks_.linked_bags(res.dst_link_)) {
      const auto track_points_{
          bags_[bag_ind]->image_tracks_.at(track_id).selected_track_points()};

      track_points.insert(track_points.end(), track_points_.begin(),
                          track_points_.end());
    }

    {
      const auto track_points_{bags_[res.src_link_.bag_ind_]
                                   ->image_tracks_.at(res.src_link_.track_id_)
                                   .selected_track_points()};

      track_points.insert(track_points.end(), track_points_.begin(),
                          track_points_.end());
    }

    return triangulate_on_boxes(track_points);
  }

  bool check_landmarks_proximity(Link link, const ImageTrack &track) const {

    if (not landmarks_.contain(link)) {
      return false;
    }

    if (not track.landmark_.has_value()) {
      return false;
    }

    return (landmarks_.at(link).enu_ - track.landmark_->enu_).norm() < 20.0;
  }

  bool check_intersecton(const TrackIndexer::Result &res) {

    std::vector<cv::Point2f> points{};

    const auto &dst_track{bags_[res.dst_link_.bag_ind_]->image_tracks_.at(
        res.dst_link_.track_id_)};

    const auto &src_track{bags_[res.src_link_.bag_ind_]->image_tracks_.at(
        res.src_link_.track_id_)};

    for (auto &&[dst_det_ind, src_det_ind] : res.det_inds_) {

      const auto p0{dst_track.dets_[dst_det_ind].enu_.value()};
      const auto p1{src_track.dets_[src_det_ind].enu_.value()};

      points.push_back(
          {static_cast<float>(p0.x()), static_cast<float>(p0.y())});
      points.push_back(
          {static_cast<float>(p1.x()), static_cast<float>(p1.y())});
    }

    const auto rect{cv::minAreaRect(points)};
    const auto max_dimension{std::max(rect.size.width, rect.size.height)};

    const auto dst_length_ratio{max_dimension / dst_track.length_};
    const auto src_length_ratio{max_dimension / src_track.length_};

    const bool track_intersection{dst_length_ratio >= 0.5 and
                                  src_length_ratio >= 0.5};

    return track_intersection;
  }

  bool check_closest_box(const TrackIndexer::Result &res) {

    const auto &dst_track{bags_[res.dst_link_.bag_ind_]->image_tracks_.at(
        res.dst_link_.track_id_)};
    const auto &src_track{bags_[res.src_link_.bag_ind_]->image_tracks_.at(
        res.src_link_.track_id_)};

    const auto &dst_dets{dst_track.dets_};
    const auto &src_dets{src_track.dets_};

    const auto dst_bag{bags_[res.dst_link_.bag_ind_]};
    const auto src_bag{bags_[res.src_link_.bag_ind_]};

    const auto &dst_calib{dst_track.calib_};
    const auto &src_calib{src_track.calib_};

    size_t num_ambiguites{0};
    std::vector<bool> requires_cheking{};

    for (const auto &[dst_det_ind, src_det_ind] : res.det_inds_) {

      const auto &this_timestamp_dets{
          dst_bag->image_detections_.at(dst_dets[dst_det_ind].timestamp_)
              .dets_};

      size_t num_candidates{0};
      for (auto &d : this_timestamp_dets) {

        if (d->code_ == src_track.code_) {
          ++num_candidates;
        }
      }

      if (num_candidates > 1) {
        ++num_ambiguites;
      }

      if (dst_bag->selected_frames_.contains(
              dst_dets[dst_det_ind].image_id_) and
          src_bag->selected_frames_.contains(src_dets[src_det_ind].image_id_)) {
        requires_cheking.push_back(num_candidates > 1);
      } else {
        requires_cheking.push_back(false);
      }
    }

    const auto ambiguites_ratio{static_cast<float>(num_ambiguites) /
                                static_cast<float>(res.det_inds_.size())};

    if (ambiguites_ratio <= 0.2f) {
      return true;
    }

    std::unordered_set<std::pair<size_t, size_t>, decltype([](const auto &p) {
                         size_t seed{0};
                         boost::hash_combine(seed, p.first);
                         boost::hash_combine(seed, p.second);
                         return seed;
                       })>
        taken{};

    size_t too_far_away{0};

    for (auto &&[dst_det_ind, src_det_ind] : res.det_inds_) {

      const auto x_dist{std::abs(dst_dets[dst_det_ind].center_undistorted_.x /
                                     dst_calib.camera_resolution_.x() -
                                 src_dets[src_det_ind].center_undistorted_.x /
                                     src_calib.camera_resolution_.x())};

      if (x_dist > 0.3) {
        ++too_far_away;
      }
    }

    const auto too_faw_away_ratio{static_cast<float>(too_far_away) /
                                  static_cast<float>(res.det_inds_.size())};

    if (too_faw_away_ratio > 0.7f) {
      return false;
    }

    size_t num_mismatches{0};
    size_t num_checked{0};

    for (auto &&[det_pair, required] : zip(res.det_inds_, requires_cheking)) {

      if (not required) {
        continue;
      }

      const auto [dst_det_ind, src_det_ind] = det_pair;

      const auto dst_image_id{dst_dets[dst_det_ind].image_id_};
      const auto src_image_id{src_dets[src_det_ind].image_id_};

      const auto tag1{
          fmt::format("bag_{}_image_{}", res.dst_link_.bag_ind_, dst_image_id)};

      const auto tag2{
          fmt::format("bag_{}_image_{}", res.src_link_.bag_ind_, src_image_id)};

      if (not dst_bag->selected_frames_.contains(dst_image_id)) {
        LOG(WARNING) << fmt::format("image {} not found", dst_image_id);
        continue;
      }

      if (not src_bag->selected_frames_.contains(src_image_id)) {
        LOG(WARNING) << fmt::format("image {} not found", src_image_id);
        continue;
      }

      if (matcher_.estimate_homography(
              dst_bag->selected_frames_.at(dst_image_id), dst_calib, tag1,
              src_bag->selected_frames_.at(src_image_id), src_calib, tag2)) {

        const auto this_timestamp_dets{
            get_synced_detections(res.dst_link_, dst_dets[dst_det_ind])};

        const auto src_center{
            cv::Vec2f{src_dets[src_det_ind].center_undistorted_}};

        std::vector<cv::Point2f> dst_points{};
        std::vector<size_t> track_ids;

        for (auto &d : this_timestamp_dets) {
          if (d->code_ == src_track.code_) {
            dst_points.push_back(d->center_undistorted_);
            track_ids.push_back(d->track_id_);
          }
        }

        const auto projected_points{
            matcher_.warp_points(dst_points, tag1, tag2)};

        size_t closest_track_id{0};
        double min_dist{std::numeric_limits<double>::max()};

        for (auto &&[p, id] : zip(projected_points, track_ids)) {
          const auto dist{cv::norm(cv::Vec2f{p}, src_center)};

          if (dist < min_dist) {
            min_dist = dist;
            closest_track_id = id;
          }
        }

        if (closest_track_id != res.dst_link_.track_id_) {
          ++num_mismatches;
        }

        ++num_checked;
      }
    }

    LOG(INFO) << fmt::format("num mistmatches: {}, num checked: {}",
                             num_mismatches, num_checked);

    if (num_checked == 0) {
      return false;
    }

    const auto mismatches_ratio{static_cast<float>(num_mismatches) /
                                static_cast<float>(num_checked)};

    return mismatches_ratio < 0.2f;
  }

  void log_current_state() const {
#if USE_RERUN
    if (rec_) {

      rerun::Collection<rerun::components::LatLon> ss;

      std::vector<rerun::components::GeoLineString> segments{};
      std::vector<rerun::components::LatLon> points{};

      for (auto &&[landmark_ind, landmark] :
           landmarks_.landmarks_ | enumerate) {

        for (auto &&link : landmarks_.landmark_to_bag_.at(landmark_ind)) {

          segments.emplace_back(rerun::components::GeoLineString::from_lat_lon(
              {to_lat_lon(landmark.latlon_),
               to_lat_lon(landmarks_.at(link).latlon_)}));
        }

        points.emplace_back(to_lat_lon(landmark.latlon_));
      }

      rec_->log("map/links", rerun::GeoLineStrings{segments}
                                 .with_colors(to_color("rgba(218, 123, 16, 1)"))
                                 .with_radii(rerun::Radius::ui_points(2.0f)));

      rec_->log("map/points", rerun::GeoPoints{points}
                                  .with_colors(to_color("rgba(37, 1, 201, 1)"))
                                  .with_radii(rerun::Radius::ui_points(5.0f)));

      [[maybe_unused]] const auto err{rec_->flush_blocking()};
    }
#endif
  }

  ImageTrack &get_track(const Link &link) {
    return bags_[link.bag_ind_]->image_tracks_.at(link.track_id_);
  }

  const ImageTrack &get_track(const Link &link) const {
    return bags_[link.bag_ind_]->image_tracks_.at(link.track_id_);
  }

  std::span<Detection *> get_synced_detections(const Link &link,
                                               const Detection &det) {
    return bags_[link.bag_ind_]->image_detections_.at(det.timestamp_).dets_;
  }

  std::vector<BagProcessor::ptr> bags_;
  CartesianConverter converter_;
  CombinedLandmarks landmarks_;
  FeatureMatcher matcher_;
#if USE_RERUN
  std::shared_ptr<rerun::RecordingStream> rec_;
#endif
};

TracksCollection::TracksCollection() : pimpl_{std::make_unique<impl>()} {}
TracksCollection::~TracksCollection() = default;

void TracksCollection::merge(BagProcessor::ptr bag) { pimpl_->merge(bag); }

#if USE_RERUN
void TracksCollection::set_rerun(std::shared_ptr<rerun::RecordingStream> rec) {
  pimpl_->rec_ = std::move(rec);
}
#endif