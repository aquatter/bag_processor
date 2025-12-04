#include <Eigen/Core>
#include <GeographicLib/LocalCartesian.hpp>
#include <algorithm>
#include <bag_loader.hpp>
#include <boost/math/constants/constants.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <flann/flann.hpp>
#include <fmt/color.h>
#include <fmt/core.h>
#include <limits>
#include <memory>
#include <ng-log/logging.h>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <queue>
#include <range/v3/algorithm/max.hpp>
#include <range/v3/algorithm/minmax.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/enumerate.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/reverse.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/view/zip.hpp>
#include <rerun.hpp>
#include <rerun_logging.hpp>
#include <span>
#include <tracks_collecton.hpp>
#include <triangulation.hpp>
#include <types.hpp>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using cv::Vec2f;
using ranges::to;
using ranges::views::enumerate;
using ranges::views::filter;
using ranges::views::reverse;
using ranges::views::transform;
using ranges::views::zip;

struct GroupedDetections {
  using map_type = std::unordered_map<int64_t, GroupedDetections>;

  static map_type collect_detections(const ImageTrack::map_type &tracks) {

    map_type res{};

    for (auto &&[track_id, track] : tracks) {
      for (auto &&det : track.dets_) {
        res[det.timestamp_].dets_[det.code_].push_back(&det);
      }
    }

    return res;
  }

  std::unordered_map<std::string, std::vector<const Detection *>> dets_;
};

constexpr double constexpr_cos(double x) {
  double cos{1.0};
  double pow{x};

  for (auto fac{1ull}, n{1ull}; n != 19; fac *= ++n, pow *= x) {
    if ((n & 1) == 0)
      cos += (n & 2 ? -pow : pow) / fac;
  }

  return cos;
}

std::tuple<size_t, size_t, double>
start_closest_poses(const ImageTrack &track1, const ImageTrack &track2) {

  std::tuple<size_t, size_t, double> res{0, 0, 0.0};

  for (auto &&[i1, d1] : track1.dets_ | enumerate) {

    if (not d1.enu_.has_value()) {
      continue;
    }

    std::get<0>(res) = i1;

    double min_dist{std::numeric_limits<double>::max()};
    size_t min_ind{0};

    for (auto &&[i2, d2] : track2.dets_ | enumerate) {

      if (not d2.enu_.has_value()) {
        continue;
      }

      const auto dist{(d1.enu_.value() - d2.enu_.value()).squaredNorm()};

      if (min_dist > dist) {
        min_dist = dist;
        min_ind = i2;
      }
    }

    std::get<1>(res) = min_ind;
    std::get<2>(res) = min_dist;
    break;
  }

  return res;
}

std::tuple<size_t, size_t, double> end_closest_poses(const ImageTrack &track1,
                                                     const ImageTrack &track2) {

  std::tuple<size_t, size_t, double> res{0, 0, 0.0};

  for (auto &&[i1, d1] : track1.dets_ | enumerate | reverse) {

    if (not d1.enu_.has_value()) {
      continue;
    }

    std::get<0>(res) = i1;

    double min_dist{std::numeric_limits<double>::max()};
    size_t min_ind{0};

    for (auto &&[i2, d2] : track2.dets_ | enumerate) {

      if (not d2.enu_.has_value()) {
        continue;
      }

      const auto dist{(d1.enu_.value() - d2.enu_.value()).squaredNorm()};
      if (min_dist > dist) {
        min_dist = dist;
        min_ind = i2;
      }
    }

    std::get<1>(res) = min_ind;
    std::get<2>(res) = min_dist;
    break;
  }

  return res;
}

size_t find_closest_index(Eigen::Vector2d p0, const ImageTrack &track) {
  double min_dist{std::numeric_limits<double>::max()};
  size_t min_ind{0};

  for (auto &&[i, d] : track.dets_ | enumerate) {

    if (not d.enu_.has_value()) {
      continue;
    }

    const auto dist{(d.enu_.value() - p0).squaredNorm()};

    if (min_dist > dist) {
      min_dist = dist;
      min_ind = i;
    }
  }

  return min_ind;
}

std::pair<size_t, size_t>
find_closest_indices_both_direction(size_t index1, const ImageTrack &track1,
                                    const ImageTrack &track2) {

  const auto index2{
      find_closest_index(track1.dets_[index1].enu_.value(), track2)};

  const auto dist1{
      (track1.dets_[index1].enu_.value() - track2.dets_[index2].enu_.value())
          .squaredNorm()};

  const auto index1_prime{
      find_closest_index(track2.dets_[index2].enu_.value(), track1)};

  const auto dist2{(track1.dets_[index1_prime].enu_.value() -
                    track2.dets_[index2].enu_.value())
                       .squaredNorm()};

  if (dist1 < dist2) {
    return {index1, index2};
  }

  return {index1_prime, index2};
}

cv::Rect find_closest_box(Eigen::Vector2d p0, const ImageTrack &track) {
  return track.dets_[find_closest_index(p0, track)].box_;
}

cv::Point2f find_closest_center(Eigen::Vector2d p0, const ImageTrack &track) {
  return track.dets_[find_closest_index(p0, track)].center_undistorted_;
}

std::pair<size_t, size_t> start_indices(const ImageTrack &track1,
                                        const ImageTrack &track2) {

  const auto [ind1, ind2, d1] = start_closest_poses(track1, track2);
  const auto [ind3, ind4, d2] = start_closest_poses(track2, track1);

  if (d1 < d2) {
    return {ind1, ind2};
  }

  return {ind4, ind3};
}

std::pair<size_t, size_t> end_indices(const ImageTrack &track1,
                                      const ImageTrack &track2) {

  const auto [ind1, ind2, d1] = end_closest_poses(track1, track2);
  const auto [ind3, ind4, d2] = end_closest_poses(track2, track1);

  if (d1 < d2) {
    return {ind1, ind2};
  }

  return {ind4, ind3};
}

std::array<size_t, 4>
start_end_indices(const ImageTrack &track1, const ImageTrack &track2,
                  const std::unordered_set<size_t> &track1_dets) {

  auto [s1, e1] = ranges::minmax(track1_dets);

  auto p1{track1.dets_[s1].enu_.value()};
  auto p2{track1.dets_[e1].enu_.value()};

  double min_dist1{std::numeric_limits<double>::max()};
  double min_dist2{std::numeric_limits<double>::max()};
  size_t s2{0};
  size_t e2{0};

  for (auto &&[i, d] : enumerate(track2.dets_)) {
    if (not d.enu_.has_value()) {
      continue;
    }

    const double dist1{(d.enu_.value() - p1).squaredNorm()};
    const double dist2{(d.enu_.value() - p2).squaredNorm()};

    if (dist1 < min_dist1) {
      min_dist1 = dist1;
      s2 = i;
    }

    if (dist2 < min_dist2) {
      min_dist2 = dist2;
      e2 = i;
    }
  }

  min_dist1 = std::numeric_limits<double>::max();
  min_dist2 = std::numeric_limits<double>::max();

  p1 = track2.dets_[s2].enu_.value();
  p2 = track2.dets_[e2].enu_.value();

  for (auto &&[i, d] : enumerate(track1.dets_)) {
    if (not d.enu_.has_value()) {
      continue;
    }

    const double dist1{(d.enu_.value() - p1).squaredNorm()};
    const double dist2{(d.enu_.value() - p2).squaredNorm()};

    if (dist1 < min_dist1) {
      min_dist1 = dist1;
      s1 = i;
    }

    if (dist2 < min_dist2) {
      min_dist2 = dist2;
      e1 = i;
    }
  }

  return {s1, e1, s2, e2};
}

struct MinMaxAccumulator {

  void add(double val) noexcept {
    min_ = std::min(min_, val);
    max_ = std::max(max_, val);
  }

  double delta() const noexcept { return max_ - min_; }

  double min_{std::numeric_limits<double>::max()};
  double max_{std::numeric_limits<double>::min()};
};

bool TracksCollection::should_be_linked(CombinedLandmarks::Link dst_link,
                                        std::span<const size_t> det_ind,
                                        size_t src_track_id) {

  auto dst_bag{bags_[dst_link.bag_ind_]};
  const auto &dst_track{dst_bag->image_tracks_.at(dst_link.track_id_)};
  const auto &dst_dets{dst_track.dets_};
  const auto &src_track{bags_.back()->image_tracks_.at(src_track_id)};
  const auto &src_dets{src_track.dets_};

  MinMaxAccumulator dst_minmax{};
  MinMaxAccumulator src_minmax{};

  if (not check_proximity(dst_link, src_track)) {
    return false;
  }

  if (not check_closest_box_and_intersecton(dst_link, det_ind, src_track_id)) {
    return false;
  }

  auto landmark{try_link(dst_link, src_track)};

  if (landmark.has_value()) {

#if 0
    if (src_track_id == 98 or src_track_id == 99) {
      BagLoader src_loader{BagLoader::Settings{
          .compressed_image_topic_ = bags_.back()->set_.compressed_image_topic_,
          .path_to_bag_ = bags_.back()->set_.bag_path_,
          .timestamp_delta_ = bags_.back()->set_.camera_gps_delta_,
          .rec_ = {}}};

      const auto src_det_ind{src_dets.size() >> 1};

      src_loader.dump_detection(
          fmt::format("/root/data/images/{}_src.png", src_track_id),
          src_track.dets_[src_det_ind]);

      BagLoader dst_loader{BagLoader::Settings{
          .compressed_image_topic_ = dst_bag->set_.compressed_image_topic_,
          .path_to_bag_ = dst_bag->set_.bag_path_,
          .timestamp_delta_ = dst_bag->set_.camera_gps_delta_,
          .rec_ = {}}};

      const auto dst_det_ind{dst_dets.size() >> 1};

      dst_loader.dump_detection(
          fmt::format("/root/data/images/{}_dst.png", dst_link.track_id_),
          dst_track.dets_[dst_det_ind]);
    }
#endif

    // if (landmark->dist_variance_ < landmarks_.at(dst_link).dist_variance_)
    {
#if 0
      landmark->latlon_ = converter_.latlon(landmark->enu_);
      landmark->code_ = landmarks_.at(dst_link).code_;
      landmark->id_ = landmarks_.at(dst_link).id_ + src_track.id_;

      log_track_map(rec_, src_track, {255, 0, 0});
      log_track_map(rec_, dst_track, {0, 255, 0});

      log_landmark(rec_, src_track.landmark_.value(), {255, 0, 0});
      log_landmark(rec_, dst_track.landmark_.value(), {0, 255, 0});
      log_landmark(rec_, landmark.value(), {0, 0, 255}, 4.0f);

      log_segment(fmt::format("link_{}", src_track.landmark_.value().id_), rec_,
                  rerun::LatLon{src_track.landmark_.value().latlon_.x(),
                                src_track.landmark_.value().latlon_.y()},
                  rerun::LatLon{landmark.value().latlon_.x(),
                                landmark.value().latlon_.y()});

      log_segment(fmt::format("link_{}", dst_track.landmark_.value().id_), rec_,
                  rerun::LatLon{dst_track.landmark_.value().latlon_.x(),
                                dst_track.landmark_.value().latlon_.y()},
                  rerun::LatLon{landmark.value().latlon_.x(),
                                landmark.value().latlon_.y()});
#endif
      std::string msg{};

      for (auto &&[bag_ind, track_id] : landmarks_.linked_bags(dst_link)) {

        msg = fmt::format("{} {}:{}:{}", msg, bag_ind, track_id,
                          bags_[bag_ind]->image_tracks_.at(track_id).code_);
      }

      landmarks_.at(dst_link) = landmark.value();
      landmarks_.link({bags_.size() - 1, src_track_id}, dst_link);

      LOG(INFO) << fmt::format(fmt::fg(fmt::color::coral), "linked ")
                << fmt::format(fmt::fg(fmt::color::light_green), "{}:{}",
                               src_track.id_, src_track.code_)
                << " ->"
                << fmt::format(fmt::fg(fmt::color::light_green), "{}", msg);

      return true;
    }
  }

  return false;

#if 0
  BagLoader dst_loader{BagLoader::Settings{
      .compressed_image_topic_ = "/camera/image_raw/compressed",
      .path_to_bag_ = dst_bag->set_.bag_path_,
      .timestamp_delta_ = dst_bag->set_.camera_gps_delta_,
      .rec_ = {}}};

  BagLoader src_loader{BagLoader::Settings{
      .compressed_image_topic_ = "/camera/image_raw/compressed",
      .path_to_bag_ = src_bag->set_.bag_path_,
      .timestamp_delta_ = src_bag->set_.camera_gps_delta_,
      .rec_ = {}}};



  {

    if (not src_track.landmark_.has_value()) {
      [[maybe_unused]] auto p{
          triangulate_on_boxes(src_track.selected_track_points())};
    }

    if (src_track.landmark_.has_value()) {
      log_landmark(rec_, src_track.landmark_.value(), {255, 0, 0});
    }

    if (dst_track.landmark_.has_value()) {
      log_landmark(rec_, dst_track.landmark_.value(), {0, 255, 0});
    }
  }

  // int not_closest_num{0};

  std::unordered_map<size_t, size_t> num_hits{};

  for (auto &&dst_det_ind : det_ind) {

    const auto [dst_ind, src_ind] =
        find_closest_indices_both_direction(dst_det_ind, dst_track, src_track);

    dst_minmax.add(dst_track.dets_[dst_ind].cumulative_length_);
    src_minmax.add(src_track.dets_[src_ind].cumulative_length_);

    const auto &this_timestamp_dets{
        dst_bag->image_detections_.at(dst_dets[dst_ind].timestamp_).dets_};
    {
      log_segment(rec_, src_track, dst_track, src_ind, dst_ind);
      log_detection(rec_, src_track, src_ind);
      log_detection(rec_, dst_track, dst_ind);
#if 0
      dst_loader.dump_detection("/root/data/images/dst.png", dst_dets[dst_ind]);
      src_loader.dump_detection("/root/data/images/src.png", src_dets[src_ind]);

#endif
    }

    const auto src_center{src_track.dets_[src_ind].center_undistorted_};

    // const auto src_center{find_closest_center(
    //     dst_dets[dst_det_ind].pose_.value().head<2>(), src_track)};

    double min_dist{std::numeric_limits<double>::max()};
    size_t closest_track{0};

    for (auto &d : this_timestamp_dets) {
      if (d->code_ == src_track.code_) {

        if (d->enu_.has_value()) {
          auto dst_center{d->center_undistorted_};

          const auto dist{cv::norm(Vec2f{dst_center}, cv::Vec2f{src_center})};

          if (dist < min_dist) {
            min_dist = dist;
            closest_track = d->track_id_;
          }
        }
      }
    }

    ++num_hits[closest_track];
  }

  const auto track_with_max_hits{
      std::max_element(
          num_hits.begin(), num_hits.end(),
          [](const auto &a, const auto &b) { return a.second < b.second; })
          ->first};

  const auto dst_length_ratio{dst_minmax.delta() / dst_track.length_};
  const auto src_length_ratio{src_minmax.delta() / src_track.length_};

  const bool track_intersection{dst_length_ratio >= 0.7 or
                                src_length_ratio >= 0.7};

  const bool closest_detection{track_with_max_hits == dst_link.track_id_};

  bool landmark_proximity{false};

  if (src_track.landmark_.has_value()) {
    if (landmarks_.contain(dst_link)) {
      landmark_proximity =
          ((landmarks_.at(dst_link).enu_ - src_track.landmark_->enu_).norm() <
           5.0);
    }
  }

  return (closest_detection or landmark_proximity) and track_intersection;
#endif
}

bool should_be_linked(const GroupedDetections::map_type &dst_detections,
                      const ImageTrack &track_dst, const ImageTrack &track_src,
                      const std::unordered_set<size_t> &track_dst_dets) {

  const float norm_w1{
      1.0f / static_cast<float>(track_dst.calib_.camera_resolution_.x())};

  const float norm_h1{
      1.0f / static_cast<float>(track_dst.calib_.camera_resolution_.y())};

  const float norm_w2{
      1.0f / static_cast<float>(track_src.calib_.camera_resolution_.x())};

  const float norm_h2{
      1.0f / static_cast<float>(track_src.calib_.camera_resolution_.y())};

  for (auto &&track_dst_det_ind : track_dst_dets) {

    const auto dst_det_timestamp{track_dst.dets_[track_dst_det_ind].timestamp_};

    const auto total_num_detections{
        dst_detections.at(dst_det_timestamp).dets_.size()};

    const auto num_detections{
        dst_detections.at(dst_det_timestamp).dets_.at(track_dst.code_).size()};

    if (num_detections > 1) {

      auto src_center{find_closest_center(
          track_dst.dets_[track_dst_det_ind].enu_.value(), track_src)};

      src_center.x *= norm_w2;
      src_center.y *= norm_h2;

      double min_dist{std::numeric_limits<double>::max()};
      size_t closest_track{0};

      for (auto &&d :
           dst_detections.at(track_dst.dets_[track_dst_det_ind].timestamp_)
               .dets_.at(track_dst.code_)) {

        auto dst_center{d->center_undistorted_};
        dst_center.x *= norm_w1;
        dst_center.y *= norm_h1;

        const auto dist{
            cv::norm(Vec2f{d->center_undistorted_}, cv::Vec2f{src_center})};

        if (dist < min_dist) {
          min_dist = dist;
          closest_track = d->track_id_;
        }
      }

      if (closest_track != track_dst.id_) {
        return false;
      }
    }
  }

  return true;
}

class TrackIndexer {
public:
  struct GeoPoint {
    size_t bag_index_;
    size_t track_id_;
    size_t detection_index_;
    Eigen::Vector2d enu_;
  };

  struct Result {

    struct Bag {
      struct Track {
        std::unordered_set<size_t> dets_;
      };

      std::unordered_map<size_t, Track> tracks_;
    };

    std::unordered_map<size_t, Bag> bags_;
  };

  struct FlattenedResult {
    CombinedLandmarks::Link link_;
    std::vector<size_t> dets_;
  };

  TrackIndexer(const std::vector<BagProcessor::ptr> &bags)
      : bags_{bags}, indexer_{flann::KDTreeSingleIndexParams{}} {

    for (auto &&[bag_ind, bag] : bags_ | enumerate) {
      for (auto &&[track_id, track] : bag->image_tracks_) {
        if (not track.valid_) {
          continue;
        }

        for (auto &&[det_ind, d] : enumerate(track.dets_)) {

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

  std::vector<FlattenedResult> find(const ImageTrack &track) {

    std::vector<size_t> other_track_indices{};
    other_track_indices.reserve(track.dets_.size());

    for (auto &&[i, d] : enumerate(track.dets_)) {

      if (not d.enu_.has_value()) {
        continue;
      }

      other_track_indices.push_back(i);
    }

    auto query_data{track.dets_ |
                    filter([](const auto &d) { return d.enu_.has_value(); }) |
                    transform([](const auto &d) {
                      return std::pair{d.enu_.value().x(), d.enu_.value().y()};
                    }) |
                    to<std::vector>()};

    const flann::Matrix<double> query_dataset{&query_data.front().first,
                                              query_data.size(), 2ul};

    std::vector<std::vector<int>> indices{};
    std::vector<std::vector<double>> distances{};

    indexer_.radiusSearch(query_dataset, indices, distances, search_rad_sqr_,
                          flann::SearchParams{});

    Result found_tracks{};

    for (auto &&[det_ind, data_ind] : enumerate(indices)) {

      for (auto &i : data_ind) {

        const auto &det{detection(i)};

        if (det.code_ != track.code_) {
          continue;
        }

        const Eigen::Vector2d dir1{det.direction_.value()};
        const Eigen::Vector2d dir2{
            track.dets_[other_track_indices[det_ind]].direction_.value()};

        const auto dot_product{dir1.dot(dir2)};

        if (dot_product < cos_angle_threshold_) {
          continue;
        }

        found_tracks.bags_[points_[i].bag_index_]
            .tracks_[points_[i].track_id_]
            .dets_.insert(points_[i].detection_index_);
      }
    }

    std::vector<FlattenedResult> res{};

    for (auto &&[bag_ind, bag] : found_tracks.bags_) {
      for (auto &&[track_id, track_] : bag.tracks_) {

        FlattenedResult r{.link_ = {bag_ind, track_id},
                          .dets_ = {track_.dets_.begin(), track_.dets_.end()}};

        std::sort(r.dets_.begin(), r.dets_.end());
        res.push_back(std::move(r));
      }
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

void TracksCollection::init(BagProcessor::ptr bag) {
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

void TracksCollection::recalculate_coords(BagProcessor::ptr bag) {

  for (auto &&[track_id, track] : bag->image_tracks_) {
    for (auto &&d : track.dets_) {

      if (not d.enu_.has_value()) {
        continue;
      }

      const Eigen::Vector2d enu{
          converter_.enu(bag->local_converter_.latlon(d.enu_.value()))};

      d.enu_ = enu;

      if (d.cam_to_world_.has_value()) {
        d.cam_to_world_->translation() = Eigen::Vector3d{enu.x(), enu.y(), 0.0};
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

void TracksCollection::merge(BagProcessor::ptr bag) {

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

    const auto found_tracks{indexer_.find(track)};

    if (found_tracks.empty()) {
      landmarks_.add({bags_.size() - 1, track_id}, track.landmark_.value());

      LOG(INFO) << fmt::format(fmt::fg(fmt::color::coral), "added ")
                << fmt::format(fmt::fg(fmt::color::light_green), "{}:{}",
                               track_id, track.code_);
      continue;
    }

    for (auto &&res : found_tracks) {
      if (should_be_linked(res.link_, res.dets_, track_id)) {
        affected_landmarks.push_back(landmarks_.landmark_index(res.link_));
        break;
      }
    }
  }

  combine_landmarks(affected_landmarks);

  log_current_state();
}

void TracksCollection::combine_landmarks(
    std::span<const size_t> affected_landmarks) {

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
              const auto index{landmarks_.landmark_index({bag_ind, linked_id})};

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

std::optional<Landmark>
TracksCollection::try_link(CombinedLandmarks::Link link,
                           const ImageTrack &new_track) const {

  if (not landmarks_.contain(link)) {
    return std::nullopt;
  }

  std::vector<TrackPoint> track_points{};

  for (auto &&[bag_ind, track_id] : landmarks_.linked_bags(link)) {
    const auto track_points_{
        bags_[bag_ind]->image_tracks_.at(track_id).selected_track_points()};

    track_points.insert(track_points.end(), track_points_.begin(),
                        track_points_.end());
  }

  {
    const auto track_points_{new_track.selected_track_points()};
    track_points.insert(track_points.end(), track_points_.begin(),
                        track_points_.end());
  }

  return triangulate_on_boxes(track_points);
}

bool TracksCollection::check_proximity(CombinedLandmarks::Link link,
                                       const ImageTrack &track) const {

  if (not landmarks_.contain(link)) {
    return false;
  }

  if (not track.landmark_.has_value()) {
    return false;
  }

  return (landmarks_.at(link).enu_ - track.landmark_->enu_).norm() < 20.0;
}

bool TracksCollection::check_closest_box_and_intersecton(
    CombinedLandmarks::Link dst_link, std::span<const size_t> det_ind,
    size_t src_track_id) const {

  auto dst_bag{bags_[dst_link.bag_ind_]};
  const auto &dst_track{dst_bag->image_tracks_.at(dst_link.track_id_)};
  const auto &dst_dets{dst_track.dets_};
  const auto &src_track{bags_.back()->image_tracks_.at(src_track_id)};
  const auto &src_dets{src_track.dets_};

  MinMaxAccumulator dst_minmax{};
  MinMaxAccumulator src_minmax{};

  // std::shared_ptr<BagLoader> dst_loader{};
  // std::unordered_map<size_t, size_t> num_hits{};

  std::unordered_set<std::pair<size_t, size_t>, decltype([](const auto &p) {
                       size_t seed{0};
                       boost::hash_combine(seed, p.first);
                       boost::hash_combine(seed, p.second);
                       return seed;
                     })>
      taken{};

  for (auto &&dst_det_ind : det_ind) {

    const auto [dst_ind, src_ind] =
        find_closest_indices_both_direction(dst_det_ind, dst_track, src_track);

    if (taken.contains({src_ind, dst_ind})) {
      continue;
    }

    taken.insert({src_ind, dst_ind});

#if 0
    {
      if ((src_track_id == 98 or src_track_id == 99) and
          dst_link.track_id_ == 80) {

        dst_loader = std::make_shared<BagLoader>(BagLoader::Settings{
            .compressed_image_topic_ = dst_bag->set_.compressed_image_topic_,
            .path_to_bag_ = dst_bag->set_.bag_path_,
            .timestamp_delta_ = dst_bag->set_.camera_gps_delta_,
            .rec_ = {}});

        auto img = dst_loader->load_image(dst_dets[dst_ind].timestamp_ -
                                          dst_bag->set_.camera_gps_delta_);

        cv::rectangle(img, dst_dets[dst_ind].box_, cv::Scalar(0, 255, 0), 2);
        cv::rectangle(img, src_dets[src_ind].box_, cv::Scalar(255, 0, 0), 2);

        cv::imwrite(fmt::format("/root/data/images/{}_{}_{}_{}.png",
                                src_track_id, dst_link.track_id_, src_ind,
                                dst_ind),
                    img);
      }
    }
#endif

    const auto &this_timestamp_dets{
        dst_bag->image_detections_.at(dst_dets[dst_ind].timestamp_).dets_};

    const auto src_center{src_dets[src_ind].center_undistorted_};

    double min_dist{std::numeric_limits<double>::max()};
    size_t closest_track{0};

    for (auto &d : this_timestamp_dets) {
      if (d->code_ == src_track.code_) {

        if (d->enu_.has_value()) {
          const auto dist{
              cv::norm(Vec2f{d->center_undistorted_}, cv::Vec2f{src_center})};

          if (dist < min_dist) {
            min_dist = dist;
            closest_track = d->track_id_;
          }
        }
      }
    }

    dst_minmax.add(dst_track.dets_[dst_ind].cumulative_length_);
    src_minmax.add(src_track.dets_[src_ind].cumulative_length_);

    if (closest_track != dst_link.track_id_) {
      return false;
    }

    // ++num_hits[closest_track];
  }

  const auto dst_length_ratio{dst_minmax.delta() / dst_track.length_};
  const auto src_length_ratio{src_minmax.delta() / src_track.length_};

  const bool track_intersection{dst_length_ratio >= 0.5 and
                                src_length_ratio >= 0.5};

  return track_intersection;

  // const auto track_with_max_hits{
  //     std::max_element(
  //         num_hits.begin(), num_hits.end(),
  //         [](const auto &a, const auto &b) { return a.second < b.second; })
  //         ->first};

  // return track_with_max_hits == dst_link.track_id_;
}

void TracksCollection::log_current_state() const {

  if (rec_) {

    rerun::Collection<rerun::components::LatLon> ss;

    std::vector<rerun::components::GeoLineString> segments{};
    std::vector<rerun::components::LatLon> points{};

    for (auto &&[landmark_ind, landmark] : landmarks_.landmarks_ | enumerate) {

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
}
