#include <Eigen/Core>
#include <GeographicLib/LocalCartesian.hpp>
#include <algorithm>
#include <bag_loader.hpp>
#include <boost/math/constants/constants.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <flann/flann.hpp>
#include <limits>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <range/v3/algorithm/max.hpp>
#include <range/v3/algorithm/minmax.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/enumerate.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/reverse.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/view/zip.hpp>
#include <rerun_logging.hpp>
#include <span>
#include <tracks_collecton.hpp>
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

bool TracksCollection::should_be_linked(
    size_t bag_index, size_t track_id,
    const std::unordered_set<size_t> &det_ind, BagProcessor::ptr src_bag,
    size_t src_bag_id) const {

  auto dst_bag{bags_[bag_index]};
  const auto &dst_track{dst_bag->image_tracks_.at(track_id)};
  const auto &dst_dets{dst_track.dets_};
  const auto &src_track{src_bag->image_tracks_.at(src_bag_id)};
  const auto &src_dets{src_track.dets_};

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
    log_track_map(rec_, src_track, {255, 0, 0});
    log_track_map(rec_, dst_track, {0, 255, 0});
  }

#endif

  // int not_closest_num{0};

  std::unordered_map<size_t, size_t> num_hits{};

  for (auto &&dst_det_ind : det_ind) {

    const auto [dst_ind, src_ind] =
        find_closest_indices_both_direction(dst_det_ind, dst_track, src_track);

    const auto &this_timestamp_dets{
        dst_bag->image_detections_.at(dst_dets[dst_ind].timestamp_).dets_};
#if 0
    {
      dst_loader.dump_detection("/root/data/images/dst.png", dst_dets[dst_ind]);
      src_loader.dump_detection("/root/data/images/src.png", src_dets[src_ind]);

      log_segment(rec_, src_track, dst_track, src_ind, dst_ind);
      log_detection(rec_, src_track, src_ind);
      log_detection(rec_, dst_track, dst_ind);
    }
#endif

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

  return track_with_max_hits == track_id;
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

struct GeoPoint {
  size_t bag_index_;
  size_t track_id_;
  size_t detection_index_;
  Eigen::Vector2d enu_;
};

class TrackIndexer {
public:
  struct Result {

    struct Bag {
      struct Track {
        std::unordered_set<size_t> dets_;
      };

      std::unordered_map<size_t, Track> tracks_;
    };

    std::unordered_map<size_t, Bag> bags_;
  };

  TrackIndexer(std::span<const BagProcessor::ptr> bags)
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

  Result find(const ImageTrack &track) {

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

    return found_tracks;
  }

private:
  const Detection &detection(size_t i) const {
    return bags_[points_[i].bag_index_]
        ->image_tracks_.at(points_[i].track_id_)
        .dets_[points_[i].detection_index_];
  }

  std::span<const BagProcessor::ptr> bags_;
  flann::Index<flann::L2_Simple<double>> indexer_;
  std::vector<GeoPoint> points_;
  static constexpr float search_rad_sqr_{15.0f * 15.0f};
  static constexpr double cos_angle_threshold_{
      constexpr_cos(boost::math::double_constants::degree * 15.0)};
};

void TracksCollection::init(BagProcessor::ptr bag) {
  converter_ = bag->local_converter_;
  bags_.push_back(bag);
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
      d.cam_to_world_->translation() = Eigen::Vector3d{enu.x(), enu.y(), 0.0};
    }

    track.geodetic_origin_ = converter_.origin();
  }

  for (auto &&gps : bag->gps_) {
    gps.enu_ = converter_.enu(gps.latlon_);
  }

  for (auto &&landmark : bag->found_landmarks_) {
    landmark.enu_ = converter_.enu(landmark.latlon_);
  }
}

void TracksCollection::merge(BagProcessor::ptr bag) {

  if (bags_.empty()) {
    init(bag);
    return;
  }

  recalculate_coords(bag);

  TrackIndexer indexer_{bags_};

  std::unordered_map<size_t, std::vector<size_t>> should_be_added{};
  ImageTrack::map_type other_tracks_copy{};

  for (auto &&[track_id, track] : bag->image_tracks_) {

    if (not track.valid_) {
      continue;
    }

    const auto found_tracks{indexer_.find(track)};

    for (auto &&[bag_ind_, bag_] : found_tracks.bags_) {
      for (auto &&[track_ind_, track_] : bag_.tracks_) {
        should_be_linked(bag_ind_, track_ind_, track_.dets_, bag, track_id);
      }
    }
  }

#if 0 
  for (auto &&[other_track_id, other_track] : other_tracks) {

    ImageTrack track_copy{other_track};
    recalculate_coords(track_copy);

    const auto found_tracks{indexer_.query(track_copy)};

    if (found_tracks.empty()) {
      should_be_added[other_track_id] = {};
    }

    for (auto &&[track_id, dets_ind] : found_tracks) {
      if (should_be_linked(image_detections, tracks_.at(track_id), track_copy,
                           dets_ind)) {
        should_be_added[other_track_id].push_back(track_id);
      }
    }

    if (should_be_added.contains(other_track_id)) {
      other_tracks_copy[other_track_id] = std::move(track_copy);
    }
  }

  std::unordered_map<size_t, size_t> id_remapping;

  for (auto &&[other_track_id, other_track] : other_tracks_copy) {

    id_remapping[other_track_id] = last_track_id_;
    other_track.id_ = last_track_id_;
    tracks_[last_track_id_] = std::move(other_track);
    ++last_track_id_;
  }

  for (auto &&[other_track_id, ids_to_link] : should_be_added) {

    std::unordered_set<size_t> linked_ids{};

    for (auto &&linked_id :
         tracks_[id_remapping[other_track_id]].linked_tracks_) {

      if (id_remapping.contains(linked_id)) {
        linked_ids.insert(id_remapping[linked_id]);
      }
    }

    for (auto &&id_to_link : ids_to_link) {
      linked_ids.insert(id_to_link);
      tracks_[id_to_link].linked_tracks_.insert(id_remapping[other_track_id]);
    }

    tracks_[id_remapping[other_track_id]].linked_tracks_ =
        std::move(linked_ids);
  }
#endif
}
