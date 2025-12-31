#include <bag_processor.hpp>
#include <boost/math/constants/constants.hpp>
#include <cstddef>
#include <feature_matcher.hpp>
#include <flann/flann.hpp>
#include <fmt/format.h>
#include <memory>
#include <opencv2/core.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/enumerate.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/join.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/view/zip.hpp>
#include <track_merger.hpp>
#include <unordered_map>
#include <utility>
#include <vector>

using ranges::to;
using ranges::views::enumerate;
using ranges::views::filter;
using ranges::views::join;
using ranges::views::transform;
using ranges::views::zip;

struct TrackMerger::impl {

  struct IndexPoint {
    size_t track_id_;
    size_t det_ind_;
    Eigen::Vector2d enu_;
  };

  struct SearchResult {
    struct DetPair {
      size_t dst_ind_;
      size_t src_ind_;
    };

    size_t dst_track_id_;
    size_t src_track_id_;
    std::vector<DetPair> det_indixes_;
  };

  impl(std::shared_ptr<BagProcessor> bag)
      : bag_{bag}, indexer_{flann::KDTreeSingleIndexParams{}} {

    for (auto &&[track_id, track] : bag->image_tracks_) {
      if (not track.valid_) {
        continue;
      }

      for (auto &&det_ind : track.selected_detections_) {
        points_.emplace_back(track_id, det_ind,
                             track.dets_[det_ind].enu_.value());
      }
    }

    auto data_vec{points_ | transform([](const IndexPoint &p) {
                    return std::pair{p.enu_.x(), p.enu_.y()};
                  }) |
                  to<std::vector>()};

    const flann::Matrix<double> dataset{&data_vec.front().first,
                                        data_vec.size(), 2ul};

    indexer_.buildIndex(dataset);
  }

  bool should_be_linked(const SearchResult &res) { return false; }

  void process() {

    std::vector<std::pair<size_t, size_t>> link_result{};

    for (auto &&[src_track_id, track] : bag_->image_tracks_) {
      const auto found_tracks{find(track)};

      for (auto &&index_result : found_tracks) {

        if (should_be_linked(index_result)) {
          link_result.emplace_back(index_result.dst_track_id_, src_track_id);
        }
      }
    }
  }

  std::vector<SearchResult> find(const ImageTrack &track) {

    const auto query_points{
        track.selected_detections_ | transform([&](const size_t det_ind) {
          return IndexPoint{.track_id_ = track.id_,
                            .det_ind_ = det_ind,
                            .enu_ = track.dets_[det_ind].enu_.value()};
        }) |
        to<std::vector>()};

    auto query_vec{points_ | transform([](const IndexPoint &p) {
                     return std::pair{p.enu_.x(), p.enu_.y()};
                   }) |
                   to<std::vector>()};

    const flann::Matrix<double> dataset{&query_vec.front().first,
                                        query_vec.size(), 2ul};

    std::vector<std::vector<int>> indices{};
    std::vector<std::vector<double>> distances{};

    std::unordered_map<size_t, std::vector<std::pair<size_t, size_t>>>
        found_tracks{};

    indexer_.radiusSearch(dataset, indices, distances, search_rad_sqr_,
                          flann::SearchParams{});

    for (auto &&[query_ind, dataset_indices] : enumerate(indices)) {

      std::unordered_map<size_t, size_t> found_indices{};

      for (auto &i : dataset_indices) {

        if (points_[i].track_id_ == track.id_) {
          continue;
        }

        if (bag_->image_tracks_.at(points_[i].track_id_).code_ != track.code_) {
          continue;
        }

        if (found_indices.contains(points_[i].track_id_)) {
          continue;
        }

        const auto &det{bag_->image_tracks_.at(points_[i].track_id_)
                            .dets_[points_[i].det_ind_]};

        const Eigen::Vector2d dir1{det.direction_.value()};
        const Eigen::Vector2d dir2{
            track.dets_[query_points[query_ind].det_ind_].direction_.value()};

        const auto dot_product{dir1.dot(dir2)};

        if (dot_product < cos_angle_threshold_) {
          continue;
        }

        found_indices[points_[i].track_id_] = points_[i].det_ind_;
      }

      for (auto &&[dst_track_id, dst_det_ind] : found_indices) {
        found_tracks[dst_track_id].push_back(
            {dst_det_ind, query_points[query_ind].det_ind_});
      }
    }

    return found_tracks | transform([&track](const auto &val) {
             return SearchResult{.dst_track_id_ = val.first,
                                 .src_track_id_ = track.id_,
                                 .det_indixes_ = val.second |
                                                 transform([](const auto &p) {
                                                   return SearchResult::DetPair{
                                                       .dst_ind_ = p.first,
                                                       .src_ind_ = p.second};
                                                 }) |
                                                 to<std::vector>()};
           }) |
           to<std::vector>();
  }

  bool check_boxes(const SearchResult &res) {

    const auto &dst_track{bag_->image_tracks_.at(res.dst_track_id_)};
    const auto &src_track{bag_->image_tracks_.at(res.src_track_id_)};
    const auto &dst_dets{dst_track.dets_};
    const auto &src_dets{src_track.dets_};

    size_t num_ambiguites{0};
    std::vector<bool> requires_cheking{};

    for (const auto &[dst_det_ind, src_det_ind] : res.det_indixes_) {
      const auto &this_timestamp_dets{
          bag_->image_detections_.at(dst_dets[dst_det_ind].timestamp_).dets_};

      size_t num_candidates{0};
      for (auto &d : this_timestamp_dets) {

        if (d->code_ == src_track.code_) {
          ++num_candidates;
        }
      }

      if (num_candidates > 1) {
        ++num_ambiguites;
      }

      requires_cheking.push_back(num_candidates > 1);
    }

    const auto ambiguites_ratio{static_cast<float>(num_ambiguites) /
                                static_cast<float>(res.det_indixes_.size())};

    size_t num_mismatches{0};
    size_t num_checked{0};

    if (ambiguites_ratio > 0.2f) {

      for (const auto &[dst_det_ind, src_det_ind] :
           res.det_indixes_ | enumerate |
               filter([&requires_cheking](auto &&val) {
                 return requires_cheking[std::get<0>(val)];
               }) |
               transform([](auto &&val) { return std::get<1>(val); })) {

        const auto tag1{fmt::format("track_{}_image_{}", dst_track.id_,
                                    dst_dets[dst_det_ind].image_id_)};

        const auto tag2{fmt::format("track_{}_image_{}", src_track.id_,
                                    src_dets[src_det_ind].image_id_)};

        const auto dst_image_id{dst_dets[dst_det_ind].image_id_};
        const auto src_image_id{src_dets[src_det_ind].image_id_};

        if (bag_->selected_frames_.contains(dst_image_id) and
            bag_->selected_frames_.contains(src_image_id)) {

          if (matcher_.estimate_homography(
                  bag_->selected_frames_.at(dst_image_id), bag_->calib_, tag1,
                  bag_->selected_frames_.at(src_image_id), bag_->calib_,
                  tag2)) {

            const auto &this_timestamp_dets{
                bag_->image_detections_.at(dst_dets[dst_det_ind].timestamp_)
                    .dets_};

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

            size_t closest_track{0};
            double min_dist{std::numeric_limits<double>::max()};

            for (auto &&[p, id] : zip(projected_points, track_ids)) {
              const auto dist{cv::norm(cv::Vec2f{p}, src_center)};

              if (dist < min_dist) {
                min_dist = dist;
                closest_track = id;
              }
            }

            if (closest_track != dst_track.id_) {
              ++num_mismatches;
            }

            ++num_checked;
          }
        }
      }
    } else {
      return true;
    }

    if (num_checked == 0) {
      return false;
    }

    return false;
  }

  bool check_intersection(const SearchResult &res) { return false; }
  bool check_landmarks(const SearchResult &res) { return false; }

  std::shared_ptr<BagProcessor> bag_;
  flann::Index<flann::L2_Simple<double>> indexer_;
  std::vector<IndexPoint> points_;
  static constexpr float search_rad_sqr_{15.0f * 15.0f};
  static constexpr double cos_angle_threshold_{
      constexpr_cos(boost::math::double_constants::degree * 15.0)};

  FeatureMatcher matcher_;
};

TrackMerger::TrackMerger(std::shared_ptr<BagProcessor> bag)
    : pimpl_{std::make_unique<impl>(bag)} {}

TrackMerger::~TrackMerger() = default;