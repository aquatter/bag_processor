#include <algorithm>
#include <atomic>
#include <bag_processor.hpp>
#include <boost/container_hash/hash.hpp>
#include <boost/math/constants/constants.hpp>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <feature_matcher.hpp>
#include <flann/flann.hpp>
#include <fmt/color.h>
#include <fmt/format.h>
#include <geo_json.hpp>
#include <limits>
#include <memory>
#include <mp4_image_loader.hpp>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <progress_bar.hpp>
#include <range/v3/algorithm/count_if.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/enumerate.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/join.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/view/zip.hpp>
#include <string>
#include <string_view>
#include <threads.hpp>
#include <track_merger.hpp>
#include <triangulation.hpp>
#include <types.hpp>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using ranges::to;
using ranges::views::enumerate;
using ranges::views::filter;
using ranges::views::ints;
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
      : bag_{bag}, indexer_{flann::KDTreeSingleIndexParams{}},
        image_loader_{bag_->set_.bag_path_} {

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

  void process() {
    std::vector<std::pair<size_t, size_t>> link_result{};

    std::unordered_set<std::pair<size_t, size_t>, decltype([](const auto &p) {
                         size_t seed{0};
                         boost::hash_combine(seed, p.first);
                         boost::hash_combine(seed, p.second);
                         return seed;
                       })>
        taken{};

    std::vector<SearchResult> tracks_to_check{};
    std::vector<size_t> failed_to_link{};

    for (auto &&[src_track_id, track] : bag_->image_tracks_) {
      if (not track.valid_) {
        continue;
      }

      const auto found_tracks{find(track)};

      for (auto &&index_result : found_tracks) {

        if (taken.contains(
                {index_result.src_track_id_, index_result.dst_track_id_}) or
            taken.contains(
                {index_result.dst_track_id_, index_result.src_track_id_})) {
          continue;
        }

        taken.insert({index_result.src_track_id_, index_result.dst_track_id_});

        if (not check_frame_indices(index_result)) {
          continue;
        }

        if (not check_intersection(index_result)) {
          continue;
        }

        LOG(INFO) << fmt::format("Analyzing tracks {} -> {}",
                                 index_result.src_track_id_,
                                 index_result.dst_track_id_);

        if (boxes_should_be_checked(index_result)) {
          tracks_to_check.push_back(index_result);
#if 0
          cv::Mat_<cv::Vec3b> img = image_loader_.load_image(
              bag_->image_tracks_.at(index_result.dst_track_id_)
                  .dets_[index_result.det_indixes_[0].dst_ind_]
                  .image_id_);

          cv::imwrite("dst_img.png", img);

          img = image_loader_.load_image(
              bag_->image_tracks_.at(index_result.src_track_id_)
                  .dets_[index_result.det_indixes_[0].src_ind_]
                  .image_id_);

          cv::imwrite("src_img.png", img);
#endif
          continue;
        }

        // if (not check_boxes(index_result)) {
        //   LOG(INFO) << fmt::format(fmt::fg(fmt::color::coral), "failed");
        //   continue;
        // }

        // LOG(INFO) << fmt::format(fmt::fg(fmt::color::yellow_green),
        // "success");

        LOG(INFO) << "Trying to triangulate";
        if (not try_triangulate(
                    {index_result.dst_track_id_, index_result.src_track_id_})
                    .has_value()) {
          LOG(INFO) << fmt::format(fmt::fg(fmt::color::coral), "failed");
          failed_to_link.push_back(index_result.src_track_id_);
          continue;
        }

        LOG(INFO) << fmt::format(fmt::fg(fmt::color::yellow_green), "success");

        link_result.emplace_back(index_result.dst_track_id_,
                                 index_result.src_track_id_);

        // dump_search_result(index_result, "linked_tracks.geojson");
#if 0
        dump_image(
            index_result.dst_track_id_, index_result.det_indixes_[5].dst_ind_,
            fmt::format("dst_track_{}_det_{}.png", index_result.dst_track_id_,
                        index_result.det_indixes_[5].dst_ind_));

        dump_image(
            index_result.src_track_id_, index_result.det_indixes_[5].src_ind_,
            fmt::format("src_track_{}_det_{}.png", index_result.src_track_id_,
                        index_result.det_indixes_[5].src_ind_));

        if (should_be_linked(index_result)) {
          link_result.emplace_back(index_result.dst_track_id_, src_track_id);
        }
#endif
      }
    }
    {
      std::unordered_set<size_t> image_ids{};

      for (auto &&search_res : tracks_to_check) {
        for (auto &&[dst_det_ind, src_det_ind] : search_res.det_indixes_) {

          image_ids.insert(bag_->image_tracks_.at(search_res.dst_track_id_)
                               .dets_[dst_det_ind]
                               .image_id_);
          image_ids.insert(bag_->image_tracks_.at(search_res.src_track_id_)
                               .dets_[src_det_ind]
                               .image_id_);
        }
      }

      auto images_to_load{image_ids | to<std::vector>()};
      std::sort(images_to_load.begin(), images_to_load.end());

      ProgressBar progress{images_to_load.size(), "loading images..."};
      progress.draw();
      image_loader_.set_progress([&progress]() { progress.advance(); });
      progress.done();

      auto images{image_loader_.extract(images_to_load)};

      for (auto &&[id, img] : zip(images_to_load, images)) {
        image_cache_[id] = std::move(img);
      }

      ProgressBar progress2{images_to_load.size(), "extracting descriptors..."};
      progress2.draw();

      RunInThreads{images_to_load.size()}(
          [this, &progress2,
           &images_to_load](const RunInThreads::Context &ctx) {
            for (auto &&i : ints(ctx.beg_, ctx.end_)) {
              matcher_.extract_descriptors(
                  image_cache_.at(images_to_load[i]),
                  fmt::format("image_{}", images_to_load[i]));
              progress2.advance();
            }
          });

      progress2.done();
    }

    for (auto &&[i, search_res] : tracks_to_check | enumerate) {
      LOG(INFO) << fmt::format(
          "Analyzing tracks {} -> {}, ({}/{})", search_res.src_track_id_,
          search_res.dst_track_id_, i + 1, tracks_to_check.size());

      if (not check_boxes(search_res)) {
        LOG(INFO) << fmt::format(fmt::fg(fmt::color::coral), "failed");
        continue;
      }

      LOG(INFO) << fmt::format(fmt::fg(fmt::color::yellow_green), "success");

      LOG(INFO) << "Trying to triangulate";
      if (not try_triangulate(
                  {search_res.dst_track_id_, search_res.src_track_id_})
                  .has_value()) {
        LOG(INFO) << fmt::format(fmt::fg(fmt::color::coral), "failed");
        failed_to_link.push_back(search_res.src_track_id_);
        continue;
      }

      LOG(INFO) << fmt::format(fmt::fg(fmt::color::yellow_green), "success");

      link_result.emplace_back(search_res.dst_track_id_,
                               search_res.src_track_id_);
    }

    LOG(INFO) << "Combining landmarks";
    combine_landmarks(link_result, failed_to_link);
  }

  std::vector<SearchResult> find(const ImageTrack &track) {

    const auto query_points{
        track.selected_detections_ | transform([&](const size_t det_ind) {
          return IndexPoint{.track_id_ = track.id_,
                            .det_ind_ = det_ind,
                            .enu_ = track.dets_[det_ind].enu_.value()};
        }) |
        to<std::vector>()};

    auto query_vec{query_points | transform([](const IndexPoint &p) {
                     return std::pair{p.enu_.x(), p.enu_.y()};
                   }) |
                   to<std::vector>()};

    const flann::Matrix<double> dataset{&query_vec.front().first,
                                        query_vec.size(), 2ul};

    std::vector<std::vector<double>> distances{};
    std::vector<std::vector<int>> indices{};

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

    std::vector<SearchResult> res{};

    for (auto &&[dst_track_id, ind_pairs] : found_tracks) {

      const auto &dst_track{bag_->image_tracks_.at(dst_track_id)};

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

      res.emplace_back(SearchResult{
          .dst_track_id_ = dst_track_id,
          .src_track_id_ = track.id_,
          .det_indixes_ = m | transform([](const auto &p) {
                            return SearchResult::DetPair{.dst_ind_ = p.first,
                                                         .src_ind_ =
                                                             p.second.first};
                          }) |
                          to<std::vector>()

      });
    }

    return res;
#if 0
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
#endif
  }

  bool boxes_should_be_checked(const SearchResult &res) {

    const auto &dst_track{bag_->image_tracks_.at(res.dst_track_id_)};
    const auto &src_track{bag_->image_tracks_.at(res.src_track_id_)};
    const auto &dst_dets{dst_track.dets_};

    size_t num_ambiguites{0};

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
    }

    const auto ambiguites_ratio{static_cast<float>(num_ambiguites) /
                                static_cast<float>(res.det_indixes_.size())};

    return ambiguites_ratio > 0.2f;
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

    std::atomic_size_t num_mismatches{0};
    std::atomic_size_t num_checked{0};

    if (ambiguites_ratio > 0.2f) {

      std::vector<SearchResult::DetPair> pairs_to_check{};

      for (auto &&[det_pair, required] :
           zip(res.det_indixes_, requires_cheking)) {

        if (not required) {
          continue;
        }

        pairs_to_check.push_back(det_pair);
      }

      ProgressBar progress{pairs_to_check.size(),
                           fmt::format("Processing ({} -> {})",
                                       res.src_track_id_, res.dst_track_id_)};
      progress.draw();

      RunInThreads{pairs_to_check.size()}(
          [this, &pairs_to_check, &dst_track, &dst_dets, &src_track, &src_dets,
           &progress, &num_mismatches,
           &num_checked](const RunInThreads::Context &ctx) {
            for (auto &&ind : ints(ctx.beg_, ctx.end_)) {

              const size_t dst_det_ind{pairs_to_check[ind].dst_ind_};
              const size_t src_det_ind{pairs_to_check[ind].src_ind_};
              const auto dst_image_id{dst_dets[dst_det_ind].image_id_};
              const auto src_image_id{src_dets[src_det_ind].image_id_};

              const auto tag1{fmt::format("image_{}", dst_image_id)};
              const auto tag2{fmt::format("image_{}", src_image_id)};

              if (not image_cache_.contains(dst_image_id)) {
                LOG(WARNING) << fmt::format("image {} not found", dst_image_id);
                continue;
              }

              if (not image_cache_.contains(src_image_id)) {
                LOG(WARNING) << fmt::format("image {} not found", src_image_id);
                continue;
              }

              if (matcher_.estimate_homography(
                      image_cache_.at(dst_image_id), bag_->calib_, tag1,
                      image_cache_.at(src_image_id), bag_->calib_, tag2)) {

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

                size_t closest_track_id{0};
                double min_dist{std::numeric_limits<double>::max()};

                for (auto &&[p, id] : zip(projected_points, track_ids)) {
                  const auto dist{cv::norm(cv::Vec2f{p}, src_center)};

                  if (dist < min_dist) {
                    min_dist = dist;
                    closest_track_id = id;
                  }
                }

                if (closest_track_id != dst_track.id_) {
                  num_mismatches.fetch_add(1);
                }

                num_checked.fetch_add(1);
              }

              progress.advance();
            }
          });

      progress.done();
#if 0
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

        if (not image_cache_.contains(dst_image_id)) {
          throw std::runtime_error{
              fmt::format("image {} not found", dst_image_id)};
          // image_cache_[dst_image_id] =
          // image_loader_.load_image(dst_image_id);
        }

        if (not image_cache_.contains(src_image_id)) {
          throw std::runtime_error{
              fmt::format("image {} not found", src_image_id)};
          // image_cache_[src_image_id] =
          // image_loader_.load_image(src_image_id);
        }

        if (matcher_.estimate_homography(
                image_cache_.at(dst_image_id), bag_->calib_, tag1,
                image_cache_.at(src_image_id), bag_->calib_, tag2)) {

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

          // dump_homography(res.src_track_id_, src_det_ind, res.dst_track_id_,
          //                 dst_det_ind, projected_points);

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

        progress.advance();
      }

#endif

    } else {
      return true;
    }

    LOG(INFO) << fmt::format("num mistmatches: {}, num checked: {}",
                             num_mismatches.load(), num_checked.load());

    if (num_checked == 0) {
      return false;
    }

    const auto mismatches_ratio{static_cast<float>(num_mismatches.load()) /
                                static_cast<float>(num_checked.load())};

    return mismatches_ratio < 0.2f;
  }

  std::optional<Landmark> try_triangulate(const std::vector<size_t> &tracks) {

    std::vector<TrackPoint> track_points{};

    for (auto &&track_id : tracks) {
      const auto track_points_{
          bag_->image_tracks_.at(track_id).selected_track_points()};
      track_points.insert(track_points.end(), track_points_.begin(),
                          track_points_.end());
    }

    return triangulate_on_boxes(track_points);
  }

  bool check_frame_indices(const SearchResult &res) {
    const auto &dst_track{bag_->image_tracks_.at(res.dst_track_id_)};
    const auto &src_track{bag_->image_tracks_.at(res.src_track_id_)};

    float mean_ind_diff{0.0f};

    for (auto &&[dst_ind, src_ind] : res.det_indixes_) {
      mean_ind_diff +=
          std::abs(static_cast<float>(dst_track.dets_[dst_ind].image_id_) -
                   static_cast<float>(src_track.dets_[src_ind].image_id_));
    }

    mean_ind_diff /= static_cast<float>(res.det_indixes_.size());

    return mean_ind_diff > 500.0f;
  }

  bool check_intersection(const SearchResult &res) {

    const auto &dst_track{bag_->image_tracks_.at(res.dst_track_id_)};
    const auto &src_track{bag_->image_tracks_.at(res.src_track_id_)};

    double dst_intersection_ratio{0.0};

    {
      const auto min_max_it{
          std::minmax_element(res.det_indixes_.begin(), res.det_indixes_.end(),
                              [](const auto &a, const auto &b) {
                                return a.dst_ind_ < b.dst_ind_;
                              })};

      dst_intersection_ratio =
          std::abs(
              dst_track.dets_[min_max_it.second->dst_ind_].cumulative_length_ -
              dst_track.dets_[min_max_it.first->dst_ind_].cumulative_length_) /
          dst_track.length_;
    }

    double src_intersection_ratio{0.0};

    {
      const auto min_max_it{
          std::minmax_element(res.det_indixes_.begin(), res.det_indixes_.end(),
                              [](const auto &a, const auto &b) {
                                return a.src_ind_ < b.src_ind_;
                              })};

      src_intersection_ratio =
          std::abs(
              src_track.dets_[min_max_it.second->src_ind_].cumulative_length_ -
              src_track.dets_[min_max_it.first->src_ind_].cumulative_length_) /
          src_track.length_;
    }

    return dst_intersection_ratio > 0.5 and src_intersection_ratio > 0.5;
  }

  void dump_image(size_t track_id, size_t det_ind,
                  const std::string_view path) {
    cv::Mat_<cv::Vec3b> img = image_loader_.load_image(
        bag_->image_tracks_.at(track_id).dets_[det_ind].image_id_);

    cv::rectangle(img, bag_->image_tracks_.at(track_id).dets_[det_ind].box_,
                  {0.0, 0.0, 255.0}, 2);

    cv::imwrite(path.data(), img);
  }

  void dump_homography(size_t src_track_id, size_t src_det_ind,
                       size_t dst_track_id, size_t dst_det_ind,
                       std::span<const cv::Point2f> projected_dst_points) {

    const auto &src_track{bag_->image_tracks_.at(src_track_id)};
    const auto &dst_track{bag_->image_tracks_.at(dst_track_id)};
    const auto &src_dets{src_track.dets_};
    const auto &dst_dets{dst_track.dets_};
    const auto src_image_id{src_dets[src_det_ind].image_id_};
    const auto dst_image_id{dst_dets[dst_det_ind].image_id_};

    cv::Mat_<cv::Vec3b> img{};

    {
      const auto &buf{image_cache_.at(src_image_id)};
      cv::Mat_<uint8_t> gray_img = cv::imdecode(
          {buf.data(), static_cast<int>(buf.size())}, cv::IMREAD_GRAYSCALE);

      cv::cvtColor(gray_img, img, cv::COLOR_GRAY2BGR);
    }

    for (auto &&d :
         bag_->image_detections_.at(src_dets[src_det_ind].timestamp_).dets_) {
      cv::rectangle(img, d->box_, {0.0, 0.0, 255.0}, 2);
    }

    cv::rectangle(img, src_dets[src_det_ind].box_, {0.0, 255.0, 0.0}, 2);

    img = bag_->calib_.undistort_image(img);

    for (auto &&p : projected_dst_points) {
      cv::circle(img, p, 10, {255.0, 255.0, 25.0}, cv::FILLED, cv::LINE_AA);
      cv::circle(img, p, 10, cv::Scalar::all(0.0), 2, cv::LINE_AA);
    }

    cv::imwrite("src_img.png", img);

    {
      const auto &buf{image_cache_.at(dst_image_id)};
      cv::Mat_<uint8_t> gray_img = cv::imdecode(
          {buf.data(), static_cast<int>(buf.size())}, cv::IMREAD_GRAYSCALE);

      cv::cvtColor(gray_img, img, cv::COLOR_GRAY2BGR);
    }

    for (auto &&d :
         bag_->image_detections_.at(dst_dets[dst_det_ind].timestamp_).dets_) {
      cv::rectangle(img, d->box_, {0.0, 0.0, 255.0}, 2);
    }

    cv::rectangle(img, dst_dets[dst_det_ind].box_, {0.0, 255.0, 0.0}, 2);

    cv::imwrite("dst_img.png", img);
  }

  void dump_search_result(const SearchResult &res,
                          const std::string_view path) {

    const auto &dst_track{bag_->image_tracks_.at(res.dst_track_id_)};
    const auto &src_track{bag_->image_tracks_.at(res.src_track_id_)};

    GeoJson geo_json{};
    {

      const auto vec{dst_track.selected_detections_ |
                     transform([this, &dst_track](size_t i) {
                       return bag_->local_converter_.latlon(
                           dst_track.dets_[i].enu_.value());
                     }) |
                     to<std::vector>()};

      geo_json.add_element(GeoJson::LineString{}
                               .with_stroke_color("#002791")
                               .with_stroke_width(2)
                               .with_stroke_opacity(1.0)
                               .with_coordinates_latlon(vec));

      for (auto &ind : dst_track.selected_detections_) {

        geo_json.add_element(
            GeoJson::Square{}
                .with_size(0.1)
                .with_stroke_color("#555555")
                .with_stroke_width(2)
                .with_stroke_opacity(1.0)
                .with_fill_color("#002791")
                .with_fill_opacity(0.5)
                .with_coordinate_latlon(bag_->local_converter_.latlon(
                    dst_track.dets_[ind].enu_.value())));
      }
    }

    {
      const auto vec{src_track.selected_detections_ |
                     transform([this, &src_track](size_t i) {
                       return bag_->local_converter_.latlon(
                           src_track.dets_[i].enu_.value());
                     }) |
                     to<std::vector>()};

      geo_json.add_element(GeoJson::LineString{}
                               .with_stroke_color("#299100")
                               .with_stroke_width(2)
                               .with_stroke_opacity(1.0)
                               .with_coordinates_latlon(vec));

      for (auto &ind : src_track.selected_detections_) {

        geo_json.add_element(
            GeoJson::Square{}
                .with_size(0.1)
                .with_stroke_color("#555555")
                .with_stroke_width(2)
                .with_stroke_opacity(1.0)
                .with_fill_color("#299100")
                .with_fill_opacity(0.5)
                .with_coordinate_latlon(bag_->local_converter_.latlon(
                    src_track.dets_[ind].enu_.value())));
      }
    }

    for (auto &&[dst_ind, src_ind] : res.det_indixes_) {

      const auto p_src{
          bag_->local_converter_.latlon(src_track.dets_[src_ind].enu_.value())};

      const auto p_dst{
          bag_->local_converter_.latlon(dst_track.dets_[dst_ind].enu_.value())};

      geo_json.add_element(GeoJson::Line{}
                               .with_stroke_color("#ea1717")
                               .with_stroke_opacity(1.0)
                               .with_stroke_width(2)
                               .with_coordinates_latlon(p_src, p_dst));
    }

    geo_json.save(path);
  }

  void
  combine_landmarks(std::span<const std::pair<size_t, size_t>> linked_pairs,
                    std::span<const size_t> failed_to_link) {

    if (linked_pairs.empty()) {
      for (auto &&track_id : failed_to_link) {
        bag_->image_tracks_.at(track_id).valid_ = false;
      }
      return;
    }

    std::unordered_map<size_t, std::vector<size_t>> g{};
    size_t next_track_id{0};

    for (auto &&[id, track] : bag_->image_tracks_) {
      if (id > next_track_id) {
        next_track_id = id;
      }
    }

    for (auto &&[dst, src] : linked_pairs) {
      g[dst].push_back(src);
      g[src].push_back(dst);
    }

    std::unordered_set<size_t> taken{};

    for (auto &&[key, value] : g) {

      if (taken.contains(key)) {
        continue;
      }

      std::unordered_map<std::string, std::vector<size_t>> tracks{};
      tracks[bag_->image_tracks_.at(key).code_].push_back(key);
      taken.insert(key);
      std::deque<size_t> q{};
      q.push_back(key);

      while (not q.empty()) {
        const size_t current_track_id{q.front()};
        q.pop_front();

        if (g.contains(current_track_id)) {
          for (auto &&linked_track_id : g.at(current_track_id)) {
            if (taken.contains(linked_track_id)) {
              continue;
            }

            taken.insert(linked_track_id);
            tracks[bag_->image_tracks_.at(linked_track_id).code_].push_back(
                linked_track_id);

            q.push_back(linked_track_id);
          }
        }

        for (auto &&linked_track_id :
             bag_->image_tracks_.at(current_track_id).linked_tracks_) {
          if (taken.contains(linked_track_id)) {
            continue;
          }

          taken.insert(linked_track_id);
          tracks[bag_->image_tracks_.at(linked_track_id).code_].push_back(
              linked_track_id);

          q.push_back(linked_track_id);
        }
      }

      std::unordered_map<std::string, Landmark> landmarks{};

      size_t max_num_samples{0};
      std::string best_code{};

      for (auto &&[code, track_ids] : tracks) {
        auto landmark{try_triangulate(track_ids)};

        if (landmark.has_value()) {
          landmarks[code] = landmark.value();
          landmarks.at(code).latlon_ =
              bag_->local_converter_.latlon(landmarks.at(code).enu_);

          if (track_ids.size() > max_num_samples) {
            max_num_samples = track_ids.size();
            best_code = code;
          }
        }
      }

      Landmark best_landmark{};

      if (not best_code.empty()) {
        best_landmark = landmarks.at(best_code);
      } else {
        size_t max_num_samples{0};

        for (auto &&[code, track_ids] : tracks) {
          for (auto &&track_id : track_ids) {
            auto &track{bag_->image_tracks_.at(track_id)};

            if (not track.valid_) {
              continue;
            }

            if (track.selected_detections_.size() > max_num_samples) {
              max_num_samples = track.selected_detections_.size();
              best_landmark = track.landmark_.value();
            }
          }
        }
      }

      for (auto &&[code, track_ids] : tracks) {
        for (auto &&track_id : track_ids) {
          auto &track{bag_->image_tracks_.at(track_id)};
          track.landmark_ = best_landmark;
          track.matched_tracks_ = track_ids;
        }
      }
    }

    for (auto &&track_id : failed_to_link) {
      if (not g.contains(track_id)) {
        bag_->image_tracks_.at(track_id).valid_ = false;
      }
    }
  }

  std::shared_ptr<BagProcessor> bag_;
  flann::Index<flann::L2_Simple<double>> indexer_;
  std::vector<IndexPoint> points_;
  static constexpr float search_rad_sqr_{15.0f * 15.0f};
  static constexpr double cos_angle_threshold_{
      constexpr_cos(boost::math::double_constants::degree * 15.0)};

  FeatureMatcher matcher_;
  Mp4ImageLoader image_loader_;
  std::unordered_map<size_t, std::vector<uint8_t>> image_cache_;
};

TrackMerger::TrackMerger(std::shared_ptr<BagProcessor> bag)
    : pimpl_{std::make_unique<impl>(bag)} {}

TrackMerger::~TrackMerger() = default;

void TrackMerger::process() { pimpl_->process(); }