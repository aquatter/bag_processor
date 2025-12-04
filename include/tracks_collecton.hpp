#pragma once

#include <Eigen/Core>
#include <bag_processor.hpp>
#include <boost/container_hash/hash.hpp>
#include <cartesian_converter.hpp>
#include <cstddef>
#include <memory>
#include <rerun.hpp>
#include <serialization.hpp>
#include <types.hpp>
#include <unordered_map>
#include <utility>
#include <vector>

struct CombinedLandmarks {
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

class TracksCollection {
public:
  void set_rerun(std::shared_ptr<rerun::RecordingStream> rec) {
    rec_ = std::move(rec);
  }

  void merge(BagProcessor::ptr bag);

private:
  void init(BagProcessor::ptr bag);
  void recalculate_coords(BagProcessor::ptr bag);
  bool should_be_linked(CombinedLandmarks::Link dst_link,
                        std::span<const size_t> det_ind, size_t src_track_id);

  std::optional<Landmark> try_link(CombinedLandmarks::Link link,
                                   const ImageTrack &new_track) const;

  bool check_proximity(CombinedLandmarks::Link link,
                       const ImageTrack &track) const;

  bool check_closest_box_and_intersecton(CombinedLandmarks::Link dst_link,
                                         std::span<const size_t> det_ind,
                                         size_t src_track_id) const;

  void combine_landmarks(std::span<const size_t> affected_landmarks);

  friend class boost::serialization::access;

  template <typename Archive> void serialize(Archive &ar, const unsigned int) {
    ar & bags_;
  }

  void log_current_state() const;

  std::vector<BagProcessor::ptr> bags_;
  CartesianConverter converter_;
  CombinedLandmarks landmarks_;
  std::shared_ptr<rerun::RecordingStream> rec_;
};