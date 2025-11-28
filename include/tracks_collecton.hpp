#pragma once

#include <Eigen/Core>
#include <bag_processor.hpp>
#include <cartesian_converter.hpp>
#include <cstddef>
#include <memory>
#include <rerun.hpp>
#include <serialization.hpp>
#include <types.hpp>
#include <unordered_set>
#include <utility>
#include <vector>

class TracksCollection {
public:
  void set_rerun(std::shared_ptr<rerun::RecordingStream> rec) {
    rec_ = std::move(rec);
  }

  void merge(BagProcessor::ptr bag);

private:
  void init(BagProcessor::ptr bag);
  void recalculate_coords(BagProcessor::ptr bag);
  bool should_be_linked(size_t bag_index, size_t track_id,
                        const std::unordered_set<size_t> &det_ind,
                        BagProcessor::ptr src_bag, size_t src_bag_id) const;

  friend class boost::serialization::access;

  template <typename Archive> void serialize(Archive &ar, const unsigned int) {
    ar & bags_;
  }

  std::vector<BagProcessor::ptr> bags_;
  CartesianConverter converter_;
  std::shared_ptr<rerun::RecordingStream> rec_;
};