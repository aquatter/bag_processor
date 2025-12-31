#pragma once
#include <memory>
#include <types.hpp>

class BagProcessor;

class TrackMerger {
public:
  TrackMerger(std::shared_ptr<BagProcessor> bag);
  ~TrackMerger();

private:
  struct impl;
  std::unique_ptr<impl> pimpl_;
};