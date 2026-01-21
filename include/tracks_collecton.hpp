#pragma once
#include <bag_processor.hpp>

class TracksCollection {
public:
  TracksCollection();
  ~TracksCollection();

  void merge(BagProcessor::ptr bag);
#if USE_RERUN
  void set_rerun(std::shared_ptr<rerun::RecordingStream> rec);
#endif

private:
  struct impl;
  std::unique_ptr<impl> pimpl_;
};