#include <cstddef>
#include <memory>
#include <rerun.hpp>
#include <rerun/components/lat_lon.hpp>
#include <string_view>
#include <types.hpp>

void log_track_map(std::shared_ptr<rerun::RecordingStream> rec,
                   const ImageTrack &track, rerun::Color color);

void log_segment(const std::string_view entity_path,
                 std::shared_ptr<rerun::RecordingStream> rec,
                 rerun::components::LatLon p0, rerun::components::LatLon p1);

void log_segment(std::shared_ptr<rerun::RecordingStream> rec,
                 const ImageTrack &track1, const ImageTrack &track2,
                 size_t index1, size_t index2);

void log_start_end_segments(std::shared_ptr<rerun::RecordingStream> rec,
                            const ImageTrack &track1, const ImageTrack &track2,
                            size_t start1, size_t end1, size_t start2,
                            size_t end2);

void log_detection(std::shared_ptr<rerun::RecordingStream> rec,
                   const ImageTrack &track, size_t index);