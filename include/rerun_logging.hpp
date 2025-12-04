#include <Eigen/Core>
#include <cartesian_converter.hpp>
#include <cstddef>
#include <memory>
#include <range/v3/view/enumerate.hpp>

#include <rerun.hpp>
#include <rerun/components/lat_lon.hpp>
#include <string_view>
#include <types.hpp>
#include <utils.hpp>

template <typename Scalar, int Rows>
rerun::LatLon to_lat_lon(const Eigen::Matrix<Scalar, Rows, 1> &v) {
  return rerun::LatLon{static_cast<double>(v.x()), static_cast<double>(v.y())};
}

rerun::Color to_color(const std::string_view str);

void log_track_map(std::shared_ptr<rerun::RecordingStream> rec,
                   const ImageTrack &track, rerun::Color color,
                   const std::string_view entity_path = {});

void log_segment(const std::string_view entity_path,
                 std::shared_ptr<rerun::RecordingStream> rec,
                 rerun::components::LatLon p0, rerun::components::LatLon p1,
                 rerun::Color line_color = to_color("rgba(19, 71, 184, 1)"),
                 rerun::Color point_color = to_color("rgba(184, 19, 143, 1)"));

void log_segment(std::shared_ptr<rerun::RecordingStream> rec,
                 const ImageTrack &track1, const ImageTrack &track2,
                 size_t index1, size_t index2);

void log_start_end_segments(std::shared_ptr<rerun::RecordingStream> rec,
                            const ImageTrack &track1, const ImageTrack &track2,
                            size_t start1, size_t end1, size_t start2,
                            size_t end2);

void log_detection(std::shared_ptr<rerun::RecordingStream> rec,
                   const ImageTrack &track, size_t index);

void log_landmark(std::shared_ptr<rerun::RecordingStream> rec,
                  const Landmark &landmark, rerun::Color color,
                  float ui_points = 2.0f);

void log_poly(const std::string_view entity_path,
              std::shared_ptr<rerun::RecordingStream> rec,
              const PolyResult<3> &poly, rerun::Color color,
              const CartesianConverter &converter);