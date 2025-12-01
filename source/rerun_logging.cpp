#include "rerun.hpp"
#include "rerun/archetypes/geo_points.hpp"
#include <Eigen/Core>
#include <Eigen/src/Core/Matrix.h>
#include <GeographicLib/LocalCartesian.hpp>
#include <boost/math/constants/constants.hpp>
#include <cartesian_converter.hpp>
#include <cmath>
#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/transform.hpp>
#include <rerun/components/vector2d.hpp>
#include <rerun_logging.hpp>
#include <vector>

using ranges::to;
using ranges::views::filter;
using ranges::views::transform;
using namespace rerun;

void log_track_map(std::shared_ptr<rerun::RecordingStream> rec,
                   const ImageTrack &track, rerun::Color color) {

  const GeographicLib::LocalCartesian converter{
      track.geodetic_origin_.x(), track.geodetic_origin_.y(), 0.0};

  const auto path{track.dets_ |
                  filter([](auto &&p) { return p.enu_.has_value(); }) |
                  transform([&converter](auto &&p) {
                    Eigen::Vector3d lla{Eigen::Vector3d::Zero()};
                    converter.Reverse(p.enu_.value().x(), p.enu_.value().y(),
                                      0.0, lla.x(), lla.y(), lla.z());

                    return rerun::DVec2D{lla.x(), lla.y()};
                  }) |
                  to<std::vector>()};

  rec->log(fmt::format("map/{}_{}", track.name_, track.id_),
           rerun::GeoLineStrings{
               rerun::components::GeoLineString::from_lat_lon(path)}
               .with_colors(color)
               .with_radii(rerun::Radius::ui_points(2.0f)));

  [[maybe_unused]] const auto err{rec->flush_blocking()};
}

void log_segment(const std::string_view entity_path,
                 std::shared_ptr<rerun::RecordingStream> rec,
                 rerun::components::LatLon p0, rerun::components::LatLon p1) {

  rec->log(entity_path,
           rerun::GeoLineStrings{rerun::components::GeoLineString::from_lat_lon(
                                     {p0.lat_lon, p1.lat_lon})}
               .with_colors(rerun::Color{255, 255, 0})
               .with_radii(rerun::Radius::ui_points(2.0f)));

  rec->log(entity_path, rerun::GeoPoints{{p0, p1}}
                            .with_colors(rerun::Color{0, 0, 255})
                            .with_radii(rerun::Radius::ui_points(5.0f)));
}

void log_segment(std::shared_ptr<rerun::RecordingStream> rec,
                 const ImageTrack &track1, const ImageTrack &track2,
                 size_t index1, size_t index2) {
  const GeographicLib::LocalCartesian converter1{
      track1.geodetic_origin_.x(), track1.geodetic_origin_.y(), 0.0};

  const GeographicLib::LocalCartesian converter2{
      track2.geodetic_origin_.x(), track2.geodetic_origin_.y(), 0.0};

  const auto pose0{track1.dets_[index1].enu_.value()};
  const auto pose1{track2.dets_[index2].enu_.value()};

  Eigen::Vector3d lla0{Eigen::Vector3d::Zero()};
  Eigen::Vector3d lla1{Eigen::Vector3d::Zero()};

  converter1.Reverse(pose0.x(), pose0.y(), 0.0, lla0.x(), lla0.y(), lla0.z());
  converter2.Reverse(pose1.x(), pose1.y(), 0.0, lla1.x(), lla1.y(), lla1.z());

  log_segment(fmt::format("map/segment_{}_{}", track1.name_, track2.name_), rec,
              {lla0.x(), lla0.y()}, {lla1.x(), lla1.y()});

  [[maybe_unused]] const auto err{rec->flush_blocking()};
}

void log_start_end_segments(std::shared_ptr<rerun::RecordingStream> rec,
                            const ImageTrack &track1, const ImageTrack &track2,
                            size_t start1, size_t end1, size_t start2,
                            size_t end2) {

  const GeographicLib::LocalCartesian converter{
      track1.geodetic_origin_.x(), track1.geodetic_origin_.y(), 0.0};

  const auto pose0{track1.dets_[start1].enu_.value()};
  const auto pose1{track2.dets_[start2].enu_.value()};
  const auto pose2{track1.dets_[end1].enu_.value()};
  const auto pose3{track2.dets_[end2].enu_.value()};

  Eigen::Vector3d lla0{Eigen::Vector3d::Zero()};
  Eigen::Vector3d lla1{Eigen::Vector3d::Zero()};
  Eigen::Vector3d lla2{Eigen::Vector3d::Zero()};
  Eigen::Vector3d lla3{Eigen::Vector3d::Zero()};

  converter.Reverse(pose0.x(), pose0.y(), 0.0, lla0.x(), lla0.y(), lla0.z());
  converter.Reverse(pose1.x(), pose1.y(), 0.0, lla1.x(), lla1.y(), lla1.z());
  converter.Reverse(pose2.x(), pose2.y(), 0.0, lla2.x(), lla2.y(), lla2.z());
  converter.Reverse(pose3.x(), pose3.y(), 0.0, lla3.x(), lla3.y(), lla3.z());

  log_segment(fmt::format("map/start_{}_{}", track1.name_, track2.name_), rec,
              {lla0.x(), lla0.y()}, {lla1.x(), lla1.y()});
  log_segment(fmt::format("map/end_{}_{}", track1.name_, track2.name_), rec,
              {lla2.x(), lla2.y()}, {lla3.x(), lla3.y()});
}

void log_detection(std::shared_ptr<rerun::RecordingStream> rec,
                   const ImageTrack &track, size_t index) {

  const CartesianConverter converter{track.geodetic_origin_};

  const auto &det{track.dets_[index]};

  const std::array<Eigen::Vector3d, 4> arrow{
      {{0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {-0.2, 0.7, 0.0}, {0.2, 0.7, 0.0}}};

  const auto angle{
      std::atan2(det.direction_.value().x(), det.direction_.value().y())};

  const Eigen::Isometry3d transf{
      Eigen::Translation3d{
          Eigen::Vector3d{det.enu_.value().x(), det.enu_.value().y(), 0.0}} *
      Eigen::AngleAxisd{-angle, Eigen::Vector3d::UnitZ()}};

  const std::vector<rerun::DVec2D> lla_arrow{
      arrow | transform([&transf, &converter](auto &&p) {
        const Eigen::Vector3d p0{transf * p};

        const auto latlon{converter.latlon(p0.head<2>())};
        return rerun::DVec2D{latlon.x(), latlon.y()};
      }) |
      to<std::vector>()};

  const std::string entity_path{fmt::format("map/{}_{}", track.name_, index)};

  rec->log(
      entity_path,
      rerun::GeoLineStrings{{rerun::components::GeoLineString::from_lat_lon(
                                 {lla_arrow[0], lla_arrow[1]}),
                             rerun::components::GeoLineString::from_lat_lon(
                                 {lla_arrow[2], lla_arrow[1], lla_arrow[3]})}}
          .with_colors(rerun::Color{255, 0, 0}));

  rec->log(entity_path, rerun::GeoPoints{{rerun::LatLon{lla_arrow[0]}}}
                            .with_colors(rerun::Color{0, 255, 0})
                            .with_radii(rerun::Radius::ui_points(5.0f)));

  [[maybe_unused]] const auto err{rec->flush_blocking()};
}

void log_landmark(std::shared_ptr<rerun::RecordingStream> rec,
                  const Landmark &landmark, Color color) {

  rec->log(fmt::format("map/{}_{}", landmark.code_, landmark.id_),
           GeoPoints{{LatLon{landmark.latlon_.x(), landmark.latlon_.y()}}}
               .with_colors(color)
               .with_radii(Radius::ui_points(2.0f)));

  [[maybe_unused]] const auto err{rec->flush_blocking()};
}