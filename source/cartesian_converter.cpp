#include <GeographicLib/Geocentric.hpp>
#include <GeographicLib/LocalCartesian.hpp>
#include <cartesian_converter.hpp>

CartesianConverter::CartesianConverter()
    : geodetic_origin_{Eigen::Vector2d::Zero()},
      converter_{
          GeographicLib::LocalCartesian{GeographicLib::Geocentric::WGS84()}},
      origin_set_{false} {}

CartesianConverter::CartesianConverter(Eigen::Vector2d geodetic_origin)
    : geodetic_origin_{geodetic_origin},
      converter_{GeographicLib::LocalCartesian{geodetic_origin_.x(),
                                               geodetic_origin_.y()}},
      origin_set_{true} {}

void CartesianConverter::set_origin(Eigen::Vector2d geodetic_origin) {
  geodetic_origin_ = geodetic_origin;
  converter_.Reset(geodetic_origin_.x(), geodetic_origin_.y());
  origin_set_ = true;
}

Eigen::Vector2d CartesianConverter::origin() const noexcept {
  return geodetic_origin_;
}

bool CartesianConverter::origin_set() const noexcept { return origin_set_; }

Eigen::Vector2d CartesianConverter::enu(Eigen::Vector2d latlon) const noexcept {
  Eigen::Vector2d enu{Eigen::Vector2d::Zero()};
  double z{0.0};
  converter_.Forward(latlon.x(), latlon.y(), 0.0, enu.x(), enu.y(), z);
  return enu;
}

Eigen::Vector2d CartesianConverter::latlon(Eigen::Vector2d enu) const noexcept {
  Eigen::Vector2d latlon{Eigen::Vector2d::Zero()};
  double h{0.0};
  converter_.Reverse(enu.x(), enu.y(), 0.0, latlon.x(), latlon.y(), h);
  return latlon;
}