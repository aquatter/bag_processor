#pragma once

#include <Eigen/Core>
#include <GeographicLib/LocalCartesian.hpp>
#include <serialization.hpp>

class CartesianConverter {
public:
  CartesianConverter();
  CartesianConverter(Eigen::Vector2d geodetic_origin);

  void set_origin(Eigen::Vector2d geodetic_origin);

  [[nodiscard]] Eigen::Vector2d origin() const noexcept;
  [[nodiscard]] Eigen::Vector2d enu(Eigen::Vector2d latlon) const noexcept;
  [[nodiscard]] Eigen::Vector2d latlon(Eigen::Vector2d enu) const noexcept;

  [[nodiscard]] bool origin_set() const noexcept;

private:
  friend class boost::serialization::access;

  template <typename Archive> void serialize(Archive &ar, const unsigned int) {
    ar & origin_set_;
    ar & geodetic_origin_;

    if (Archive::is_loading::value) {
      set_origin(geodetic_origin_);
    }
  }

  Eigen::Vector2d geodetic_origin_;
  GeographicLib::LocalCartesian converter_;
  bool origin_set_;
};