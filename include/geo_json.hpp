#pragma once

#include <Eigen/Core>
#include <boost/math/constants/constants.hpp>
#include <cartesian_converter.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include <range/v3/view/linear_distribute.hpp>
#include <span>
#include <string_view>

class GeoJson {
public:
  template <typename Derived> struct BaseElement {
    BaseElement() {
      element_["type"] = "Feature";
      element_["properties"]["stroke"] = "#002791";
      element_["properties"]["stroke-width"] = 1;
      element_["properties"]["stroke-opacity"] = 1.0;
    }

    template <typename T>
    Derived &with_property(const std::string_view key, T value) {
      element_["geometry"][key.data()] = value;
      return static_cast<Derived &>(*this);
    }

    Derived &with_stroke_color(const std::string_view color) {
      element_["properties"]["stroke"] = color.data();
      return static_cast<Derived &>(*this);
    }

    Derived &with_stroke_width(int width) {
      element_["properties"]["stroke-width"] = width;
      return static_cast<Derived &>(*this);
    }

    Derived &with_stroke_opacity(double opacity) {
      element_["properties"]["stroke-opacity"] = opacity;
      return static_cast<Derived &>(*this);
    }

    Derived &with_description(const std::string_view str) {
      element_["properties"]["description"] = str.data();
      return static_cast<Derived &>(*this);
    }

    nlohmann::json element_{};
  };

  struct LineString : public BaseElement<LineString> {

    LineString() { element_["geometry"]["type"] = "LineString"; }

    LineString &
    with_coordinates_latlon(std::span<const Eigen::Vector2d> coords) {

      for (auto &&p : coords) {
        element_["geometry"]["coordinates"].push_back({p.y(), p.x()});
      }

      return *this;
    }
  };

  template <typename Derived> struct Polygon : public BaseElement<Derived> {
    using base = BaseElement<Derived>;
    Polygon() {
      base::element_["geometry"]["type"] = "Polygon";
      base::element_["properties"]["fill"] = "#299100";
      base::element_["properties"]["fill-opacity"] = 0.5;
    }

    Derived &with_fill_color(const std::string_view color) {
      base::element_["properties"]["fill"] = color.data();
      return static_cast<Derived &>(*this);
    }

    Derived &with_fill_opacity(double opacity) {
      base::element_["properties"]["fill-opacity"] = opacity;
      return static_cast<Derived &>(*this);
    }
  };

  struct Line : public BaseElement<Line> {
    Line() { element_["geometry"]["type"] = "LineString"; }

    Line &with_coordinates_latlon(const Eigen::Vector2d &from,
                                  const Eigen::Vector2d &to) {

      element_["geometry"]["coordinates"].push_back({from.y(), from.x()});
      element_["geometry"]["coordinates"].push_back({to.y(), to.x()});

      return *this;
    }
  };

  struct Square : public Polygon<Square> {
    Square &with_size(double size) {
      marker_size_ = size;
      return *this;
    }

    Square &with_coordinate_latlon(const Eigen::Vector2d &p) {
      const CartesianConverter converter{p};
      std::vector<std::vector<double>> d{};

      auto latlon{converter.latlon({-marker_size_, -marker_size_})};
      d.push_back({latlon.y(), latlon.x()});
      latlon = converter.latlon({marker_size_, -marker_size_});
      d.push_back({latlon.y(), latlon.x()});
      latlon = converter.latlon({marker_size_, marker_size_});
      d.push_back({latlon.y(), latlon.x()});
      latlon = converter.latlon({-marker_size_, marker_size_});
      d.push_back({latlon.y(), latlon.x()});
      latlon = converter.latlon({-marker_size_, -marker_size_});
      d.push_back({latlon.y(), latlon.x()});

      element_["geometry"]["coordinates"].push_back(d);

      return *this;
    }

    double marker_size_{0.1};
  };

  struct Circle : public Polygon<Circle> {

    Circle &with_size(double size) {
      marker_size_ = size;
      return *this;
    }

    Circle &with_coordinate_latlon(const Eigen::Vector2d &p) {
      const CartesianConverter converter{p};
      std::vector<std::vector<double>> d{};

      for (auto &&angle : ranges::views::linear_distribute(
               0.0, boost::math::double_constants::two_pi, 30)) {

        const auto latlon{converter.latlon(
            {marker_size_ * std::cos(angle), marker_size_ * std::sin(angle)})};

        d.push_back({latlon.y(), latlon.x()});
      }

      element_["geometry"]["coordinates"].push_back(d);
      return *this;
    }

    double marker_size_{0.1};
  };

  struct Point : public BaseElement<Point> {
    enum MarkerSize { Small, Medium, Large };

    Point() {
      element_["properties"]["marker-color"] = "#002791";
      element_["properties"]["marker-size"] = "medium";
      element_["geometry"]["type"] = "Point";
    }

    Point &with_marker_color(const std::string_view color) {
      element_["properties"]["marker-color"] = color.data();
      return *this;
    }

    Point &with_coordinate_latlon(const Eigen::Vector2d &p) {
      element_["geometry"]["coordinates"].push_back(p.y());
      element_["geometry"]["coordinates"].push_back(p.x());
      return *this;
    }

    Point &with_description(const std::string_view str) {
      element_["properties"]["description"] = str.data();
      return *this;
    }

    Point &with_title(const std::string_view str) {
      element_["properties"]["title"] = str.data();
      return *this;
    }

    Point &with_marker_size(MarkerSize size) {
      switch (size) {
      case Small:
        element_["properties"]["marker-size"] = "small";
        break;
      case Medium:
        element_["properties"]["marker-size"] = "medium";
        break;
      case Large:
        element_["properties"]["marker-size"] = "large";
        break;
      }
      return *this;
    }

    Point &with_sign_id(const std::string_view str) {
      element_["properties"]["sign_id"] = str.data();
      return *this;
    }
  };

  GeoJson(const std::string_view name = "Some Shitty GeoJson") {
    root_["type"] = "FeatureCollection";
    root_["name"] = name.data();
  }

  template <typename T> void add_element(const T &element) {
    root_["features"].push_back(element.element_);
  }

  void save(const std::string_view path) const {
    std::ofstream f{path.data()};
    f << root_.dump(4);
  }

private:
  nlohmann::json root_;
};