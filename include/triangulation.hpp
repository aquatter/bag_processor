#pragma once
#include <span>
#include <types.hpp>

[[nodiscard]] Plane3d fit_a_plane(std::span<const Eigen::Vector3d> points);
[[nodiscard]] std::optional<Landmark>
triangulate_on_boxes(std::span<const TrackPoint> track, double sigma_pix = 5.0);