#pragma once
#include <span>
#include <types.hpp>

Plane3d fit_a_plane(std::span<const Eigen::Vector3d> points);
std::optional<Landmark> triangulate_on_boxes(std::span<const TrackPoint> track);