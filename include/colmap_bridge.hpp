#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <colmap_bridge_export.h>
#include <memory>
#include <string_view>

namespace colmap_bridge {

class ColmapDatabase {
public:
  COLMAP_BRIDGE_EXPORT ColmapDatabase(const std::string_view path_to_db);
  COLMAP_BRIDGE_EXPORT void set_pose_prior(uint32_t image_id,
                                           const Eigen::Vector3d &pos,
                                           const Eigen::Matrix3d &cov);

  COLMAP_BRIDGE_EXPORT void clear_pose_priors();
  COLMAP_BRIDGE_EXPORT ~ColmapDatabase();

private:
  struct impl;
  std::unique_ptr<impl> pimpl_;
};

COLMAP_BRIDGE_EXPORT std::unordered_map<size_t, Eigen::Isometry3d>
read_track(const std::string_view path);

} // namespace colmap_bridge