#include <colmap/geometry/pose_prior.h>
#include <colmap/scene/database.h>
#include <colmap/scene/database_sqlite.h>
#include <colmap/scene/reconstruction.h>
#include <colmap/scene/reconstruction_io_binary.h>
#include <colmap_bridge.hpp>
#include <cstddef>
#include <fmt/color.h>
#include <fmt/format.h>
#include <memory>
#include <string_view>
#include <unordered_map>

namespace colmap_bridge {

struct ColmapDatabase::impl {
  impl(const std::string_view path_to_db)
      : db_{colmap::OpenSqliteDatabase(path_to_db.data())} {}

  void set_pose_prior(uint32_t image_id, const Eigen::Vector3d &pos,
                      const Eigen::Matrix3d &cov) {
    db_->WritePosePrior(
        image_id,
        colmap::PosePrior{pos, cov,
                          colmap::PosePrior::CoordinateSystem::CARTESIAN});
  }

  ~impl() { db_->Close(); }

  std::shared_ptr<colmap::Database> db_;
};

ColmapDatabase::ColmapDatabase(const std::string_view path_to_db)
    : pimpl_{std::make_unique<impl>(path_to_db)} {}

void ColmapDatabase::set_pose_prior(uint32_t image_id,
                                    const Eigen::Vector3d &pos,
                                    const Eigen::Matrix3d &cov) {
  pimpl_->set_pose_prior(image_id, pos, cov);
}

void ColmapDatabase::clear_pose_priors() { pimpl_->db_->ClearPosePriors(); }

ColmapDatabase::~ColmapDatabase() = default;

std::unordered_map<size_t, Eigen::Isometry3d>
read_track(const std::string_view path) {

  colmap::Reconstruction rec{};
  colmap::ReadCamerasBinary(rec, fmt::format("{}/cameras.bin", path.data()));
  colmap::ReadImagesBinary(rec, fmt::format("{}/images.bin", path.data()));

  std::unordered_map<size_t, Eigen::Isometry3d> res{};

  for (auto &&[image_id, image] : rec.Images()) {

    res[static_cast<size_t>(image_id)] = Eigen::Isometry3d{
        Eigen::Translation3d{image.CamFromWorld().translation} *
        Eigen::AngleAxisd{image.CamFromWorld().rotation}};
  }

  return res;
}

} // namespace colmap_bridge