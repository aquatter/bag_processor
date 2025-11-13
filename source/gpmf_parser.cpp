
#include <Eigen/Core>
#include <Eigen/src/Core/Matrix.h>
#include <cam/CamRadtan.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <feat/Feature.h>
#include <feat/FeatureDatabase.h>
#include <fmt/color.h>
#include <fmt/format.h>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/concat.hpp>
#include <range/v3/view/enumerate.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/view/zip.hpp>
#include <string_view>
#include <track/TrackBase.h>
#include <track/TrackKLT.h>
#include <unordered_map>
#include <unordered_set>
#include <utils/sensor_data.h>
#include <vector>
#include <yaml-cpp/node/node.h>
#include <yaml-cpp/node/parse.h>
#include <yaml-cpp/yaml.h>

using ranges::to;
using ranges::views::concat;
using ranges::views::enumerate;
using ranges::views::ints;
using ranges::views::transform;
using ranges::views::zip;

std::shared_ptr<ov_core::CamRadtan>
load_calibration(const std::string_view path) {

  YAML::Node calib{YAML::LoadFile(path.data())};

  const auto intrinsics{calib["cam0"]["intrinsics"].as<std::vector<double>>()};
  const auto distortion{
      calib["cam0"]["distortion_coeffs"].as<std::vector<double>>()};

  auto cam{std::make_shared<ov_core::CamRadtan>(
      calib["cam0"]["resolution"][0].as<int>(),
      calib["cam0"]["resolution"][1].as<int>())};

  Eigen::MatrixXd m;
  m.resize(8, 1);

  for (auto &&[i, val] : enumerate(concat(intrinsics, distortion))) {
    m(i) = val;
  }

  cam->set_value(m);
  return cam;
}

std::vector<cv::Rect> load_boxes(const std::string_view path) {

  std::vector<cv::Rect> res{};
  std::ifstream f{path.data()};

  if (f.is_open() and f.good()) {
    nlohmann::json json_root = nlohmann::json::parse(f);

    for (auto &&json_box : json_root["boxes"]) {

      res.emplace_back(json_box["x"].get<int>(), json_box["y"].get<int>(),
                       json_box["width"].get<int>(),
                       json_box["height"].get<int>());
    }

    return res;
  }

  return {};
}

struct FeatureInTheImage {
  size_t id_;
  Eigen::Vector2f uv_;
};

int main() {

  std::unordered_map<size_t, std::shared_ptr<ov_core::CamBase>> cameras{};
  cameras[0] =
      load_calibration("/root/data/calib_bmi160_1204x768_29_09/"
                       "rosbag2_2025_09_26-22_19_00_converted-camchain.yaml");

  const auto boxes{load_boxes("/root/data/images/boxes.json")};

  ov_core::TrackKLT tracker{
      cameras, 368'640, 0,  false, ov_core::TrackBase::HISTOGRAM,
      20,      128,     96, 1};

  for (auto &&[i, box] : enumerate(boxes)) {

    cv::Mat_<cv::Vec3b> img = cv::imread(
        fmt::format("/root/data/images/image_{}.png", i), cv::IMREAD_UNCHANGED);

    cv::Mat_<uint8_t> img_gray{};
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

    // cv::Mat_<uint8_t> mask = cv::imread(
    //     fmt::format("/root/data/images/mask_{}.png", i),
    //     cv::IMREAD_UNCHANGED);

    cv::Mat_<uint8_t> mask = cv::Mat_<uint8_t>::zeros(img.size());

    tracker.feed_new_camera(
        ov_core::CameraData{.timestamp = static_cast<double>(i),
                            .sensor_ids = {0},
                            .images = {img_gray},
                            .masks = {mask},
                            .boxes = {}});

    // cv::Mat_<cv::Vec3b> img_tmp;
    // img.copyTo(img_tmp);

    // tracker.display_active(img_tmp, 255, 0, 0, 255, 255, 255);

    // cv::imwrite(fmt::format("/root/data/images/img_active_{}.png", i),
    // img_tmp);

    // img.copyTo(img_tmp);
    // tracker.display_history(img_tmp, 255, 0, 0, 255, 255, 255);

    // cv::imwrite(fmt::format("/root/data/images/img_history_{}.png", i),
    //             img_tmp);

    // cv::Mat_<uint8_t> img_grey;
    // cv::cvtColor(img, img_grey, cv::COLOR_BGR2GRAY);

    // std::vector<cv::KeyPoint> key_points;

    // cv::FAST(img_grey, key_points, 20, true);
    // auto pts_refined{key_points | transform([](auto &&val) { return val.pt;
    // }) |
    //                  to<std::vector>()};

    // if (not key_points.empty()) {
    //   cv::cornerSubPix(
    //       img_grey, pts_refined, {5, 5}, {-1, -1},
    //       cv::TermCriteria{cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
    //       20,
    //                        0.001});

    //   for (auto &&[p0, p1] : zip(key_points, pts_refined)) {
    //     p0.pt = p1;

    //     cv::circle(
    //         img, {static_cast<int>(p1.x + 0.5f), static_cast<int>(p1.y +
    //         0.5f)}, 1, cv::Scalar{0.0, 255.0, 0.0}, cv::FILLED, cv::LINE_AA);
    //   }

    //   cv::imwrite(fmt::format("/root/data/images/keypoints_{}.png", i), img);
  }

  const auto feature_db{tracker.get_feature_database()->get_internal_data()};

  std::unordered_map<size_t, std::vector<FeatureInTheImage>> stamp_to_id{};

  for (auto &&[id, feature] : feature_db) {
    for (auto &&[stamp, uv] : zip(feature->timestamps[0], feature->uvs[0])) {
      stamp_to_id[static_cast<int>(stamp)].emplace_back(id, uv);
    }
  }

  std::unordered_set<size_t> features_in_the_box{};

  for (auto &&[stamp, features] : stamp_to_id) {

    for (auto &&feature : features) {
      if (boxes[stamp].contains({static_cast<int>(feature.uv_.x() + 0.5f),
                                 static_cast<int>(feature.uv_.y() + 0.5f)})) {
        features_in_the_box.insert(feature.id_);
      }
    }
  }

  {
    cv::Mat_<cv::Vec3b> img =
        cv::imread("/root/data/images/image_90.png", cv::IMREAD_UNCHANGED);

    for (auto &&id : features_in_the_box) {

      // fmt::print("id: {}, size: {}\n", id, feature_db.at(id)->uvs[0].size());

      if (feature_db.at(id)->uvs[0].size() > 120) {

        bool feature_is_valid{true};

        for (auto &&[stamp, p] :
             zip(feature_db.at(id)->timestamps[0], feature_db.at(id)->uvs[0])) {

          if (not boxes[stamp].contains({static_cast<int>(p.x() + 0.5f),
                                         static_cast<int>(p.y() + 0.5f)})) {

            feature_is_valid = false;
            break;
          }
        }

        if (not feature_is_valid) {
          continue;
        }

        for (auto i{0}; i < feature_db.at(id)->uvs[0].size() - 1; ++i) {

          const Eigen::Vector2i p0{feature_db.at(id)->uvs[0][i].cast<int>()};
          const Eigen::Vector2i p1{
              feature_db.at(id)->uvs[0][i + 1].cast<int>()};

          cv::line(img, {p0.x(), p0.y()}, {p1.x(), p1.y()}, {0.0, 255.0, 255.0},
                   1, cv::LINE_AA);

          cv::circle(img, {p0.x(), p0.y()}, 2, cv::Scalar{0.0, 255.0, 0.0},
                     cv::FILLED, cv::LINE_AA);
        }

        const Eigen::Vector2i p0{feature_db.at(id)->uvs[0].back().cast<int>()};
        cv::circle(img, {p0.x(), p0.y()}, 2, cv::Scalar{0.0, 255.0, 0.0},
                   cv::FILLED, cv::LINE_AA);

        cv::imwrite("/root/data/images/track.png", img);
      }
    }
  }

  // constexpr static int track_id{30231};

  // for (auto &&[ts, p] : zip(feature_db.at(track_id)->timestamps[0],
  //                           feature_db.at(track_id)->uvs[0])) {
  //   cv::Mat_<cv::Vec3b> img = cv::imread(
  //       fmt::format("/root/data/images/image_{}.png",
  //       static_cast<int>(ts)), cv::IMREAD_UNCHANGED);

  //   cv::circle(img,
  //              {static_cast<int>(p.x() + 0.5f), static_cast<int>(p.y() +
  //              0.5f)}, 2, cv::Scalar{0.0, 255.0, 0.0}, cv::FILLED,
  //              cv::LINE_AA);

  //   cv::imwrite(fmt::format("/root/data/images/track_{}_{}.png", track_id,
  //                           static_cast<int>(ts)),
  //               img);
  // }

  // for (auto &&[id, feature] : feature_db) {
  //   fmt::print("id: {}, size: {}\n", id, feature->uvs[0].size());
  // }

  return EXIT_SUCCESS;
  // fmt::print("image #{}, found: {}\n", i, key_points.size());
}
