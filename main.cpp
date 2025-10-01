#if 0
#include "rerun/archetypes/line_strips3d.hpp"
#include "rerun/archetypes/pinhole.hpp"
#include "rerun/archetypes/transform3d.hpp"
#include "rerun/collection.hpp"
#include "rerun/components/pinhole_projection.hpp"
#include "rerun/image_utils.hpp"
#include <Eigen/Geometry>
#include <GeographicLib/LocalCartesian.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <csv_parser.hpp>
#include <fmt/color.h>
#include <fmt/format.h>
#include <fstream>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Quaternion.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/triangulation.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/ISAM2Params.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/BetweenFactor.h>
#include <iterator>
#include <memory>
#include <numbers>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <range/v3/algorithm/max.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/enumerate.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/linear_distribute.hpp>
#include <range/v3/view/transform.hpp>
#include <rerun.hpp>
#include <rerun/archetypes/geo_line_strings.hpp>
#include <rerun/archetypes/geo_points.hpp>
#include <rerun/components/lat_lon.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <tracker.h>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <yaml-cpp/node/node.h>
#include <yaml-cpp/node/parse.h>
#include <yaml-cpp/yaml.h>

using ranges::to;
using ranges::views::enumerate;
using ranges::views::ints;
using ranges::views::linear_distribute;
using ranges::views::transform;

using gtsam::symbol_shorthand::B;
using gtsam::symbol_shorthand::G;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::X;

struct TestData {
  rerun::RecordingStream rec_;
  rosbag2_cpp::Reader reader_{};
};

struct ImuMeasurement {
  double time;
  double dt;
  gtsam::Vector3 accelerometer;
  gtsam::Vector3 gyroscope;
};

struct OpenVinsMeasurements {
  double time;
  Eigen::Vector3d p_;
  Eigen::Quaterniond q_;
  Eigen::Vector3d lla_;
};

std::shared_ptr<gtsam::PreintegratedCombinedMeasurements::Params>
get_imu_params(gtsam::Vector3 n_gravity) {

  const double accel_noise_sigma{0.10939632};
  const double gyro_noise_sigma{0.003664236374078847};
  const double accel_bias_rw_sigma{0.01453129807358673};
  const double gyro_bias_rw_sigma{3.99199279153564e-05};

  auto params{
      std::make_shared<gtsam::PreintegratedCombinedMeasurements::Params>(
          n_gravity)};

  params->accelerometerCovariance =
      gtsam::I_3x3 * std::pow(accel_noise_sigma, 2.0);
  params->integrationCovariance = gtsam::I_3x3 * 1.0e-8;
  params->gyroscopeCovariance = gtsam::I_3x3 * std::pow(gyro_noise_sigma, 2.0);
  params->biasAccCovariance = gtsam::I_3x3 * std::pow(accel_bias_rw_sigma, 2.0);
  params->biasOmegaCovariance =
      gtsam::I_3x3 * std::pow(gyro_bias_rw_sigma, 2.0);
  params->biasAccOmegaInt = gtsam::I_6x6 * 1.0e-5;

  return params;
}

void test_lm(TestData &test_data) {

  static constexpr int num_imu_init_frames{100};
  gtsam::Vector3 n_gravity{0.0, 0.0, 0.0};
  int imu_init_count{0};

  GeographicLib::LocalCartesian local_converter{};

  rclcpp::Serialization<sensor_msgs::msg::NavSatFix> serialization_gps;
  rclcpp::Serialization<sensor_msgs::msg::Imu> serialization_imu;
  rclcpp::Serialization<sensor_msgs::msg::CompressedImage> serialization_image;

  std::vector<rerun::LatLon> points;
  std::vector<rerun::DVec2D> imu_points;
  std::vector<rerun::Vec3D> path;
  std::vector<rerun::Position3D> cam_poses;

  Eigen::Matrix3d imu_transf{Eigen::Matrix3d::Zero()};
  imu_transf(0, 1) = -1.0;
  imu_transf(1, 0) = 1.0;
  imu_transf(2, 2) = 1.0;

  imu_transf = Eigen::AngleAxisd{-25.0 * std::numbers::pi / 180.0,
                                 Eigen::Vector3d::UnitZ()}
                   .matrix() *
               imu_transf;

  uint64_t index{0};
  gtsam::Values initial_values;

  const gtsam::Pose3 prior_pose{
      gtsam::Rot3{gtsam::Quaternion{1.0, 0.0, 0.0, 0.0}},
      gtsam::Point3{0.0, 0.0, 0.0}};

  const gtsam::Vector3 prior_velocity{0.0, 0.0, 0.0};

  gtsam::imuBias::ConstantBias prior_bias;

  initial_values.insert(X(index), prior_pose);
  initial_values.insert(V(index), prior_velocity);
  initial_values.insert(B(index), prior_bias);

  const auto pose_noise_model{gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector6{} << 0.01, 0.01, 0.01, 0.5, 0.5, 0.5).finished())};

  const auto velocity_noise_model{gtsam::noiseModel::Isotropic::Sigma(3, 0.1)};
  const auto bias_noise_model{gtsam::noiseModel::Isotropic::Sigma(6, 1.0e-3)};
  const auto gps_noise_model{gtsam::noiseModel::Isotropic::Sigma(3, 1.0)};

  gtsam::NonlinearFactorGraph graph;

  graph.addPrior(X(index), prior_pose, pose_noise_model);
  graph.addPrior(V(index), prior_velocity, velocity_noise_model);
  graph.addPrior(B(index), prior_bias, bias_noise_model);

  std::shared_ptr<gtsam::PreintegratedCombinedMeasurements> preintegrated;

  gtsam::NavState prev_state{prior_pose, prior_velocity};
  gtsam::NavState prop_state{prev_state};
  gtsam::imuBias::ConstantBias prev_bias{prior_bias};

  gtsam::Pose3 gps_to_imu{gtsam::Rot3::Identity(), gtsam::Vector3::Zero()};

  bool gps_initialized{false};
  bool imu_initialized{false};

  double pre_imu_time{-1.0};

  std::vector<GpsMeasurement> gps_measurements;
  std::vector<CameraMeasurement> camera_measurements;

  std::ofstream cam_stamps{"/root/data/cam_stamps.txt"};
  std::ofstream gps_stamps{"/root/data/gps_stamps.txt"};

  while (test_data.reader_.has_next()) {
    auto msg{test_data.reader_.read_next()};
    const rclcpp::SerializedMessage serialized_msg{*msg->serialized_data};

    if (msg->topic_name == "/camera/image_raw/compressed") {
      sensor_msgs::msg::CompressedImage image_msg;
      serialization_image.deserialize_message(&serialized_msg, &image_msg);

      const auto timestamp{
          static_cast<uint64_t>(image_msg.header.stamp.sec) * 1'000'000'000ull +
          static_cast<uint64_t>(image_msg.header.stamp.nanosec)};

      cam_stamps << timestamp << std::endl;
      continue;

      // camera_measurements.emplace_back(CameraMeasurement{
      //     .time = timestamp, .img = {}
      //     // .img = cv::imdecode(image_msg.data, cv::IMREAD_UNCHANGED)
      // });

      std::sort(gps_measurements.begin(), gps_measurements.end(),
                [](const auto &a, const auto &b) { return a.time < b.time; });

      const auto it{std::upper_bound(
          gps_measurements.begin(), gps_measurements.end(), timestamp,
          [](const double val, const GpsMeasurement &m) {
            return val < m.time;
          })};

      if (it != gps_measurements.end()) {
        const auto ind{std::distance(gps_measurements.begin(), it)};

        if (ind > 0) {

          const auto t{
              (timestamp - gps_measurements[ind - 1].time) /
              (gps_measurements[ind].time - gps_measurements[ind - 1].time)};

          const Eigen::Vector3d cam_pose{gps_measurements[ind - 1].position *
                                             (1.0 - t) +
                                         gps_measurements[ind].position * t};

          cam_poses.emplace_back(
              rerun::Position3D{static_cast<float>(cam_pose.x()),
                                static_cast<float>(cam_pose.y()),
                                static_cast<float>(cam_pose.z())});

          test_data.rec_.log("world/cam_poses",
                             rerun::Points3D{cam_poses}
                                 .with_radii(rerun::Radius::ui_points(10.0f))
                                 .with_colors(rerun::Color{255, 0, 0}));
        }
      }

    } else if (msg->topic_name == "/fix") {

      sensor_msgs::msg::NavSatFix gps_msg;
      serialization_gps.deserialize_message(&serialized_msg, &gps_msg);

      const auto timestamp{static_cast<int64_t>(gps_msg.header.stamp.sec) *
                               1'000'000'000 +
                           static_cast<int64_t>(gps_msg.header.stamp.nanosec)};

      // points.emplace_back(rerun::LatLon{gps_msg.latitude,
      // gps_msg.longitude}); fmt::print("{}, {}\n", gps_msg.latitude,
      // gps_msg.longitude); test_data.rec_.log("gps_track",
      // rerun::GeoPoints{points}.with_radii(
      //                                     {rerun::Radius::ui_points(10.0f)}));

      if (gps_initialized == false) {
        local_converter = GeographicLib::LocalCartesian{
            gps_msg.latitude, gps_msg.longitude, gps_msg.altitude};
        gps_initialized = true;
      } else if (gps_initialized and imu_initialized) {
        ++index;

        gtsam::CombinedImuFactor imu_factor{
            X(index - 1), V(index - 1), X(index),      V(index),
            B(index - 1), B(index),     *preintegrated};

        graph.add(imu_factor);

        gtsam::Point3 enu{};

        local_converter.Forward(gps_msg.latitude, gps_msg.longitude,
                                gps_msg.altitude, enu.x(), enu.y(), enu.z());

        gtsam::GPSFactor gps_factor{G(index), enu, gps_noise_model};

        graph.add(gtsam::BetweenFactor<gtsam::Pose3>{
            G(index), X(index), gps_to_imu, pose_noise_model});

        // fmt::print(fmt::fg(fmt::color::yellow_green), "ENU:({}, {}, {})\n",
        //            enu.x(), enu.y(), enu.z());

        graph.add(gps_factor);

        prop_state = preintegrated->predict(prev_state, prev_bias);

        // fmt::print(fmt::fg(fmt::color::yellow_green),
        //            "Prop State:({}, {}, {})\n",
        //            prop_state.pose().translation().x(),
        //            prop_state.pose().translation().y(),
        //            prop_state.pose().translation().z());

        initial_values.insert(X(index), prop_state.pose());
        initial_values.insert(V(index), prop_state.v());
        initial_values.insert(B(index), prev_bias);
        initial_values.insert(G(index), enu);

        gtsam::LevenbergMarquardtParams params;
        params.setVerbosityLM("SUMMARY");
        gtsam::LevenbergMarquardtOptimizer optimizer{graph, initial_values,
                                                     params};

        auto res{optimizer.optimize()};

        prev_state = gtsam::NavState{res.at<gtsam::Pose3>(X(index)),
                                     res.at<gtsam::Vector3>(V(index))};

        prev_bias = res.at<gtsam::imuBias::ConstantBias>(B(index));

        gps_to_imu = res.at<gtsam::Pose3>(G(index));

        // preintegrated =
        //     std::make_shared<gtsam::PreintegratedCombinedMeasurements>(
        //         get_imu_params(-n_gravity), prev_bias);

        preintegrated->resetIntegrationAndSetBias(prev_bias);

        {
          double lat{0.0};
          double lon{0.0};
          double alt{0.0};

          local_converter.Reverse(prev_state.pose().x(), prev_state.pose().y(),
                                  prev_state.pose().y(), lat, lon, alt);

          // imu_points.emplace_back(gps_msg.latitude, gps_msg.longitude);
          imu_points.emplace_back(lat, lon);
        }

        test_data.rec_.log_static(
            "imu_track",
            rerun::GeoLineStrings{
                rerun::components::GeoLineString::from_lat_lon(imu_points)}
                .with_radii(rerun::Radius::ui_points(5.0f))
                .with_colors(rerun::Color{0, 255, 255}));
      } else if (gps_initialized) {

        Eigen::Vector3d enu{};
        local_converter.Forward(gps_msg.latitude, gps_msg.longitude,
                                gps_msg.altitude, enu.x(), enu.y(), enu.z());

        gps_measurements.push_back(GpsMeasurement{
            .time = timestamp,
            .position = enu,
            .lla = Eigen::Vector3d{gps_msg.latitude, gps_msg.longitude,
                                   gps_msg.altitude}});

        path.emplace_back(enu.x(), enu.y(), enu.z());

        test_data.rec_.log(
            "world/path",
            rerun::LineStrips3D{rerun::LineStrip3D{path}}.with_colors(
                rerun::Color{0, 255, 0}));
      }

    } else if (msg->topic_name == "/imu/mpu6050") {

      continue;

      sensor_msgs::msg::Imu imu_msg;
      serialization_imu.deserialize_message(&serialized_msg, &imu_msg);

      const auto timestamp{static_cast<double>(imu_msg.header.stamp.sec) +
                           static_cast<double>(imu_msg.header.stamp.nanosec) *
                               1.0e-9};

      const Eigen::Vector3d acc{imu_transf *
                                Eigen::Vector3d{imu_msg.linear_acceleration.x,
                                                imu_msg.linear_acceleration.y,
                                                imu_msg.linear_acceleration.z}};

      const Eigen::Vector3d omega{imu_transf *
                                  Eigen::Vector3d{imu_msg.angular_velocity.x,
                                                  imu_msg.angular_velocity.y,
                                                  imu_msg.angular_velocity.z}};

      if (imu_initialized == false) {

        pre_imu_time = timestamp;

        if (imu_init_count < num_imu_init_frames) {
          n_gravity += acc;
          ++imu_init_count;
        } else {
          n_gravity /= static_cast<double>(num_imu_init_frames);

          preintegrated =
              std::make_shared<gtsam::PreintegratedCombinedMeasurements>(
                  get_imu_params(-n_gravity), prior_bias);

          imu_initialized = true;
          continue;
        }

      } else if (imu_initialized and gps_initialized) {
        const double dt{timestamp - pre_imu_time};
        pre_imu_time = timestamp;

        preintegrated->integrateMeasurement(acc, omega, dt);
      }
    }
  }

  cam_stamps.close();
  gps_stamps.close();

  // test_data.rec_.log("track_gps",
  //                    rerun::GeoPoints{points}
  //                        .with_radii(rerun::Radius::ui_points(5.0f))
  //                        .with_colors(rerun::Color{255, 0, 0}));
}

void test_isam(TestData &test_data) {

  std::vector<rerun::DVec2D> imu_points;

  rclcpp::Serialization<sensor_msgs::msg::NavSatFix> serialization_gps;
  rclcpp::Serialization<sensor_msgs::msg::Imu> serialization_imu;

  std::vector<ImuMeasurement> imu_measurements;
  std::vector<GpsMeasurement> gps_measurements;

  GeographicLib::LocalCartesian local_converter{};

  bool gps_initialized{false};
  double pre_imu_time{-1.0};

  while (test_data.reader_.has_next()) {
    auto msg{test_data.reader_.read_next()};
    const rclcpp::SerializedMessage serialized_msg{*msg->serialized_data};

    if (msg->topic_name == "/fix") {

      sensor_msgs::msg::NavSatFix gps_msg;
      serialization_gps.deserialize_message(&serialized_msg, &gps_msg);

      const auto timestamp{static_cast<int64_t>(gps_msg.header.stamp.sec) *
                               1'000'000'000 +
                           static_cast<int64_t>(gps_msg.header.stamp.nanosec)};

      if (not gps_initialized) {
        local_converter = GeographicLib::LocalCartesian{
            gps_msg.latitude, gps_msg.longitude, gps_msg.altitude};
        gps_initialized = true;
        gps_measurements.emplace_back(GpsMeasurement{
            .time = timestamp,
            .position = gtsam::Vector3::Zero(),
            .lla = gtsam::Vector3{gps_msg.latitude, gps_msg.longitude,
                                  gps_msg.altitude}});
      } else {
        gtsam::Point3 enu{};
        local_converter.Forward(gps_msg.latitude, gps_msg.longitude,
                                gps_msg.altitude, enu.x(), enu.y(), enu.z());
        gps_measurements.emplace_back(GpsMeasurement{
            .time = timestamp,
            .position = enu,
            .lla = gtsam::Vector3{gps_msg.latitude, gps_msg.longitude,
                                  gps_msg.altitude}});
      }

    } else if (msg->topic_name == "/imu/mpu6050") {

      sensor_msgs::msg::Imu imu_msg;
      serialization_imu.deserialize_message(&serialized_msg, &imu_msg);

      const auto timestamp{static_cast<double>(imu_msg.header.stamp.sec) +
                           static_cast<double>(imu_msg.header.stamp.nanosec) *
                               1.0e-9};

      if (pre_imu_time == -1.0) {
        pre_imu_time = timestamp;
      } else {
        imu_measurements.emplace_back(ImuMeasurement{
            .time = timestamp,
            .dt = timestamp - pre_imu_time,
            .accelerometer = gtsam::Vector3{imu_msg.linear_acceleration.x,
                                            imu_msg.linear_acceleration.y,
                                            imu_msg.linear_acceleration.z},
            .gyroscope = gtsam::Vector3{imu_msg.angular_velocity.x,
                                        imu_msg.angular_velocity.y,
                                        imu_msg.angular_velocity.z}});
      }
    }
  }

  {
    std::vector<rerun::DVec2D> ov_points;

    std::ifstream f{"/root/data/ov_msckf/state_estimate.txt"};

    std::string line{};
    while (std::getline(f, line)) {
      if (line.front() == '#') {
        continue;
      }

      std::stringstream ss{line};

      OpenVinsMeasurements ov;
      ss >> ov.time >> ov.q_.x() >> ov.q_.y() >> ov.q_.z() >> ov.q_.w() >>
          ov.p_.x() >> ov.p_.y() >> ov.p_.z();

      local_converter.Reverse(ov.p_.x(), ov.p_.y(), ov.p_.z(), ov.lla_.x(),
                              ov.lla_.y(), ov.lla_.z());

      ov_points.emplace_back(ov.lla_.x(), ov.lla_.y());
    }

    for (auto &&val : gps_measurements) {
      imu_points.emplace_back(val.lla.x(), val.lla.y());
    }

    test_data.rec_.log(
        "imu_track",
        rerun::GeoLineStrings{
            rerun::components::GeoLineString::from_lat_lon(imu_points)}
            .with_radii(rerun::Radius::ui_points(5.0f))
            .with_colors(rerun::Color{0, 255, 255}));

    test_data.rec_.log(
        "imu_track",
        rerun::GeoLineStrings{
            rerun::components::GeoLineString::from_lat_lon(ov_points)}
            .with_radii(rerun::Radius::ui_points(5.0f))
            .with_colors(rerun::Color{255, 0, 255}));
  }

  return;

  size_t first_gps_pose{10};

  auto current_pose_global{
      gtsam::Pose3(gtsam::Rot3(), gps_measurements[first_gps_pose].position)};
  gtsam::Vector3 current_velocity_global{gtsam::Vector3::Zero()};
  auto current_bias{gtsam::imuBias::ConstantBias()};

  const auto sigma_init_x{gtsam::noiseModel::Diagonal::Precisions(
      (gtsam::Vector6() << gtsam::Vector3::Constant(0),
       gtsam::Vector3::Constant(1.0))
          .finished())};

  const auto sigma_init_v{
      gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3::Constant(1000.0))};

  const auto sigma_init_b{gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector6() << gtsam::Vector3::Constant(0.100),
       gtsam::Vector3::Constant(5.00e-05))
          .finished())};

  auto noise_model_gps{gtsam::noiseModel::Diagonal::Precisions(
      (gtsam::Vector6() << gtsam::Vector3::Constant(0),
       gtsam::Vector3::Constant(1.0 / 0.07))
          .finished())};

  std::shared_ptr<gtsam::PreintegratedImuMeasurements>
      current_summarized_measurement{};

  gtsam::ISAM2 isam{};
  gtsam::NonlinearFactorGraph new_factors;
  gtsam::Values new_values;
  size_t j{0};

  auto imu_params{gtsam::PreintegratedImuMeasurements::Params::MakeSharedU()};

  gtsam::Matrix33 measured_acc_cov{gtsam::I_3x3 * std::pow(0.010939632, 2)};
  gtsam::Matrix33 measured_omega_cov{gtsam::I_3x3 *
                                     std::pow(0.0003664236374078847, 2)};
  gtsam::Matrix33 integration_error_cov{gtsam::I_3x3 * 1.0e-8};

  imu_params->accelerometerCovariance = measured_acc_cov;
  imu_params->integrationCovariance = integration_error_cov;
  imu_params->gyroscopeCovariance = measured_omega_cov;
  imu_params->omegaCoriolis = gtsam::Vector3::Zero();

  const double accelerometer_bias_sigma{0.01453129807358673};
  const double gyroscope_bias_sigma{3.99199279153564e-05};

  auto previous_pose_key{gtsam::Symbol{}};
  auto previous_vel_key{gtsam::Symbol{}};
  auto previous_bias_key{gtsam::Symbol{}};
  double t_previous{0.0};

  for (size_t i{first_gps_pose}; i < gps_measurements.size() - 1; ++i) {

    auto current_pose_key{X(i)};
    auto current_vel_key{V(i)};
    auto current_bias_key{B(i)};

    const int64_t t{gps_measurements[i].time};
    size_t included_imu_measurement_count{0};

    if (i == first_gps_pose) {
      new_values.insert(current_pose_key, current_pose_global);
      new_values.insert(current_vel_key, current_velocity_global);
      new_values.insert(current_bias_key, current_bias);

      new_factors.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
          current_pose_key, current_pose_global, sigma_init_x);

      new_factors.emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(
          current_vel_key, current_velocity_global, sigma_init_v);

      new_factors
          .emplace_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
              current_bias_key, current_bias, sigma_init_b);

      previous_pose_key = current_pose_key;
      previous_vel_key = current_vel_key;
      previous_bias_key = current_bias_key;
      t_previous = t;
    } else {

      current_summarized_measurement =
          std::make_shared<gtsam::PreintegratedImuMeasurements>(imu_params,
                                                                current_bias);

      while (j < imu_measurements.size() && imu_measurements[j].time <= t) {
        if (imu_measurements[j].time >= t_previous) {
          current_summarized_measurement->integrateMeasurement(
              imu_measurements[j].accelerometer, imu_measurements[j].gyroscope,
              imu_measurements[j].dt);
          included_imu_measurement_count++;
        }
        ++j;
      }

      if (included_imu_measurement_count == 0) {
        continue;
      }

      new_factors.emplace_shared<gtsam::ImuFactor>(
          previous_pose_key, previous_vel_key, current_pose_key,
          current_vel_key, previous_bias_key, *current_summarized_measurement);

      const auto sigma_between_b{gtsam::noiseModel::Diagonal::Sigmas(
          (gtsam::Vector6() << gtsam::Vector3::Constant(
               std::sqrt(included_imu_measurement_count) *
               accelerometer_bias_sigma),
           gtsam::Vector3::Constant(std::sqrt(included_imu_measurement_count) *
                                    gyroscope_bias_sigma))
              .finished())};

      new_factors
          .emplace_shared<gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>>(
              previous_bias_key, current_bias_key,
              gtsam::imuBias::ConstantBias(), sigma_between_b);

      const auto gps_pose{gtsam::Pose3(current_pose_global.rotation(),
                                       gps_measurements[i].position)};

      new_factors.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
          current_pose_key, gps_pose, noise_model_gps);

      new_values.insert(current_pose_key, gps_pose);
      new_values.insert(current_vel_key, current_velocity_global);
      new_values.insert(current_bias_key, current_bias);

      previous_pose_key = current_pose_key;
      previous_vel_key = current_vel_key;
      previous_bias_key = current_bias_key;
      t_previous = t;

      if (i > 30) {
        isam.update(new_factors, new_values);
        isam.update();

        new_factors.resize(0);
        new_values.clear();

        gtsam::Values result{isam.calculateEstimate()};

        current_pose_global = result.at<gtsam::Pose3>(current_pose_key);
        current_velocity_global = result.at<gtsam::Vector3>(current_vel_key);
        current_bias =
            result.at<gtsam::imuBias::ConstantBias>(current_bias_key);

        {
          double lat{0.0};
          double lon{0.0};
          double alt{0.0};

          local_converter.Reverse(current_pose_global.x(),
                                  current_pose_global.y(),
                                  current_pose_global.y(), lat, lon, alt);

          imu_points.emplace_back(lat, lon);
        }

        test_data.rec_.log_static(
            "imu_track",
            rerun::GeoLineStrings{
                rerun::components::GeoLineString::from_lat_lon(imu_points)}
                .with_radii(rerun::Radius::ui_points(5.0f))
                .with_colors(rerun::Color{0, 255, 255}));
      }
    }
  }
}

cv::Mat_<cv::Vec3b> load_image(const std::string_view path, int64_t timestamp) {
  rclcpp::Serialization<sensor_msgs::msg::CompressedImage> serialization_image;
  rosbag2_cpp::Reader reader{};
  reader.open(path.data());

  while (reader.has_next()) {
    auto msg{reader.read_next()};

    if (msg->topic_name == "/camera/image_raw/compressed") {
      const rclcpp::SerializedMessage serialized_msg{*msg->serialized_data};

      sensor_msgs::msg::CompressedImage image_msg;
      serialization_image.deserialize_message(&serialized_msg, &image_msg);

      const auto bag_timestamp{
          static_cast<int64_t>(image_msg.header.stamp.sec) * 1'000'000'000 +
          static_cast<int64_t>(image_msg.header.stamp.nanosec)};

      if (bag_timestamp == timestamp) {
        return cv::imdecode(image_msg.data, cv::IMREAD_UNCHANGED);
      }
    }
  }

  return {};
}

std::vector<ImageTrack> create_tracks(const std::string_view path) {
  auto detections{parse_csv(path)};

  std::unordered_map<std::string, Tracker> trackers;
  std::unordered_map<std::string, ImageTrack> track_map{};

  for (auto &&dets : detections) {

    std::unordered_map<std::string, std::vector<cv::Rect>> det_map;

    for (auto &&d : dets.dets_) {
      det_map[d.code_].push_back(d.box_);

      if (not trackers.contains(d.code_)) {
        trackers[d.code_] = {};
      }
    }

    // const auto det_vec{dets.dets_ |
    //                    transform([](auto &&val) { return val.box_; }) |
    //                    to<std::vector>()};

    // tracker.Run(det_vec);
    // const auto tracks{tracker.GetTracks()};

    for (auto &&[code, tracker] : trackers) {

      const auto &current_dets{det_map[code]};

      tracker.Run(current_dets);
      const auto tracks{tracker.GetTracks()};

      for (auto &&[id, track] : tracks) {

        if (track.coast_cycles_ < kMaxCoastCycles and
            track.hit_streak_ >= kMinHits) {

          size_t det_ind{0};
          float max_iou{0.0f};

          for (auto &&[i, d] : enumerate(dets.dets_)) {
            const float iou{Tracker::CalculateIou(d.box_, track)};
            if (max_iou < iou) {
              det_ind = i;
              max_iou = iou;
            }
          }

          Detection d;
          d.timestamp_ = dets.timestamp_;
          d.box_ = track.GetStateAsBbox();
          d.id_ = id;
          d.class_ = dets.dets_[det_ind].class_;
          d.code_ = code;
          d.confidence_ = dets.dets_[det_ind].confidence_;

          const auto track_id{fmt::format("{}_{}", code, id)};

          if (not track_map.contains(track_id)) {
            track_map[track_id].id_ = track_id;
          }

          track_map[track_id].dets_.emplace_back(d);
        }
      }
    }

#if 0
    if (dets.dets_.size() > 10) {

      cv::Mat_<cv::Vec3b> img =
          load_image("/root/data/rosbag2_2025_08_21-09_32_11", dets.timestamp_);

      if (not img.empty()) {

        for (auto &&d : dets.dets_) {
          cv::rectangle(img, d.box_, {255.0, 0.0, 200.0}, 5);
        }

        cv::imwrite("test.png", img);
      }
    }
#endif
  }

  return track_map | transform([](auto &&val) { return val.second; }) |
         to<std::vector>();
}

void load_measurements(std::vector<CameraMeasurement> &camera,
                       std::vector<GpsMeasurement> &gps,
                       const std::string_view path) {

  rclcpp::Serialization<sensor_msgs::msg::CompressedImage> serialization_image;
  rclcpp::Serialization<sensor_msgs::msg::NavSatFix> serialization_gps;
  rosbag2_cpp::Reader reader{};

  reader.open(path.data());

  GeographicLib::LocalCartesian local_converter{};
  bool gps_initialized{false};

  while (reader.has_next()) {
    auto msg{reader.read_next()};
    const rclcpp::SerializedMessage serialized_msg{*msg->serialized_data};

    if (msg->topic_name == "/camera/image_raw/compressed") {
      sensor_msgs::msg::CompressedImage image_msg;
      serialization_image.deserialize_message(&serialized_msg, &image_msg);

      const auto timestamp{
          static_cast<int64_t>(image_msg.header.stamp.sec) * 1'000'000'000 +
          static_cast<int64_t>(image_msg.header.stamp.nanosec)};

      camera.emplace_back(timestamp);

    } else if (msg->topic_name == "/fix") {
      sensor_msgs::msg::NavSatFix gps_msg;
      serialization_gps.deserialize_message(&serialized_msg, &gps_msg);

      const auto timestamp{static_cast<int64_t>(gps_msg.header.stamp.sec) *
                               1'000'000'000 +
                           static_cast<int64_t>(gps_msg.header.stamp.nanosec)};

      if (not gps_initialized) {
        local_converter = GeographicLib::LocalCartesian{
            gps_msg.latitude, gps_msg.longitude, gps_msg.altitude};
        gps_initialized = true;
      }

      Eigen::Vector3d enu{};
      local_converter.Forward(gps_msg.latitude, gps_msg.longitude,
                              gps_msg.altitude, enu.x(), enu.y(), enu.z());

      gps.emplace_back(timestamp, enu,
                       Eigen::Vector3d{gps_msg.latitude, gps_msg.longitude,
                                       gps_msg.altitude});
    }
  }

  std::sort(camera.begin(), camera.end(),
            [](const auto &a, const auto &b) { return a.time < b.time; });
  std::sort(gps.begin(), gps.end(),
            [](const auto &a, const auto &b) { return a.time < b.time; });
}

std::tuple<gtsam::Cal3_S2, cv::Mat_<double>, cv::Mat_<double>>
load_calibration(const std::string_view path) {
  YAML::Node calib{YAML::LoadFile(path.data())};
  const auto intrinsics{calib["cam0"]["intrinsics"].as<std::vector<double>>()};
  const auto distortion{
      calib["cam0"]["distortion_coeffs"].as<std::vector<double>>()};

  cv::Mat_<double> camera_matrix = cv::Mat_<double>::eye(3, 3);
  camera_matrix(0, 0) = intrinsics[0];
  camera_matrix(1, 1) = intrinsics[1];
  camera_matrix(0, 2) = intrinsics[2];
  camera_matrix(1, 2) = intrinsics[3];

  const cv::Mat_<double> dist_coeffs(distortion, true);

  return {gtsam::Cal3_S2{intrinsics[0], intrinsics[1], 0.0, intrinsics[2],
                         intrinsics[3]},
          camera_matrix, dist_coeffs};
}

template <int degree>
std::tuple<Eigen::Vector2d, std::array<double, degree + 1>>
estimate_direction(std::span<const Eigen::Vector2d> points,
                   Eigen::Vector2d query_point) {
  const int n{static_cast<int>(points.size())};

  Eigen::MatrixXd A;
  Eigen::MatrixXd b;
  A.resize(n, degree + 1);
  b.resize(n, 1);

  for (auto &&[i, p] : enumerate(points)) {
    double x_val{1.0};

    for (int &&j : ints(0, degree + 1)) {
      A(i, j) = x_val;
      x_val *= static_cast<double>(p.x());
    }

    b(i) = p.y();
  }

  const Eigen::MatrixXd p{(A.transpose() * A).ldlt().solve(A.transpose() * b)};

  double y_prime{0.0};
  double x_val{1.0};

  for (auto &&i : ints(1, degree + 1)) {
    y_prime += p(i) * x_val * static_cast<double>(i);
    x_val *= query_point.x();
  }

  std::array<double, degree + 1> poly{};
  for (auto &&i : ints(0, degree + 1)) {
    poly[i] = p(i);
  }

  return {Eigen::Vector2d{1.0, y_prime}.normalized(), poly};
}

std::tuple<std::vector<Eigen::Vector2d>, Eigen::Vector2d>
get_points_in_the_radius(std::span<const GpsMeasurement> points, double rad,
                         Eigen::Vector3d query_point) {

  const double rad_squared{rad * rad};

  std::vector<Eigen::Vector2d> res;
  std::optional<Eigen::Vector2d> first_point{};
  Eigen::Vector2d last_point{};

  for (auto &&p : points) {

    if ((p.position - query_point).squaredNorm() < rad_squared) {
      res.emplace_back(p.position.x(), p.position.y());

      if (not first_point.has_value()) {
        first_point = p.position.head(2);
      }
      last_point = p.position.head(2);
    }
  }

  last_point = (last_point - first_point.value()).normalized();

  return {res, last_point};
}

int main(const int argc, const char *const *argv) {

  auto [gtsam_cal3_s2, camera_matrix, dist_coeffs] = load_calibration(
      "/root/data/calibration_20_08/imu_camera_20_08-camchain.yaml");

  // {
  //   auto img = cv::imread("test.png", cv::IMREAD_UNCHANGED);
  //   cv::Mat_<cv::Vec3b> img_und{};
  //   cv::undistort(img, img_und, camera_matrix, dist_coeffs);
  //   cv::imwrite("test_undist.png", img_und);
  // }

  std::vector<ImageTrack> image_tracks{
      create_tracks("/root/data/detects_for_rosbag2_2025_08_21-09_32_11.csv")};

  std::vector<CameraMeasurement> camera;
  std::vector<GpsMeasurement> gps;

  load_measurements(camera, gps, "/root/data/rosbag2_2025_08_21-09_32_11");

  rerun::RecordingStream rec{"bag_converter"};
  rec.connect_grpc().exit_on_failure();

  const std::vector<rerun::Vec3D> gps_path{
      gps | transform([](auto &&val) {
        return rerun::Vec3D{static_cast<float>(val.position.x()),
                            static_cast<float>(val.position.y()), 0.0f};
      }) |
      to<std::vector>()};

  static constexpr int cam_num{329};

  {

    rerun::Mat3x3 rerun_camera{rerun::Mat3x3::IDENTITY};
    rerun_camera.flat_columns[0] = gtsam_cal3_s2.fx();
    rerun_camera.flat_columns[4] = gtsam_cal3_s2.fy();
    rerun_camera.flat_columns[6] = gtsam_cal3_s2.px();
    rerun_camera.flat_columns[7] = gtsam_cal3_s2.py();

    const auto &cam{camera[cam_num]};

    cv::Mat_<cv::Vec3b> img =
        load_image("/root/data/rosbag2_2025_08_21-09_32_11", cam.time);

    cv::Mat_<cv::Vec3b> img_undist;
    cv::undistort(img, img_undist, camera_matrix, dist_coeffs);
    cv::cvtColor(img_undist, img_undist, cv::COLOR_BGR2RGB);

    const auto it{
        std::upper_bound(gps.begin(), gps.end(), cam.time,
                         [](const double val, const GpsMeasurement &m) {
                           return val < m.time;
                         })};

    if (it != gps.end()) {
      const auto ind{std::distance(gps.begin(), it)};

      if (ind > 0) {

        const auto t{static_cast<double>(cam.time - gps[ind - 1].time) /
                     static_cast<double>(gps[ind].time - gps[ind - 1].time)};

        const Eigen::Vector3d cam_pose{gps[ind - 1].position * (1.0 - t) +
                                       gps[ind].position * t};

        const auto [points_in_the_radius, direction] =
            get_points_in_the_radius(gps, 10.0, cam_pose);

        auto [estimated_direction, poly] =
            estimate_direction<2>(points_in_the_radius, cam_pose.head(2));

        if (estimated_direction.dot(direction) < 0.0) {
          estimated_direction *= -1.0;
        }

        {

          const auto alpha{
              180.0 *
              std::atan2(estimated_direction.y(), estimated_direction.x()) /
              std::numbers::pi};

          fmt::print(fmt::fg(fmt::color::bisque), "angle: {}\n", alpha);
        }
#if 0
        {

          auto [min_it, max_it] = std::minmax_element(
              points_in_the_radius.begin(), points_in_the_radius.end(),
              [](const auto &a, const auto &b) { return a.x() < b.x(); });

          std::vector<rerun::Vec3D> poly_points;
          for (auto &&x : linear_distribute(min_it->x(), max_it->x(), 30)) {

            double y{0.0};
            double x_val{1.0};

            for (auto &&p : poly) {
              y += p * x_val;
              x_val *= x;
            }

            poly_points.emplace_back(x, y, 0.0);
          }

          rec.log("world/poly",
                  rerun::LineStrips3D{rerun::LineStrip3D{poly_points}}
                      .with_colors(rerun::Color{120, 255, 40})
                      .with_radii(rerun::Radius::ui_points(4.0f)));

          poly_points.clear();
          for (auto &&p : points_in_the_radius) {
            poly_points.emplace_back(static_cast<float>(p.x()),
                                     static_cast<float>(p.y()), 0.0f);
          }

          rec.log("world/selected_points",
                  rerun::LineStrips3D{rerun::LineStrip3D{poly_points}}
                      .with_colors(rerun::Color{255, 0, 0})
                      .with_radii(rerun::Radius::ui_points(2.0f)));
        }
#endif

        const Eigen::Matrix3f r{Eigen::Isometry3f{
            Eigen::AngleAxisf{
                static_cast<float>(-std::atan2(estimated_direction.x(),
                                               estimated_direction.y())),
                Eigen::Vector3f::UnitZ()} *
            Eigen::AngleAxisf{-0.5f * std::numbers::pi_v<float>,
                              Eigen::Vector3f::UnitX()}}
                                    .linear()};

        rec.log("world/camera",
                rerun::Transform3D{
                    rerun::Vec3D{static_cast<float>(cam_pose.x()),
                                 static_cast<float>(cam_pose.y()), 0.0f},
                    rerun::Mat3x3{r.data()}});

        rec.log(
            "world/camera/image",
            rerun::Pinhole{rerun::components::PinholeProjection{rerun_camera}});

        rec.log(
            "world/camera/image",
            rerun::Image::from_rgb24(
                rerun::Collection<uint8_t>::borrow(
                    img_undist.data,
                    img_undist.cols * img_undist.rows * img_undist.channels()),
                rerun::WidthHeight{static_cast<uint32_t>(img_undist.cols),
                                   static_cast<uint32_t>(img_undist.rows)}));

        rec.log("world/path",
                rerun::LineStrips3D{rerun::LineStrip3D{gps_path}}.with_colors(
                    rerun::Color{0, 255, 0}));

        rec.log("world/X",
                rerun::LineStrips3D{
                    rerun::LineStrip3D{{rerun::Vec3D{0.0f, 0.0f, 0.0f},
                                        rerun::Vec3D{100.0f, 0.0f, 0.0f}}}}
                    .with_colors(rerun::Color{255, 0, 0})
                    .with_radii(rerun::Radius::ui_points(10.0f)));

        rec.log("world/Y",
                rerun::LineStrips3D{
                    rerun::LineStrip3D{{rerun::Vec3D{0.0f, 0.0f, 0.0f},
                                        rerun::Vec3D{0.0f, 100.0f, 0.0f}}}}
                    .with_colors(rerun::Color{0, 255, 0})
                    .with_radii(rerun::Radius::ui_points(10.0f)));

        rec.log("world/Z",
                rerun::LineStrips3D{
                    rerun::LineStrip3D{{rerun::Vec3D{0.0f, 0.0f, 0.0f},
                                        rerun::Vec3D{0.0f, 0.0f, 100.0f}}}}
                    .with_colors(rerun::Color{0, 0, 255})
                    .with_radii(rerun::Radius::ui_points(10.0f)));

        {

          const Eigen::Vector2d p1{cam_pose.head(2) +
                                   30.0 * estimated_direction};

          rec.log("world/direction",
                  rerun::LineStrips3D{
                      rerun::LineStrip3D{
                          {rerun::Vec3D{static_cast<float>(cam_pose.x()),
                                        static_cast<float>(cam_pose.y()), 0.0},
                           rerun::Vec3D{static_cast<float>(p1.x()),
                                        static_cast<float>(p1.y()), 0.0f}}}}
                      .with_colors(rerun::Color{255, 255, 0})
                      .with_radii(rerun::Radius::ui_points(3.0f)));
        }
      }
    }
  }

  return EXIT_SUCCESS;

  std::vector<rerun::Position3D> cam_poses;
  std::vector<rerun::Text> labels;

  for (auto &&[i, cam] : enumerate(camera)) {

    const auto it{
        std::upper_bound(gps.begin(), gps.end(), cam.time,
                         [](const double val, const GpsMeasurement &m) {
                           return val < m.time;
                         })};

    if (it != gps.end()) {
      const auto ind{std::distance(gps.begin(), it)};

      if (ind > 0) {

        const auto t{(cam.time - gps[ind - 1].time) /
                     (gps[ind].time - gps[ind - 1].time)};

        const Eigen::Vector3d cam_pose{gps[ind - 1].position * (1.0 - t) +
                                       gps[ind].position * t};

        cam_poses.emplace_back(
            rerun::Position3D{static_cast<float>(cam_pose.x()),
                              static_cast<float>(cam_pose.y()), 0.0f});

        labels.push_back(fmt::format("camera_{}", i));
      }
    }
  }
  // const std::vector<rerun::DVec2D> gps_points{gps | transform([](auto &&val)
  // {
  //                                               return rerun::DVec2D{
  //                                                   val.lla.x(),
  //                                                   val.lla.y()};
  //                                             }) |
  //                                             to<std::vector>()};

  rec.log("world/cam_poses", rerun::Points3D{cam_poses}
                                 .with_radii(rerun::Radius::ui_points(2.0f))
                                 .with_colors(rerun::Color{255, 0, 0})
                                 .with_labels(labels));

  // rec.log("shit33",
  //         rerun::GeoLineStrings{
  //             rerun::components::GeoLineString::from_lat_lon(gps_points)}
  //             .with_radii(rerun::Radius::ui_points(5.0f))
  //             .with_colors(rerun::Color{0, 255, 255}));

  return EXIT_SUCCESS;

  TestData test_data{.rec_ = rerun::RecordingStream{"bag_converter"}};
  test_data.rec_.connect_grpc().exit_on_failure();

  test_data.reader_.open("/root/data/rosbag2_2025_08_21-09_09_38");
  test_lm(test_data);

  return EXIT_SUCCESS;
}

#endif

#include <bag_processor.hpp>
#include <cstdlib>
#include <fmt/color.h>
#include <fmt/format.h>

void LogFormatter(std::ostream &s, const nglog::LogMessage &m, void *) {

  std::string prefix_str{};

  switch (m.severity()) {
  case nglog::NGLOG_INFO:
    prefix_str = fmt::format(fmt::fg(fmt::color::spring_green), "INFO");
    break;
  case nglog::NGLOG_WARNING:
    prefix_str =
        fmt::format(fmt::fg(fmt::color::light_golden_rod_yellow), "WARNING");
    break;
  case nglog::NGLOG_ERROR:
    prefix_str = fmt::format(fmt::fg(fmt::color::indian_red), "ERROR");
    break;
  case nglog::NGLOG_FATAL:
    prefix_str = fmt::format(fmt::fg(fmt::color::medium_violet_red), "FATAL");
    break;
  }

  s << fmt::format("[{} {}.{}.{} {}:{}:{} {}:{}]", prefix_str, m.time().day(),
                   1 + m.time().month(), 1900 + m.time().year(),
                   m.time().hour(), m.time().min(), m.time().sec(),
                   m.basename(), m.line());
}

int main(const int argc, const char *const *argv) {

  nglog::InitializeLogging(argv[0]);
  nglog::InstallPrefixFormatter(&LogFormatter);
  FLAGS_stderrthreshold = 0;

  constexpr static std::string_view file_name{"video20250831-125017"};
  constexpr static std::string_view folder_name{"domodedovo/video_domodedovo"};

  try {
    BagProcessor bag_proc{
        {.bag_path_ = fmt::format("/root/data/{}/{}", folder_name.data(),
                                  file_name.data()),
         .annotations_path_ = fmt::format("/root/data/{}/detects_{}.jsonl",
                                          folder_name.data(), file_name.data()),
         .calibration_path_ = "/root/data/domodedovo/video_domodedovo/calib/"
                              "VID_20250929_134020-camchain.yaml",
         .ground_truth_path_ =
             fmt::format("/root/data/{}/gt.geojson", folder_name.data()),
         .use_logger_ = true}};

    LOG(INFO) << "Bag processor initialized";

    const int64_t from_timestamp{1759256481426434007 - 2 * 1'000'000'000l};
    const int64_t to_timestamp{from_timestamp + 10 * 1'000'000'000l};

    // const auto landmark{bag_proc.triangulate("8.1.1_11")};

    const auto found_landmarks{bag_proc.triangulate_tracks()};
    bag_proc.log_gps_path_map()
        .log_ground_truth_landmarks("3.27")
        .log_track("3.27_43")
        .log_gps_path()
        .log_axis()
        .log_track_directions("3.27_43", 150.0f);

    for (auto &&landmark : found_landmarks) {
      if (landmark.code_ == "3.27") {
        Landmark l{landmark};
        l.code_ += "__detected";
        bag_proc.log_landmark_map(l, {0, 255, 0});
      }
    }

    // .log_landmarks_map(found_landmarks)
    // .save_geojson(found_landmarks,
    //               fmt::format("/root/data/{}/{}.geojson",
    //                           folder_name.data(), file_name.data()));

    // bag_proc.log_axis().log_gps_path_map();
    // bag_proc.log_axis().log_gps_path().log_poly(1756128239494864977);

    // bag_proc.log_axis().log_gps_path().log_images(from_timestamp,
    // to_timestamp); .log_track("3.27_116");

    // .log_direction("8.1.1_37", 1755768830363105351, 30.0f);

    // .log_track("8.1.1_37");
    // .log_poly(1755768834763552351)

    // .log_poly(1755768834405726351);
    // .log_camera(1755768834405726351);
    // bag_proc.log_images(from_timestamp, to_timestamp);
    /// world/camera_1755768834405726351
  } catch (const std::exception &ex) {
    LOG(ERROR) << ex.what();
  }

  return EXIT_SUCCESS;
}