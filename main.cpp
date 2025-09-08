#include "sensor_msgs/msg/compressed_image.hpp"
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fmt/color.h>
#include <fmt/format.h>
#include <fstream>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Quaternion.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <iterator>
#include <memory>
#include <numbers>
#include <opencv2/core.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <rerun.hpp>
#include <rerun/archetypes/geo_line_strings.hpp>
#include <rerun/archetypes/geo_points.hpp>
#include <rerun/components/lat_lon.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>

#include <rosbag2_cpp/reader.hpp>

#include <GeographicLib/LocalCartesian.hpp>
#include <gtsam/geometry/triangulation.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/ISAM2Params.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/BetweenFactor.h>
#include <sstream>
#include <string>
#include <vector>

using gtsam::symbol_shorthand::B;
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

struct GpsMeasurement {
  double time;
  gtsam::Vector3 position;
  gtsam::Vector3 lla;
};

struct CameraMeasurement {
  double time;
  cv::Mat_<cv::Vec3b> img;
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

      const auto timestamp{static_cast<uint64_t>(gps_msg.header.stamp.sec) *
                               1'000'000'000 +
                           static_cast<uint64_t>(gps_msg.header.stamp.nanosec)};

      gps_stamps << timestamp << std::endl;
      continue;

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

        gtsam::GPSFactor gps_factor{X(index), enu, gps_noise_model};

        fmt::print(fmt::fg(fmt::color::yellow_green), "ENU:({}, {}, {})\n",
                   enu.x(), enu.y(), enu.z());

        graph.add(gps_factor);
        prop_state = preintegrated->predict(prev_state, prev_bias);

        fmt::print(fmt::fg(fmt::color::yellow_green),
                   "Prop State:({}, {}, {})\n",
                   prop_state.pose().translation().x(),
                   prop_state.pose().translation().y(),
                   prop_state.pose().translation().z());

        initial_values.insert(X(index), prop_state.pose());
        initial_values.insert(V(index), prop_state.v());
        initial_values.insert(B(index), prev_bias);

        gtsam::LevenbergMarquardtParams params;
        params.setVerbosityLM("SUMMARY");
        gtsam::LevenbergMarquardtOptimizer optimizer{graph, initial_values,
                                                     params};

        auto res{optimizer.optimize()};

        prev_state = gtsam::NavState{res.at<gtsam::Pose3>(X(index)),
                                     res.at<gtsam::Vector3>(V(index))};

        prev_bias = res.at<gtsam::imuBias::ConstantBias>(B(index));

        preintegrated =
            std::make_shared<gtsam::PreintegratedCombinedMeasurements>(
                get_imu_params(-n_gravity), prev_bias);

        // preintegrated->resetIntegrationAndSetBias(prev_bias);

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

      const auto timestamp{static_cast<double>(gps_msg.header.stamp.sec) +
                           static_cast<double>(gps_msg.header.stamp.nanosec) *
                               1.0e-9};

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

    const double t{gps_measurements[i].time};
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

int main(const int argc, const char *const *argv) {

  TestData test_data{.rec_ = rerun::RecordingStream{"bag_converter"}};
  test_data.rec_.connect_grpc().exit_on_failure();

  test_data.reader_.open("/root/data/rosbag2_2025_08_21-09_09_38");
  test_lm(test_data);

  return EXIT_SUCCESS;
}
