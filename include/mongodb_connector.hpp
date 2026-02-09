#pragma once
#include <Eigen/Core>
#include <memory>
#include <string_view>
#include <types.hpp>

class MongoDBConnector {
public:
  MongoDBConnector(const std::string_view uri);
  ~MongoDBConnector();

  void add_landmark(const Landmark &landmark);
  void query_2d_sphere(Eigen::Vector2d query_latlon, double dist);

private:
  struct impl;
  std::unique_ptr<impl> pimpl_;
};