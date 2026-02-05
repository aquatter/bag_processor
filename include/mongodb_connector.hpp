#pragma once
#include <memory>
#include <string_view>

class MongoDBConnector {
public:
  MongoDBConnector(const std::string_view uri);
  ~MongoDBConnector();

private:
  struct impl;
  std::unique_ptr<impl> pimpl_;
};