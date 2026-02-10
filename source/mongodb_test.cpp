#include <Eigen/Core>
#include <aws/core/Aws.h>
#include <cstdlib>
#include <exception>
#include <fmt/color.h>
#include <fmt/format.h>
#include <fstream>
#include <log_formatter.hpp>
#include <mongodb_connector.hpp>
#include <ng-log/logging.h>
#include <nlohmann/json.hpp>
#include <string>
#include <types.hpp>
#include <vector>

int main(const int argc, const char *const *argv) {

  Aws::SDKOptions options{};
  Aws::InitAPI(options);

  nglog::InitializeLogging(argv[0]);
  nglog::InstallPrefixFormatter([](std::ostream &s, const nglog::LogMessage &m,
                                   void *ptr) { LogFormatter(s, m, ptr); });
  FLAGS_stderrthreshold = 0;

  try {

    MongoDBConnector db_connector{"mongodb://admin:admin123@localhost:27017"};

    db_connector.query_2d_sphere({41.341414, 69.287815}, 150.0);
    return EXIT_SUCCESS;

    std::ifstream f{"/root/data/records/old_lomakin_20.geojson"};
    nlohmann::json json = nlohmann::json::parse(f);

    size_t landmark_id{0};

    for (auto &&feature : json["features"]) {
      if (feature["geometry"]["type"].get<std::string>() == "Point") {

        const auto lonlat{
            feature["geometry"]["coordinates"].get<std::vector<double>>()};

        Landmark landmark{};

        landmark.id_ = landmark_id;
        landmark.code_ = feature["properties"]["sign_id"].get<std::string>();
        landmark.azimuth_ = feature["properties"]["azimuth"].get<double>();
        landmark.latlon_ = Eigen::Vector2d{lonlat[1], lonlat[0]};
        landmark.dist_variance_ = 0.0;

        db_connector.add_landmark(landmark);
        ++landmark_id;
      }
    }

  } catch (const std::exception &ex) {
    LOG(ERROR) << ex.what();
  }

  return EXIT_SUCCESS;
}