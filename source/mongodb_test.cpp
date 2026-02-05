#include <cstdlib>
#include <exception>
#include <fmt/color.h>
#include <fmt/format.h>
#include <log_formatter.hpp>
#include <mongodb_connector.hpp>
#include <ng-log/logging.h>

int main(const int argc, const char *const *argv) {

  nglog::InitializeLogging(argv[0]);
  nglog::InstallPrefixFormatter([](std::ostream &s, const nglog::LogMessage &m,
                                   void *ptr) { LogFormatter(s, m, ptr); });
  FLAGS_stderrthreshold = 0;

  try {

    MongoDBConnector db_connector{"mongodb://admin:admin123@localhost:27017"};

  } catch (const std::exception &ex) {
    LOG(ERROR) << ex.what();
  }

  return EXIT_SUCCESS;
}