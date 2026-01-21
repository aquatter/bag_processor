#include <CLI/CLI.hpp>
#include <cstdlib>
#include <fmt/color.h>
#include <fmt/format.h>
#include <iostream>
#if USE_RERUN
#include <rerun.hpp>
#endif
#include <tracks_collecton.hpp>
#include <vector>

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

  s << fmt::format("[{}]", prefix_str);
}

int main(const int argc, const char *const *argv) {

  nglog::InitializeLogging(argv[0]);
  nglog::InstallPrefixFormatter(&LogFormatter);
  FLAGS_stderrthreshold = 0;

  try {
    CLI::App app{"Combine multiple bags"};

    std::vector<std::string> input_bags{};

    app.add_option("-i, --input", input_bags, "Specify input path to bags")
        ->required()
        ->check(CLI::ExistingFile);

    bool use_logger{false};

    app.add_flag("--use-logger", use_logger, "Enable rerun logger")
        ->default_val(false);

    CLI11_PARSE(app, argc, argv);

    TracksCollection track_collection{};

#if USE_RERUN
    if (use_logger) {
      auto rec{std::make_shared<rerun::RecordingStream>("bag_converter")};
      rec->connect_grpc().exit_on_failure();
      track_collection.set_rerun(rec);
    }
#endif

    for (auto &&path : input_bags) {
      track_collection.merge(BagProcessor::load(path));
    }

  } catch (const std::exception &ex) {
    LOG(ERROR) << ex.what();
    return EXIT_FAILURE;
  }

  LOG(INFO) << "Processing completed successfully";
  return EXIT_SUCCESS;
}