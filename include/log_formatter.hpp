#pragma once
#include <fmt/color.h>
#include <fmt/format.h>
#include <ng-log/logging.h>

inline void LogFormatter(std::ostream &s, const nglog::LogMessage &m, void *) {

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