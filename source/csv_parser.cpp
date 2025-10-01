#include <charconv>
#include <cstddef>
#include <cstdint>
#include <csv_parser.hpp>
#include <exception>
#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

std::vector<ImageDetections> parse_csv(const std::string_view path) {
  std::vector<ImageDetections> res;

  std::ifstream f{path.data()};

  if (f.is_open() and f.good()) {

    std::string str_line{};

    bool header{true};
    while (std::getline(f, str_line)) {

      if (header) {
        header = false;
        continue;
      }

      std::stringstream string_stream{str_line};
      std::string token{};

      int index{0};
      int64_t sec{0};
      int64_t nanosec{0};
      ImageDetections dets;

      std::string json_string{};

      while (std::getline(string_stream, token, ',')) {

        switch (index) {
        case 0: {
          auto [ptr, ecc] =
              std::from_chars(token.data(), token.data() + token.size(), sec);

          if (ecc == std::errc::invalid_argument or
              ecc == std::errc::result_out_of_range) {
            throw std::runtime_error{
                fmt::format("unable to parse token: {}", token)};
          }
          break;
        }
        case 1: {
          auto [ptr, ecc] = std::from_chars(
              token.data(), token.data() + token.size(), nanosec);

          if (ecc == std::errc::invalid_argument or
              ecc == std::errc::result_out_of_range) {
            throw std::runtime_error{
                fmt::format("unable to parse token: {}", token)};
          }
          break;
        }

        default: {
          if (index >= 10) {
            json_string += token + ',';
          }
        }
        }

        ++index;
      }

      if (json_string.size() < 15) {
        continue;
      }

      json_string[0] = ' ';
      json_string.pop_back();
      json_string.pop_back();

      for (auto &&c : json_string) {
        if (c == '\'') {
          c = '\"';
        }
      }

      try {

        nlohmann::json json = nlohmann::json::parse(json_string);

        nlohmann::json json_dets =
            json.begin().value()["frames"]["photo.jpg_0"]["detections"];

        if (json_dets.empty()) {
          continue;
        }

        dets.dets_.resize(json_dets.size());

        for (size_t i{0}; auto det : json_dets) {

          dets.dets_[i].code_ = det["attributes"]["code"].get<std::string>();
          dets.dets_[i].class_ = det["attributes"]["class"].get<std::string>();

          dets.dets_[i].confidence_ = det["confidence"].get<float>();
          dets.dets_[i].box_.x = det["bbox"][0].get<int>();
          dets.dets_[i].box_.y = det["bbox"][1].get<int>();
          dets.dets_[i].box_.width =
              det["bbox"][2].get<int>() - dets.dets_[i].box_.x + 1;
          dets.dets_[i].box_.height =
              det["bbox"][3].get<int>() - dets.dets_[i].box_.y + 1;

          ++i;
        }

        if (not dets.dets_.empty()) {
          dets.timestamp_ = sec * 1'000'000'000 + nanosec;
          res.emplace_back(std::move(dets));
        }
      } catch (const std::exception &ex) {
        fmt::print(fmt::fg(fmt::color::red), "{}\n", ex.what());
      }
    }
  } else {
    throw std::runtime_error{fmt::format("unable to open: {}", path)};
  }

  std::sort(res.begin(), res.end(), [](const auto &a, const auto &b) {
    return a.timestamp_ < b.timestamp_;
  });

  return res;
}