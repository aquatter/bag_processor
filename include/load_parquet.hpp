#pragma once
#include <string_view>
#include <types.hpp>
#include <vector>

[[nodiscard]] std::vector<Detection>
load_detections_from_parquet(const std::string_view path);