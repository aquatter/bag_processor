#pragma once
#include <string_view>
#include <types.hpp>
#include <vector>

std::vector<ImageDetections> parse_csv(const std::string_view path);
