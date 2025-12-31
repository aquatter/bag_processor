#pragma once
#include <filesystem>
#include <types.hpp>
#include <vector>

std::vector<ImageTrack>
load_tracks_from_parquet(const std::filesystem::path &path);