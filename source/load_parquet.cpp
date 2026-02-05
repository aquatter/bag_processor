#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/io/file.h>
#include <arrow/io/interfaces.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/type_fwd.h>
#include <cls_idx_mapping.hpp>
#include <cstddef>
#include <fmt/core.h>
#include <fmt/format.h>
#include <load_parquet.hpp>
#include <memory>
#include <ng-log/logging.h>
#include <nlohmann/json.hpp>
#include <parquet/arrow/reader.h>
#include <parquet/exception.h>
#include <range/v3/view/iota.hpp>
#include <string>
#include <unordered_map>
#include <vector>

using ranges::views::ints;

std::vector<Detection>
load_detections_from_parquet(const std::string_view path) {

  nlohmann::json root = nlohmann::json::parse(cls_idx_mapping.data());
  std::unordered_map<size_t, std::string> classes_mapping{};
  const size_t unknown_class_index{root["unknown_class_index"].get<size_t>()};

  for (auto &&cls : root["reverse_mapping"].items()) {
    classes_mapping[std::stoul(cls.key())] = cls.value().get<std::string>();
  }

  std::shared_ptr<arrow::io::RandomAccessFile> input_file{};
  PARQUET_ASSIGN_OR_THROW(input_file,
                          arrow::io::ReadableFile::Open(path.data()));

  std::unique_ptr<parquet::arrow::FileReader> parquet_reader{};
  PARQUET_THROW_NOT_OK(parquet::arrow::OpenFile(
      input_file, arrow::default_memory_pool(), &parquet_reader));

  std::shared_ptr<arrow::Table> table{};
  PARQUET_THROW_NOT_OK(parquet_reader->ReadTable(&table));

  size_t max_sign_track_id{0};
  size_t detection_id{0};
  std::vector<Detection> dets{};
  std::vector<size_t> barrier_dets{};

  arrow::TableBatchReader batch_reader{*table};

  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch{};
    PARQUET_THROW_NOT_OK(batch_reader.ReadNext(&batch));

    if (!batch) {
      break;
    }

    auto frame_index{std::static_pointer_cast<arrow::Int64Array>(
        batch->GetColumnByName("frame_index"))};

    auto track_id{std::static_pointer_cast<arrow::StringArray>(
        batch->GetColumnByName("track_id"))};

    auto class_index{std::static_pointer_cast<arrow::Int64Array>(
        batch->GetColumnByName("cls_idx"))};

    auto confidence{std::static_pointer_cast<arrow::DoubleArray>(
        batch->GetColumnByName("conf"))};

    auto box{std::static_pointer_cast<arrow::ListArray>(
        batch->GetColumnByName("box"))};

    auto box_values{std::static_pointer_cast<arrow::Int64Array>(box->values())};

    for (auto &&row_num : ints(0l, batch->num_rows())) {

      const auto class_idx{class_index->Value(row_num)};

      if (class_idx == unknown_class_index) {
        continue;
      }

      const auto track_id_str{track_id->Value(row_num)};
      const size_t track_id{std::stoul(track_id_str.substr(2).data())};

      if (track_id_str[0] == 'b') {
        barrier_dets.push_back(dets.size());
      }

      if (track_id_str[0] == 's') {
        max_sign_track_id = std::max(track_id, max_sign_track_id);
      }

      const auto box_value_offset{box->value_offset(row_num)};

      Detection det{};
      det.track_id_ = track_id;
      det.det_id_ = dets.size();
      det.image_id_ = frame_index->Value(row_num);
      det.code_ = classes_mapping.at(class_idx);
      det.confidence_ = confidence->Value(row_num);
      det.box_.x = box_values->Value(box_value_offset);
      det.box_.y = box_values->Value(box_value_offset + 1);
      det.box_.width = box_values->Value(box_value_offset + 2) - det.box_.x + 1;
      det.box_.height =
          box_values->Value(box_value_offset + 3) - det.box_.y + 1;
      det.center_ = {det.box_.x + (det.box_.width >> 1),
                     det.box_.y + (det.box_.height >> 1)};
      det.cumulative_length_ = 0.0;
      det.det_id_ = detection_id;
      ++detection_id;

      dets.push_back(det);
    }
  }

  LOG(INFO) << "num loaded detections: " << detection_id;

  for (auto &&i : barrier_dets) {
    dets[i].track_id_ += max_sign_track_id;
  }

  return dets;
}