#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/io/file.h>
#include <arrow/table.h>
#include <arrow/type_fwd.h>
#include <memory>
#include <parquet/arrow/reader.h>
#include <parquet_reader.hpp>

std::vector<ImageTrack>
load_tracks_from_parquet(const std::filesystem::path &path) {

  auto maybe_file{arrow::io::ReadableFile::Open(path)};

  if (not maybe_file.ok()) {
  }

  auto reader{parquet::arrow::OpenFile(maybe_file.ValueOrDie(),
                                       arrow::default_memory_pool())
                  .ValueOrDie()};

  std::shared_ptr<arrow::Table> table{};
  if (not reader->ReadTable(&table).ok()) {
  }

  std::cout << table->schema()->ToString() << std::endl;
}