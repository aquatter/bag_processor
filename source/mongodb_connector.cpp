#include "geo_json.hpp"
#include <bsoncxx/builder/basic/array.hpp>
#include <bsoncxx/builder/basic/document.hpp>
#include <bsoncxx/builder/basic/kvp.hpp>
#include <bsoncxx/builder/stream/array.hpp>
#include <bsoncxx/builder/stream/document.hpp>
#include <bsoncxx/builder/stream/helpers.hpp>
#include <bsoncxx/json.hpp>
#include <bsoncxx/types.hpp>
#include <chrono>
#include <cls_idx_mapping.hpp>
#include <cstdint>
#include <cstdlib>
#include <fmt/color.h>
#include <fmt/format.h>
#include <geo_json.hpp>
#include <memory>
#include <mongocxx/client.hpp>
#include <mongocxx/database.hpp>
#include <mongocxx/exception/exception.hpp>
#include <mongocxx/exception/operation_exception.hpp>
#include <mongocxx/instance.hpp>
#include <mongocxx/uri.hpp>
#include <mongocxx/validation_criteria.hpp>
#include <mongodb_connector.hpp>
#include <mongodb_defines.hpp>
#include <ng-log/logging.h>
#include <nlohmann/json.hpp>
#include <range/v3/view/iota.hpp>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace bsoncxx::builder::stream;

using bsoncxx::builder::basic::array;
using bsoncxx::builder::basic::kvp;
using bsoncxx::builder::basic::make_array;
using bsoncxx::builder::basic::make_document;
using bsoncxx::stdx::string_view;
using bsoncxx::types::b_bool;
using bsoncxx::types::b_date;
using bsoncxx::types::b_double;
using bsoncxx::types::b_int32;
using bsoncxx::types::b_int64;
using bsoncxx::types::b_string;
using mongocxx::collection;
using ranges::views::ints;

struct MongoDBConnector::impl {
  impl(const std::string_view uri)
      : inst_{}, client_{mongocxx::uri{uri.data()}},
        db_{client_[MONGODB_NAME.data()]} {
    ensure_geodb_schema();
    nlohmann::json root = nlohmann::json::parse(cls_idx_mapping.data());

    for (auto &&val : root["sign_class_indices"].items()) {
      sign_class_indices_[val.key()] = val.value().get<int>();
    }
  }

  template <collection_enum name> int64_t get_next_id() {
    auto counters{db_[ct<COUNTERS>::name]};

    mongocxx::options::find_one_and_update opts{};
    opts.return_document(mongocxx::options::return_document::k_after);
    opts.upsert(true);

    const auto res{counters.find_one_and_update(
        make_document(kvp("_id", ct<name>::name)),
        make_document(kvp("$inc", make_document(kvp("seq", b_int64{1})))),
        opts)};

    return res.value().view()["seq"].get_int64().value;
  }

  template <collection_enum name> int64_t get_last_id() {
    return db_[ct<COUNTERS>::name]
        .find_one(make_document(kvp("_id", ct<name>::name)))
        .value()
        .view()["seq"]
        .get_int64()
        .value;
  }

  std::unordered_set<std::string> list_collection_names() {
    std::unordered_set<std::string> out;
    for (auto &&name : db_.list_collection_names()) {
      out.insert(name);
    }
    return out;
  }

  void ensure_geodb_schema() {

    if (not db_.has_collection(ct<COUNTERS>::name)) {
      db_.create_collection(ct<COUNTERS>::name,
                            bsoncxx::from_json(ct<COUNTERS>::validator));
      db_[ct<COUNTERS>::name].insert_one(make_document(
          kvp("_id", b_string{ct<RECORDS>::name}), kvp("seq", b_int64{0})));
      db_[ct<COUNTERS>::name].insert_one(
          make_document(kvp("_id", b_string{ct<CALIBRATIONS>::name}),
                        kvp("seq", b_int64{0})));
      db_[ct<COUNTERS>::name].insert_one(make_document(
          kvp("_id", b_string{ct<LANDMARKS>::name}), kvp("seq", b_int64{0})));
      db_[ct<COUNTERS>::name].insert_one(
          make_document(kvp("_id", b_string{ct<CAMERA_POSES>::name}),
                        kvp("seq", b_int64{0})));

      db_[ct<COUNTERS>::name].insert_one(make_document(
          kvp("_id", b_string{ct<DETECTIONS>::name}), kvp("seq", b_int64{0})));
      db_[ct<COUNTERS>::name].insert_one(make_document(
          kvp("_id", b_string{ct<TRACKS>::name}), kvp("seq", b_int64{0})));
    }

    if (not db_.has_collection(ct<RECORDS>::name)) {
      db_.create_collection(ct<RECORDS>::name,
                            bsoncxx::from_json(ct<RECORDS>::validator));

      auto col{db_[ct<RECORDS>::name]};
      create_index_safe(col, make_document(kvp("created_at", -1)).view(),
                        "rec_created");
      create_index_safe(col, make_document(kvp("session_name", 1)).view(),
                        "rec_session");
      create_index_safe(col, make_document(kvp("calib_id", 1)).view(),
                        "rec_calib");
    }

    if (not db_.has_collection(ct<CALIBRATIONS>::name)) {
      db_.create_collection(ct<CALIBRATIONS>::name,
                            bsoncxx::from_json(ct<CALIBRATIONS>::validator));

      auto col{db_[ct<CALIBRATIONS>::name]};
      create_index_safe(col, make_document(kvp("created_at", -1)).view(),
                        "cal_created_at");
    }

    if (not db_.has_collection(ct<LANDMARKS>::name)) {
      db_.create_collection(ct<LANDMARKS>::name,
                            bsoncxx::from_json(ct<LANDMARKS>::validator));

      auto col{db_[ct<LANDMARKS>::name]};
      create_index_safe(col, make_document(kvp("loc", "2dsphere")).view(),
                        "lm_loc_2dsphere");
      create_index_safe(
          col, make_document(kvp("valid", 1), kvp("created_at", -1)).view(),
          "lm_valid_created");
    }

    if (not db_.has_collection(ct<CAMERA_POSES>::name)) {
      db_.create_collection(ct<CAMERA_POSES>::name,
                            bsoncxx::from_json(ct<CAMERA_POSES>::validator));

      auto col{db_[ct<CAMERA_POSES>::name]};
      create_index_safe(col, make_document(kvp("loc", "2dsphere")).view(),
                        "cam_loc_2dsphere");
    }

    if (not db_.has_collection(ct<DETECTIONS>::name)) {
      db_.create_collection(ct<DETECTIONS>::name,
                            bsoncxx::from_json(ct<DETECTIONS>::validator));

      auto col{db_[ct<DETECTIONS>::name]};

      create_index_safe(col, make_document(kvp("frame_id", 1)).view(),
                        "det_frame");
      create_index_safe(
          col, make_document(kvp("track_id", 1), kvp("frame_id", 1)).view(),
          "det_track_frame");

      create_index_safe(col, make_document(kvp("pose_id", 1)).view(),
                        "det_pose");

      create_index_safe(col,
                        make_document(kvp("frame_id", 1), kvp("cls", 1)).view(),
                        "det_frame_cls");
    }

    if (not db_.has_collection(ct<TRACKS>::name)) {
      db_.create_collection(ct<TRACKS>::name,
                            bsoncxx::from_json(ct<TRACKS>::validator));

      auto col{db_[ct<TRACKS>::name]};

      create_index_safe(col, make_document(kvp("record_id", 1)).view(),
                        "trk_record");
      create_index_safe(col, make_document(kvp("landmark_id", 1)).view(),
                        "trk_landmark");
      create_index_safe(col, make_document(kvp("cls", 1)).view(), "trk_cls");
    }
  }

  void create_index_safe(mongocxx::collection &col,
                         const bsoncxx::document::view &keys,
                         const std::string_view name = "",
                         bool background = true) {
    mongocxx::options::index idx_opts;

    if (!name.empty()) {
      idx_opts.name(name.data());
    }

    idx_opts.background(background);

    try {
      col.create_index(keys, idx_opts);
    } catch (const mongocxx::exception &ex) {
      LOG(ERROR) << "Index create warning on " << col.name() << ": "
                 << ex.what();
    }
  }

  void add_landmark(const Landmark &landmark) {

    if (not sign_class_indices_.contains(landmark.code_)) {
      return;
    }

    const auto landmark_id{get_next_id<LANDMARKS>()};

    // clang-format off
    auto doc = document{} << 
      "_id" << b_int64(landmark_id) << 
      "track_ids" << open_array<< b_int64{0} << close_array << 
      "cls" << b_int32{sign_class_indices_.at(landmark.code_)} << 
      "text" << b_string{""} << 
      "latlon" << open_array << landmark.latlon_.x() << landmark.latlon_.y() << close_array << 
      "loc" << open_document <<
        "type" << "Point" <<
        "coordinates" << open_array << landmark.latlon_.y() << landmark.latlon_.x() << close_array <<
      close_document <<
      "azimuth" << b_double{landmark.azimuth_} << 
      "variance" << b_double{landmark.dist_variance_} << 
      "valid" << b_bool{true} << 
      "prev_id" << b_int64{-1} << 
      "created_at" << b_date{std::chrono::system_clock::now()} << 
    finalize;
    // clang-format on

    try {
      db_[ct<LANDMARKS>::name].insert_one(doc.view());
    } catch (const mongocxx::operation_exception &ex) {
      LOG(ERROR) << ex.what();
    }
  }

  void query_2d_sphere(Eigen::Vector2d query_latlon, double dist) {

    // clang-format off
    auto query = document{} << 
      "loc" << open_document <<
        "$nearSphere" << open_document <<
          "$geometry" << open_document <<
            "type" << "Point" <<
            "coordinates" << open_array <<  query_latlon.y() << query_latlon.x() << close_array <<
          close_document <<
          "$maxDistance" << dist <<
        close_document <<
      close_document << 
    finalize;
    // clang-format on

    mongocxx::options::find opts{};
    opts.projection(bsoncxx::from_json(R"({"latlon" : 1})"));

    auto cursor{db_[ct<LANDMARKS>::name].find(query.view(), opts)};

    GeoJson geojson{};

    for (auto &&doc : cursor) {

      const auto lat{doc["latlon"].get_array().value[0].get_double()};
      const auto lon{doc["latlon"].get_array().value[1].get_double()};

      geojson.add_element(
          GeoJson::Point{}
              .with_coordinate_latlon({lat, lon})
              .with_marker_size(GeoJson::Point::MarkerSize::Small)
              .with_marker_color("#1acb11"));
    }

    geojson.save("query_result.geojson");
  }

  mongocxx::instance inst_;
  mongocxx::client client_;
  mongocxx::database db_;
  std::unordered_map<std::string, int> sign_class_indices_;
};

MongoDBConnector::MongoDBConnector(const std::string_view uri)
    : pimpl_{std::make_unique<impl>(uri)} {}

void MongoDBConnector::add_landmark(const Landmark &landmark) {
  pimpl_->add_landmark(landmark);
}

void MongoDBConnector::query_2d_sphere(Eigen::Vector2d query_latlon,
                                       double dist) {
  pimpl_->query_2d_sphere(query_latlon, dist);
}

MongoDBConnector::~MongoDBConnector() = default;