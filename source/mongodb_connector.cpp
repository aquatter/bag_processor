#include <bsoncxx/builder/basic/array.hpp>
#include <bsoncxx/builder/basic/document.hpp>
#include <bsoncxx/builder/basic/kvp.hpp>
#include <bsoncxx/json.hpp>
#include <cstdlib>
#include <fmt/color.h>
#include <fmt/format.h>
#include <memory>
#include <mongocxx/client.hpp>
#include <mongocxx/exception/exception.hpp>
#include <mongocxx/instance.hpp>
#include <mongocxx/uri.hpp>
#include <mongocxx/validation_criteria.hpp>
#include <mongodb_connector.hpp>
#include <ng-log/logging.h>
#include <string_view>
#include <unordered_set>

using bsoncxx::builder::basic::kvp;
using bsoncxx::builder::basic::make_array;
using bsoncxx::builder::basic::make_document;
using bsoncxx::stdx::string_view;
using mongocxx::collection;

struct MongoDBConnector::impl {
  impl(const std::string_view uri) : inst_{}, client_ {
    mongocxx::uri{uri.data()} { ensure_geodb_schema(); }

    std::unordered_set<std::string> list_collection_names() {
      std::unordered_set<std::string> out;
      for (auto &&name : db_.list_collection_names()) {
        out.insert(name);
      }
      return out;
    }

    void create_collection_if_missing(
        const std::string_view name, const bsoncxx::document::view &validator) {
      if (list_collection_names().count(name.data())) {
        return;
      }

      try {
        db_.create_collection(name, make_document(validator));
      } catch (const mongocxx::exception &ex) {
        LOG(ERROR) << "Failed to create collection " << name << ": "
                   << ex.what();
      }
    }

    bsoncxx::document::value schema_array_fixed(
        const std::string_view item_bson_type, int n) const {
      return make_document(
          kvp("bsonType", "array"), kvp("minItems", n), kvp("maxItems", n),
          kvp("items", make_document(kvp("bsonType", item_bson_type))));
    }

    bsoncxx::document::value schema_geopoint() const {
      return make_document(
          kvp("bsonType", "object"),
          kvp("required", make_array("type", "coordinates")),
          kvp("properties",
              make_document(
                  kvp("type", make_document(kvp("enum", make_array("Point")))),
                  kvp("coordinates",
                      make_document(
                          kvp("bsonType", "array"), kvp("minItems", 2),
                          kvp("maxItems", 2),
                          kvp("items",
                              make_document(kvp("bsonType", "double"))))))));
    }

    void ensure_geodb_schema() {
      {
        const auto validator{make_document(kvp(
            "$jsonSchema",
            make_document(
                kvp("bsonType", "object"),
                kvp("required",
                    make_array("_id", "camera_type", "model", "K", "D",
                               "resolution", "created_at", "source")),
                kvp("properties",
                    make_document(
                        kvp("_id", make_document(kvp("bsonType", "long"))),
                        kvp("camera_type",
                            make_document(kvp("bsonType", "string"))),
                        kvp("model", make_document(kvp("bsonType", "string"))),
                        kvp("K", schema_array_fixed("double", 4).view()),
                        kvp("D", schema_array_fixed("double", 5).view()),
                        kvp("resolution",
                            make_document(
                                kvp("bsonType", "object"),
                                kvp("required", make_array("width", "height")),
                                kvp("properties",
                                    make_document(
                                        kvp("width", make_document(kvp(
                                                         "bsonType", "int"))),
                                        kvp("height",
                                            make_document(
                                                kvp("bsonType", "int"))))))),
                        kvp("created_at",
                            make_document(kvp("bsonType", "date"))),
                        kvp("source",
                            make_document(kvp("bsonType", "string"))))))))};

        create_collection_if_missing("calibrations", validator.view());

        auto col{db_["calibrations"]};
        create_index_safe(col, make_document(kvp("created_at", -1)).view(),
                          "cal_created_at");
      }
      {
        const auto validator{make_document(kvp(
            "$jsonSchema",
            make_document(
                kvp("bsonType", "object"),
                kvp("required", make_array("_id", "track_ids", "cls", "text",
                                           "latlon", "loc", "azimuth",
                                           "variance", "created_at", "valid")),
                kvp("properties",
                    make_document(
                        kvp("_id", make_document(kvp("bsonType", "long"))),
                        kvp("track_ids",
                            make_document(
                                kvp("bsonType", "array"), kvp("minItems", 1),
                                kvp("items",
                                    make_document(kvp("bsonType", "long"))))),
                        kvp("cls", make_document(kvp("bsonType", "int"))),
                        kvp("text", make_document(kvp("bsonType", "string"))),
                        kvp("latlon", schema_array_fixed("double", 2).view()),
                        kvp("loc", schema_geopoint().view()),
                        kvp("azimuth",
                            make_document(kvp("bsonType", "double"))),
                        kvp("variance",
                            make_document(kvp("bsonType", "double"))),
                        kvp("created_at",
                            make_document(kvp("bsonType", "date"))),
                        kvp("valid", make_document(kvp("bsonType", "bool"))),
                        kvp("prev_id",
                            make_document(kvp(
                                "bsonType", make_array("long", "null")))))))))};

        create_collection_if_missing("landmarks", validator.view());

        auto col{db_["landmarks"]};
        create_index_safe(col, make_document(kvp("loc", "2dsphere")).view(),
                          "lm_loc_2dsphere");
        create_index_safe(
            col, make_document(kvp("valid", 1), kvp("created_at", -1)).view(),
            "lm_valid_created");
      }
      {
        const auto validator{make_document(kvp(
            "$jsonSchema",
            make_document(
                kvp("bsonType", "object"),
                kvp("required", make_array("_id", "T_cam_to_world", "latlon",
                                           "enu", "loc")),
                kvp("properties",
                    make_document(
                        kvp("_id", make_document(kvp("bsonType", "long"))),
                        kvp("T_cam_to_world",
                            make_document(
                                kvp("bsonType", "object"),
                                kvp("required", make_array("q", "t")),
                                kvp("properties",
                                    make_document(
                                        kvp("q", schema_array_fixed("double", 4)
                                                     .view()),
                                        kvp("t", schema_array_fixed("double", 3)
                                                     .view()))))),
                        kvp("latlon", schema_array_fixed("double", 2).view()),
                        kvp("enu", schema_array_fixed("double", 2).view()),
                        kvp("loc", schema_geopoint().view()))))))};

        create_collection_if_missing("frame_poses", validator.view());

        auto col{db_["frame_poses"]};
        create_index_safe(col, make_document(kvp("loc", "2dsphere")).view(),
                          "fp_loc_2dsphere");
      }
      {
        const auto validator{make_document(kvp(
            "$jsonSchema",
            make_document(
                kvp("bsonType", "object"),
                kvp("required",
                    make_array("_id", "track_id", "frame_id", "pose_id", "cls",
                               "bbox", "center", "center_undistorted",
                               "cumulative_length", "angle", "confidence")),
                kvp("properties",
                    make_document(
                        kvp("_id", make_document(kvp("bsonType", "long"))),
                        kvp("track_id", make_document(kvp("bsonType", "long"))),
                        kvp("frame_id", make_document(kvp("bsonType", "long"))),
                        kvp("pose_id", make_document(kvp("bsonType", "long"))),
                        kvp("cls", make_document(kvp("bsonType", "int"))),

                        kvp("bbox", make_document(
                                        kvp("bsonType", "array"),
                                        kvp("minItems", 4), kvp("maxItems", 4),
                                        kvp("items", make_document(kvp(
                                                         "bsonType", "int"))))),

                        kvp("center",
                            make_document(
                                kvp("bsonType", "array"), kvp("minItems", 2),
                                kvp("maxItems", 2),
                                kvp("items",
                                    make_document(kvp("bsonType", "int"))))),

                        kvp("center_undistorted",
                            schema_array_fixed("double", 2).view()),

                        kvp("cumulative_length",
                            make_document(kvp("bsonType", "double"))),
                        kvp("angle", make_document(kvp("bsonType", "double"))),

                        kvp("confidence",
                            make_document(kvp("bsonType", "double"),
                                          kvp("minimum", 0.0),
                                          kvp("maximum", 1.0))))))))};

        create_collection_if_missing("detections", validator.view());

        auto col{db_["detections"]};
        create_index_safe(col, make_document(kvp("frame_id", 1)).view(),
                          "det_frame");
        create_index_safe(
            col, make_document(kvp("track_id", 1), kvp("frame_id", 1)).view(),
            "det_track_frame");

        create_index_safe(col, make_document(kvp("pose_id", 1)).view(),
                          "det_pose");

        create_index_safe(
            col, make_document(kvp("frame_id", 1), kvp("cls", 1)).view(),
            "det_frame_cls");
      }
      {
        const auto validator{make_document(kvp(
            "$jsonSchema",
            make_document(
                kvp("bsonType", "object"),
                kvp("required",
                    make_array("_id", "record_id", "cls", "text", "length",
                               "parallax_angle", "created_at")),
                kvp("properties",
                    make_document(
                        kvp("_id", make_document(kvp("bsonType", "long"))),
                        kvp("record_id",
                            make_document(kvp("bsonType", "long"))),
                        kvp("cls", make_document(kvp("bsonType", "int"))),
                        kvp("text", make_document(kvp("bsonType", "string"))),
                        kvp("length", make_document(kvp("bsonType", "double"))),
                        kvp("parallax_angle",
                            make_document(kvp("bsonType", "double"))),
                        kvp("detections",
                            make_document(
                                kvp("bsonType", make_array("array", "null")),
                                kvp("items",
                                    make_document(kvp("bsonType", "long"))))),

                        kvp("landmark_id",
                            make_document(
                                kvp("bsonType", make_array("long", "null")))),
                        kvp("created_at",
                            make_document(kvp("bsonType", "date"))))))))};

        create_collection_if_missing("tracks", validator.view());

        auto col{db_["tracks"]};
        create_index_safe(col, make_document(kvp("record_id", 1)).view(),
                          "trk_record");
        create_index_safe(col, make_document(kvp("landmark_id", 1)).view(),
                          "trk_landmark");
        create_index_safe(
            col, make_document(kvp("cls", 1), kvp("created_at", -1)).view(),
            "trk_cls_created");
      }
      {
        const auto validator{make_document(kvp(
            "$jsonSchema",
            make_document(
                kvp("bsonType", "object"),
                kvp("required",
                    make_array("_id", "calib_id", "video_paths", "session_name",
                               "geodetic_origin", "created_at")),
                kvp("properties",
                    make_document(
                        kvp("_id", make_document(kvp("bsonType", "long"))),
                        kvp("calib_id", make_document(kvp("bsonType", "long"))),
                        kvp("video_paths",
                            make_document(
                                kvp("bsonType", "array"), kvp("minItems", 1),
                                kvp("items",
                                    make_document(kvp("bsonType", "string"))))),
                        kvp("session_name",
                            make_document(kvp("bsonType", "string"))),
                        kvp("geodetic_origin",
                            schema_array_fixed("double", 2).view()),
                        kvp("created_at",
                            make_document(kvp("bsonType", "date"))))))))};

        create_collection_if_missing("records", validator.view());

        auto col{db_["records"]};
        create_index_safe(col, make_document(kvp("created_at", -1)).view(),
                          "rec_created");
        create_index_safe(col, make_document(kvp("session_name", 1)).view(),
                          "rec_session");
        create_index_safe(col, make_document(kvp("calib_id", 1)).view(),
                          "rec_calib");
      }
    }

    void create_index_safe(
        mongocxx::collection & col, const bsoncxx::document::view &keys,
        const std::string_view name = "", bool background = true) {
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

    mongocxx::instance inst_;
    mongocxx::client client_;
    mongocxx::database db_;
  };

  MongoDBConnector::MongoDBConnector(const std::string_view uri)
      : pimpl_{std::make_unique<impl>(uri)} {}

  MongoDBConnector::~MongoDBConnector() = default;