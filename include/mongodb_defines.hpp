#pragma once
#include <string>
#include <string_view>

// namespace details
namespace details {

constexpr static std::string_view RECORDS_COLLECION_NAME{"records"};
constexpr static std::string_view RECORDS_VALIDATOR{R"( 
{
    "validationLevel": "strict",
    "validationAction": "error",
    "validator": {
        "$jsonSchema": {
            "bsonType": "object",
            "required": [
                "_id",
                "calib_id",
                "video_paths",
                "session_name",
                "geodetic_origin",
                "created_at"
            ],
            "additionalProperties": false,
            "properties": {
                "_id": {
                    "bsonType": "long"
                },
                "calib_id": {
                    "bsonType": "long"
                },
                "video_paths": {
                    "bsonType": [
                        "array"
                    ],
                    "items": {
                        "bsonType": "string"
                    }
                },
                "session_name": {
                    "bsonType": "string"
                },
                "geodetic_origin": {
                    "bsonType": [
                        "array"
                    ],
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {
                        "bsonType": "double"
                    }
                },
                "created_at": {
                    "bsonType": "date"
                }
            }
        }
    }
})"};

constexpr static std::string_view CALIBRATIONS_COLLECION_NAME{"calibrations"};
constexpr static std::string_view CALIBRATIONS_VALIDATOR{R"(
{
    "validationLevel": "strict",
    "validationAction": "error",
    "validator": {
        "$jsonSchema": {
            "bsonType": "object",
            "required": [
                "_id",
                "camera_type",
                "model",
                "K",
                "D",
                "resolution",
                "source",
                "created_at"
            ],
            "additionalProperties": false,
            "properties": {
                "_id": {
                    "bsonType": "long"
                },
                "camera_type": {
                    "bsonType": "string"
                },
                "model": {
                    "bsonType": "string"
                },
                "K": {
                    "bsonType": "array",
                    "minItems": 4,
                    "maxItems": 4,
                    "items": {
                        "bsonType": "double"
                    }
                },
                "D": {
                    "bsonType": "array",
                    "minItems": 5,
                    "maxItems": 5,
                    "items": {
                        "bsonType": "double"
                    }
                },
                "resolution": {
                    "bsonType": "object",
                    "required": [
                        "width",
                        "height"
                    ],
                    "additionalProperties": false,
                    "properties": {
                        "width": {
                            "bsonType": "int"
                        },
                        "height": {
                            "bsonType": "int"
                        }
                    }
                },
                "source": {
                    "bsonType": "string"
                },
                "created_at": {
                    "bsonType": "date"
                }
            }
        }
    }
})"};

constexpr static std::string_view LANDMARKS_COLLECION_NAME{"landmarks"};
constexpr static std::string_view LANDMARKS_VALIDATOR{R"(
{
    "validationLevel": "strict",
    "validationAction": "error",
    "validator": {
        "$jsonSchema": {
            "bsonType": "object",
            "required": [
                "_id",
                "track_ids",
                "cls",
                "text",
                "latlon",
                "loc",
                "azimuth",
                "variance",
                "valid",
                "created_at"
            ],
            "additionalProperties": false,
            "properties": {
                "_id": {
                    "bsonType": "long"
                },
                "track_ids": {
                    "bsonType": "array",
                    "minItems": 1,
                    "items": {
                        "bsonType": "long"
                    }
                },
                "cls": {
                    "bsonType": "int"
                },
                "text": {
                    "bsonType": "string"
                },
                "latlon": {
                    "bsonType": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {
                        "bsonType": "double"
                    }
                },
                "loc": {
                    "bsonType": "object",
                    "required": [
                        "type",
                        "coordinates"
                    ],
                    "properties": {
                        "type": { 
                            "enum": [
                                "Point"
                            ]
                        },
                        "coordinates": {
                            "bsonType": "array",
                            "minItems": 2,
                            "maxItems": 2,
                            "items": {
                                "bsonType": "double"
                            }
                        }
                    }
                },
                "azimuth": {
                    "bsonType": "double"
                },
                "variance": {
                    "bsonType": "double"
                },
                "valid": {
                    "bsonType": "bool"
                },
                "prev_id": {
                    "bsonType": [
                        "long"
                    ]
                },
                "created_at": {
                    "bsonType": "date"
                }
            }
        }
    }
})"};

constexpr static std::string_view CAMERA_POSES_COLLECION_NAME{"camera_poses"};
constexpr static std::string_view CAMERA_POSES_VALIDATOR{R"(
{
    "validationLevel": "strict",
    "validationAction": "error",
    "validator": {
        "$jsonSchema": {
            "bsonType": "object",
            "required": [
                "_id",
                "T_cam_to_world",
                "latlon",
                "loc"
            ],
            "additionalProperties": false,
            "properties": {
                "_id": {
                    "bsonType": "long"
                },
                "T_cam_to_world": {
                    "bsonType": "object",
                    "required": [
                        "q",
                        "t"
                    ],
                    "properties": {
                        "q": {
                            "bsonType": "array",
                            "minItems": 4,
                            "maxItems": 4,
                            "items": {
                                "bsonType": "double"
                            }
                        },
                        "t": {
                            "bsonType": "array",
                            "minItems": 3,
                            "maxItems": 3,
                            "items": {
                                "bsonType": "double"
                            }
                        }
                    }
                },
                "latlon": {
                    "bsonType": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {
                        "bsonType": "double"
                    }
                },
                "loc": {
                    "bsonType": "object",
                    "required": [
                        "type",
                        "coordinates"
                    ],
                    "properties": {
                        "type": {
                            "enum": [
                                "Point"
                            ]
                        },
                        "coordinates": {
                            "bsonType": "array",
                            "minItems": 2,
                            "maxItems": 2,
                            "items": {
                                "bsonType": "double"
                            }
                        }
                    }
                }
            }
        }
    }
})"};

constexpr static std::string_view DETECTIONS_COLLECION_NAME{"detections"};
constexpr static std::string_view DETECTIONS_VALIDATOR{R"(
{
    "validationLevel": "strict",
    "validationAction": "error",
    "validator": {
        "$jsonSchema": {
            "bsonType": "object",
            "required": [
                "_id",
                "track_id",
                "frame_id",
                "pose_id",
                "cls",
                "text",
                "bbox",
                "center",
                "center_undistorted",
                "cumulative_length",
                "angle",
                "confidence"
            ],
            "additionalProperties": false,
            "properties": {
                "_id": {
                    "bsonType": "long"
                },
                "track_id": {
                    "bsonType": "long"
                },
                "frame_id": {
                    "bsonType": "long"
                },
                "pose_id": {
                    "bsonType": "long"
                },
                "cls": {
                    "bsonType": "int"
                },
                "text": {
                    "bsonType": "string"
                },
                "bbox": {
                    "bsonType": "array",
                    "minItems": 4,
                    "maxItems": 4,
                    "items": {
                        "bsonType": "int"
                    }
                },
                "center": {
                    "bsonType": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {
                        "bsonType": "int"
                    }
                },
                "center_undistorted": {
                    "bsonType": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {
                        "bsonType": "double"
                    }
                },
                "cumulative_length": {
                    "bsonType": "double"
                },
                "angle": {
                    "bsonType": "double"
                },
                "confidence": {
                    "bsonType": "double"
                }
            }
        }
    }
})"};

constexpr static std::string_view TRACKS_COLLECION_NAME{"tracks"};
constexpr static std::string_view TRACKS_VALIDATOR{R"(
{
    "validationLevel": "strict",
    "validationAction": "error",
    "validator": {
        "$jsonSchema": {
            "bsonType": "object",
            "required": [
                "_id",
                "record_id",
                "landmark_id",
                "cls",
                "text",
                "length",
                "detections",
                "parallax_angle"
            ],
            "additionalProperties": false,
            "properties": {
                "_id": {
                    "bsonType": "long"
                },
                "record_id": {
                    "bsonType": "long"
                },
                "landmark_id": {
                    "bsonType": "long"
                },
                "cls": {
                    "bsonType": "int"
                },
                "text": {
                    "bsonType": "string"
                },
                "length": {
                    "bsonType": "double"
                },
                "detections": {
                    "bsonType": "array",
                    "items": {
                        "bsonType": "long"
                    }
                },
                "parallax_angle": {
                    "bsonType": "double"
                }
            }
        }
    }
})"};

constexpr static std::string_view COUNTERS_COLLECION_NAME{"counters"};
constexpr static std::string_view COUNTERS_VALIDATOR{R"(
{
    "validationLevel": "strict",
    "validationAction": "error",
    "validator": {
        "$jsonSchema": {
            "bsonType": "object",
            "required": [
                "_id",
                "seq"
            ],
            "additionalProperties": false,
            "properties": {
                "_id": {
                    "bsonType": "string"
                },
                "seq": {
                    "bsonType": "long"
                }
            }
        }
    }
})"};

} // namespace details

constexpr static std::string_view MONGODB_NAME{"geodb"};

enum collection_enum {
  RECORDS,
  CALIBRATIONS,
  LANDMARKS,
  CAMERA_POSES,
  DETECTIONS,
  TRACKS,
  COUNTERS,
  COLLECTIONS_COUNT
};

#define COLLECTION_INFO(F)                                                     \
  F(COUNTERS)                                                                  \
  F(RECORDS)                                                                   \
  F(CALIBRATIONS)                                                              \
  F(LANDMARKS)                                                                 \
  F(CAMERA_POSES)                                                              \
  F(DETECTIONS)                                                                \
  F(TRACKS)

template <collection_enum> struct ct;
#define DECLARE_COLLECTION_TRAITS_(c)                                          \
  template <> struct ct<c> {                                                   \
    inline static std::string name{details::c##_COLLECION_NAME};               \
    static constexpr std::string_view validator{details::c##_VALIDATOR};       \
  };

COLLECTION_INFO(DECLARE_COLLECTION_TRAITS_)