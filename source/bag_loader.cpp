#include <bag_loader.hpp>
#include <cstddef>
#include <fmt/color.h>
#include <fmt/format.h>
#include <memory>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>

struct BagLoader::impl {
  impl(const BagLoader::Settings &set, BagLoader *parent)
      : set_{set}, parent_{parent} {
    reader_.open(set_.path_to_bag_);
  }

  cv::Mat_<cv::Vec3b> load_image(int64_t timestamp) {

    reader_.seek(
        reader_.get_metadata().starting_time.time_since_epoch().count());

    reader_.seek(timestamp);

    while (reader_.has_next()) {
      auto msg{reader_.read_next()};

      if (msg->topic_name == set_.compressed_image_topic_) {
        const rclcpp::SerializedMessage serialized_msg{*msg->serialized_data};

        sensor_msgs::msg::CompressedImage image_msg;
        serialization_image_.deserialize_message(&serialized_msg, &image_msg);

        const auto bag_timestamp{
            static_cast<int64_t>(image_msg.header.stamp.sec) * 1'000'000'000l +
            static_cast<int64_t>(image_msg.header.stamp.nanosec)};

        if (bag_timestamp == timestamp) {
          return cv::imdecode(image_msg.data, cv::IMREAD_UNCHANGED);
        }
      }
    }

    return {};
  }

  cv::Mat_<cv::Vec3b> load_image(size_t image_id) {

    reader_.seek(
        reader_.get_metadata().starting_time.time_since_epoch().count());

    size_t current_id{0};

    while (reader_.has_next()) {
      auto msg{reader_.read_next()};

      if (msg->topic_name == set_.compressed_image_topic_) {
        if (current_id == image_id) {
          const rclcpp::SerializedMessage serialized_msg{*msg->serialized_data};

          sensor_msgs::msg::CompressedImage image_msg;
          serialization_image_.deserialize_message(&serialized_msg, &image_msg);

          return cv::imdecode(image_msg.data, cv::IMREAD_UNCHANGED);
        }

        ++current_id;
      }
    }

    return {};
  }

  void dump_tracks(const ImageTrack::map_type &tracks,
                   const ImageDetections::map_type &detections,
                   size_t track_id) {

    for (auto &&d : tracks.at(track_id).dets_) {
      const auto img = load_image(d.timestamp_);

      if (img.empty()) {
        continue;
      }

      int baseline{0};
      const int thickness{1};
      const auto text_size{cv::getTextSize(fmt::format("{}", track_id),
                                           cv::FONT_HERSHEY_COMPLEX, 0.3,
                                           thickness, &baseline)};

      baseline += thickness;

      cv::rectangle(img, d.box_, {0.0, 0.0, 255.0}, 1, cv::LINE_AA);
      cv::putText(img, fmt::format("{}", track_id),
                  {d.center_.x - (text_size.width >> 1),
                   d.center_.y - (text_size.height >> 1) + baseline},
                  cv::FONT_HERSHEY_COMPLEX, 0.3, {0.0, 0.0, 255.0}, thickness,
                  cv::LINE_AA);

      for (auto &&d : detections.at(d.timestamp_).dets_) {
        if (d->track_id_ == track_id) {
          continue;
        }

        if (tracks.at(track_id).linked_tracks_.contains(d->track_id_)) {
          continue;
        }

        const auto text_size{cv::getTextSize(fmt::format("{}", d->track_id_),
                                             cv::FONT_HERSHEY_COMPLEX, 0.3,
                                             thickness, &baseline)};
        baseline += thickness;

        cv::rectangle(img, d->box_, {100.0, 100.0, 100.0}, 1, cv::LINE_AA);
        cv::putText(img, fmt::format("{}", d->track_id_),
                    {d->center_.x - (text_size.width >> 1),
                     d->center_.y - (text_size.height >> 1) + baseline},
                    cv::FONT_HERSHEY_COMPLEX, 0.3, {0.0, 255.0, 255.0},
                    thickness, cv::LINE_AA);
      }

      for (auto &&linked_id : tracks.at(track_id).linked_tracks_) {

        if (tracks.at(linked_id).stamp_to_detection_.contains(d.timestamp_)) {

          const auto box{
              tracks.at(linked_id).stamp_to_detection_.at(d.timestamp_)->box_};

          const auto center{tracks.at(linked_id)
                                .stamp_to_detection_.at(d.timestamp_)
                                ->center_};

          const auto text_size{cv::getTextSize(fmt::format("{}", linked_id),
                                               cv::FONT_HERSHEY_COMPLEX, 0.3,
                                               thickness, &baseline)};

          baseline += thickness;

          cv::rectangle(img, box, {0.0, 255.0, 0.0}, 1, cv::LINE_AA);
          cv::putText(img, fmt::format("{}", linked_id),
                      {center.x - (text_size.width >> 1),
                       center.y - (text_size.height >> 1) + baseline},
                      cv::FONT_HERSHEY_COMPLEX, 0.3, {0.0, 255.0, 0.0},
                      thickness, cv::LINE_AA);
        }
      }

      cv::imwrite(
          fmt::format("/root/data/images/linked_tracks_{}.png", d.timestamp_),
          img);
    }
  }

  void dump_detection(const std::string_view path, const Detection &det) {
    auto img = load_image(det.timestamp_ - set_.timestamp_delta_);
    cv::rectangle(img, det.box_, {0.0, 255.0, 0.0}, 2);
    cv::imwrite(path.data(), img);
  }

  std::vector<std::vector<uint8_t>>
  extract(std::span<const size_t> frame_list) {

    reader_.seek(
        reader_.get_metadata().starting_time.time_since_epoch().count());

    size_t current_id{0};
    size_t frame_list_index{0};

    std::vector<std::vector<uint8_t>> res{};
    res.reserve(frame_list.size());

    while (reader_.has_next()) {
      auto msg{reader_.read_next()};

      if (msg->topic_name == set_.compressed_image_topic_) {
        if (current_id == frame_list[frame_list_index]) {
          const rclcpp::SerializedMessage serialized_msg{*msg->serialized_data};

          sensor_msgs::msg::CompressedImage image_msg;
          serialization_image_.deserialize_message(&serialized_msg, &image_msg);

          res.push_back(image_msg.data);
          parent_->progress();
          ++frame_list_index;

          if (frame_list_index >= frame_list.size()) {
            break;
          }
        }

        ++current_id;
      }
    }

    return res;
  };

  BagLoader::Settings set_;
  rosbag2_cpp::Reader reader_;
  rclcpp::Serialization<sensor_msgs::msg::CompressedImage> serialization_image_;
  BagLoader *parent_;
};

BagLoader::BagLoader(const BagLoader::Settings &set)
    : pimpl_{std::make_unique<impl>(set, this)} {}

cv::Mat_<cv::Vec3b> BagLoader::load_image(int64_t timestamp) {
  return pimpl_->load_image(timestamp);
}

cv::Mat_<cv::Vec3b> BagLoader::load_image(size_t image_id) {
  return pimpl_->load_image(image_id);
}

void BagLoader::dump_tracks(const ImageTrack::map_type &tracks,
                            const ImageDetections::map_type &detections,
                            size_t track_id) {

  pimpl_->dump_tracks(tracks, detections, track_id);
}

void BagLoader::dump_detection(const std::string_view path,
                               const Detection &det) {
  pimpl_->dump_detection(path, det);
}

std::vector<std::vector<uint8_t>>
BagLoader::extract(std::span<const size_t> frame_list) {
  return pimpl_->extract(frame_list);
}

BagLoader::~BagLoader() = default;