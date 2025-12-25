// #include <CLI/CLI.hpp>
// #include <Eigen/Core>
// #include <GeographicLib/UTMUPS.hpp>

// clang-format off
extern "C" {
#include <libavutil/mathematics.h>
#include <libavutil/rational.h>
#include <libavutil/error.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}
// clang-format on

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
// #include <fmt/core.h>
// #include <fmt/format.h>
// #include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
// #include <opencv2/features2d.hpp>
// #include <range/v3/view/enumerate.hpp>
// #include <range/v3/view/filter.hpp>
// #include <range/v3/view/transform.hpp>
#include <chrono>
#include <vector>

// using ranges::views::enumerate;
// using ranges::views::filter;
// using ranges::views::transform;

#if 0 

cv::Mat_<double> estimate_homography(const cv::Mat_<cv::Vec3b> &img1,
                                     const cv::Mat_<cv::Vec3b> &img2) {

  cv::Mat_<uint8_t> gray1{};
  cv::Mat_<uint8_t> gray2{};

  cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
  cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);

  std::vector<cv::KeyPoint> kA, kB;
  cv::Mat_<uint8_t> dA, dB;
  //   auto detector{cv::SIFT::create(8'000)};
  auto detector{cv::AKAZE::create()};

  detector->detectAndCompute(gray1, cv::noArray(), kA, dA);
  detector->detectAndCompute(gray2, cv::noArray(), kB, dB);

  //   cv::BFMatcher matcher{cv::NORM_L2};
  cv::BFMatcher matcher{cv::NORM_HAMMING};
  std::vector<std::vector<cv::DMatch>> matches;

  matcher.knnMatch(dA, dB, matches, 2);

  std::vector<cv::DMatch> good_matches;

  for (const auto &match : matches) {

    if (match.size() < 2) {
      continue;
    }

    if (match[0].distance < 0.75 * match[1].distance) {
      good_matches.push_back(match[0]);
    }
  }

  std::vector<cv::Point2f> pointsA;
  std::vector<cv::Point2f> pointsB;

  pointsA.reserve(good_matches.size());
  pointsB.reserve(good_matches.size());

  for (const auto &match : good_matches) {
    pointsA.push_back(kA[match.queryIdx].pt);
    pointsB.push_back(kB[match.trainIdx].pt);
  }

  std::vector<uint8_t> inlier_mask{};
  cv::Mat_<double> H =
      cv::findHomography(pointsA, pointsB, cv::RANSAC, 3.0, inlier_mask);

  // Draw inlier matches
  std::vector<cv::DMatch> inlier_matches;
  for (size_t i = 0; i < good_matches.size(); ++i) {
    if (inlier_mask[i] == 1) {
      inlier_matches.push_back(good_matches[i]);
    }
  }

  cv::Mat img_matches{};
  cv::drawMatches(img1, kA, img2, kB, inlier_matches, img_matches);
  cv::imwrite("/root/data/comparison/matches.png", img_matches);

  return H;
}

#endif

void ffcheck(int err) {
  if (err < 0) {
    char buf[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(err, buf, sizeof(buf));
    throw std::runtime_error{buf};
  }
}

double get_fps(AVStream *str) {
  const auto fr{str->avg_frame_rate.num != 0 ? str->avg_frame_rate
                                             : str->r_frame_rate};

  if (fr.num == 0 or fr.den == 0) {
    return 0.0;
  }

  return av_q2d(fr);
}

int64_t frame_to_pts(int64_t frame_idx, AVStream *str) {
  const auto fr{str->avg_frame_rate.num != 0 ? str->avg_frame_rate
                                             : str->r_frame_rate};

  AVRational inv_fr{fr.den, fr.num};
  return av_rescale_q(frame_idx, inv_fr, str->time_base);
}

int64_t pts_to_frame(int64_t pts, AVStream *str, double fps) {
  if (pts == AV_NOPTS_VALUE || fps <= 0.0) {
    return 0;
  }

  const double t{pts * av_q2d(str->time_base)};
  return static_cast<int64_t>(llround(t * fps));
}

void tight_seek_to_frame(int64_t target_frame, AVStream *str,
                         AVFormatContext *fmt, AVCodecContext *dec,
                         int stream_index) {
  auto target_pts{frame_to_pts(target_frame, str)};

  const auto e{avformat_index_get_entry_from_timestamp(str, target_pts,
                                                       AVSEEK_FLAG_BACKWARD)};
  if (e != nullptr) {
    target_pts = e->timestamp;
  }

  ffcheck(av_seek_frame(fmt, stream_index, target_pts, AVSEEK_FLAG_BACKWARD));
  avformat_flush(fmt);
  avcodec_flush_buffers(dec);
}

cv::Mat_<uint8_t> to_bgr(const AVFrame *frame, SwsContext *sws, int w, int h) {
  cv::Mat_<uint8_t> img = cv::Mat_<uint8_t>::zeros(h, w);

  const std::array<uint8_t *, 4> dst_data{img.data + (h - 1) * img.step,
                                          nullptr, nullptr, nullptr};

  const std::array<int, 4> dst_linesize{-static_cast<int>(img.step), 0, 0, 0};

  sws_scale(sws, frame->data, frame->linesize, 0, h, dst_data.data(),
            dst_linesize.data());

  cv::flip(img, img, 1);
  return img;
}

int main(int argc, const char **argv) {

  try {
#if 1
    {
      const auto start_time{std::chrono::system_clock::now()};

      cv::VideoCapture cap{"/root/data/domodedovo/gopro_01_11/GX010004.MP4"};
      cap.set(cv::CAP_PROP_POS_FRAMES, 923.0);
      cv::Mat_<cv::Vec3b> img;
      cap >> img;

      // cv::Mat_<uint8_t> img_gray{};
      // cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

      cv::imwrite("/root/data/comparison/image_cv.png", img);

      std::cout << "opencv: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::system_clock::now() - start_time)
                       .count()
                << std::endl;
    }
#endif

    const auto start_time{std::chrono::system_clock::now()};

    std::vector<int64_t> wanted = {923};
    size_t want_i{0};

    AVFormatContext *fmt_context{nullptr};

    ffcheck(avformat_open_input(
        &fmt_context, "/root/data/domodedovo/gopro_01_11/GX010004.MP4", nullptr,
        nullptr));

    ffcheck(avformat_find_stream_info(fmt_context, nullptr));

    const auto video_stream_ind{av_find_best_stream(
        fmt_context, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0)};

    if (video_stream_ind < 0) {
      throw std::runtime_error{"No video stream"};
    }

    const auto video_stream{fmt_context->streams[video_stream_ind]};
    auto decoder{avcodec_find_decoder(video_stream->codecpar->codec_id)};
    auto decoder_context{avcodec_alloc_context3(decoder)};

    decoder_context->thread_count = 0;
    decoder_context->thread_type = FF_THREAD_FRAME;

    avcodec_parameters_to_context(decoder_context, video_stream->codecpar);
    ffcheck(avcodec_open2(decoder_context, decoder, nullptr));

    auto pkt{av_packet_alloc()};
    auto frame{av_frame_alloc()};

    int current_frame{0};
    bool frame_base_set{false};
    bool need_reseek{false};
    int64_t reseek_to{0};

    auto sws{sws_getContext(decoder_context->width, decoder_context->height,
                            decoder_context->pix_fmt, decoder_context->width,
                            decoder_context->height, AV_PIX_FMT_GRAY8,
                            SWS_BILINEAR, nullptr, nullptr, nullptr)};

    if (sws == nullptr) {
      throw std::runtime_error{"sws_getContext failed"};
    }

    const auto fps{get_fps(video_stream)};
    std::cout << "fps: " << fps << std::endl;

    static constexpr int64_t MARGIN{10};
    static constexpr int64_t GAP_SEEK{300};

    const int64_t start_frame{wanted.front()};

    tight_seek_to_frame(std::max(0l, wanted.front() - MARGIN), video_stream,
                        fmt_context, decoder_context, video_stream_ind);

    while (want_i < wanted.size()) {

      if (need_reseek) {
        tight_seek_to_frame(reseek_to, video_stream, fmt_context,
                            decoder_context, video_stream_ind);

        frame_base_set = false;
        need_reseek = false;
      }

      if (av_read_frame(fmt_context, pkt) < 0) {
        break;
      }

      if (pkt->stream_index != video_stream_ind) {
        av_packet_unref(pkt);
        continue;
      }

      ffcheck(avcodec_send_packet(decoder_context, pkt));
      av_packet_unref(pkt);

      while (true) {
        const int ret{avcodec_receive_frame(decoder_context, frame)};

        if (ret == AVERROR(EAGAIN) or ret == AVERROR_EOF) {
          break;
        }

        ffcheck(ret);

        if (not frame_base_set) {
          const int64_t pts{(frame->best_effort_timestamp == AV_NOPTS_VALUE)
                                ? frame->pts
                                : frame->best_effort_timestamp};

          current_frame = pts_to_frame(pts, video_stream, fps);
          std::cout << "current frame: " << current_frame << std::endl;
          frame_base_set = true;
        }

        if (current_frame > wanted[want_i]) {
          reseek_to = std::max(0l, wanted[want_i] - MARGIN);
          need_reseek = true;
          av_frame_unref(frame);
          break;
        }

        if (current_frame == wanted[want_i]) {

          cv::Mat_<uint8_t> img = to_bgr(frame, sws, decoder_context->width,
                                         decoder_context->height);

          // cv::Mat_<cv::Vec3b> img = cv::Mat_<cv::Vec3b>::zeros(
          //     decoder_context->height, decoder_context->width);

          // std::array<uint8_t *, 4> dst_data{img.data, nullptr, nullptr,
          //                                   nullptr};
          // std::array<int, 4> dst_linesize{static_cast<int>(img.step), 0, 0,
          // 0};

          // sws_scale(sws, frame->data, frame->linesize, 0,
          //           decoder_context->height, dst_data.data(),
          //           dst_linesize.data());

          cv::imwrite("/root/data/comparison/image_av.png", img);

          ++want_i;
          if (want_i >= wanted.size()) {
            av_frame_unref(frame);
            break;
          }

          if (wanted[want_i] - wanted[want_i - 1] > GAP_SEEK) {
            reseek_to = std::max(0l, wanted[want_i] - MARGIN);
            need_reseek = true;
            av_frame_unref(frame);
            break;
          }
        }

        ++current_frame;
        av_frame_unref(frame);
      }
    }

    sws_freeContext(sws);
    av_frame_free(&frame);
    av_packet_free(&pkt);
    avcodec_free_context(&decoder_context);
    avformat_close_input(&fmt_context);

    std::cout << "ffmpeg: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::system_clock::now() - start_time)
                     .count()
              << std::endl;

  } catch (const std::exception &ex) {
    // fmt::print("{}\n", ex.what());
  }

#if 0
  int zone{0};
  bool northp{false};
  double x{0.0};
  double y{0.0};

  Eigen::Vector2d p0{0.0, 0.0};

  GeographicLib::UTMUPS::Forward(41.3234407, 69.2429921, zone, northp, p0.x(),
                                 p0.y());

  Eigen::Vector2d p1{0.0, 0.0};
  GeographicLib::UTMUPS::Forward(41.3242735, 69.243773, zone, northp, p1.x(),
                                 p1.y());

  fmt::print("{}\n", (p1 - p0).norm());

  CLI::App app{"Test exec"};

  std::vector<std::pair<int, int>> vec{};
  app.add_option("-p", vec)->required();

  CLI11_PARSE(app, argc, argv);

  fmt::print("tuple params:\n");

  for (auto &&v : vec) {
    fmt::print("{} {}\n", v.first, v.second);
  }

  return EXIT_SUCCESS;

  cv::Mat_<cv::Vec3b> img1 =
      cv::imread("/root/data/comparison/image1.png", cv::IMREAD_UNCHANGED);
  cv::Mat_<cv::Vec3b> img2 =
      cv::imread("/root/data/comparison/image2.png", cv::IMREAD_UNCHANGED);

  // estimate_homography(img1, img2);
#endif

  return EXIT_SUCCESS;
}