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

#include <cstddef>
#include <cstdint>
#include <extract_frames.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

struct ExtractFrames::impl {
  impl(std::filesystem::path video_path)
      : video_path_{video_path}, fmt_context_{nullptr}, video_stream_ind_{0},
        video_stream_{nullptr}, codec_{nullptr}, codec_context_{nullptr},
        sws_{nullptr}, fps_{0.0}, frame_{nullptr}, pkt_{nullptr} {
    init();
  }

  void init() {
    ffcheck(avformat_open_input(&fmt_context_, video_path_.c_str(), nullptr,
                                nullptr));

    ffcheck(avformat_find_stream_info(fmt_context_, nullptr));

    video_stream_ind_ = av_find_best_stream(fmt_context_, AVMEDIA_TYPE_VIDEO,
                                            -1, -1, nullptr, 0);

    if (video_stream_ind_ < 0) {
      throw std::runtime_error{"No video stream"};
    }

    video_stream_ = fmt_context_->streams[video_stream_ind_];
    codec_ = avcodec_find_decoder(video_stream_->codecpar->codec_id);
    codec_context_ = avcodec_alloc_context3(codec_);
    codec_context_->thread_count = 0;
    codec_context_->thread_type = FF_THREAD_FRAME;

    avcodec_parameters_to_context(codec_context_, video_stream_->codecpar);
    ffcheck(avcodec_open2(codec_context_, codec_, nullptr));

    pkt_ = av_packet_alloc();
    frame_ = av_frame_alloc();

    sws_ = sws_getContext(codec_context_->width, codec_context_->height,
                          codec_context_->pix_fmt, codec_context_->width,
                          codec_context_->height, AV_PIX_FMT_GRAY8,
                          SWS_BILINEAR, nullptr, nullptr, nullptr);

    if (sws_ == nullptr) {
      throw std::runtime_error{"sws_getContext failed"};
    }

    fps_ = get_fps();
  }

  void ffcheck(int err) {
    if (err < 0) {
      char buf[AV_ERROR_MAX_STRING_SIZE];
      av_strerror(err, buf, sizeof(buf));
      throw std::runtime_error{buf};
    }
  }

  double get_fps() {
    const auto fr{video_stream_->avg_frame_rate.num != 0
                      ? video_stream_->avg_frame_rate
                      : video_stream_->r_frame_rate};

    if (fr.num == 0 or fr.den == 0) {
      return 0.0;
    }

    return av_q2d(fr);
  }

  int64_t frame_to_pts(int64_t frame_idx) {
    const auto fr{video_stream_->avg_frame_rate.num != 0
                      ? video_stream_->avg_frame_rate
                      : video_stream_->r_frame_rate};

    AVRational inv_fr{fr.den, fr.num};
    return av_rescale_q(frame_idx, inv_fr, video_stream_->time_base);
  }

  int64_t pts_to_frame(int64_t pts) {
    if (pts == AV_NOPTS_VALUE || fps_ <= 0.0) {
      return 0;
    }

    const double t{pts * av_q2d(video_stream_->time_base)};
    return static_cast<int64_t>(llround(t * fps_));
  }

  void tight_seek_to_frame(int64_t target_frame) {
    auto target_pts{frame_to_pts(target_frame)};

    const auto e{avformat_index_get_entry_from_timestamp(
        video_stream_, target_pts, AVSEEK_FLAG_BACKWARD)};

    if (e != nullptr) {
      target_pts = e->timestamp;
    }

    ffcheck(av_seek_frame(fmt_context_, video_stream_ind_, target_pts,
                          AVSEEK_FLAG_BACKWARD));

    avformat_flush(fmt_context_);
    avcodec_flush_buffers(codec_context_);
  }

  std::vector<uint8_t> to_jpeg_buffer() {
    cv::Mat_<uint8_t> img =
        cv::Mat_<uint8_t>::zeros(codec_context_->height, codec_context_->width);

    const std::array<uint8_t *, 4> dst_data{
        img.data + (codec_context_->height - 1) * img.step, nullptr, nullptr,
        nullptr};

    const std::array<int, 4> dst_linesize{-static_cast<int>(img.step), 0, 0, 0};

    sws_scale(sws_, frame_->data, frame_->linesize, 0, codec_context_->height,
              dst_data.data(), dst_linesize.data());

    cv::flip(img, img, 1);

    std::vector<uint8_t> buf{};
    cv::imencode(".jpeg", img, buf, {cv::IMWRITE_JPEG_QUALITY, 50});

    return buf;
  }

  ~impl() {
    sws_freeContext(sws_);
    av_frame_free(&frame_);
    av_packet_free(&pkt_);
    avcodec_free_context(&codec_context_);
    avformat_close_input(&fmt_context_);
  }

  std::vector<std::vector<uint8_t>>
  extract(std::span<const size_t> frame_list) {

    size_t current_frame{0};
    bool frame_base_set{false};
    bool need_reseek{false};
    int64_t reseek_to{0};

    std::vector<std::vector<uint8_t>> res{};
    res.reserve(frame_list.size());

    static constexpr int64_t MARGIN{100};
    static constexpr int64_t GAP_SEEK{300};

    tight_seek_to_frame(
        std::max(0l, static_cast<int64_t>(frame_list.front()) - MARGIN));

    size_t current_index{0};

    while (current_index < frame_list.size()) {

      if (need_reseek) {
        tight_seek_to_frame(reseek_to);

        frame_base_set = false;
        need_reseek = false;
      }

      if (av_read_frame(fmt_context_, pkt_) < 0) {
        break;
      }

      if (pkt_->stream_index != video_stream_ind_) {
        av_packet_unref(pkt_);
        continue;
      }

      ffcheck(avcodec_send_packet(codec_context_, pkt_));
      av_packet_unref(pkt_);

      while (true) {
        const int ret{avcodec_receive_frame(codec_context_, frame_)};

        if (ret == AVERROR(EAGAIN) or ret == AVERROR_EOF) {
          break;
        }

        ffcheck(ret);

        if (not frame_base_set) {
          const int64_t pts{(frame_->best_effort_timestamp == AV_NOPTS_VALUE)
                                ? frame_->pts
                                : frame_->best_effort_timestamp};

          current_frame = pts_to_frame(pts);
          frame_base_set = true;
        }

        if (current_frame > frame_list[current_index]) {
          reseek_to = std::max(
              0l, static_cast<int64_t>(frame_list[current_index]) - MARGIN);
          need_reseek = true;
          av_frame_unref(frame_);
          break;
        }

        if (current_frame == frame_list[current_index]) {

          res.push_back(to_jpeg_buffer());

          ++current_index;
          progress();

          if (current_index >= frame_list.size()) {
            av_frame_unref(frame_);
            break;
          }

          if (frame_list[current_index] - frame_list[current_index - 1] >
              GAP_SEEK) {
            reseek_to = std::max(
                0l, static_cast<int64_t>(frame_list[current_index]) - MARGIN);
            need_reseek = true;
            av_frame_unref(frame_);
            break;
          }
        }

        ++current_frame;
        av_frame_unref(frame_);
      }
    }

    return res;
  }

  void progress() {
    if (prog_) {
      prog_();
    }
  }

  std::filesystem::path video_path_;
  AVFormatContext *fmt_context_;
  int video_stream_ind_;
  AVStream *video_stream_;
  const AVCodec *codec_;
  AVCodecContext *codec_context_;
  SwsContext *sws_;
  double fps_;
  AVFrame *frame_;
  AVPacket *pkt_;
  std::function<void()> prog_;
};

ExtractFrames::ExtractFrames(std::filesystem::path video_path)
    : pimpl_{std::make_unique<impl>(video_path)} {}

ExtractFrames::~ExtractFrames() = default;

std::vector<std::vector<uint8_t>>
ExtractFrames::extract(std::span<const size_t> frame_list) {
  return pimpl_->extract(frame_list);
}

void ExtractFrames::set_progress(std::function<void()> f) {
  pimpl_->prog_ = std::move(f);
}