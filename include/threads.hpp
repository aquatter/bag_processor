#pragma once
#include <concepts>
#include <functional>
#include <ranges>
#include <thread>

struct RunInThreads {
  explicit RunInThreads(const int problem_size)
      : problem_size_{problem_size},
        num_threads_{static_cast<int>(std::thread::hardware_concurrency())},
        chunk_size_{(problem_size_ + num_threads_ - 1) / num_threads_},
        problem_counter_{0} {}

  struct Context {
    int beg_;
    int end_;
    int id_;
    std::function<int()> get_progress_;
  };

  template <std::invocable<const Context &> F> void operator()(F &&fun) {
    std::vector<std::jthread> threads;

    for (auto &&i : std::views::iota(0, num_threads_)) {
      const int beg{i * chunk_size_};
      const int end{std::min((i + 1) * chunk_size_, problem_size_)};

      if (beg < end) {
        threads.emplace_back([body = fun, beg, end, i, this]() {
          body(Context{beg, end, i, [this]() { return get_progress(); }});
        });
      }
    }
  }

  int get_progress() noexcept {
    const int curr_item{problem_counter_.fetch_add(1)};
    return static_cast<int>(100.0 * static_cast<double>(curr_item + 1) /
                            problem_size_);
  }

  int problem_size_;
  int num_threads_;
  int chunk_size_;
  std::atomic_int problem_counter_;
};
