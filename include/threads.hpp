#pragma once
#include <atomic>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <fmt/color.h>
#include <functional>
#include <ranges>
#include <thread>

struct RunInThreads {
  explicit RunInThreads(
      size_t problem_size,
      uint32_t num_threads = std::thread::hardware_concurrency())
      : problem_size_{problem_size}, num_threads_{num_threads},
        chunk_size_{(problem_size_ + num_threads_ - 1) / num_threads_},
        problem_counter_{0} {}

  struct Context {
    struct Iterator {
      using value_type = size_t;
      using difference_type = std::ptrdiff_t;

      explicit Iterator(size_t pos) : value_{pos} {}
      size_t operator*() const { return value_; }

      Iterator &operator++() {
        ++value_;
        return *this;
      }

      friend bool operator==(const Iterator &a, const Iterator &b) {
        return a.value_ == b.value_;
      }

      friend bool operator!=(const Iterator &a, const Iterator &b) {
        return !(a == b);
      }

      size_t value_;
    };

    Iterator begin() const { return Iterator{beg_}; }
    Iterator end() const { return Iterator{end_}; }

    size_t beg_;
    size_t end_;
    size_t id_;
  };

  template <std::invocable<const Context &> F> void operator()(F &&fun) {
    std::vector<std::jthread> threads;

    for (auto &&i : std::views::iota(0ul, num_threads_)) {
      const size_t beg{i * chunk_size_};
      const size_t end{std::min((i + 1) * chunk_size_, problem_size_)};

      if (beg < end) {
        threads.emplace_back(
            [body = fun, beg, end, i, this]() { body(Context{beg, end, i}); });
      }
    }
  }

  size_t problem_size_;
  size_t num_threads_;
  size_t chunk_size_;
  std::atomic_size_t problem_counter_;
};
