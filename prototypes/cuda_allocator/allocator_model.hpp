#pragma once

#include "../../csrc/src/tensor/cuda/allocator_core.hpp"

#include <algorithm>
#include <cstddef>
#include <map>
#include <optional>

namespace lmp::cuda_allocator_prototype {

using tensor::detail::cuda::allocator::Address;
using tensor::detail::cuda::allocator::Allocation;
using tensor::detail::cuda::allocator::BlockArena;
using tensor::detail::cuda::allocator::ReleasedSegment;
using tensor::detail::cuda::allocator::align_up;

struct Config {
  std::size_t allocation_alignment = 256;
  std::size_t segment_alignment = 2 * 1024 * 1024;
  std::size_t segment_size = 256 * 1024 * 1024;
};

class MockSegmentProvider {
 public:
  explicit MockSegmentProvider(std::size_t capacity,
                               Address first_address = 0x10000000)
      : capacity_(capacity), next_address_(first_address) {}

  std::optional<Address> acquire(std::size_t size, std::size_t alignment) {
    if (size > capacity_ - reserved_) {
      return std::nullopt;
    }
    next_address_ = align_up(next_address_, alignment);
    const Address result = next_address_;
    next_address_ += size + alignment;
    live_.emplace(result, size);
    reserved_ += size;
    ++acquire_count_;
    return result;
  }

  void release(Address address, std::size_t size) {
    auto segment = live_.find(address);
    if (segment == live_.end() || segment->second != size) {
      throw std::logic_error("releasing an unknown mock segment");
    }
    live_.erase(segment);
    reserved_ -= size;
    ++release_count_;
  }

  std::size_t reserved() const { return reserved_; }
  std::size_t acquire_count() const { return acquire_count_; }
  std::size_t release_count() const { return release_count_; }

 private:
  std::size_t capacity_;
  Address next_address_;
  std::map<Address, std::size_t> live_;
  std::size_t reserved_ = 0;
  std::size_t acquire_count_ = 0;
  std::size_t release_count_ = 0;
};

class HostAllocatorModel {
 public:
  HostAllocatorModel(Config config, MockSegmentProvider& provider)
      : config_(config),
        arena_(config.allocation_alignment),
        provider_(provider) {
    (void)align_up(config_.segment_size, config_.allocation_alignment);
    (void)align_up(config_.segment_size, config_.segment_alignment);
  }

  std::optional<Allocation> allocate(std::size_t size) {
    if (size == 0) {
      return Allocation{};
    }
    if (auto allocation = arena_.allocate(size)) {
      return allocation;
    }

    const std::size_t segment_size =
        align_up(std::max(config_.segment_size,
                          align_up(size, config_.allocation_alignment)),
                 config_.segment_alignment);

    auto address = provider_.acquire(segment_size, config_.segment_alignment);
    if (!address) {
      release_fully_free_segments();
      address = provider_.acquire(segment_size, config_.segment_alignment);
      if (!address) {
        return std::nullopt;
      }
    }

    arena_.add_segment(*address, segment_size);
    return arena_.allocate(size);
  }

  bool deallocate(Address address) { return arena_.deallocate(address); }

  void empty_cache() { release_fully_free_segments(); }

 private:
  void release_fully_free_segments() {
    for (const ReleasedSegment& segment :
         arena_.extract_fully_free_segments()) {
      provider_.release(segment.address, segment.size);
    }
  }

  Config config_;
  BlockArena arena_;
  MockSegmentProvider& provider_;
};

}  // namespace lmp::cuda_allocator_prototype
