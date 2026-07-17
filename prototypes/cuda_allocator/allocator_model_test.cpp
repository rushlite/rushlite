#include "allocator_model.hpp"

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string_view>
#include <vector>

namespace {

using lmp::cuda_allocator_prototype::Allocation;
using lmp::cuda_allocator_prototype::BlockArena;
using lmp::cuda_allocator_prototype::Config;
using lmp::cuda_allocator_prototype::HostAllocatorModel;
using lmp::cuda_allocator_prototype::MockSegmentProvider;

#define CHECK(expression)                                                \
  do {                                                                   \
    if (!(expression)) {                                                 \
      std::cerr << __func__ << ':' << __LINE__                           \
                << ": check failed: " #expression << '\n';              \
      std::exit(1);                                                      \
    }                                                                    \
  } while (false)

void test_zero_size_and_null_free() {
  MockSegmentProvider provider(4096);
  HostAllocatorModel allocator(
      Config{.allocation_alignment = 256,
             .segment_alignment = 256,
             .segment_size = 2048},
      provider);

  const auto allocation = allocator.allocate(0);
  CHECK(allocation.has_value());
  CHECK(allocation->address == 0);
  CHECK(allocator.deallocate(0));
  CHECK(provider.acquire_count() == 0);
}

void test_alignment_split_and_reuse() {
  MockSegmentProvider provider(4096);
  HostAllocatorModel allocator(
      Config{.allocation_alignment = 256,
             .segment_alignment = 256,
             .segment_size = 2048},
      provider);

  const Allocation first = *allocator.allocate(257);
  CHECK(first.size == 512);
  CHECK(first.address % 256 == 0);
  CHECK(provider.reserved() == 2048);
  CHECK(allocator.deallocate(first.address));

  const Allocation second = *allocator.allocate(512);
  CHECK(second.address == first.address);
  CHECK(provider.acquire_count() == 1);
}

void test_best_fit_and_complete_coalescing() {
  BlockArena arena(256);
  arena.add_segment(0x10000000, 4096);
  const Allocation a = *arena.allocate(512);
  const Allocation b = *arena.allocate(1024);
  const Allocation c = *arena.allocate(512);
  const Allocation d = *arena.allocate(1024);
  CHECK(arena.deallocate(b.address));
  CHECK(arena.deallocate(d.address));

  const Allocation chosen = *arena.allocate(768);
  CHECK(chosen.address == b.address);
  CHECK(arena.deallocate(a.address));
  CHECK(arena.deallocate(c.address));
  CHECK(arena.deallocate(chosen.address));

  const Allocation whole = *arena.allocate(4096);
  CHECK(whole.address == 0x10000000);
}

void test_best_fit_across_segments() {
  BlockArena arena(256);
  arena.add_segment(0x10000000, 4096);
  const Allocation first_segment = *arena.allocate(4096);
  arena.add_segment(0x20000000, 2048);
  const Allocation second_segment = *arena.allocate(2048);

  CHECK(arena.deallocate(first_segment.address));
  CHECK(arena.deallocate(second_segment.address));

  const Allocation chosen = *arena.allocate(1536);
  CHECK(chosen.address == second_segment.address);
}

void test_two_sided_coalescing() {
  BlockArena arena(256);
  arena.add_segment(0x10000000, 2048);
  const Allocation left = *arena.allocate(512);
  const Allocation middle = *arena.allocate(512);
  const Allocation right = *arena.allocate(512);

  CHECK(arena.deallocate(left.address));
  CHECK(arena.deallocate(right.address));
  CHECK(arena.deallocate(middle.address));

  const Allocation whole = *arena.allocate(2048);
  CHECK(whole.address == 0x10000000);
}

void test_invalid_deallocation_is_rejected() {
  BlockArena arena(256);
  arena.add_segment(0x10000000, 2048);
  const Allocation allocation = *arena.allocate(512);

  CHECK(!arena.deallocate(allocation.address + 1));
  CHECK(!arena.deallocate(0x20000000));
  CHECK(arena.deallocate(allocation.address));
  CHECK(!arena.deallocate(allocation.address));
}

void test_empty_cache_only_releases_free_segments() {
  MockSegmentProvider provider(8192);
  HostAllocatorModel allocator(
      Config{.allocation_alignment = 256,
             .segment_alignment = 256,
             .segment_size = 2048},
      provider);

  const Allocation live = *allocator.allocate(1024);
  const Allocation second_segment = *allocator.allocate(3072);
  CHECK(provider.reserved() == 5120);
  CHECK(allocator.deallocate(second_segment.address));

  allocator.empty_cache();
  CHECK(provider.reserved() == 2048);
  CHECK(provider.release_count() == 1);
  CHECK(allocator.deallocate(live.address));
}

void test_oom_release_and_retry() {
  MockSegmentProvider provider(4096);
  HostAllocatorModel allocator(
      Config{.allocation_alignment = 256,
             .segment_alignment = 256,
             .segment_size = 2048},
      provider);

  const Allocation first = *allocator.allocate(1024);
  const Allocation second = *allocator.allocate(1024);
  CHECK(allocator.deallocate(first.address));
  CHECK(allocator.deallocate(second.address));

  const auto recovered = allocator.allocate(3072);
  CHECK(recovered.has_value());
  CHECK(provider.reserved() == 3072);
  CHECK(provider.release_count() == 1);
}

void test_hard_oom_preserves_live_allocation() {
  MockSegmentProvider provider(4096);
  HostAllocatorModel allocator(
      Config{.allocation_alignment = 256,
             .segment_alignment = 256,
             .segment_size = 2048},
      provider);

  const Allocation live = *allocator.allocate(2048);
  CHECK(!allocator.allocate(3072).has_value());
  CHECK(provider.reserved() == 2048);
  CHECK(allocator.deallocate(live.address));
}

void test_runahead_scratch_plateaus() {
  constexpr std::size_t kSegmentSize = 64 * 1024;
  MockSegmentProvider provider(16 * kSegmentSize);
  HostAllocatorModel allocator(
      Config{.allocation_alignment = 256,
             .segment_alignment = 4096,
             .segment_size = kSegmentSize},
      provider);

  for (std::size_t iteration = 0; iteration < 50000; ++iteration) {
    const Allocation a = *allocator.allocate(4096);
    const Allocation b = *allocator.allocate(8192);
    const Allocation c = *allocator.allocate(2304);
    CHECK(allocator.deallocate(b.address));
    const Allocation d = *allocator.allocate(6144);
    CHECK(allocator.deallocate(a.address));
    CHECK(allocator.deallocate(c.address));
    CHECK(allocator.deallocate(d.address));
  }

  CHECK(provider.reserved() == kSegmentSize);
  CHECK(provider.acquire_count() == 1);
  allocator.empty_cache();
  CHECK(provider.reserved() == 0);
}

bool overlaps(const Allocation& left, const Allocation& right) {
  return left.address < right.address + right.size &&
         right.address < left.address + left.size;
}

void test_randomized_allocations_do_not_overlap() {
  MockSegmentProvider provider(1024 * 1024);
  HostAllocatorModel allocator(
      Config{.allocation_alignment = 256,
             .segment_alignment = 4096,
             .segment_size = 64 * 1024},
      provider);
  std::mt19937 generator(0xC0FFEE);
  std::vector<Allocation> live;

  for (std::size_t operation = 0; operation < 20000; ++operation) {
    const bool should_allocate =
        live.empty() || (live.size() < 128 && generator() % 100 < 60);
    if (should_allocate) {
      const auto allocation = allocator.allocate(1 + generator() % 8192);
      CHECK(allocation.has_value());
      for (const Allocation& other : live) {
        CHECK(!overlaps(*allocation, other));
      }
      live.push_back(*allocation);
    } else {
      const std::size_t index = generator() % live.size();
      CHECK(allocator.deallocate(live[index].address));
      live[index] = live.back();
      live.pop_back();
    }
  }

  for (const Allocation& allocation : live) {
    CHECK(allocator.deallocate(allocation.address));
  }
  allocator.empty_cache();
  CHECK(provider.reserved() == 0);
}

using Test = void (*)();

struct TestCase {
  std::string_view name;
  Test run;
};

}  // namespace

int main() {
  constexpr TestCase tests[] = {
      {"zero_size_and_null_free", test_zero_size_and_null_free},
      {"alignment_split_and_reuse", test_alignment_split_and_reuse},
      {"best_fit_and_complete_coalescing",
       test_best_fit_and_complete_coalescing},
      {"best_fit_across_segments", test_best_fit_across_segments},
      {"two_sided_coalescing", test_two_sided_coalescing},
      {"invalid_deallocation_is_rejected",
       test_invalid_deallocation_is_rejected},
      {"empty_cache_only_releases_free_segments",
       test_empty_cache_only_releases_free_segments},
      {"oom_release_and_retry", test_oom_release_and_retry},
      {"hard_oom_preserves_live_allocation",
       test_hard_oom_preserves_live_allocation},
      {"runahead_scratch_plateaus", test_runahead_scratch_plateaus},
      {"randomized_allocations_do_not_overlap",
       test_randomized_allocations_do_not_overlap},
  };

  for (const TestCase& test : tests) {
    test.run();
    std::cout << "PASS " << test.name << '\n';
  }
}
