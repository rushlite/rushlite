#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "lamp3/common/assert.hpp"

namespace lmp::tensor::detail::cuda::allocator {

using Address = std::uintptr_t;

struct Allocation {
  Address address = 0;
  std::size_t size = 0;
};

struct ReleasedSegment {
  Address address = 0;
  std::size_t size = 0;
};

inline std::size_t align_up(std::size_t value, std::size_t alignment) {
  LMP_INTERNAL_ASSERT(alignment != 0 && (alignment & (alignment - 1)) == 0)
      << "alignment must be a power of two";
  return (value + alignment - 1) & ~(alignment - 1);
}

class BlockArena {
 private:
  struct Segment;
  struct Block;
  using FreeIndex = std::multimap<std::size_t, Block*>;

  struct Block {
    Address address = 0;
    std::size_t size = 0;
    bool allocated = false;
    bool indexed = false;
    Segment* segment = nullptr;
    Block* previous = nullptr;
    Block* next = nullptr;
    FreeIndex::iterator free_position;
  };

  struct Segment {
    Address address = 0;
    std::size_t size = 0;
    Block* first = nullptr;
  };

 public:
  explicit BlockArena(std::size_t allocation_alignment)
      : allocation_alignment_(allocation_alignment) {}

  void add_segment(Address address, std::size_t size) {
    LMP_INTERNAL_ASSERT(address != 0 && size != 0 &&
                        address % allocation_alignment_ == 0 &&
                        size % allocation_alignment_ == 0)
        << "segment must be nonzero and aligned";

    auto segment = std::make_unique<Segment>();
    segment->address = address;
    segment->size = size;
    Segment* segment_ptr = segment.get();

    Block* block = make_block();
    block->address = address;
    block->size = size;
    block->segment = segment_ptr;
    segment_ptr->first = block;

    segments_.emplace(address, std::move(segment));
    insert_free(block);
  }

  std::optional<Allocation> allocate(std::size_t size) {
    if (size == 0) {
      return Allocation{};
    }

    const std::size_t aligned_size = align_up(size, allocation_alignment_);
    auto candidate = free_index_.lower_bound(aligned_size);
    if (candidate == free_index_.end()) {
      return std::nullopt;
    }

    Block* block = candidate->second;
    erase_free(block);

    if (block->size > aligned_size) {
      Block* remainder = make_block();
      remainder->address = block->address + aligned_size;
      remainder->size = block->size - aligned_size;
      remainder->segment = block->segment;
      remainder->previous = block;
      remainder->next = block->next;
      if (remainder->next != nullptr) {
        remainder->next->previous = remainder;
      }
      block->next = remainder;
      block->size = aligned_size;
      insert_free(remainder);
    }

    block->allocated = true;
    allocated_.emplace(block->address, block);
    return Allocation{block->address, block->size};
  }

  bool deallocate(Address address) {
    if (address == 0) {
      return true;
    }

    auto allocation = allocated_.find(address);
    if (allocation == allocated_.end()) {
      return false;
    }

    Block* block = allocation->second;
    allocated_.erase(allocation);
    block->allocated = false;

    if (block->previous != nullptr && !block->previous->allocated) {
      Block* previous = block->previous;
      erase_free(previous);
      previous->size += block->size;
      previous->next = block->next;
      if (block->next != nullptr) {
        block->next->previous = previous;
      }
      destroy_block(block);
      block = previous;
    }

    if (block->next != nullptr && !block->next->allocated) {
      Block* next = block->next;
      erase_free(next);
      block->size += next->size;
      block->next = next->next;
      if (block->next != nullptr) {
        block->next->previous = block;
      }
      destroy_block(next);
    }

    insert_free(block);
    return true;
  }

  std::vector<ReleasedSegment> extract_fully_free_segments() {
    std::vector<ReleasedSegment> released;
    for (auto iterator = segments_.begin(); iterator != segments_.end();) {
      Segment* segment = iterator->second.get();
      Block* block = segment->first;
      if (!block->allocated && block->next == nullptr &&
          block->address == segment->address && block->size == segment->size) {
        erase_free(block);
        released.push_back({segment->address, segment->size});
        destroy_block(block);
        iterator = segments_.erase(iterator);
      } else {
        ++iterator;
      }
    }
    return released;
  }

 private:
  Block* make_block() {
    auto owner = std::make_unique<Block>();
    Block* block = owner.get();
    nodes_.emplace(block, std::move(owner));
    return block;
  }

  void destroy_block(Block* block) {
    LMP_INTERNAL_ASSERT(!block->allocated && !block->indexed)
        << "destroying a live block";
    nodes_.erase(block);
  }

  void insert_free(Block* block) {
    LMP_INTERNAL_ASSERT(!block->allocated && !block->indexed)
        << "invalid free-index insertion";
    block->free_position = free_index_.emplace(block->size, block);
    block->indexed = true;
  }

  void erase_free(Block* block) {
    LMP_INTERNAL_ASSERT(!block->allocated && block->indexed)
        << "invalid free-index removal";
    free_index_.erase(block->free_position);
    block->indexed = false;
  }

  std::size_t allocation_alignment_;
  std::map<Address, std::unique_ptr<Segment>> segments_;
  std::unordered_map<Block*, std::unique_ptr<Block>> nodes_;
  FreeIndex free_index_;
  std::unordered_map<Address, Block*> allocated_;
};

}  // namespace lmp::tensor::detail::cuda::allocator
