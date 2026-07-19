#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "lamp3/tensor/offset_util.hpp"

namespace lmp::tensor::test {

template <size_t NArgs>
struct OffsetCase {
  std::string name;
  detail::shape_list iteration_shape;
  std::array<detail::OperandLayout, NArgs> operands;
  std::vector<size_t> logical_indices;
  std::vector<std::array<detail::stride_t, NArgs>> expected_offsets;
};

inline std::vector<OffsetCase<1>> unary_offset_cases() {
  using detail::OperandLayout;

  return {
      {"rank-one contiguous",
       {5},
       std::array{OperandLayout{{5}, {1}}},
       {0, 3, 4},
       {{{0}}, {{3}}, {{4}}}},
      {"rank-two contiguous",
       {2, 3},
       std::array{OperandLayout{{2, 3}, {3, 1}}},
       {0, 4, 5},
       {{{0}}, {{4}}, {{5}}}},
      {"contiguous",
       {2, 3, 4},
       std::array{OperandLayout{{2, 3, 4}, {12, 4, 1}}},
       {0, 13, 23},
       {{{0}}, {{13}}, {{23}}}},
      {"rank-two transpose",
       {3, 2},
       std::array{OperandLayout{{3, 2}, {1, 3}}},
       {0, 1, 2, 5},
       {{{0}}, {{3}}, {{1}}, {{5}}}},
      {"higher-rank permutation",
       {3, 4, 2},
       std::array{OperandLayout{{3, 4, 2}, {4, 1, 12}}},
       {0, 1, 13, 22},
       {{{0}}, {{12}}, {{18}}, {{11}}}},
      {"size-one dimension with arbitrary stride",
       {2, 1, 3},
       std::array{OperandLayout{{2, 1, 3}, {3, 99, 1}}},
       {0, 2, 5},
       {{{0}}, {{2}}, {{5}}}},
      {"reduction projection axis zero",
       {1, 4, 2},
       std::array{OperandLayout{{3, 4, 2}, {4, 1, 12}}},
       {0, 1, 7},
       {{{0}}, {{12}}, {{15}}}},
      {"reduction projection axis one",
       {3, 1, 2},
       std::array{OperandLayout{{3, 4, 2}, {4, 1, 12}}},
       {0, 1, 2, 5},
       {{{0}}, {{12}}, {{4}}, {{20}}}},
      {"reduction projection axis two",
       {3, 4, 1},
       std::array{OperandLayout{{3, 4, 2}, {4, 1, 12}}},
       {0, 3, 11},
       {{{0}}, {{3}}, {{11}}}},
      {"scalar broadcast",
       {2, 3},
       std::array{OperandLayout{{1}, {7}}},
       {0, 1, 5},
       {{{0}}, {{0}}, {{0}}}},
      {"rank-zero batch",
       {},
       std::array{OperandLayout{{}, {}}},
       {0},
       {{{0}}}},
      {"maximum rank",
       {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
       std::array{OperandLayout{
           {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
           {31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18,
            17, 16}}},
       {0, 1},
       {{{0}}, {{31}}}},
  };
}

inline std::vector<OffsetCase<2>> binary_offset_cases() {
  using detail::OperandLayout;

  return {
      {"equal contiguous",
       {2, 3},
       std::array{OperandLayout{{2, 3}, {3, 1}},
                  OperandLayout{{2, 3}, {3, 1}}},
       {0, 4, 5},
       {{{0, 0}}, {{4, 4}}, {{5, 5}}}},
      {"matching transposes",
       {3, 2},
       std::array{OperandLayout{{3, 2}, {1, 3}},
                  OperandLayout{{3, 2}, {1, 3}}},
       {1, 4, 5},
       {{{3, 3}}, {{2, 2}}, {{5, 5}}}},
      {"mixed layouts",
       {3, 2},
       std::array{OperandLayout{{3, 2}, {1, 3}},
                  OperandLayout{{3, 2}, {2, 1}}},
       {1, 4},
       {{{3, 1}}, {{2, 4}}}},
      {"row broadcast",
       {2, 3},
       std::array{OperandLayout{{2, 3}, {3, 1}},
                  OperandLayout{{3}, {1}}},
       {1, 4, 5},
       {{{1, 1}}, {{4, 1}}, {{5, 2}}}},
      {"column broadcast",
       {2, 3},
       std::array{OperandLayout{{2, 3}, {3, 1}},
                  OperandLayout{{2, 1}, {1, 9}}},
       {2, 3, 5},
       {{{2, 0}}, {{3, 1}}, {{5, 1}}}},
      {"scalar broadcast",
       {2, 3},
       std::array{OperandLayout{{2, 3}, {3, 1}},
                  OperandLayout{{1}, {11}}},
       {0, 5},
       {{{0, 0}}, {{5, 0}}}},
      {"missing leading dimensions",
       {2, 3, 4},
       std::array{OperandLayout{{3, 4}, {4, 1}},
                  OperandLayout{{4}, {1}}},
       {0, 19, 23},
       {{{0, 0}}, {{7, 3}}, {{11, 3}}}},
      {"different leading broadcasts",
       {4, 7},
       std::array{OperandLayout{{4, 1}, {1, 8}},
                  OperandLayout{{1, 7}, {7, 1}}},
       {0, 24, 27},
       {{{0, 0}}, {{3, 3}}, {{3, 6}}}},
      {"permutation plus broadcast",
       {2, 3, 4},
       std::array{OperandLayout{{2, 3, 4}, {1, 8, 2}},
                  OperandLayout{{3, 1}, {1, 5}}},
       {0, 19, 23},
       {{{0, 0}}, {{15, 1}}, {{23, 2}}}},
      {"rank-zero batches",
       {},
       std::array{OperandLayout{{}, {}}, OperandLayout{{}, {}}},
       {0},
       {{{0, 0}}}},
      {"matching one-dimensional batches",
       {5},
       std::array{OperandLayout{{5}, {12}},
                  OperandLayout{{5}, {20}}},
       {0, 2, 4},
       {{{0, 0}}, {{24, 40}}, {{48, 80}}}},
      {"shared rank-two operand",
       {4, 7},
       std::array{OperandLayout{{4, 1}, {12, 12}},
                  OperandLayout{{}, {}}},
       {0, 24, 27},
       {{{0, 0}}, {{36, 0}}, {{36, 0}}}},
      {"noncanonical batch strides",
       {3, 2},
       std::array{OperandLayout{{3, 2}, {1, 3}},
                  OperandLayout{{1, 2}, {5, 1}}},
       {0, 4, 5},
       {{{0, 0}}, {{2, 0}}, {{5, 1}}}},
  };
}

inline std::vector<OffsetCase<3>> ternary_offset_cases() {
  using detail::OperandLayout;

  return {
      {"arity-three smoke",
       {2, 2},
       std::array{OperandLayout{{2, 2}, {2, 1}},
                  OperandLayout{{2, 2}, {1, 2}},
                  OperandLayout{{1}, {9}}},
       {0, 2, 3},
       {{{0, 0, 0}}, {{2, 1, 0}}, {{3, 3, 0}}}},
  };
}

}  // namespace lmp::tensor::test
