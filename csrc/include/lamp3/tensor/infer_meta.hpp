#pragma once
#include <cstddef>
#include <vector>
#include "lamp3/tensor/data_type.hpp"

namespace lmp::tensor {
class TensorImpl;
namespace detail {

struct OpMeta {
  DataType dtype;
  size_t size;
  std::vector<size_t> shape;
  bool expand;
};

struct MatmulMeta {
  DataType dtype;
  size_t size;
  std::vector<size_t> shape;
  std::vector<size_t> batch_shape;
  size_t batch_count;
  size_t m;
  size_t n;
  size_t k;
};

OpMeta infer_unary(const TensorImpl* a);
OpMeta infer_binary(const TensorImpl* a, const TensorImpl* b);
OpMeta infer_reduct(const TensorImpl* a, size_t axis);
MatmulMeta infer_matmul(const TensorImpl* a, const TensorImpl* b);

}  // namespace detail
}  // namespace lmp::tensor
