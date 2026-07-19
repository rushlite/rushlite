#include "lamp3/tensor/infer_meta.hpp"

#include <algorithm>
#include <functional>
#include <numeric>

#include "lamp3/common/assert.hpp"
#include "lamp3/tensor/data_type.hpp"
#include "lamp3/tensor/tensor_impl.hpp"
#include "lamp3/tensor/utils/align_utils.hpp"

namespace lmp::tensor::detail {

OpMeta infer_unary(const TensorImpl* a) {
  OpMeta m;
  m.dtype = a->type();
  m.size = a->numel();
  m.shape = a->shape();
  m.expand = false;
  return m;
}

OpMeta infer_binary(const TensorImpl* a, const TensorImpl* b) {
  LMP_INTERNAL_ASSERT(a->device() == b->device())
      << "Should have asserted already";
  OpMeta m;
  m.dtype = type_upcast(a->type(), b->type());
  m.size = a->numel();
  m.shape = a->shape();
  m.expand = (a->shape() != b->shape());
  if (m.expand) {
    detail::AlignUtil expand_dims(a->shape(), b->shape());
    m.size = expand_dims.aligned_size_;
    m.shape = expand_dims.aligned_shape_;
  }
  return m;
}

OpMeta infer_reduct(const TensorImpl* a, size_t axis) {
  LMP_CHECK(axis < a->shape().size())
      << "Reduction axis " << axis << " out of range for rank "
      << a->shape().size();
  OpMeta m;
  m.dtype = a->type();
  m.shape = a->shape();
  m.expand = false;
  m.shape[axis] = 1;
  m.size = std::accumulate(m.shape.begin(), m.shape.end(), size_t{1},
                           std::multiplies<>());
  return m;
}

MatmulMeta infer_matmul(const TensorImpl* a, const TensorImpl* b) {
  LMP_INTERNAL_ASSERT(a->device() == b->device())
      << "Matmul operands must be on the same device";
  LMP_CHECK(a->shape().size() == a->strides().size() &&
            b->shape().size() == b->strides().size())
      << "Matmul operand shape and stride ranks must match";
  LMP_CHECK(a->shape().size() >= 2 && b->shape().size() >= 2)
      << "Matmul requires both operands to have rank at least two";

  const size_t a_rank = a->shape().size();
  const size_t b_rank = b->shape().size();
  const size_t m = a->shape()[a_rank - 2];
  const size_t k = a->shape()[a_rank - 1];
  const size_t b_k = b->shape()[b_rank - 2];
  const size_t n = b->shape()[b_rank - 1];
  LMP_CHECK(k == b_k)
      << "Incompatible matrix contraction dimensions: " << k << " and "
      << b_k;

  const size_t a_batch_rank = a_rank - 2;
  const size_t b_batch_rank = b_rank - 2;
  const size_t batch_rank = std::max(a_batch_rank, b_batch_rank);
  LMP_CHECK(batch_rank <= LMP_MAX_DIMS)
      << "Matmul batch rank " << batch_rank << " exceeds LMP_MAX_DIMS ("
      << LMP_MAX_DIMS << ')';

  std::vector<size_t> batch_shape(batch_rank);
  for (size_t dim = 0; dim < batch_rank; ++dim) {
    const size_t a_leading = batch_rank - a_batch_rank;
    const size_t b_leading = batch_rank - b_batch_rank;
    const size_t a_size = dim < a_leading ? 1 : a->shape()[dim - a_leading];
    const size_t b_size = dim < b_leading ? 1 : b->shape()[dim - b_leading];
    LMP_CHECK(a_size == 1 || b_size == 1 || a_size == b_size)
        << "Matmul batch dimensions are not broadcast-compatible at output "
           "dimension "
        << dim << ": " << a_size << " and " << b_size;
    batch_shape[dim] = a_size == 1 ? b_size : a_size;
  }

  size_t batch_count = 1;
  for (const size_t dim : batch_shape) batch_count *= dim;

  MatmulMeta meta;
  meta.dtype = type_upcast(a->type(), b->type());
  meta.batch_shape = batch_shape;
  meta.batch_count = batch_count;
  meta.m = m;
  meta.n = n;
  meta.k = k;
  meta.shape = std::move(batch_shape);
  meta.shape.push_back(m);
  meta.shape.push_back(n);
  meta.size = batch_count * m * n;
  return meta;
}

}  // namespace lmp::tensor::detail
