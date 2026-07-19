#pragma once

#include "lamp3/tensor/tensor.hpp"

namespace lmp::tensor::ops {

/**
 * @brief Reshape a tensor
 * @param a The tensor to reshape
 * @param new_shape The new shape of the tensor
 * @return A new view with the result of the reshape operation
 * @note this function returns a view, not a new tensor.
 */
Tensor reshape(const Tensor& a, std::vector<size_t> new_shape);

/**
 * @brief Swap two dimensions of a tensor
 * @return A storage-sharing metadata view
 */
Tensor transpose(const Tensor& a, size_t dim0, size_t dim1);

/**
 * @brief Reorder all dimensions of a tensor
 * @return A storage-sharing metadata view
 */
Tensor permute(const Tensor& a, const std::vector<size_t>& dims);

/**
 * @brief Squeeze a tensor
 * @param a The tensor to squeeze
 * @param dim The dimension to squeeze
 * @return A new view with the result of the squeeze operation
 * @note this function returns a view, not a new tensor.
 */
Tensor squeeze(const Tensor& a, size_t dim);

/**
 * @brief Expand a tensor
 * @param a The tensor to expand
 * @param dim The dimension to expand
 * @return A new view with the result of the expand operation
 * @note this function returns a view, not a new tensor.
 */
Tensor expand_dims(const Tensor& a, size_t dim);

}  // namespace lmp::tensor::ops
