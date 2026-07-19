#pragma once

#include "lamp3/tensor/data_type.hpp"
#include "lamp3/tensor/device_type.hpp"
#include "lamp3/tensor/dispatch_type.hpp"
#include "lamp3/tensor/native/expand_ops.hpp"
#include "lamp3/tensor/native/matrix_ops.hpp"
#include "lamp3/tensor/native/overloads.hpp"
#include "lamp3/tensor/native/reduct_ops.hpp"
#include "lamp3/tensor/native/shape_ops.hpp"
#include "lamp3/tensor/native/unary_ops.hpp"
#include "lamp3/tensor/tensor.hpp"
#include "lamp3/tensor/tensor_impl.hpp"

#include "lamp3/tensor/lazy/capture_mode.hpp"
#include "lamp3/tensor/lazy/lazy_backend.hpp"
#include "lamp3/tensor/lazy/lazy_function.hpp"
#include "lamp3/tensor/lazy/realize.hpp"
#include "lamp3/tensor/lazy/record.hpp"
#include "lamp3/tensor/lazy/functions/elementwise_binary.hpp"
#include "lamp3/tensor/lazy/functions/elementwise_unary.hpp"

#ifdef LMP_ENABLE_CUDA
#include "lamp3/tensor/cuda/binary.cuh"
#include "lamp3/tensor/cuda/expand.cuh"
#include "lamp3/tensor/cuda/kernels.cuh"
#include "lamp3/tensor/cuda/list_ptr.cuh"
#include "lamp3/tensor/cuda/matrix.cuh"
#include "lamp3/tensor/cuda/memory.cuh"
#include "lamp3/tensor/cuda/offset_util.cuh"
#include "lamp3/tensor/cuda/ptr_pack.cuh"
#include "lamp3/tensor/cuda/reduct.cuh"
#include "lamp3/tensor/cuda/unary.cuh"
#endif
