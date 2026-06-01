/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/searchsorted_op.h"

#include <algorithm>
#include <limits>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

namespace functor {
template <typename T, typename OutType, typename SearchFn>
absl::Status ComputeCpu(OpKernelContext* context,
                        const typename TTypes<T, 1>::ConstTensor& sorted_inputs,
                        const typename TTypes<T, 1>::ConstTensor& values,
                        int64_t batch_size, int64_t num_inputs,
                        int64_t num_values,
                        typename TTypes<OutType, 1>::Tensor* output,
                        SearchFn search_fn) {
  auto work_fn = [&](int64_t first_b, int64_t last_b) {
    for (int64_t b = first_b; b < last_b; ++b) {
      const T* sorted_inputs_ptr = sorted_inputs.data() + b * num_inputs;
      OutType* output_ptr = output->data() + b * num_values;
      const T* values_ptr = values.data() + b * num_values;
      for (int64_t i = 0; i < num_values; ++i) {
        output_ptr[i] = static_cast<OutType>(
            search_fn(sorted_inputs_ptr, sorted_inputs_ptr + num_inputs,
                      values_ptr[i]) -
            sorted_inputs_ptr);
      }
    }
  };
  const auto* worker_threads =
      context->device()->tensorflow_cpu_worker_threads();
  thread::ThreadPool* thread_pool = worker_threads->workers;
  const int64_t kCostMultiplier = 1;
  int64_t cost_per_batch =
      kCostMultiplier * num_values * Log2Ceiling64(num_inputs);
  thread_pool->ParallelFor(batch_size, cost_per_batch, work_fn);
  return absl::OkStatus();
}

template <typename T, typename OutType>
struct UpperBoundFunctor<CPUDevice, T, OutType> {
  static absl::Status Compute(
      OpKernelContext* context,
      const typename TTypes<T, 1>::ConstTensor& sorted_inputs,
      const typename TTypes<T, 1>::ConstTensor& values, int64_t batch_size,
      int64_t num_inputs, int64_t num_values,
      typename TTypes<OutType, 1>::Tensor* output) {
    return ComputeCpu<T, OutType>(
        context, sorted_inputs, values, batch_size, num_inputs, num_values,
        output, [](const T* first, const T* last, const T& val) {
          return std::upper_bound(first, last, val);
        });
  }
};

template <typename T, typename OutType>
struct LowerBoundFunctor<CPUDevice, T, OutType> {
  static absl::Status Compute(
      OpKernelContext* context,
      const typename TTypes<T, 1>::ConstTensor& sorted_inputs,
      const typename TTypes<T, 1>::ConstTensor& values, int64_t batch_size,
      int64_t num_inputs, int64_t num_values,
      typename TTypes<OutType, 1>::Tensor* output) {
    return ComputeCpu<T, OutType>(
        context, sorted_inputs, values, batch_size, num_inputs, num_values,
        output, [](const T* first, const T* last, const T& val) {
          return std::lower_bound(first, last, val);
        });
  }
};
}  // namespace functor

template <typename Device, typename T, typename OutType>
class SearchSortedOpBase : public OpKernel {
 public:
  explicit SearchSortedOpBase(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& sorted_inputs_t = ctx->input(0);
    const Tensor& values_t = ctx->input(1);

    // Inputs must be a matrix
    // This replicates the shape requirements for the op in array_ops.cc
    OP_REQUIRES(
        ctx, sorted_inputs_t.shape().dims() == 2,
        absl::InvalidArgumentError(absl::StrCat(
            "Shape must be rank 2 but is rank ", sorted_inputs_t.shape().dims(),
            " for `sorted_inputs` argument")));
    // Values must be a matrix
    // This replicates the shape requirements for the op in array_ops.cc
    OP_REQUIRES(ctx, values_t.shape().dims() == 2,
                absl::InvalidArgumentError(absl::StrCat(
                    "Shape must be rank 2 but is rank ",
                    values_t.shape().dims(), " for `values` argument")));
    // must have same batch dim_size for both
    OP_REQUIRES(ctx, sorted_inputs_t.dim_size(0) == values_t.dim_size(0),
                absl::InvalidArgumentError(
                    "Leading dim_size of both Tensors must match."));

    Tensor* output_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, values_t.shape(), &output_t));

    if (output_t->dtype() == DT_INT32) {
      OP_REQUIRES(ctx,
                  FastBoundsCheck(sorted_inputs_t.dim_size(1),
                                  std::numeric_limits<int>::max() + 1ll),
                  absl::InvalidArgumentError(absl::StrCat(
                      "trailing dim_size must be less than or equal to "
                      "INT_MAX for int32 output type, was ",
                      sorted_inputs_t.dim_size(1))));
    }

    auto output = output_t->template flat<OutType>();
    const auto sorted_inputs = sorted_inputs_t.template flat<T>();
    const auto values = values_t.template flat<T>();

    // For empty inputs, all values will be placed at the zeroth position.
    if (sorted_inputs.size() == 0) {
      functor::SetZeroFunctor<Device, OutType> set_zero;
      set_zero(ctx->eigen_device<Device>(), output);
      return;
    }

    ComputeFunctor(ctx, sorted_inputs, values, sorted_inputs_t.dim_size(0),
                   sorted_inputs_t.dim_size(1), values_t.dim_size(1), &output);
  }

 protected:
  virtual void ComputeFunctor(
      OpKernelContext* ctx,
      const typename TTypes<T, 1>::ConstTensor& sorted_inputs,
      const typename TTypes<T, 1>::ConstTensor& values, int64_t batch_size,
      int64_t num_inputs, int64_t num_values,
      typename TTypes<OutType, 1>::Tensor* output) = 0;
};

template <typename Device, typename T, typename OutType>
class UpperBoundOp : public SearchSortedOpBase<Device, T, OutType> {
 public:
  explicit UpperBoundOp(OpKernelConstruction* ctx)
      : SearchSortedOpBase<Device, T, OutType>(ctx) {}

 protected:
  void ComputeFunctor(OpKernelContext* ctx,
                      const typename TTypes<T, 1>::ConstTensor& sorted_inputs,
                      const typename TTypes<T, 1>::ConstTensor& values,
                      int64_t batch_size, int64_t num_inputs,
                      int64_t num_values,
                      typename TTypes<OutType, 1>::Tensor* output) override {
    OP_REQUIRES_OK(ctx, functor::UpperBoundFunctor<Device, T, OutType>::Compute(
                            ctx, sorted_inputs, values, batch_size, num_inputs,
                            num_values, output));
  }
};

template <typename Device, typename T, typename OutType>
class LowerBoundOp : public SearchSortedOpBase<Device, T, OutType> {
 public:
  explicit LowerBoundOp(OpKernelConstruction* ctx)
      : SearchSortedOpBase<Device, T, OutType>(ctx) {}

 protected:
  void ComputeFunctor(OpKernelContext* ctx,
                      const typename TTypes<T, 1>::ConstTensor& sorted_inputs,
                      const typename TTypes<T, 1>::ConstTensor& values,
                      int64_t batch_size, int64_t num_inputs,
                      int64_t num_values,
                      typename TTypes<OutType, 1>::Tensor* output) override {
    OP_REQUIRES_OK(ctx, functor::LowerBoundFunctor<Device, T, OutType>::Compute(
                            ctx, sorted_inputs, values, batch_size, num_inputs,
                            num_values, output));
  }
};

#define REGISTER_KERNELS(type)                                    \
  REGISTER_KERNEL_BUILDER(Name("UpperBound")                      \
                              .Device(DEVICE_CPU)                 \
                              .TypeConstraint<type>("T")          \
                              .TypeConstraint<int32>("out_type"), \
                          UpperBoundOp<CPUDevice, type, int32>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#define REGISTER_KERNELS(type)                                      \
  REGISTER_KERNEL_BUILDER(Name("UpperBound")                        \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<type>("T")            \
                              .TypeConstraint<int64_t>("out_type"), \
                          UpperBoundOp<CPUDevice, type, int64_t>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_KERNELS(type)                                    \
  REGISTER_KERNEL_BUILDER(Name("UpperBound")                      \
                              .Device(DEVICE_GPU)                 \
                              .TypeConstraint<type>("T")          \
                              .TypeConstraint<int32>("out_type"), \
                          UpperBoundOp<GPUDevice, type, int32>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#define REGISTER_KERNELS(type)                                      \
  REGISTER_KERNEL_BUILDER(Name("UpperBound")                        \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<type>("T")            \
                              .TypeConstraint<int64_t>("out_type"), \
                          UpperBoundOp<GPUDevice, type, int64_t>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_KERNELS(type)                                    \
  REGISTER_KERNEL_BUILDER(Name("LowerBound")                      \
                              .Device(DEVICE_CPU)                 \
                              .TypeConstraint<type>("T")          \
                              .TypeConstraint<int32>("out_type"), \
                          LowerBoundOp<CPUDevice, type, int32>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#define REGISTER_KERNELS(type)                                      \
  REGISTER_KERNEL_BUILDER(Name("LowerBound")                        \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<type>("T")            \
                              .TypeConstraint<int64_t>("out_type"), \
                          LowerBoundOp<CPUDevice, type, int64_t>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_KERNELS(type)                                    \
  REGISTER_KERNEL_BUILDER(Name("LowerBound")                      \
                              .Device(DEVICE_GPU)                 \
                              .TypeConstraint<type>("T")          \
                              .TypeConstraint<int32>("out_type"), \
                          LowerBoundOp<GPUDevice, type, int32>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#define REGISTER_KERNELS(type)                                      \
  REGISTER_KERNEL_BUILDER(Name("LowerBound")                        \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<type>("T")            \
                              .TypeConstraint<int64_t>("out_type"), \
                          LowerBoundOp<GPUDevice, type, int64_t>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}  // namespace tensorflow
