/* Copyright 2025 The OpenXLA Authors.

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

#if defined(__has_attribute) && __has_attribute(ext_vector_type) && \
    defined(__has_builtin) && __has_builtin(__builtin_vectorelements)

#include "xla/codegen/intrinsic/cpp/eigen_unary.h"

#include <cmath>

#include "Eigen/Core"
#include "xla/codegen/intrinsic/cpp/vector_ops.h"

namespace xla::codegen {

//===--------------------------------------------------------------------===//
// Generic conversion and operation
//===--------------------------------------------------------------------===//

template <typename VecType>
inline VecType VectorTanh(const VecType x) {
  using ArrayType = typename ArrayMap<VecType>::type;
  ArrayType x_array = *reinterpret_cast<const ArrayType*>(&x);
  ArrayType result = x_array.tanh();
  return *reinterpret_cast<const VecType*>(&result);
}

template <typename VecType>
inline VecType VectorAtan(const VecType x) {
  using ArrayType = typename ArrayMap<VecType>::type;
  ArrayType x_array = *reinterpret_cast<const ArrayType*>(&x);
  ArrayType result = x_array.atan();
  return *reinterpret_cast<const VecType*>(&result);
}

//===--------------------------------------------------------------------===//
// XLA entrypoints, renamed with asm in header file.
//===--------------------------------------------------------------------===//

using FloatArrayType = Eigen::Array<float, 16, 1>;

// Single precision
float tanh_f32(float x) { return Eigen::internal::ptanh_float(x); }
Vec4f tanh_v4f32(Vec4f x) {
  FloatArrayType buffer;
  *reinterpret_cast<Vec4f*>(&buffer) = x;
  buffer = buffer.tanh();
  return *reinterpret_cast<Vec4f*>(&buffer);
}
Vec8f tanh_v8f32(Vec8f x) {
  FloatArrayType buffer;
  *reinterpret_cast<Vec8f*>(&buffer) = x;
  buffer = buffer.tanh();
  return *reinterpret_cast<Vec8f*>(&buffer);
}
Vec16f tanh_v16f32(Vec16f x) { return VectorTanh(x); }

// Double precision
using DoubleArrayType = Eigen::Array<double, 8, 1>;
double tanh_f64(double x) { return Eigen::internal::ptanh_double(x); }
Vec4d tanh_v4f64(Vec4d x) {
  DoubleArrayType buffer;
  *reinterpret_cast<Vec4d*>(&buffer) = x;
  buffer = buffer.tanh();
  return *reinterpret_cast<Vec4d*>(&buffer);
}
Vec8d tanh_v8f64(Vec8d x) { return VectorTanh(x); }

// Single precision
float atan_f32(float x_in) {
  constexpr float kPiOverTwo = 1.5707963267948966f;

  float abs_x = std::abs(x_in);
  float x = (abs_x > 1.0f) ? (1.0f / abs_x) : abs_x;

  constexpr float alpha[] = {1.12026982009410858154296875e-01f,
                             7.296695709228515625e-01f,
                             8.109951019287109375e-01f};

  constexpr float beta[] = {1.00917108356952667236328125e-02f,
                            2.8318560123443603515625e-01f, 1.0f,
                            8.109951019287109375e-01f};

  float x2 = x * x;
  float p = (alpha[0] * x2 + alpha[1]) * x2 + alpha[2];
  float q = ((beta[0] * x2 + beta[1]) * x2 + beta[2]) * x2 + beta[3];
  float r = x * (p / q);

  float result = (abs_x > 1.0f) ? (kPiOverTwo - r) : r;
  return std::copysign(result, x_in);
}
Vec4f atan_v4f32(Vec4f x) {
  FloatArrayType buffer;
  *reinterpret_cast<Vec4f*>(&buffer) = x;
  buffer = buffer.atan();
  return *reinterpret_cast<Vec4f*>(&buffer);
}
Vec8f atan_v8f32(Vec8f x) {
  FloatArrayType buffer;
  *reinterpret_cast<Vec8f*>(&buffer) = x;
  buffer = buffer.atan();
  return *reinterpret_cast<Vec8f*>(&buffer);
}
Vec16f atan_v16f32(Vec16f x) { return VectorAtan(x); }

// Double precision
double atan_f64(double x_in) {
  constexpr double kPiOverTwo = 1.57079632679489661923;

  double abs_x = std::abs(x_in);
  double x = (abs_x > 1.0) ? (1.0 / abs_x) : abs_x;

  constexpr double alpha[] = {2.6667153866462208e-05, 3.0917513112462781e-03,
                              5.2574296781008604e-02, 3.0409318473444424e-01,
                              7.5365702534987022e-01, 8.2704055405494614e-01,
                              3.3004361289279920e-01};

  constexpr double beta[] = {2.7311202462436667e-04,
                             1.0899150928962708e-02,
                             1.1548932646420353e-01,
                             4.9716458728465573e-01,
                             1.0,
                             9.3705509168587852e-01,
                             3.3004361289279920e-01};

  double x2 = x * x;
  double p =
      (((((alpha[0] * x2 + alpha[1]) * x2 + alpha[2]) * x2 + alpha[3]) * x2 +
        alpha[4]) *
           x2 +
       alpha[5]) *
          x2 +
      alpha[6];

  double q = (((((beta[0] * x2 + beta[1]) * x2 + beta[2]) * x2 + beta[3]) * x2 +
               beta[4]) *
                  x2 +
              beta[5]) *
                 x2 +
             beta[6];

  double r = x * (p / q);
  double result = (abs_x > 1.0) ? (kPiOverTwo - r) : r;
  return std::copysign(result, x_in);
}
Vec4d atan_v4f64(Vec4d x) {
  DoubleArrayType buffer;
  *reinterpret_cast<Vec4d*>(&buffer) = x;
  buffer = buffer.atan();
  return *reinterpret_cast<Vec4d*>(&buffer);
}
Vec8d atan_v8f64(Vec8d x) { return VectorAtan(x); }

}  // namespace xla::codegen
#endif  // defined(__has_attribute) && __has_attribute(vector_size) &&
        // defined(__has_builtin) && __has_builtin(__builtin_vectorelements)
