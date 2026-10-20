//===- TensorView.cpp - Template stubs ------------------------------------===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// These entry points deliberately perform no memory access. Replace their
// bodies before using the passthrough path for numerical execution.

#include "Utils.h"
#include <cstdint>

#define TV_DIM(D)                                                              \
  int64_t size##D, int64_t stride##D, int64_t tile##D, int64_t traversal##D,   \
      int64_t localSize##D, int64_t localStride##D, int64_t coordinate##D,     \
      int64_t indexAddress##D, int64_t indexStride##D, int64_t indexBits##D
#define TV_PTR_DIM(D)                                                          \
  int64_t size##D, int64_t stride##D, int64_t indexAddress##D,                 \
      int64_t indexStride##D
#define TV_DIMS_1 TV_DIM(0)
#define TV_DIMS_2 TV_DIMS_1, TV_DIM(1)
#define TV_DIMS_3 TV_DIMS_2, TV_DIM(2)
#define TV_DIMS_4 TV_DIMS_3, TV_DIM(3)
#define TV_PTR_DIMS_1 TV_PTR_DIM(0)
#define TV_PTR_DIMS_2 TV_PTR_DIMS_1, TV_PTR_DIM(1)
#define TV_PTR_DIMS_3 TV_PTR_DIMS_2, TV_PTR_DIM(2)
#define TV_PTR_DIMS_4 TV_PTR_DIMS_3, TV_PTR_DIM(3)

#define TV_ACCESS(KIND, RANK)                                                  \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_tensor_view_##KIND##_r##RANK(                               \
          int64_t base, int64_t buffer, int64_t elementType, int64_t padding,  \
          int64_t sparseMask, TV_DIMS_##RANK) {}
#define TV_PTR_ACCESS(KIND, RANK)                                              \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_tensor_view_ptr_##KIND##_r##RANK(                           \
          int64_t pointers, int64_t buffer, int64_t elementType,               \
          int64_t length, int64_t localStride, TV_PTR_DIMS_##RANK) {}
#define TV_RANK(RANK)                                                          \
  TV_ACCESS(load, RANK)                                                        \
  TV_ACCESS(store, RANK)                                                       \
  TV_PTR_ACCESS(load, RANK)                                                    \
  TV_PTR_ACCESS(store, RANK)

extern "C" {
TV_RANK(1)
TV_RANK(2)
TV_RANK(3)
TV_RANK(4)
}
