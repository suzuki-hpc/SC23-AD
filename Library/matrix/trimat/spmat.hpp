#ifndef SENKPP_MATRIX_TRIMAT_SPMAT_HPP
#define SENKPP_MATRIX_TRIMAT_SPMAT_HPP

#include <type_traits>

#include "enums.hpp"
#include "utils/alloc.hpp"
#include "helper/helper_matrix.hpp"

#define ILUB_FORWARD(a, b, left, right) { \
    _Pragma("omp simd simdlen(Bnl)") \
    for(int k=0; k<Bnl; k++) { \
        (left)[i+k] -= val[j*Bsize+Bnl*(a)+k] * (right)[x_ind+(b)]; \
    } \
}

#define ILUB_BACKWARD(a, b, left, right) { \
    _Pragma("omp simd simdlen(Bnl)") \
    for(int k=0; k<Bnl; k++) { \
        (left)[i+k] -= val[j*Bsize+Bnl*(a)+k] * (right)[x_ind+(b)]; \
    } \
}

#define TRIMAT_FLEX_PARAM 30
#define TRIMAT_FLEX_PRINT false

#include "ltri.hpp"
#include "ldtri.hpp"
#include "dutri.hpp"

#endif
