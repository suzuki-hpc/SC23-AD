/**
 * @file blas2.hpp
 * @brief Level2 BLAS-style functions are written.
 * @author Kengo Suzuki
 * @date 5/8/2021
 */
#ifndef SENKPP_BLAS2_HPP
#define SENKPP_BLAS2_HPP

namespace senk {
/**
 * @brief This namespace contains Level2 BLAS-style functions.
 */
namespace blas2 {
/**
 * @brief Upper triangular solver.
 * @tparam T The type of vectors.
 * @param U A 2D-array of size n * m that represents an upper triangular matrix.
 * @param b A 1D-array of size m.
 * @param x A 1D-array of size m.
 * @param n The number of rows of the matrix.
 * @param m The number of columns of the matrix.
 */
template <typename T>
inline void Trsv(T *U, T *b, T *x, int n, int m)
{
    for(int i=m-1; i>=0; i--) {
        T temp = b[i];
        for(int j=m-1; j>i; j--) {
            temp -= U[j*n+i] * x[j];
        }
        x[i] = temp / U[i*n+i];
    }
}

namespace int32 {

template <int bit>
inline void Trsv(long *U, double b_scal, long *b, double *x, int n, int m)
{
    constexpr double fact = (double)((long)1 << bit);
    for(int i=m-1; i>=0; i--) {
        double temp = b_scal * (double)b[i]/fact;
        for(int j=m-1; j>i; j--) {
            temp -= (double)U[j*n+i]/fact * x[j];
        }
        x[i] = temp / ((double)U[i*n+i]/fact);
    }
}

inline void Trsv(long *U, double b_scal, long *b, double *x, int n, int m, int8_t bit)
{
    const double fact = (double)((long)1 << bit);
    for(int i=m-1; i>=0; i--) {
        double temp = b_scal * (double)b[i]/fact;
        for(int j=m-1; j>i; j--) {
            temp -= (double)U[j*n+i]/fact * x[j];
        }
        x[i] = temp / ((double)U[i*n+i]/fact);
    }
}

}

}

}

#endif
