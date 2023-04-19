/**
 * @file array.hpp
 * @brief Template functions that operate to arrays are defined.
 * @author Kengo Suzuki
 * @date 02/02/2023
 */
#ifndef SENKPP_UTILS_ARRAY_HPP
#define SENKPP_UTILS_ARRAY_HPP

namespace senk {

namespace utils {

template <typename T>
inline void Copy(T *in, T *out, int size)
{
    #pragma omp parallel for simd
    for(int i=0; i<size; i++) { out[i] = in[i]; }
}

template <typename T>
inline void Set(T val, T *out, int size)
{
    #pragma omp parallel for simd
    for(int i=0; i<size; i++) { out[i] = val; }
}

template <typename T1, typename T2>
inline void Convert(T1 *in, T2 *out, int size)
{
    #pragma omp parallel for simd
    for(int i=0; i<size; i++) { out[i] = (T2)in[i]; }
}

template <typename T, int bit>
inline void Convert_ID(T *in, double *out, int size)
{
    const double fact_inv = 1.0 / (1 << bit);
    #pragma omp parallel for simd
    for(int i=0; i<size; i++) { out[i] = (double)in[i] * fact_inv; }
}

template <typename T, int bit>
inline void Convert_DI(double *in, T *out, int size)
{
    const double fact = (double)(1 << bit);
    #pragma omp parallel for simd
    for(int i=0; i<size; i++) { out[i] = (T)(in[i] * fact); }
}

}

}

#endif
