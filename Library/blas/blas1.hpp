#ifndef SENKPP_BLAS1_HPP
#define SENKPP_BLAS1_HPP

#include <cmath>

namespace senk {

namespace blas1 {

template <typename T>
inline void Copy(T *x, T *y, int N) {
    #pragma omp parallel for simd
    for(int i=0; i<N; i++) { y[i] = x[i]; }
}

template <typename T>
inline void Scal(T a, T *x, int N) {
    #pragma omp parallel for simd
    for(int i=0; i<N; i++) { x[i] *= a; }
}

template <typename T>
inline void Axpy(T a, T *x, T *y, int N) {
    #pragma omp parallel for simd
    for(int i=0; i<N; i++) { y[i] += a * x[i]; }
}

template <typename T>
inline void Axpby(T a, T *x, T b, T *y, int N) {
    #pragma omp parallel for simd
    for(int i=0; i<N; i++) { y[i] = a * x[i] + b * y[i]; }
}

template <typename T>
inline void Axpyz(T a, T *x, T *y, T *z, int N) {
    #pragma omp parallel for simd
    for(int i=0; i<N; i++) { z[i] = a * x[i] + y[i]; }
}

template <typename T>
inline void Sub(T *x, T *y, int N) {
    #pragma omp parallel for simd
    for(int i=0; i<N; i++) { y[i] = x[i] - y[i]; }
}

template <typename T>
inline void Add(T *x, T *y, int N) {
    #pragma omp parallel for simd
    for(int i=0; i<N; i++) { y[i] = x[i] + y[i]; }
}

template <typename T>
inline T Dot(T *x, T *y, int N) {
    T res = 0;
    #pragma omp parallel for simd reduction(+: res)
    for(int i=0; i<N; i++) { 
        res += x[i] * y[i];
    }
    return res;
}

template <typename T>
inline T Nrm2(T *x, int N) {
    T res = 0;
    #pragma omp parallel for simd reduction(+: res)
    for(int i=0; i<N; i++) { res += x[i] * x[i]; }
    return std::sqrt(res);
}

template <typename T>
inline void HadProd(T *x, T *y, int N) {
    #pragma omp parallel for simd
    for(int i=0; i<N; i++) { y[i] *= x[i]; }
}

template <typename T>
inline void HadDiv(T *x, T *y, int N) {
    #pragma omp parallel for simd
    for(int i=0; i<N; i++) { y[i] /= x[i]; }
}

template <typename T>
inline T Ggen(T a, T b, T *c, T *s) {
    T r;
    r = std::sqrt(a*a + b*b);
    c[0] =  a / r;
    s[0] = -b / r;
    return r;
}

template <typename T>
inline void Grot(T c, T s, T *a, T *b) {
    T temp = a[0];
    a[0] = c * temp - s * b[0];
    b[0] = s * temp + c * b[0];
}

inline void AxpyFD(float a, float *x, double *y, int N) {
    #pragma omp parallel for simd
    for(int i=0; i<N; i++) { y[i] += (double)(a * x[i]); }
}

// ---- For int-gmres ---- //
namespace int32 {

template <typename T>
inline T lsqrt(T x) {
    T s, t;
    if(x <= 0) return 0;
    s = 1; t = x;
    while(s < t) {
        s <<= 1; t >>= 1;
    }
    do {
        t = s;
        s = (x / s + s) >> 1;
    } while (s < t);
    return t;
}
/*
template <int bit>
void Copy_I2D(int *x, double *y, int N) {
    const double fact_inv = 1.0/(double)((long)1 << bit);
    #pragma omp parallel for simd
    for(int i=0; i<N; i++) {
        y[i] = (double)x[i] * fact_inv;
    }
}

template <int bit>
void Copy_D2I(double *x, int *y, int N) {
    constexpr double fact = (double)((long)1 << bit);
    #pragma omp parallel for simd
    for(int i=0; i<N; i++) {
        y[i] = (int)(x[i] * fact);
    }
}
*/
template <int bit>
inline void Scal(long a, int *x, int N) {
    #pragma omp parallel for simd
    for(int i=0; i<N; i++) {
        x[i] = (int)(a * (long)x[i] >> bit);
    }
}

template <int bit>
inline void Scal_D(double a, int *x, double *y, int N) {
    constexpr double fact_inv = 1.0 / (double)((long)1 << bit);
    #pragma omp parallel for simd
    for(int i=0; i<N; i++) {
        y[i] = a * (double)x[i] * fact_inv;
    }
}

template <int bit>
inline void Axpy(long a, int *x, int *y, int N) {
    #pragma omp parallel for simd
    for(int i=0; i<N; i++) {
        y[i] += (int)(a * (long)x[i] >> bit);
    }
}

template <int bit>
inline void Axpy_D(double a, int *x, double *y, int N) {
    constexpr double fact_inv = 1.0 / (double)((long)1 << bit);
    #pragma omp parallel for simd
    for(int i=0; i<N; i++) {
        y[i] += a * (double)x[i] * fact_inv;
    }
}

template <int bit>
inline void Axpby(long a, int *x, long b, int *y, int N) {
    #pragma omp parallel for simd
    for(int i=0; i<N; i++) {
        y[i] = (int)((a * (long)x[i] + b * (long)y[i]) >> bit);
    }
}

template <int bit>
inline void Axpyz(long a, int *x, int *y, int *z, int N) {
    #pragma omp parallel for simd
    for(int i=0; i<N; i++) {
        z[i] = (int)(a * (long)x[i] >> bit) + y[i];
    }
}

template <int bit>
inline long Dot(int *x, int *y, int N) {
    long res = 0;
    #pragma omp parallel for simd reduction(+: res)
    for(int i=0; i<N; i++) { 
        res += (long)((long)x[i] * (long)y[i]);
    }
    return res >> bit;
}

template <int bit>
inline long Nrm2(int *x, int N) {
    long res = 0;
    #pragma omp parallel for simd reduction(+: res)
    for(int i=0; i<N; i++) {
        res += (long)((long)x[i] * (long)x[i]);
    }
    return lsqrt<long>(res);
}

template <int bit>
inline long Ggen(long a, long b, long *c, long *s)
{
    long r = a * a + b * b;
    r = lsqrt(r);
    long inv_r = ((long)1 << 62) / r >> (62-bit-bit);
    c[0] =  (a * inv_r) >> bit;
    s[0] = -(b * inv_r) >> bit;
    return r;
}

template <int bit>
inline void Grot(long c, long s, long *a, long *b)
{
    long temp = a[0];
    a[0] = (c * temp - s * b[0]) >> bit;
    b[0] = (s * temp + c * b[0]) >> bit;
}


/* With a bit parameter */
inline void Scal(long a, int *x, int N, int8_t bit) {
    #pragma omp parallel for simd
    for(int i=0; i<N; i++) {
        x[i] = (int)(a * (long)x[i] >> bit);
    }
}

inline void Scal_D(double a, int *x, double *y, int N, int8_t bit) {
    const double fact_inv = 1.0 / (double)((long)1 << bit);
    #pragma omp parallel for simd
    for(int i=0; i<N; i++) {
        y[i] = a * (double)x[i] * fact_inv;
    }
}

inline void Axpy(long a, int *x, int *y, int N, int8_t bit) {
    #pragma omp parallel for simd
    for(int i=0; i<N; i++) {
        y[i] += (int)(a * (long)x[i] >> bit);
    }
}

inline void Axpy_D(double a, int *x, double *y, int N, int8_t bit) {
    const double fact_inv = 1.0 / (double)((long)1 << bit);
    #pragma omp parallel for simd
    for(int i=0; i<N; i++) {
        y[i] += a * (double)x[i] * fact_inv;
    }
}

inline void Axpby(long a, int *x, long b, int *y, int N, int8_t bit) {
    #pragma omp parallel for simd
    for(int i=0; i<N; i++) {
        y[i] = (int)((a * (long)x[i] + b * (long)y[i]) >> bit);
    }
}

inline void Axpyz(long a, int *x, int *y, int *z, int N, int8_t bit) {
    #pragma omp parallel for simd
    for(int i=0; i<N; i++) {
        z[i] = (int)(a * (long)x[i] >> bit) + y[i];
    }
}

inline long Dot(int *x, int *y, int N, int8_t bit) {
    long res = 0;
    #pragma omp parallel for simd reduction(+: res)
    for(int i=0; i<N; i++) { 
        res += (long)((long)x[i] * (long)y[i]);
    }
    return res >> bit;
}

inline long Nrm2(int *x, int N, int8_t bit) {
    long res = 0;
    #pragma omp parallel for simd reduction(+: res)
    for(int i=0; i<N; i++) {
        res += (long)((long)x[i] * (long)x[i]);
    }
    return lsqrt<long>(res);
}

inline long Ggen(long a, long b, long *c, long *s, int8_t bit)
{
    long r = a * a + b * b;
    r = lsqrt(r);
    long inv_r = ((long)1 << 62) / r >> (62-bit-bit);
    c[0] =  (a * inv_r) >> bit;
    s[0] = -(b * inv_r) >> bit;
    return r;
}

inline void Grot(long c, long s, long *a, long *b, int8_t bit)
{
    long temp = a[0];
    a[0] = (c * temp - s * b[0]) >> bit;
    b[0] = (s * temp + c * b[0]) >> bit;
}

} // int

} // blas1

} // senk

#endif
