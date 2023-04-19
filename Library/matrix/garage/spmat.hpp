#ifndef SENKPP_MATRIX_SPMAT_HPP
#define SENKPP_MATRIX_SPMAT_HPP

#include "concepts.hpp"
#include "matrix/csrmat.hpp"
#include "helper/helper_matrix.hpp"

namespace senk {

template <Floating T>
class SpMat {
public:
    int N, M; // N denots #rows; M denots #columns.
    virtual void SpMV(T *in, T *out) = 0;
    virtual ~SpMat() {}
};

template <Floating T>
class CSR : public SpMat<T> {
    T *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
public:
    CSR(CSRMat *A) {
        A->Copy<T>(&val, &cind, &rptr);
        SpMat<T>::N = A->N;
        SpMat<T>::M = A->M;
    }
    ~CSR() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
    }
    void SpMV(T *in, T *out) {
        #pragma omp parallel for
        for(int i=0; i<SpMat<T>::N; i++) {
            T temp = 0;
            for(int j=rptr[i]; j<rptr[i+1]; j++) {
                temp += val[j] * in[cind[j]];
            }
            out[i] = temp;
        }
    }
};

template <Floating T, int C>
class SELL : public SpMat<T> {
    T *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
public:
    SELL(CSRMat *A) {
        A->CopyAsSell<T>(&val, &cind, &rptr, C);
        SpMat<T>::N = A->N;
        SpMat<T>::M = A->M;
    }
    ~SELL() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
    }
    void SpMV(T *in, T *out) {
        int N_C = SpMat<T>::N/C;
        #pragma omp parallel for
        for(int i=0; i<N_C; i++) {
            int off = rptr[i] * C;
            for(int k=0; k<C; k++) { out[i*C+k] = 0; }
            for(int j=0; j<rptr[i+1]-rptr[i]; j++) {
                for(int k=0; k<C; k++) {
                    out[i*C+k] += val[off+j*C+k] * in[cind[off+j*C+k]];
                }
            }
        }
    }
};

template <Floating T, int Bnl, int Bnw>
class BCSR : public SpMat<T> {
    T *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
public:
    BCSR(CSRMat *A) {
        A->CopyAsBCSR<T>(&val, &cind, &rptr, Bnl, Bnw);
        SpMat<T>::N = A->N;
        SpMat<T>::M = A->M;
    }
    ~BCSR() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
    }
    void SpMV(T *in, T *out) {
        int b_size = Bnl * Bnw;
        #pragma omp parallel for
        for(int i=0; i<SpMat<T>::N; i+=Bnl) {
            int bidx = i / Bnl;
            T temp[Bnl] = {0};
            #pragma omp simd simdlen(Bnl)
            for(int j=0; j<Bnl; j++) { temp[j] = 0; }
            for(int j=rptr[bidx]; j<rptr[bidx+1]; j++) {
                int x_ind = cind[j] * Bnw;
                int off = j*b_size;
                #pragma omp simd simdlen(Bnl)
                for(int k=0; k<Bnl; k++) {
                    temp[k] += val[off+k] * in[x_ind];
                }
                if constexpr(Bnw == 2 || Bnw == 4 || Bnw == 8) {
                    #pragma omp simd simdlen(Bnl)
                    for(int k=0; k<Bnl; k++) {
                        temp[k] += val[off+Bnl+k] * in[x_ind+1];
                    } 
                }
                if constexpr(Bnw == 4 || Bnw == 8) {
                    #pragma omp simd simdlen(Bnl)
                    for(int k=0; k<Bnl; k++) {
                        temp[k] += val[off+Bnl*2+k] * in[x_ind+2];
                    } 
                    #pragma omp simd simdlen(Bnl)
                    for(int k=0; k<Bnl; k++) {
                        temp[k] += val[off+Bnl*3+k] * in[x_ind+3];
                    } 
                }
                if constexpr(Bnw == 8) {
                    #pragma omp simd simdlen(Bnl)
                    for(int k=0; k<Bnl; k++) {
                        temp[k] += val[off+Bnl*4+k] * in[x_ind+2];
                    } 
                    #pragma omp simd simdlen(Bnl)
                    for(int k=0; k<Bnl; k++) {
                        temp[k] += val[off+Bnl*5+k] * in[x_ind+3];
                    }
                    for(int k=0; k<Bnl; k++) {
                        temp[k] += val[off+Bnl*6+k] * in[x_ind+2];
                    }
                    #pragma omp simd simdlen(Bnl)
                    for(int k=0; k<Bnl; k++) {
                        temp[k] += val[off+Bnl*7+k] * in[x_ind+3];
                    }
                }
            }
            #pragma omp simd simdlen(Bnl)
            for(int j=0; j<Bnl; j++) { out[i+j] = temp[j]; }
        }
    }
};

// --------------- //

template <typename T1, typename T2>
class SpMat2 {
public:
    int N, M; // N denots #rows; M denots #columns.
    virtual void SpMV(T1 *in, T1 *out) = 0;
    virtual void SpMV2(T2 *in, T2 *out) = 0;
    virtual ~SpMat2() {}
};

template <int bit>
class CSR_DI : public SpMat2<double, int> {
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    int32_t *val32 = nullptr;
    int16_t *val16 = nullptr;
    int8_t *val8 = nullptr;
public:
    CSR_DI(CSRMat *A) {
        if constexpr(bit > 30) {
            std::cout << "bit must be less than 30.\n";
            exit(0);
        }
        A->Copy<double>(&val, &cind, &rptr);
        SpMat2<double, int>::N = A->N;
        SpMat2<double, int>::M = A->M;
        if constexpr(bit > 14) {
            val32 = utils::SafeMalloc<int32_t>(A->rptr[A->N]);
            utils::Convert_DI<int32_t, bit>(val, val32, A->rptr[A->N]);
        }else if constexpr(bit > 6) {
            val16 = utils::SafeMalloc<int16_t>(A->rptr[A->N]);
            utils::Convert_DI<int16_t, bit>(val, val16, A->rptr[A->N]);
        }else {
            val8 = utils::SafeMalloc<int8_t>(A->rptr[A->N]);
            utils::Convert_DI<int8_t, bit>(val, val8, A->rptr[A->N]);
        }
    }
    ~CSR_DI() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
        utils::SafeFree(&val8);
        utils::SafeFree(&val16);
        utils::SafeFree(&val32);
    }
    void SpMV(double *in, double *out) {
        #pragma omp parallel for
        for(int i=0; i<N; i++) {
            double temp = 0;
            for(int j=rptr[i]; j<rptr[i+1]; j++) {
                temp += val[j] * in[cind[j]];
            }
            out[i] = temp;
        }
    }
    void SpMV2(int *in, int *out) {
        #pragma omp parallel for
        for(int i=0; i<N; i++) {
            long temp = 0;
            for(int j=rptr[i]; j<rptr[i+1]; j++) {
                if constexpr(bit > 14) {
                    temp += (long)val32[j] * (long)in[cind[j]];
                }else if constexpr(bit > 6) {
                    temp += (long)val16[j] * (long)in[cind[j]];
                }else {
                    temp += (long)val8[j] * (long)in[cind[j]];
                }
            }
            out[i] = (int)(temp >> bit);
        }
    }
};

template <int C, int bit>
class SELL_DI : public SpMat2<double, int> {
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    int32_t *val32 = nullptr;
    int16_t *val16 = nullptr;
    int8_t *val8 = nullptr;
public:
    SELL_DI(CSRMat *A) {
        if(bit > 30) {
            std::cout << "bit must be less than 30.\n";
            exit(0);
        }
        if(A->N % C != 0) {
            std::cerr << "N must be a multiple of "<< C <<"." << std::endl;
            exit(1);
        }
        A->CopyAsSell(&val, &cind, &rptr, C);
        SpMat2<double, int>::N = A->N;
        SpMat2<double, int>::M = A->M;
        const int nnz = rptr[A->N/C]*C;
        if constexpr(bit > 14) {
            val32 = utils::SafeMalloc<int32_t>(nnz);
            utils::Convert_DI<int32_t, bit>(val, val32, nnz);
        }else if constexpr(bit > 6) {
            val16 = utils::SafeMalloc<int16_t>(nnz);
            utils::Convert_DI<int16_t, bit>(val, val16, nnz);
        }else {
            val8 = utils::SafeMalloc<int8_t>(nnz);
            utils::Convert_DI<int8_t, bit>(val, val8, nnz);
        }
    }
    ~SELL_DI() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
        utils::SafeFree(&val8);
        utils::SafeFree(&val16);
        utils::SafeFree(&val32);
    }
    void SpMV(double *in, double *out) {
        int N_C = SpMat2<double, int>::N/C;
        #pragma omp parallel for
        for(int i=0; i<N_C; i++) {
            int off = rptr[i] * C;
            for(int k=0; k<C; k++) { out[i*C+k] = 0; }
            for(int j=0; j<rptr[i+1]-rptr[i]; j++) {
                for(int k=0; k<C; k++) {
                    out[i*C+k] += val[off+j*C+k] * in[cind[off+j*C+k]];
                }
            }
        }
    }
    void SpMV2(int *in, int *out) {
        int N_C = SpMat2<double, int>::N/C;
        #pragma omp parallel for
        for(int i=0; i<N_C; i++) {
            int off = rptr[i] * C;
            long temp[C] = {0};
            for(int j=0; j<rptr[i+1]-rptr[i]; j++) {
                for(int k=0; k<C; k++) {
                    if constexpr(bit > 14) {
                        temp[k] += (long)val32[off+j*C+k] * (long)in[cind[off+j*C+k]];
                    }else if constexpr(bit > 6) {
                        temp[k] += (long)val16[off+j*C+k] * (long)in[cind[off+j*C+k]];
                    }else {
                        temp[k] += (long)val8[off+j*C+k] * (long)in[cind[off+j*C+k]];
                    }
                }
            }
            for(int k=0; k<C; k++) { out[i*C+k] = (int)(temp[k] >> bit); }
        }
    }
};

/*
template <Floating T=double>
class SELL_32 : public SpMat<T> {
    T *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
public:
    SELL_32(CSRMat *A) {
        helper::matrix::csr_to_sell32(
            A->val, A->cind, A->rptr,
            &val, &cind, &rptr, A->N);
        SpMat<T>::N = A->N; SpMat<T>::M = A->M;
    }
    ~SELL_32() {
        utils::SafeFree<T>(&val);
        utils::SafeFree<int>(&cind);
        utils::SafeFree<int>(&rptr);
    }
    void SpMV(T *in, T *out) {
        int N_32 = SpMat<T>::N/32;
        #pragma omp parallel for
        for(int i=0; i<N_32; i++) {
            int off = rptr[i] * 32;
            for(int k=0; k<32; k++) { out[i*32+k] = 0; }
            for(int j=0; j<rptr[i+1]-rptr[i]; j++) {
                for(int k=0; k<32; k++) {
                    out[i*32+k] += val[off+j*32+k] * in[cind[off+j*32+k]];
                }
            }
        }
    }
};
*/

} // senk


#endif
