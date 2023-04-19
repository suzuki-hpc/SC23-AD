#ifndef SENKPP_MATRIX_SPMAT_HPP
#define SENKPP_MATRIX_SPMAT_HPP

#include "enums.hpp"
#include "matrix/csrmat.hpp"
#include "helper/helper_matrix.hpp"

#define SPMAT_FLEX_PARAM 30
#define SPMAT_FLEX_PRINT false

namespace senk {

template <typename T>
class SpMat {
public:
    virtual int GetN() = 0;
    virtual void SpMV(T *in, T *out) = 0;
    virtual ~SpMat() {}
};

template <typename T>
class CSR : public SpMat<T> {
    int N, M; // N denots #rows; M denots #columns.
    T *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
public:
    CSR(CSRMat *Mat) {
        double *tval;    
        Mat->Copy(&tval, &cind, &rptr);
        N = Mat->N; M = Mat->M;
        int nnz = rptr[N];
        val = utils::SafeMalloc<T>(nnz);
        utils::Convert<double, T>(tval, val, nnz);
        free(tval);
    }
    ~CSR() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
    }
    int GetN() { return N; }
    void SpMV(T *in, T *out) {
        #pragma omp parallel for
        for(int i=0; i<N; i++) {
            T temp = 0;
            for(int j=rptr[i]; j<rptr[i+1]; j++) {
                temp += val[j] * in[cind[j]];
            }
            out[i] = temp;
        }
    }
};

template <int bit>
class CSR<Fixed<bit>> : public SpMat<int> {
    int N, M; // N denots #rows; M denots #columns.
    int *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
public:
    CSR(CSRMat *Mat) {
        double *tval;
        Mat->Copy(&tval, &cind, &rptr);
        N = Mat->N; M = Mat->M;
        int nnz = rptr[N];
        val = utils::SafeMalloc<int>(nnz);
        utils::Convert_DI<int, bit>(tval, val, nnz);
        free(tval);
    }
    ~CSR() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
    }
    int GetN() { return N; }
    void SpMV(int *in, int *out) {
        #pragma omp parallel for
        for(int i=0; i<N; i++) {
            long temp = 0;
            for(int j=rptr[i]; j<rptr[i+1]; j++) {
                temp += (long)val[j] * (long)in[cind[j]];
            }
            out[i] = (int)(temp >> bit);
        }
    }
};

class CSRflex : public SpMat<int> {
    int N, M; // N denots #rows; M denots #columns.
    int *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    int bit = 0;
public:
    CSRflex(CSRMat *Mat) {
        double *tval;
        Mat->Copy(&tval, &cind, &rptr);
        N = Mat->N; M = Mat->M;
        int nnz = rptr[N];
        val = utils::SafeMalloc<int>(nnz);
        double max = 0;
        for(int i=0; i<N; i++) {
            for(int j=rptr[i]; j<rptr[i+1]; j++) {            
                if(std::fabs(tval[j]) > max) max = std::fabs(tval[j]);
            }
        }
        bit = (max < 1)? 30 : 32 - ((int)std::ceil(std::log2(int(max)+1)) + 1);
        // while( bit < SPMAT_FLEX_PARAM && max * (1 << bit) < (1 << SPMAT_FLEX_PARAM) ) {
        //     bit++;
        // }
#if SPMAT_FLEX_PRINT
        printf("# spmat %d\n", bit);
#endif
        for(int i=0; i<N; i++) {
            for(int j=rptr[i]; j<rptr[i+1]; j++) {
                val[j] = (int)(tval[j] * (1 << bit));
            }
        }
        free(tval);
    }
    ~CSRflex() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
    }
    int GetN() { return N; }
    void SpMV(int *in, int *out) {
        #pragma omp parallel for
        for(int i=0; i<N; i++) {
            long temp = 0;
            for(int j=rptr[i]; j<rptr[i+1]; j++) {
                temp += (long)val[j] * (long)in[cind[j]];
            }
            out[i] = (int)(temp >> bit);
        }
    }
};

template <int bit>
class CSRi8 : public SpMat<int> {
    int N, M; // N denots #rows; M denots #columns.
    int8_t *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
public:
    CSRi8(CSRMat *Mat) {
        double *tval;
        Mat->Copy(&tval, &cind, &rptr);
        N = Mat->N; M = Mat->M;
        int nnz = rptr[N];
        val = utils::SafeMalloc<int8_t>(nnz);
        utils::Convert_DI<int8_t, bit>(tval, val, nnz);
        free(tval);
    }
    ~CSRi8() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
    }
    int GetN() { return N; }
    void SpMV(int *in, int *out) {
        #pragma omp parallel for
        for(int i=0; i<N; i++) {
            long temp = 0;
            for(int j=rptr[i]; j<rptr[i+1]; j++) {
                temp += (long)val[j] * (long)in[cind[j]];
            }
            out[i] = (int)(temp >> bit);
        }
    }
};

template <int bit>
class CSRi16 : public SpMat<int> {
    int N, M; // N denots #rows; M denots #columns.
    int16_t *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
public:
    CSRi16(CSRMat *Mat) {
        double *tval;
        Mat->Copy(&tval, &cind, &rptr);
        N = Mat->N; M = Mat->M;
        int nnz = rptr[N];
        val = utils::SafeMalloc<int16_t>(nnz);
        utils::Convert_DI<int16_t, bit>(tval, val, nnz);
        free(tval);
    }
    ~CSRi16() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
    }
    int GetN() { return N; }
    void SpMV(int *in, int *out) {
        #pragma omp parallel for
        for(int i=0; i<N; i++) {
            long temp = 0;
            for(int j=rptr[i]; j<rptr[i+1]; j++) {
                temp += (long)val[j] * (long)in[cind[j]];
            }
            out[i] = (int)(temp >> bit);
        }
    }
};

template <typename T, int C>
class SELL : public SpMat<T> {
    int N, M; // N denots #rows; M denots #columns.
    T *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
public:
    SELL(CSRMat *Mat) {
        double *tval;
        Mat->CopyAsSell(&tval, &cind, &rptr, C);
        N = Mat->N; M = Mat->M;
        int nnz = rptr[N/C]*C;
        val = utils::SafeMalloc<T>(nnz);
        utils::Convert<double, T>(tval, val, nnz);
        free(tval);
    }
    ~SELL() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
    }
    int GetN() { return N; }
    void SpMV(T *in, T *out) {
        const int N_C = N/C;
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

template <int bit, int C>
class SELL<Fixed<bit>, C> : public SpMat<int> {
    int N, M; // N denots #rows; M denots #columns.
    int *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
public:
    SELL(CSRMat *Mat) {
        double *tval;
        Mat->CopyAsSell(&tval, &cind, &rptr, C);
        N = Mat->N; M = Mat->M;
        int nnz = rptr[N/C]*C;
        val = utils::SafeMalloc<int>(nnz);
        utils::Convert_DI<int, bit>(tval, val, nnz);
        free(tval);
    }
    ~SELL() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
    }
    int GetN() { return N; }
    void SpMV(int *in, int *out) {
        const int N_C = N/C;
        #pragma omp parallel for
        for(int i=0; i<N_C; i++) {
            int off = rptr[i] * C;
            long temp[C] = {0};
            for(int j=0; j<rptr[i+1]-rptr[i]; j++) {
                for(int k=0; k<C; k++) {
                    temp[k] += (long)val[off+j*C+k] * (long)in[cind[off+j*C+k]];
                }
            }
            for(int k=0; k<C; k++) { out[i*C+k] = (int)(temp[k] >> bit); }
        }
    }
};

template <typename T, int Bnl, int Bnw>
class BCSR : public SpMat<T> {
    int N, M; // N denots #rows; M denots #columns.
    T *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
public:
    BCSR(CSRMat *Mat) {
        double *tval;
        Mat->CopyAsBCSR(&tval, &cind, &rptr, Bnl, Bnw);
        N = Mat->N; M = Mat->M;
        int nnz = rptr[N/Bnl]*Bnl*Bnw;
        val = utils::SafeMalloc<T>(nnz);
        utils::Convert<double, T>(tval, val, nnz);
        free(tval);
    }
    ~BCSR() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
    }
    int GetN() { return N; }
    void SpMV(T *in, T *out) {
        const int b_size = Bnl * Bnw;
        #pragma omp parallel for
        for(int i=0; i<N; i+=Bnl) {
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
                    #pragma omp simd simdlen(Bnl)
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

template <int bit, int Bnl, int Bnw>
class BCSR<Fixed<bit>, Bnl, Bnw> : public SpMat<int> {
    int N, M; // N denots #rows; M denots #columns.
    int *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
public:
    BCSR(CSRMat *Mat) {
        double *tval;
        Mat->CopyAsBCSR(&tval, &cind, &rptr, Bnl, Bnw);
        N = Mat->N; M = Mat->M;
        int nnz = rptr[N/Bnl]*Bnl*Bnw;
        val = utils::SafeMalloc<int>(nnz);
        utils::Convert_DI<int, bit>(tval, val, nnz);
        free(tval);
    }
    ~BCSR() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
    }
    int GetN() { return N; }
    void SpMV(int *in, int *out) {
        const int b_size = Bnl * Bnw;
        #pragma omp parallel for
        for(int i=0; i<N; i+=Bnl) {
            int bidx = i / Bnl;
            long temp[Bnl] = {0};
            #pragma omp simd simdlen(Bnl)
            for(int j=0; j<Bnl; j++) { temp[j] = 0; }
            for(int j=rptr[bidx]; j<rptr[bidx+1]; j++) {
                int x_ind = cind[j] * Bnw;
                int off = j*b_size;
                #pragma omp simd simdlen(Bnl)
                for(int k=0; k<Bnl; k++) {
                    temp[k] += (long)val[off+k] * (long)in[x_ind];
                }
                if constexpr(Bnw == 2 || Bnw == 4 || Bnw == 8) {
                    #pragma omp simd simdlen(Bnl)
                    for(int k=0; k<Bnl; k++) {
                        temp[k] += (long)val[off+Bnl+k] * (long)in[x_ind+1];
                    } 
                }
                if constexpr(Bnw == 4 || Bnw == 8) {
                    #pragma omp simd simdlen(Bnl)
                    for(int k=0; k<Bnl; k++) {
                        temp[k] += (long)val[off+Bnl*2+k] * (long)in[x_ind+2];
                    } 
                    #pragma omp simd simdlen(Bnl)
                    for(int k=0; k<Bnl; k++) {
                        temp[k] += (long)val[off+Bnl*3+k] * (long)in[x_ind+3];
                    } 
                }
                if constexpr(Bnw == 8) {
                    #pragma omp simd simdlen(Bnl)
                    for(int k=0; k<Bnl; k++) {
                        temp[k] += (long)val[off+Bnl*4+k] * (long)in[x_ind+2];
                    } 
                    #pragma omp simd simdlen(Bnl)
                    for(int k=0; k<Bnl; k++) {
                        temp[k] += (long)val[off+Bnl*5+k] * (long)in[x_ind+3];
                    }
                    #pragma omp simd simdlen(Bnl)
                    for(int k=0; k<Bnl; k++) {
                        temp[k] += (long)val[off+Bnl*6+k] * (long)in[x_ind+2];
                    }
                    #pragma omp simd simdlen(Bnl)
                    for(int k=0; k<Bnl; k++) {
                        temp[k] += (long)val[off+Bnl*7+k] * (long)in[x_ind+3];
                    }
                }
            }
            #pragma omp simd simdlen(Bnl)
            for(int j=0; j<Bnl; j++) { out[i+j] = (int)(temp[j] >> bit); }
        }
    }
};

//** Double and float **//
/*
template <int C, bool isConsistent>
class DF_SELL : public SpMat<double>, public SpMat<float>{
    int N, M; // N denots #rows; M denots #columns.
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    float *val32 = nullptr;
public:
    DF_SELL(CSRMat *A) {
        A->CopyAsSell(&val, &cind, &rptr, C);
        N = A->N;
        M = A->M;
        const int nnz = rptr[A->N/C]*C;
        val32 = utils::SafeMalloc<float>(nnz);
        utils::Convert<double, float>(val, val32, nnz);
        if constexpr(isConsistent) {
            utils::Convert<float, double>(val32, val, nnz);
        }
    }
    ~DF_SELL() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
        utils::SafeFree(&val32);
    }
    int GetN() { return N; }
    void SpMV(double *in, double *out) {
        const int N_C = N/C;
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
    void SpMV(float *in, float *out) {
        const int N_C = N/C;
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
*/
} // senk

#endif
