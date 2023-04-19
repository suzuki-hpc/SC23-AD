#ifndef SENKPP_MATRIX_SPMAT2_HPP
#define SENKPP_MATRIX_SPMAT2_HPP

#include "matrix/csrmat.hpp"
#include "helper/helper_matrix.hpp"

namespace senk {

template <typename T>
class SpMat2 {
public:
    virtual int GetN() = 0;
    virtual void SpMV(double *in, double *out) = 0;
    virtual void SpMV2(T *in, T *out) = 0;
    virtual ~SpMat2() {}
};

template <int bit, bool isConsistent>
class CSR_DI : public SpMat2<int> {
    int N, M; // N denots #rows; M denots #columns.
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
        A->Copy(&val, &cind, &rptr);
        N = A->N;
        M = A->M;
        int nnz = A->rptr[A->N];
        if constexpr(bit > 14) {
            val32 = utils::SafeMalloc<int32_t>(nnz);
            utils::Convert_DI<int32_t, bit>(val, val32, nnz);
            if constexpr(isConsistent) {
                utils::Convert_ID<int32_t, bit>(val32, val, nnz);
            }
        }else if constexpr(bit > 6) {
            val16 = utils::SafeMalloc<int16_t>(nnz);
            utils::Convert_DI<int16_t, bit>(val, val16, nnz);
            if constexpr(isConsistent) {
                utils::Convert_ID<int16_t, bit>(val16, val, nnz);
            }
        }else {
            val8 = utils::SafeMalloc<int8_t>(nnz);
            utils::Convert_DI<int8_t, bit>(val, val8, nnz);
            if constexpr(isConsistent) {
                utils::Convert_ID<int8_t, bit>(val8, val, nnz);
            }
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
    int GetN() { return N; }
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

template <int C, int bit, bool isConsistent>
class SELL_DI : public SpMat2<int> {
    int N, M; // N denots #rows; M denots #columns.
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
        A->CopyAsSell(&val, &cind, &rptr, C);
        N = A->N;
        M = A->M;
        const int nnz = rptr[A->N/C]*C;
        if constexpr(bit > 14) {
            val32 = utils::SafeMalloc<int32_t>(nnz);
            utils::Convert_DI<int32_t, bit>(val, val32, nnz);
            if constexpr(isConsistent) {
                utils::Convert_ID<int32_t, bit>(val32, val, nnz);
            }
        }else if constexpr(bit > 6) {
            val16 = utils::SafeMalloc<int16_t>(nnz);
            utils::Convert_DI<int16_t, bit>(val, val16, nnz);
            if constexpr(isConsistent) {
                utils::Convert_ID<int16_t, bit>(val16, val, nnz);
            }
        }else {
            val8 = utils::SafeMalloc<int8_t>(nnz);
            utils::Convert_DI<int8_t, bit>(val, val8, nnz);
            if constexpr(isConsistent) {
                utils::Convert_ID<int8_t, bit>(val8, val, nnz);
            }
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
    void SpMV2(int *in, int *out) {
        const int N_C = N/C;
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

/** Float **/
template <bool isConsistent>
class CSR_DF : public SpMat2<float> {
    int N, M; // N denots #rows; M denots #columns.
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    float *val32 = nullptr;
public:
    CSR_DF(CSRMat *A) {        
        A->Copy(&val, &cind, &rptr);
        N = A->N;
        M = A->M;
        int nnz = A->rptr[A->N];
        val32 = utils::SafeMalloc<float>(nnz);
        utils::Convert<double, float>(val, val32, nnz);
        if constexpr(isConsistent) {
            utils::Convert<float, double>(val32, val, nnz);
        }
    }
    ~CSR_DF() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
        utils::SafeFree(&val32);
    }
    int GetN() { return N; }
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
    void SpMV2(float *in, float *out) {
        #pragma omp parallel for
        for(int i=0; i<N; i++) {
            float temp = 0;
            for(int j=rptr[i]; j<rptr[i+1]; j++) {
                temp += val[j] * in[cind[j]];
            }
            out[i] = temp;
        }
    }
};

template <int C, bool isConsistent>
class SELL_DF : public SpMat2<float> {
    int N, M; // N denots #rows; M denots #columns.
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    float *val32 = nullptr;
public:
    SELL_DF(CSRMat *A) {
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
    ~SELL_DF() {
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
    void SpMV2(float *in, float *out) {
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

} // senk


#endif
