#ifndef SENKPP_MATRIX_TRISPMAT2_HPP
#define SENKPP_MATRIX_TRISPMAT2_HPP

#include "enums.hpp"
#include "utils/alloc.hpp"
#include "utils/array.hpp"
#include "helper/helper_matrix.hpp"

namespace senk {

namespace ltri {

template <typename T>
class SpMat2 {
public:
    virtual int GetN() = 0;
    virtual void SpTRSV(double *in, double *out) = 0;
    virtual void SpTRSV2(T *in, T *out) = 0;
    virtual ~SpMat2() {}
};

template <int bit, bool isConsistent>
class CSR_DI : public ltri::SpMat2<int> {
    int N, M; // N denots #rows; M denots #columns.
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    int32_t *val32 = nullptr;
public:
    CSR_DI(CSRMat *Mat) {
        Mat->CopyL(&val, &cind, &rptr);
        N = Mat->N; M = Mat->M;
        val32 = utils::SafeMalloc<int>(rptr[Mat->N]);
        utils::Convert_DI<int32_t, bit>(val, val32, rptr[Mat->N]);
        if constexpr(isConsistent) {
            utils::Convert_ID<int32_t, bit>(val32, val, rptr[Mat->N]);
        }
    }
    ~CSR_DI() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
        utils::SafeFree(&val32);
    }
    int GetN() { return N; }
    void SpTRSV(double *in, double *out) {
        for(int i=0; i<N; i++) {
            double t = in[i];
            for(int j=rptr[i]; j<rptr[i+1]; j++) {
                t -= val[j] * out[cind[j]];
            }
            out[i] = t;
        }
    }
    void SpTRSV2(int *in, int *out) {
        for(int i=0; i<N; i++) {
            long t = (long)in[i] << bit ;
            for(int j=rptr[i]; j<rptr[i+1]; j++) {
                t -= (long)val32[j] * (long)out[cind[j]];
            }
            out[i] = (int)(t >> bit);
        }
    }
};

template <int bit, bool isConsistent>
class BJCSR_DI : public ltri::SpMat2<int> {
    int N, M; // N denots #rows; M denots #columns.
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    int32_t *val32 = nullptr;
    int *bj_size, bj_num;
public:
    BJCSR_DI(CSRMat *Mat) {
        if(Mat->order != MMOrder::BJ) {
            printf("In ltri::BJCSR, the order of Mat must be BJ.\n");
            exit(1);
        }
        Mat->CopyL(&val, &cind, &rptr);
        N = Mat->N; M = Mat->M;
        bj_size = Mat->bj_size;
        bj_num  = Mat->bj_num;
        val32 = utils::SafeMalloc<int>(rptr[Mat->N]);
        utils::Convert_DI<int32_t, bit>(val, val32, rptr[Mat->N]);
        if constexpr(isConsistent) {
            utils::Convert_ID<int32_t, bit>(val32, val, rptr[Mat->N]);
        }
    }
    ~BJCSR_DI() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
        utils::SafeFree(&val32);
    }
    int GetN() { return N; }
    void SpTRSV(double *in, double *out) {
        #pragma omp parallel for
        for(int id=0; id<bj_num; id++) {
            int s = bj_size[id];
            int e = bj_size[id+1];
            for(int i=s; i<e; i++) {
                double temp = in[i];
                for(int j=rptr[i]; j<rptr[i+1]; j++) {
                    temp -= val[j] * out[cind[j]];
                }
                out[i] = temp;
            }
        }
    }
    void SpTRSV2(int *in, int *out) {
        #pragma omp parallel for
        for(int id=0; id<bj_num; id++) {
            int s = bj_size[id];
            int e = bj_size[id+1];
            for(int i=s; i<e; i++) {
                long t = (long)in[i] << bit;
                for(int j=rptr[i]; j<rptr[i+1]; j++) {
                    t -= (long)val[j] * (long)out[cind[j]];
                }
                out[i] = (int)(t >> bit);
            }
        }
    }
};

} // namespace ltri

namespace ldtri {

template <typename T>
class SpMat2 {
public:
    virtual int GetN() = 0;
    virtual void SpTRSV(double *in, double *out) = 0;
    virtual void SpTRSV2(T *in, T *out) = 0;
    virtual ~SpMat2() {}
};

template <int bit, bool isConsistent>
class CSR_DI : public ldtri::SpMat2<int> {
    int N, M; // N denots #rows; M denots #columns.
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    int32_t *val32 = nullptr;
public:
    CSR_DI(CSRMat *Mat) {
        helper::matrix::extractLDinv(
            Mat->val, Mat->cind, Mat->rptr,
            &val, &cind, &rptr, Mat->N);
        N = Mat->N; M = Mat->M;
        val32 = utils::SafeMalloc<int>(rptr[Mat->N]);
        utils::Convert_DI<int32_t, bit>(val, val32, rptr[Mat->N]);
        if constexpr(isConsistent) {
            utils::Convert_ID<int32_t, bit>(val32, val, rptr[Mat->N]);
        }
    }
    ~CSR_DI() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
        utils::SafeFree(&val32);
    }
    int GetN() { return N; }
    void SpTRSV(double *in, double *out) {
       for(int i=0; i<N; i++) {
            double t = in[i];
            int j;
            for(j=rptr[i]; j<rptr[i+1]-1; j++) {
                t -= val[j] * out[cind[j]];
            }
            out[i] = t * val[j];
        }
    }
    void SpTRSV2(int *in, int *out) {
        for(int i=0; i<N; i++) {
            long t = (long)in[i] << bit;
            int j;
            for(j=rptr[i]; j<rptr[i+1]-1; j++) {
                t -= (long)val32[j] * (long)out[cind[j]];
            }
            out[i] = (int)((t >> bit) * (long)val32[j] >> bit);
        }
        
    }
};

} // namespace ldtri

namespace dutri {

template <typename T>
class SpMat2 {
public:
    virtual int GetN() = 0;
    virtual void SpTRSV(double *in, double *out) = 0;
    virtual void SpTRSV2(T *in, T *out) = 0;
    virtual ~SpMat2() {}
};

template <int bit, bool isConsistent>
class CSR_DI : public dutri::SpMat2<int> {
    int N, M; // N denots #rows; M denots #columns.
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    int32_t *val32 = nullptr;
public:
    CSR_DI(CSRMat *Mat) {
        Mat->CopyDinvU(&val, &cind, &rptr);
        N = Mat->N; M = Mat->M;
        val32 = utils::SafeMalloc<int>(rptr[Mat->N]);
        utils::Convert_DI<int32_t, bit>(val, val32, rptr[Mat->N]);
        if constexpr(isConsistent) {
            utils::Convert_ID<int32_t, bit>(val32, val, rptr[Mat->N]);
        }
    }
    ~CSR_DI() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
        utils::SafeFree(&val32);
    }
    int GetN() { return N; }
    void SpTRSV(double *in, double *out) {
        for(int i=N-1; i>=0; i--) {
            double t = in[i];
            int j;
            for(j=rptr[i+1]-1; j>=rptr[i]+1; j--) {
                t -= val[j] * out[cind[j]];
            }
            out[i] = t * val[j];
        }
    }
    void SpTRSV2(int *in, int *out) {
        for(int i=N-1; i>=0; i--) {
            long t = (long)in[i] << bit;
            int j;
            for(j=rptr[i+1]-1; j>=rptr[i]+1; j--) {
                t -= (long)val32[j] * (long)out[cind[j]];
            }
            out[i] = (int)((t >> bit) * (long)val32[j] >> bit);
        }
    }
};

template <int bit, bool isConsistent>
class BJCSR_DI : public dutri::SpMat2<int> {
    int N, M; // N denots #rows; M denots #columns.
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    int32_t *val32 = nullptr;
    int *bj_size, bj_num;
public:
    BJCSR_DI(CSRMat *Mat) {
        if(Mat->order != MMOrder::BJ) {
            printf("In ltri::BJBCSR, the order of Mat must be BJ.\n");
            exit(1);
        }
        Mat->CopyDinvU(&val, &cind, &rptr);
        N = Mat->N; M = Mat->M;
        bj_size = Mat->bj_size;
        bj_num  = Mat->bj_num;
        val32 = utils::SafeMalloc<int>(rptr[Mat->N]);
        utils::Convert_DI<int32_t, bit>(val, val32, rptr[Mat->N]);
        if constexpr(isConsistent) {
            utils::Convert_ID<int32_t, bit>(val32, val, rptr[Mat->N]);
        }
    }
    ~BJCSR_DI() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
        utils::SafeFree(&val32);
    }
    int GetN() { return N; }
    void SpTRSV(double *in, double *out) {
        #pragma omp parallel for
        for(int id=0; id<bj_num; id++) {
            int s = bj_size[id];
            int e = bj_size[id+1];
            for(int i=e-1; i>=s; i--) {
                double temp = in[i];
                int j;
                for(j=rptr[i+1]-1; j>=rptr[i]+1; j--) {
                    temp -= val[j] * out[cind[j]];
                }
                out[i] = temp * val[j];
            }
        }
    }
    void SpTRSV2(int *in, int *out) {
        #pragma omp parallel for
        for(int id=0; id<bj_num; id++) {
            int s = bj_size[id];
            int e = bj_size[id+1];
            for(int i=e-1; i>=s; i--) {
                long t = (long)in[i] << bit;
                int j;
                for(j=rptr[i+1]-1; j>=rptr[i]+1; j--) {
                    t -= (long)val[j] * (long)out[cind[j]];
                }
                out[i] = (int)((t >> bit) * (long)val[j] >> bit);
            }
        }
    }
};

} // namespace dutri
/*
template <typename T>
class TriSpMat2 {
public:
    virtual int GetN() = 0;
    virtual void SpTRSV(double *in, double *out) = 0;
    virtual void SpTRSV2(T *in, T *out) = 0;
    virtual ~TriSpMat2() {}
};

template <int bit, bool isConsistent>
class TriLCSR_DI : public TriSpMat2<int> {
    int N, M; // N denots #rows; M denots #columns.
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    int32_t *val32 = nullptr;
public:
    TriLCSR_DI(CSRMat *Mat) {
        Mat->CopyL(&val, &cind, &rptr);
        N = Mat->N; M = Mat->M;
        val32 = utils::SafeMalloc<int>(rptr[Mat->N]);
        utils::Convert_DI<int32_t, bit>(val, val32, rptr[Mat->N]);
        if constexpr(isConsistent) {
            utils::Convert_ID<int32_t, bit>(val32, val, rptr[Mat->N]);
        }
    }
    ~TriLCSR_DI() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
        utils::SafeFree(&val32);
    }
    int GetN() { return N; }
    void SpTRSV(double *in, double *out) {
        for(int i=0; i<N; i++) {
            double t = in[i];
            for(int j=rptr[i]; j<rptr[i+1]; j++) {
                t -= val[j] * out[cind[j]];
            }
            out[i] = t;
        }
    }
    void SpTRSV2(int *in, int *out) {
        for(int i=0; i<N; i++) {
            long t = (long)in[i] << bit ;
            for(int j=rptr[i]; j<rptr[i+1]; j++) {
                t -= (long)val32[j] * (long)out[cind[j]];
            }
            out[i] = (int)(t >> bit);
        }
    }
};

template <int bit, bool isConsistent>
class TriLDinvCSR_DI : public TriSpMat2<int> {
    int N, M; // N denots #rows; M denots #columns.
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    int32_t *val32 = nullptr;
public:
    TriLDinvCSR_DI(CSRMat *Mat) {
        helper::matrix::extractLDinv(
            Mat->val, Mat->cind, Mat->rptr,
            &val, &cind, &rptr, Mat->N);
        N = Mat->N; M = Mat->M;
        val32 = utils::SafeMalloc<int>(rptr[Mat->N]);
        utils::Convert_DI<int32_t, bit>(val, val32, rptr[Mat->N]);
        if constexpr(isConsistent) {
            utils::Convert_ID<int32_t, bit>(val32, val, rptr[Mat->N]);
        }
    }
    ~TriLDinvCSR_DI() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
        utils::SafeFree(&val32);
    }
    int GetN() { return N; }
    void SpTRSV(double *in, double *out) {
       for(int i=0; i<N; i++) {
            double t = in[i];
            int j;
            for(j=rptr[i]; j<rptr[i+1]-1; j++) {
                t -= val[j] * out[cind[j]];
            }
            out[i] = t * val[j];
        }
    }
    void SpTRSV2(int *in, int *out) {
        for(int i=0; i<N; i++) {
            long t = (long)in[i] << bit;
            int j;
            for(j=rptr[i]; j<rptr[i+1]-1; j++) {
                t -= (long)val32[j] * (long)out[cind[j]];
            }
            out[i] = (int)((t >> bit) * (long)val32[j] >> bit);
        }
        
    }
};

template <int bit, bool isConsistent>
class TriDinvUCSR_DI : public TriSpMat2<int> {
    int N, M; // N denots #rows; M denots #columns.
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    int32_t *val32 = nullptr;
public:
    TriDinvUCSR_DI(CSRMat *Mat) {
        Mat->CopyDinvU(&val, &cind, &rptr);
        N = Mat->N; M = Mat->M;
        val32 = utils::SafeMalloc<int>(rptr[Mat->N]);
        utils::Convert_DI<int32_t, bit>(val, val32, rptr[Mat->N]);
        if constexpr(isConsistent) {
            utils::Convert_ID<int32_t, bit>(val32, val, rptr[Mat->N]);
        }
    }
    ~TriDinvUCSR_DI() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
        utils::SafeFree(&val32);
    }
    int GetN() { return N; }
    void SpTRSV(double *in, double *out) {
        for(int i=N-1; i>=0; i--) {
            double t = in[i];
            int j;
            for(j=rptr[i+1]-1; j>=rptr[i]+1; j--) {
                t -= val[j] * out[cind[j]];
            }
            out[i] = t * val[j];
        }
    }
    void SpTRSV2(int *in, int *out) {
        for(int i=N-1; i>=0; i--) {
            long t = (long)in[i] << bit;
            int j;
            for(j=rptr[i+1]-1; j>=rptr[i]+1; j--) {
                t -= (long)val32[j] * (long)out[cind[j]];
            }
            out[i] = (int)((t >> bit) * (long)val32[j] >> bit);
        }
    }
};

template <typename T>
class TriBJSpMat2 {
public:
    virtual int GetN() = 0;
    virtual void SpTRSV(double *in, double *out) = 0;
    virtual void SpTRSV2(T *in, T *out) = 0;
    virtual ~TriBJSpMat2() {}
};

template <int bit, bool isConsistent>
class TriLBJCSR_DI : public TriBJSpMat2<int> {
    int N, M; // N denots #rows; M denots #columns.
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    int32_t *val32 = nullptr;
    int *bj_size, bj_num;
public:
    TriLBJCSR_DI(CSRMat *Mat, int *_bj_size, int _bj_num) {
        Mat->CopyL(&val, &cind, &rptr);
        N = Mat->N; M = Mat->M;
        bj_size = _bj_size;
        bj_num  = _bj_num;
        val32 = utils::SafeMalloc<int>(rptr[Mat->N]);
        utils::Convert_DI<int32_t, bit>(val, val32, rptr[Mat->N]);
        if constexpr(isConsistent) {
            utils::Convert_ID<int32_t, bit>(val32, val, rptr[Mat->N]);
        }
    }
    ~TriLBJCSR_DI() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
        utils::SafeFree(&val32);
    }
    int GetN() { return N; }
    void SpTRSV(double *in, double *out) {
        #pragma omp parallel for
        for(int id=0; id<bj_num; id++) {
            int s = bj_size[id];
            int e = bj_size[id+1];
            for(int i=s; i<e; i++) {
                double temp = in[i];
                for(int j=rptr[i]; j<rptr[i+1]; j++) {
                    temp -= val[j] * out[cind[j]];
                }
                out[i] = temp;
            }
        }
    }
    void SpTRSV2(int *in, int *out) {
        #pragma omp parallel for
        for(int id=0; id<bj_num; id++) {
            int s = bj_size[id];
            int e = bj_size[id+1];
            for(int i=s; i<e; i++) {
                long t = (long)in[i] << bit;
                for(int j=rptr[i]; j<rptr[i+1]; j++) {
                    t -= (long)val[j] * (long)out[cind[j]];
                }
                out[i] = (int)(t >> bit);
            }
        }
    }
};

template <int bit, bool isConsistent>
class TriDinvUBJCSR_DI : public TriBJSpMat2<int> {
    int N, M; // N denots #rows; M denots #columns.
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    int *bj_size, bj_num;
    int32_t *val32 = nullptr;
public:
    TriDinvUBJCSR_DI(CSRMat *Mat, int *_bj_size, int _bj_num) {
        Mat->CopyDinvU(&val, &cind, &rptr);
        N = Mat->N; M = Mat->M;
        bj_size = _bj_size;
        bj_num  = _bj_num;
        val32 = utils::SafeMalloc<int>(rptr[Mat->N]);
        utils::Convert_DI<int32_t, bit>(val, val32, rptr[Mat->N]);
        if constexpr(isConsistent) {
            utils::Convert_ID<int32_t, bit>(val32, val, rptr[Mat->N]);
        }
    }
    ~TriDinvUBJCSR_DI() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
        utils::SafeFree(&val32);
    }
    int GetN() { return N; }
    void SpTRSV(double *in, double *out) {
        #pragma omp parallel for
        for(int id=0; id<bj_num; id++) {
            int s = bj_size[id];
            int e = bj_size[id+1];
            for(int i=e-1; i>=s; i--) {
                double temp = in[i];
                int j;
                for(j=rptr[i+1]-1; j>=rptr[i]+1; j--) {
                    temp -= val[j] * out[cind[j]];
                }
                out[i] = temp * val[j];
            }
        }
    }
    void SpTRSV2(int *in, int *out) {
        #pragma omp parallel for
        for(int id=0; id<bj_num; id++) {
            int s = bj_size[id];
            int e = bj_size[id+1];
            for(int i=e-1; i>=s; i--) {
                long t = (long)in[i] << bit;
                int j;
                for(j=rptr[i+1]-1; j>=rptr[i]+1; j--) {
                    t -= (long)val[j] * (long)out[cind[j]];
                }
                out[i] = (int)((t >> bit) * (long)val[j] >> bit);
            }
        }
    }
};
*/
} // senk

#endif
