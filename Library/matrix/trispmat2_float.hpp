#ifndef SENKPP_MATRIX_TRISPMAT2_FLOAT_HPP
#define SENKPP_MATRIX_TRISPMAT2_FLOAT_HPP

#include "enums.hpp"
#include "utils/alloc.hpp"
#include "utils/array.hpp"
#include "helper/helper_matrix.hpp"

#include "matrix/trispmat2.hpp"

namespace senk {

namespace ltri {

template <bool isConsistent>
class CSR_DF : public ltri::SpMat2<float> {
    int N, M; // N denots #rows; M denots #columns.
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    float *fval = nullptr;
public:
    CSR_DF(CSRMat *Mat) {
        Mat->CopyL(&val, &cind, &rptr);
        N = Mat->N; M = Mat->M;
        fval = utils::SafeMalloc<float>(rptr[Mat->N]);
        utils::Convert<double, float>(val, fval, rptr[Mat->N]);
        if constexpr(isConsistent) {
            utils::Convert<float, double>(fval, val, rptr[Mat->N]);
        }
    }
    ~CSR_DF() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
        utils::SafeFree(&fval);
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
    void SpTRSV2(float *in, float *out) {
        for(int i=0; i<N; i++) {
            double t = in[i];
            for(int j=rptr[i]; j<rptr[i+1]; j++) {
                t -= val[j] * out[cind[j]];
            }
            out[i] = t;
        }
    }
};

template <bool isConsistent>
class MCCSR_DF : public ltri::SpMat2<float> {
    int N, M; // N denots #rows; M denots #columns.
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    float *fval = nullptr;
    int *c_size, c_num;
public:
    MCCSR_DF(CSRMat *Mat)
    {
        if(Mat->order != MMOrder::MC) {
            printf("In ltri::MCCSR, the order of Mat must be MC.\n");
            exit(1);
        }
        Mat->CopyL(&val, &cind, &rptr);
        N = Mat->N; M = Mat->M;
        c_size = Mat->c_size;
        c_num  = Mat->c_num;
        fval = utils::SafeMalloc<float>(rptr[Mat->N]);
        utils::Convert<double, float>(val, fval, rptr[Mat->N]);
        if constexpr(isConsistent) {
            utils::Convert<float, double>(fval, val, rptr[Mat->N]);
        }
    }
    ~MCCSR_DF() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
        utils::SafeFree(&fval);
    }
    int GetN() { return N; }
    void SpTRSV(double *in, double *out) {
        #pragma omp parallel
        {
            for(int id=0; id<c_num; id++) {
                int s = c_size[id];
                int e = c_size[id+1];
                #pragma omp for
                for(int i=s; i<e; i++) {
                    double t = in[i];
                    for(int j=rptr[i]; j<rptr[i+1]; j++) {
                        t -= val[j] * out[cind[j]];
                    }
                    out[i] = t;
                }
            }
        }
    }
    void SpTRSV2(float *in, float *out) {
        #pragma omp parallel
        {
            for(int id=0; id<c_num; id++) {
                int s = c_size[id];
                int e = c_size[id+1];
                #pragma omp for
                for(int i=s; i<e; i++) {
                    float t = in[i];
                    for(int j=rptr[i]; j<rptr[i+1]; j++) {
                        t -= val[j] * out[cind[j]];
                    }
                    out[i] = t;
                }
            }
        }
    }
};

} // namespace ltri

namespace ldtri {

template <bool isConsistent>
class CSR_DF : public ldtri::SpMat2<float> {
    int N, M; // N denots #rows; M denots #columns.
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    float *fval = nullptr;
public:
    CSR_DF(CSRMat *Mat) {
        helper::matrix::extractLDinv(
            Mat->val, Mat->cind, Mat->rptr,
            &val, &cind, &rptr, Mat->N);
        N = Mat->N; M = Mat->M;
        fval = utils::SafeMalloc<float>(rptr[Mat->N]);
        utils::Convert<double, float>(val, fval, rptr[Mat->N]);
        if constexpr(isConsistent) {
            utils::Convert<float, double>(fval, val, rptr[Mat->N]);
        }
    }
    ~CSR_DF() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
        utils::SafeFree(&fval);
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
    void SpTRSV2(float *in, float *out) {
        for(int i=0; i<N; i++) {
            float t = in[i];
            int j;
            for(j=rptr[i]; j<rptr[i+1]-1; j++) {
                t -= val[j] * out[cind[j]];
            }
            out[i] = t * val[j];
        }
    }
};

} // namespace ldtri

namespace dutri {

template <bool isConsistent>
class CSR_DF : public dutri::SpMat2<float> {
    int N, M; // N denots #rows; M denots #columns.
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    float *fval = nullptr;
public:
    CSR_DF(CSRMat *Mat) {
        Mat->CopyDinvU(&val, &cind, &rptr);
        N = Mat->N; M = Mat->M;
        fval = utils::SafeMalloc<float>(rptr[Mat->N]);
        utils::Convert<double, float>(val, fval, rptr[Mat->N]);
        if constexpr(isConsistent) {
            utils::Convert<float, double>(fval, val, rptr[Mat->N]);
        }
    }
    ~CSR_DF() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
        utils::SafeFree(&fval);
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
    void SpTRSV2(float *in, float *out) {
        for(int i=N-1; i>=0; i--) {
            float t = in[i];
            int j;
            for(j=rptr[i+1]-1; j>=rptr[i]+1; j--) {
                t -= val[j] * out[cind[j]];
            }
            out[i] = t * val[j];
        }
    }
};

template <bool isConsistent>
class MCCSR_DF : public dutri::SpMat2<float> {
    int N, M; // N denots #rows; M denots #columns.
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    float *fval = nullptr;
    int *c_size, c_num;
public:
    MCCSR_DF(CSRMat *Mat)
    {
        if(Mat->order != MMOrder::MC) {
            printf("In dutri::MCCSR, the order of Mat must be MC.\n");
            exit(1);
        }
        Mat->CopyDinvU(&val, &cind, &rptr);
        N = Mat->N; M = Mat->M;
        c_size = Mat->c_size;
        c_num  = Mat->c_num;
        fval = utils::SafeMalloc<float>(rptr[Mat->N]);
        utils::Convert<double, float>(val, fval, rptr[Mat->N]);
        if constexpr(isConsistent) {
            utils::Convert<float, double>(fval, val, rptr[Mat->N]);
        }
    }
    ~MCCSR_DF() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
        utils::SafeFree(&fval);
    }
    int GetN() { return N; }
    void SpTRSV(double *in, double *out) {
        #pragma omp parallel
        {
            for(int id=c_num-1; id>=0; id--) {
                int s = c_size[id];
                int e = c_size[id+1];
                #pragma omp for
                for(int i=e-1; i>=s; i--) {
                    double t = in[i];
                    int j;
                    for(j=rptr[i+1]-1; j>=rptr[i]+1; j--) {
                        t -= val[j] * out[cind[j]];
                    }
                    out[i] = t * val[j];
                }
            }
        }
    }
    void SpTRSV2(float *in, float *out) {
        #pragma omp parallel
        {
            for(int id=c_num-1; id>=0; id--) {
                int s = c_size[id];
                int e = c_size[id+1];
                #pragma omp for
                for(int i=e-1; i>=s; i--) {
                    float t = in[i];
                    int j;
                    for(j=rptr[i+1]-1; j>=rptr[i]+1; j--) {
                        t -= val[j] * out[cind[j]];
                    }
                    out[i] = t * val[j];
                }
            }
        }
    }
};

} // namespace dutri

} // senk

#endif
