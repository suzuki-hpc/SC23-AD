#ifndef SENKPP_MATRIX_TRISPMAT_HPP
#define SENKPP_MATRIX_TRISPMAT_HPP

#include "enums.hpp"
#include "utils/alloc.hpp"
#include "helper/helper_matrix.hpp"

namespace senk {

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

/** With namespace **/

namespace ltri {

class SpMat {
public:
    virtual int GetN() = 0;
    virtual void SpTRSV(double *in, double *out) = 0;
    virtual ~SpMat() {}
};

class CSR : public ltri::SpMat {
    int N, M; // N denots #rows; M denots #columns.
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
public:
    CSR(CSRMat *Mat) {
        Mat->CopyL(&val, &cind, &rptr);
        N = Mat->N; M = Mat->M;
    }
    ~CSR() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
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
};

template <int Iter>
class CSRJa : public ltri::SpMat {
    int N, M; // N denots #rows; M denots #columns.
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    double *temp = nullptr;
public:
    CSRJa(CSRMat *Mat) {
        Mat->CopyL(&val, &cind, &rptr);
        N = Mat->N; M = Mat->M;
        temp = utils::SafeMalloc<double>(N);
    }
    ~CSRJa() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
        utils::SafeFree(&temp);
    }
    int GetN() { return N; }
    void SpTRSV(double *in, double *out) {
        #pragma omp parallel for
        for(int i=0; i<N; i++) { out[i] = in[i]; }
        if constexpr(Iter > 1) {
            #pragma omp parallel
            {
                for(int k=1; k<Iter; k++) {
                    #pragma omp for
                    for(int i=0; i<N; i++) {
                        double t = in[i];
                        for(int j=rptr[i]; j<rptr[i+1]; j++) {
                            t -= val[j] * out[cind[j]];
                        }
                        temp[i] = t;
                    }
                    #pragma omp for
                    for(int i=0; i<N; i++) { out[i] = temp[i]; }
                }
            }
        }
    }
};

template <int Iter>
class CSRAsync : public ltri::SpMat {
    int N, M; // N denots #rows; M denots #columns.
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    double *temp = nullptr;
public:
    CSRAsync(CSRMat *Mat) {
        Mat->CopyL(&val, &cind, &rptr);
        N = Mat->N; M = Mat->M;
        temp = utils::SafeMalloc<double>(N);
    }
    ~CSRAsync() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
        utils::SafeFree(&temp);
    }
    int GetN() { return N; }
    void SpTRSV(double *in, double *out) {
        #pragma omp parallel for
        for(int i=0; i<N; i++) { out[i] = in[i]; }
        if constexpr(Iter > 1) {
            // #pragma omp parallel
            // {
                for(int k=1; k<Iter; k++) {
                    for(int i=0; i<N; i+=32) {
                        // #pragma omp for
                        for(int l=0; l<32; l++) {
                            double t = in[i+l];
                            for(int j=rptr[i+l]; j<rptr[i+l+1]; j++) {
                                t -= val[j] * out[cind[j]];
                            }
                            temp[i+l] = t;
                        }
                        // #pragma omp for
                        for(int l=0; l<32; l++) { out[i+l] = temp[i+l]; }
                    }
                }
            // }
        }
    }
};

template <int Bnl, int Bnw>
class BCSR : public ltri::SpMat {
    int N, M; // N denots #rows; M denots #columns.
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
public:
    BCSR(CSRMat *Mat) {
        double *tval;
        int *tcind, *trptr;
        Mat->CopyL(&tval, &tcind, &trptr);
        N = Mat->N; M = Mat->M;
        helper::matrix::csr_to_bcsr(
            tval, tcind, trptr, &val, &cind, &rptr, Mat->N, Bnl, Bnw);
        free(tval);
        free(tcind);
        free(trptr);
    }
    ~BCSR() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
    }
    int GetN() { return N; }
    void SpTRSV(double *in, double *out) {
        const int l_off = (Bnw == 1)? Bnl-1 : Bnl / Bnw;
        const int Bsize = Bnl * Bnw;
        for(int i=0; i<N; i+=Bnl) {
            int bidx = i / Bnl;
            #pragma omp simd simdlen(Bnl)
            for(int k=0; k<Bnl; k++) { out[i+k] = in[i+k]; }
            int j;
            for(j=rptr[bidx]; j<rptr[bidx+1]-l_off; j++) {
                int x_ind = cind[j] * Bnw;
                ILUB_FORWARD(0,0,out,out);
                if constexpr(Bnw == 2 || Bnw == 4 || Bnw == 8) {
                    ILUB_FORWARD(1,1,out,out);
                }
                if constexpr(Bnw == 4 || Bnw == 8) {
                    ILUB_FORWARD(2,2,out,out);
                    ILUB_FORWARD(3,3,out,out);
                }if constexpr(Bnw == 8) {
                    ILUB_FORWARD(4,4,out,out);
                    ILUB_FORWARD(5,5,out,out);
                    ILUB_FORWARD(6,6,out,out);
                    ILUB_FORWARD(7,7,out,out);
                }                    
            }
            int off = j*Bsize;
            for(int k=0; k<Bnl-1; k++) {
                for(int l=k+1; l<Bnl; l++) {
                    out[i+l] -= val[off+l] * out[i+k];
                }
                off += Bnl;
            }
        }
    }
};

class BJCSR : public ltri::SpMat {
    int N, M; // N denots #rows; M denots #columns.
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    int *bj_size, bj_num;
public:
    BJCSR(CSRMat *Mat) {
        if(Mat->order != MMOrder::BJ) {
            printf("In ltri::BJCSR, the order of Mat must be BJ.\n");
            exit(1);
        }
        Mat->CopyL(&val, &cind, &rptr);
        N = Mat->N; M = Mat->M;
        bj_size = Mat->bj_size;
        bj_num  = Mat->bj_num;
    }
    ~BJCSR() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
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
};

template <int Bnl, int Bnw>
class BJBCSR : public ltri::SpMat {
    int N, M; // N denots #rows; M denots #columns.
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    int *bj_size, bj_num;
public:
    BJBCSR(CSRMat *Mat) {
        if(Mat->order != MMOrder::BJ) {
            printf("In ltri::BJBCSR, the order of Mat must be BJ.\n");
            exit(1);
        }
        bj_size = Mat->bj_size;
        bj_num  = Mat->bj_num;
        for(int i=0; i<bj_num; i++) {
            int size = bj_size[i+1] - bj_size[i];
            if( size % Bnl != 0 ) {
                printf("In ltri::BJBCSR, each bj_size must be a multiple of Bnl.\n");
                exit(1);
            }
            if( size % Bnw != 0 ) {
                printf("In ltri::BJBCSR, each bj_size must be a multiple of Bnl.\n");
                exit(1);
            }
        }
        double *tval;
        int *tcind, *trptr;
        Mat->CopyL(&tval, &tcind, &trptr);
        N = Mat->N; M = Mat->M;
        helper::matrix::csr_to_bcsr(
            tval, tcind, trptr, &val, &cind, &rptr, Mat->N, Bnl, Bnw);
        free(tval);
        free(tcind);
        free(trptr);
    }
    ~BJBCSR() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
    }
    int GetN() { return N; }
    void SpTRSV(double *in, double *out) {
        const int l_off = (Bnw == 1)? Bnl-1 : Bnl / Bnw;
        const int Bsize = Bnl * Bnw;
        #pragma omp parallel for
        for(int id=0; id<bj_num; id++) {
            int s = bj_size[id];
            int e = bj_size[id+1];
            for(int i=s; i<e; i+=Bnl) {
                int bidx = i / Bnl;
                #pragma omp simd simdlen(Bnl)
                for(int k=0; k<Bnl; k++) { out[i+k] = in[i+k]; }
                int j;
                for(j=rptr[bidx]; j<rptr[bidx+1]-l_off; j++) {
                    int x_ind = cind[j] * Bnw;
                    ILUB_FORWARD(0,0,out,out);
                    if constexpr(Bnw == 2 || Bnw == 4 || Bnw == 8) {
                        ILUB_FORWARD(1,1,out,out);
                    }
                    if constexpr(Bnw == 4 || Bnw == 8) {
                        ILUB_FORWARD(2,2,out,out);
                        ILUB_FORWARD(3,3,out,out);
                    }if constexpr(Bnw == 8) {
                        ILUB_FORWARD(4,4,out,out);
                        ILUB_FORWARD(5,5,out,out);
                        ILUB_FORWARD(6,6,out,out);
                        ILUB_FORWARD(7,7,out,out);
                    }                    
                }
                int off = j*Bsize;
                for(int k=0; k<Bnl-1; k++) {
                    for(int l=k+1; l<Bnl; l++) {
                        out[i+l] -= val[off+l] * out[i+k];
                    }
                    off += Bnl;
                }
            }
        }
    }
};

class MCCSR : public ltri::SpMat {
    int N, M; // N denots #rows; M denots #columns.
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    int *c_size, c_num;
public:
    MCCSR(CSRMat *Mat)
    {
        if(Mat->order != MMOrder::MC) {
            printf("In ltri::MCCSR, the order of Mat must be MC.\n");
            exit(1);
        }
        Mat->CopyL(&val, &cind, &rptr);
        N = Mat->N; M = Mat->M;
        c_size = Mat->c_size;
        c_num  = Mat->c_num;
    }
    ~MCCSR() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
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
};

class MCSELL32 : public ltri::SpMat {
    int N, M; // N denots #rows; M denots #columns.
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    int *nnz = nullptr;
    int *c_size, c_num, *c_s_num;
public:
    MCSELL32(CSRMat *Mat, int *_c_size, int _c_num)
    {
        double *tval;
        int *tcind, *trptr;
        Mat->CopyL(&tval, &tcind, &trptr);
        N = Mat->N; M = Mat->M;
        c_size = _c_size;
        c_num  = _c_num;

        helper::matrix::csr_to_mcsell32(
            tval, tcind, trptr,
            &val, &cind, &rptr, &nnz, &c_s_num,
            N, c_size, c_num);
    }
    ~MCSELL32() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
        utils::SafeFree(&nnz);
        free(c_s_num);
    }
    int GetN() { return N; }
    void SpTRSV(double *in, double *out) {
        for(int cid=0; cid<c_num; cid++) {
            int s = c_size[cid];
            int e = c_size[cid+1];
            int s_num = c_s_num[cid];
            int size = e - s;
            for(int i=0; i<size; i++) {
                int sid = s_num+i/32;
                int myid = i%32;
                int offset = nnz[sid];
                int slice = (sid != s_num+size/32)? 32 : size % 32;
                double temp = in[s+i];
                for(int j=0; j<rptr[sid+1]-rptr[sid]; j++) {
                    temp -= val[offset+j*slice+myid] * in[cind[offset+j*slice+myid]];
                }
                out[s+i] = temp;
            }
        }
    }
};

class BMCCSR : public ltri::SpMat {
private:
    int N, M; // N denots #rows; M denots #columns.
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    int *c_size, c_num, b_size;
public:
    BMCCSR(CSRMat *Mat)
    {
        if(Mat->order != MMOrder::BMC) {
            printf("In ltri::BMCCSR, the order of Mat must be BMC.\n");
            exit(1);
        }
        Mat->CopyL(&val, &cind, &rptr);
        N = Mat->N; M = Mat->M;
        c_size = Mat->c_size;
        c_num  = Mat->c_num;
        b_size = Mat->b_size;
    }
    ~BMCCSR() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
    }
    int GetN() { return N; }
    void SpTRSV(double *in, double *out) {
        #pragma omp parallel
        {
            for(int id=0; id<c_num; id++) {
                int s = c_size[id];
                int e = c_size[id+1];
                #pragma omp for
                for(int bid=s; bid<e; bid++) {
                    int off = bid*b_size;
                    for(int i=off; i<off+b_size; i++) {
                        double t = in[i];
                        for(int j=rptr[i]; j<rptr[i+1]; j++) {
                            t -= val[j] * out[cind[j]];
                        }
                        out[i] = t;
                    }
                }
            }
        }
    }
};

template <int Bnl, int Bnw>
class BMCBCSR : public ltri::SpMat {
private:
    int N, M; // N denots #rows; M denots #columns.
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    int *c_size, c_num, b_size;
public:
    BMCBCSR(CSRMat *Mat)
    {
        if(Mat->order != MMOrder::BMC) {
            printf("In ltri::BMCBCSR, the order of Mat must be BMC.\n");
            exit(1);
        }
        if(Mat->b_size % Bnl != 0 || Mat->b_size % Bnw) {
            printf("In ltri::BMCBCSR, b_size must be multiples of Bnl and Bnw.\n");
            exit(1);
        }
        double *tval;
        int *tcind, *trptr;
        Mat->CopyL(&tval, &tcind, &trptr);
        N = Mat->N; M = Mat->M;
        helper::matrix::csr_to_bcsr(
            tval, tcind, trptr, &val, &cind, &rptr, Mat->N, Bnl, Bnw);
        free(tval);
        free(tcind);
        free(trptr);
        c_size = Mat->c_size;
        c_num  = Mat->c_num;
        b_size = Mat->b_size;
    }
    ~BMCBCSR() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
    }
    int GetN() { return N; }
    void SpTRSV(double *in, double *out) {
        const int l_off = (Bnw == 1)? Bnl-1 : Bnl / Bnw;
        const int Bsize = Bnl * Bnw;
        #pragma omp parallel
        {
            for(int cid=0; cid<c_num; cid++) {
                int s = c_size[cid];
                int e = c_size[cid+1];
                #pragma omp for
                for(int bid=s; bid<e; bid++) {
                    int off = bid*b_size;
                    for(int i=off; i<off+b_size; i+=Bnl) {
                        int bidx = i / Bnl;
                        #pragma omp simd simdlen(Bnl)
                        for(int k=0; k<Bnl; k++) { out[i+k] = in[i+k]; }
                        int j;
                        for(j=rptr[bidx]; j<rptr[bidx+1]-l_off; j++) {
                            int x_ind = cind[j] * Bnw;
                            ILUB_FORWARD(0,0,out,out);
                            if constexpr(Bnw == 2 || Bnw == 4 || Bnw == 8) {
                                ILUB_FORWARD(1,1,out,out);
                            }
                            if constexpr(Bnw == 4 || Bnw == 8) {
                                ILUB_FORWARD(2,2,out,out);
                                ILUB_FORWARD(3,3,out,out);
                            }if constexpr(Bnw == 8) {
                                ILUB_FORWARD(4,4,out,out);
                                ILUB_FORWARD(5,5,out,out);
                                ILUB_FORWARD(6,6,out,out);
                                ILUB_FORWARD(7,7,out,out);
                            }                    
                        }
                        int off = j*Bsize;
                        for(int k=0; k<Bnl-1; k++) {
                            for(int l=k+1; l<Bnl; l++) {
                                out[i+l] -= val[off+l] * out[i+k];
                            }
                            off += Bnl;
                        }
                    }
                }
            }
        }
    }
};

} // namespace ltri

namespace ldtri {

class SpMat {
public:
    virtual int GetN() = 0;
    virtual void SpTRSV(double *in, double *out) = 0;
    virtual ~SpMat() {}
};

class CSR : public ldtri::SpMat {
    int N, M; // N denots #rows; M denots #columns.
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
public:
    CSR(CSRMat *Mat) {
        helper::matrix::extractLDinv(
            Mat->val, Mat->cind, Mat->rptr,
            &val, &cind, &rptr, Mat->N);
        N = Mat->N; M = Mat->M;
    }
    ~CSR() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
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
};

class MCCSR : public ldtri::SpMat {
    int N, M; // N denots #rows; M denots #columns.
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    int *c_size, c_num;
public:
    MCCSR(CSRMat *Mat) {
        if(Mat->order != MMOrder::MC) {
            printf("In ldtri::MCCSR, the order of Mat must be MC.\n");
            exit(1);
        }
        helper::matrix::extractLDinv(
            Mat->val, Mat->cind, Mat->rptr,
            &val, &cind, &rptr, Mat->N);
        N = Mat->N; M = Mat->M;
        c_size = Mat->c_size;
        c_num  = Mat->c_num;
    }
    ~MCCSR() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
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
                    int j;
                    for(j=rptr[i]; j<rptr[i+1]-1; j++) {
                        t -= val[j] * out[cind[j]];
                    }
                    out[i] = t * val[j];
                }
            }
        }
    }
};

class BMCCSR : public ldtri::SpMat {
    int N, M; // N denots #rows; M denots #columns.
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    int *c_size, c_num, b_size;
public:
    BMCCSR(CSRMat *Mat) {
        if(Mat->order != MMOrder::BMC) {
            printf("In ldtri::BMCCSR, the order of Mat must be BMC.\n");
            exit(1);
        }
        helper::matrix::extractLDinv(
            Mat->val, Mat->cind, Mat->rptr,
            &val, &cind, &rptr, Mat->N);
        N = Mat->N; M = Mat->M;
        c_size = Mat->c_size;
        c_num  = Mat->c_num;
        b_size = Mat->b_size;
        printf("%d %d\n", c_num, b_size);
    }
    ~BMCCSR() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
    }
    int GetN() { return N; }
    void SpTRSV(double *in, double *out) {
        #pragma omp parallel
        {
            for(int id=0; id<c_num; id++) {
                int s = c_size[id];
                int e = c_size[id+1];
                #pragma omp for
                for(int bid=s; bid<e; bid++) {
                    int off = bid*b_size;
                    for(int i=off; i<off+b_size; i++) {
                        double t = in[i];
                        int j;
                        for(j=rptr[i]; j<rptr[i+1]-1; j++) {
                            t -= val[j] * out[cind[j]];
                        }
                        out[i] = t * val[j];
                    }
                }
            }
        }
    }
};

} // namespace ldtri

namespace dutri {

class SpMat {
public:
    virtual int GetN() = 0;
    virtual void SpTRSV(double *in, double *out) = 0;
    virtual ~SpMat() {}
};

class CSR : public dutri::SpMat {
    int N, M; // N denots #rows; M denots #columns.
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
public:
    CSR(CSRMat *Mat) {
        Mat->CopyDinvU(&val, &cind, &rptr);
        N = Mat->N; M = Mat->M;
    }
    ~CSR() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
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
};

template <int Iter>
class CSRJa : public dutri::SpMat {
    int N, M; // N denots #rows; M denots #columns.
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    double *temp = nullptr;
public:
    CSRJa(CSRMat *Mat) {
        Mat->CopyDinvU(&val, &cind, &rptr);
        N = Mat->N; M = Mat->M;
        temp = utils::SafeMalloc<double>(N);
    }
    ~CSRJa() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
        utils::SafeFree(&temp);
    }
    int GetN() { return N; }
    void SpTRSV(double *in, double *out) {
        #pragma omp parallel for
        for(int i=0; i<N; i++) {
            out[i] = (in[i]) * val[rptr[i]];
        }
        if constexpr(Iter > 1) {
            #pragma omp parallel
            {
                for(int k=1; k<Iter; k++) {
                    #pragma omp for
                    for(int i=N-1; i>=0; i--) {
                        double t = in[i];
                        int j;
                        for(j=rptr[i+1]-1; j>=rptr[i]+1; j--) {
                            t -= val[j] * out[cind[j]];
                        }
                        temp[i] = t * val[j];
                    }
                    #pragma omp for
                    for(int i=0; i<N; i++) { out[i] = temp[i]; }
                }
            }
        }
    }
};

template <int Bnl, int Bnw>
class BCSR : public dutri::SpMat {
    int N, M; // N denots #rows; M denots #columns.
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
public:
    BCSR(CSRMat *Mat) {
        double *tval;
        int *tcind, *trptr;
        Mat->CopyDinvU(&tval, &tcind, &trptr);
        N = Mat->N; M = Mat->M;
        helper::matrix::csr_to_bcsr(
            tval, tcind, trptr, &val, &cind, &rptr, Mat->N, Bnl, Bnw);
        free(tval);
        free(tcind);
        free(trptr);
    }
    ~BCSR() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
    }
    int GetN() { return N; }
    void SpTRSV(double *in, double *out) {
        const int u_off = Bnl / Bnw;
        const int Bsize = Bnl * Bnw;
        for(int i=N-Bnl; i>=0; i-=Bnl) {
            int bidx = i / Bnl;
            int j;
            for(j=rptr[bidx+1]-1; j>=rptr[bidx]+u_off; j--) {
                int x_ind = cind[j] * Bnw;
                ILUB_BACKWARD(0,0,out,out);
                if constexpr(Bnw == 2 || Bnw == 4 || Bnw == 8) {
                    ILUB_BACKWARD(1,1,out,out);
                }
                if constexpr(Bnw == 4 || Bnw == 8) {
                    ILUB_BACKWARD(2,2,out,out);
                    ILUB_BACKWARD(3,3,out,out);
                }if constexpr(Bnw == 8) {
                    ILUB_BACKWARD(4,4,out,out);
                    ILUB_BACKWARD(5,5,out,out);
                    ILUB_BACKWARD(6,6,out,out);
                    ILUB_BACKWARD(7,7,out,out);
                } 
            }
            int off = (j+1)*Bsize-Bnl;
            for(int k=Bnl-1; k>=0; k--) {
                out[i+k] *= val[off+k];
                for(int l=k-1; l>=0; l--) {
                    out[i+l] -= val[off+l] * out[i+k];
                }
                off -= Bnl;
            }
        }
    }
};

class BJCSR : public dutri::SpMat {
    int N, M; // N denots #rows; M denots #columns.
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    int *bj_size, bj_num;
public:
    BJCSR(CSRMat *Mat)
    {
        if(Mat->order != MMOrder::BJ) {
            printf("In dutri::BJCSR, the order of Mat must be BJ.\n");
            exit(1);
        }
        Mat->CopyDinvU(&val, &cind, &rptr);
        N = Mat->N; M = Mat->M;
        bj_size = Mat->bj_size;
        bj_num  = Mat->bj_num;
    }
    ~BJCSR() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
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
};

template <int Bnl, int Bnw>
class BJBCSR : public dutri::SpMat {
    int N, M; // N denots #rows; M denots #columns.
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    int *bj_size, bj_num;
public:
    BJBCSR(CSRMat *Mat)
    {
        if(Mat->order != MMOrder::BJ) {
            printf("In dutri::BJBCSR, the order of Mat must be BJ.\n");
            exit(1);
        }
        bj_size = Mat->bj_size;
        bj_num  = Mat->bj_num;
        for(int i=0; i<bj_num; i++) {
            int size = bj_size[i+1] - bj_size[i];
            if( size % Bnl != 0 ) {
                printf("In dutri::BJBCSR, each bj_size must be a multiple of Bnl.\n");
                exit(1);
            }
            if( size % Bnw != 0 ) {
                printf("In dutri::BJBCSR, each bj_size must be a multiple of Bnl.\n");
                exit(1);
            }
        }
        double *tval;
        int *tcind, *trptr;
        Mat->CopyDinvU(&tval, &tcind, &trptr);
        N = Mat->N; M = Mat->M;
        helper::matrix::csr_to_bcsr(
            tval, tcind, trptr, &val, &cind, &rptr, Mat->N, Bnl, Bnw);
        free(tval);
        free(tcind);
        free(trptr);
    }
    ~BJBCSR() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
    }
    int GetN() { return N; }
    void SpTRSV(double *in, double *out) {
        const int u_off = Bnl / Bnw;
        const int Bsize = Bnl * Bnw;
        #pragma omp parallel for
        for(int id=0; id<bj_num; id++) {
            int s = bj_size[id];
            int e = bj_size[id+1];
            for(int i=e-Bnl; i>=s; i-=Bnl) {
                int bidx = i / Bnl;
                int j;
                for(j=rptr[bidx+1]-1; j>=rptr[bidx]+u_off; j--) {
                    int x_ind = cind[j] * Bnw;
                    ILUB_BACKWARD(0,0,out,out);
                    if constexpr(Bnw == 2 || Bnw == 4 || Bnw == 8) {
                        ILUB_BACKWARD(1,1,out,out);
                    }
                    if constexpr(Bnw == 4 || Bnw == 8) {
                        ILUB_BACKWARD(2,2,out,out);
                        ILUB_BACKWARD(3,3,out,out);
                    }if constexpr(Bnw == 8) {
                        ILUB_BACKWARD(4,4,out,out);
                        ILUB_BACKWARD(5,5,out,out);
                        ILUB_BACKWARD(6,6,out,out);
                        ILUB_BACKWARD(7,7,out,out);
                    } 
                }
                int off = (j+1)*Bsize-Bnl;
                for(int k=Bnl-1; k>=0; k--) {
                    out[i+k] *= val[off+k];
                    for(int l=k-1; l>=0; l--) {
                        out[i+l] -= val[off+l] * out[i+k];
                    }
                    off -= Bnl;
                }
            }
        }
    }
};

class MCCSR : public dutri::SpMat {
    int N, M; // N denots #rows; M denots #columns.
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    int *c_size, c_num;
public:
    MCCSR(CSRMat *Mat)
    {
        if(Mat->order != MMOrder::MC) {
            printf("In dutri::MCCSR, the order of Mat must be MC.\n");
            exit(1);
        }
        Mat->CopyDinvU(&val, &cind, &rptr);
        N = Mat->N; M = Mat->M;
        c_size = Mat->c_size;
        c_num  = Mat->c_num;
    }
    ~MCCSR() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
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
};

class MCSELL32 : public dutri::SpMat {
    int N, M; // N denots #rows; M denots #columns.
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    int *nnz = nullptr;
    int *c_size, c_num, *c_s_num;
public:
    MCSELL32(CSRMat *Mat, int *_c_size, int _c_num)
    {
        double *tval;
        int *tcind, *trptr;
        Mat->CopyDinvU(&tval, &tcind, &trptr);
        N = Mat->N; M = Mat->M;
        c_size = _c_size;
        c_num  = _c_num;

        helper::matrix::csr_to_mcsell32(
            tval, tcind, trptr,
            &val, &cind, &rptr, &nnz, &c_s_num,
            N, c_size, c_num);
    }
    ~MCSELL32() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
        utils::SafeFree(&nnz);
        free(c_s_num);
    }
    int GetN() { return N; }
    void SpTRSV(double *in, double *out) {
        for(int cid=c_num-1; cid>=0; cid--) {
            int s = c_size[cid];
            int e = c_size[cid+1];
            int s_num = c_s_num[cid];
            int size = e - s;
            for(int i=0; i<size; i++) {
                int sid = s_num+i/32;
                int myid = i%32;
                int offset = nnz[sid];
                int slice = (sid != s_num+size/32)? 32 : size % 32;
                double temp = in[s+i];
                for(int j=rptr[sid+1]-rptr[sid]-1; j>=1; j--) {
                    temp -= val[offset+j*slice+myid] * in[cind[offset+j*slice+myid]];
                }
                out[s+i] = temp * val[offset+myid];
            }
        }
    }
};

class BMCCSR : public dutri::SpMat {
private:
    int N, M; // N denots #rows; M denots #columns.
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    int *c_size, c_num, b_size;
public:
    BMCCSR(CSRMat *Mat)
    {
        if(Mat->order != MMOrder::BMC) {
            printf("In dutri::BMCCSR, the order of Mat must be BMC.\n");
            exit(1);
        }
        Mat->CopyDinvU(&val, &cind, &rptr);
        N = Mat->N; M = Mat->M;
        c_size = Mat->c_size;
        c_num  = Mat->c_num;
        b_size = Mat->b_size;
    }
    ~BMCCSR() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
    }
    int GetN() { return N; }
    void SpTRSV(double *in, double *out) {
        #pragma omp parallel
        {
            for(int id=c_num-1; id>=0; id--) {
                int s = c_size[id];
                int e = c_size[id+1];
                #pragma omp for
                for(int bid=e-1; bid>=s; bid--) {
                    int off = bid*b_size;
                    for(int i=off+b_size-1; i>=off; i--) {
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
    }
};

template <int Bnl, int Bnw>
class BMCBCSR : public dutri::SpMat {
private:
    int N, M; // N denots #rows; M denots #columns.
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    int *c_size, c_num, b_size;
public:
    BMCBCSR(CSRMat *Mat)
    {
        if(Mat->order != MMOrder::BMC) {
            printf("In dutri::BMCBCSR, the order of Mat must be BMC.\n");
            exit(1);
        }
        if(Mat->b_size % Bnl != 0 || Mat->b_size % Bnw) {
            printf("In ltri::BMCBCSR, b_size must be multiples of Bnl and Bnw.\n");
            exit(1);
        }
        double *tval;
        int *tcind, *trptr;
        Mat->CopyDinvU(&tval, &tcind, &trptr);
        N = Mat->N; M = Mat->M;
        helper::matrix::csr_to_bcsr(
            tval, tcind, trptr, &val, &cind, &rptr, Mat->N, Bnl, Bnw);
        free(tval);
        free(tcind);
        free(trptr);
        c_size = Mat->c_size;
        c_num  = Mat->c_num;
        b_size = Mat->b_size;
    }
    ~BMCBCSR() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
    }
    int GetN() { return N; }
    void SpTRSV(double *in, double *out) {
        const int u_off = Bnl / Bnw;
        const int Bsize = Bnl * Bnw;
        #pragma omp parallel
        {
            for(int cid=c_num-1; cid>=0; cid--) {
                int s = c_size[cid];
                int e = c_size[cid+1];
                #pragma omp for
                for(int bid=e-1; bid>=s; bid--) {
                    int off = bid*b_size;
                    for(int i=off+b_size-Bnl; i>=off; i-=Bnl) {
                        int bidx = i / Bnl;
                        int j;
                        for(j=rptr[bidx+1]-1; j>=rptr[bidx]+u_off; j--) {
                            int x_ind = cind[j] * Bnw;
                            ILUB_BACKWARD(0,0,out,out);
                            if constexpr(Bnw == 2 || Bnw == 4 || Bnw == 8) {
                                ILUB_BACKWARD(1,1,out,out);
                            }
                            if constexpr(Bnw == 4 || Bnw == 8) {
                                ILUB_BACKWARD(2,2,out,out);
                                ILUB_BACKWARD(3,3,out,out);
                            }if constexpr(Bnw == 8) {
                                ILUB_BACKWARD(4,4,out,out);
                                ILUB_BACKWARD(5,5,out,out);
                                ILUB_BACKWARD(6,6,out,out);
                                ILUB_BACKWARD(7,7,out,out);
                            } 
                        }
                        int off = (j+1)*Bsize-Bnl;
                        for(int k=Bnl-1; k>=0; k--) {
                            out[i+k] *= val[off+k];
                            for(int l=k-1; l>=0; l--) {
                                out[i+l] -= val[off+l] * out[i+k];
                            }
                            off -= Bnl;
                        }
                    }
                }
            }
        }
    }
};

} // namespace udtri

} // senk

#endif
