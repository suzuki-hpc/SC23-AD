#ifndef SENKPP_MATRIX_TRISPMAT_HPP
#define SENKPP_MATRIX_TRISPMAT_HPP

#include "enums.hpp"
#include "utils/alloc.hpp"
#include "concepts.hpp"
#include "helper/helper_matrix.hpp"

namespace senk {

#define ILUB_FORWARD(a, b, left, right) { \
    _Pragma("omp simd simdlen(Bnl)") \
    for(int k=0; k<Bnl; k++) { \
        (left)[i+k] -= val[j*b_size+Bnl*(a)+k] * (right)[x_ind+(b)]; \
    } \
}

#define ILUB_BACKWARD(a, b, left, right) { \
    _Pragma("omp simd simdlen(Bnl)") \
    for(int k=0; k<Bnl; k++) { \
        (left)[i+k] -= val[j*b_size+Bnl*(a)+k] * (right)[x_ind+(b)]; \
    } \
}

template <Floating T>
class TriSpMat {
public:
    int N, M; // N denots #rows; M denots #columns.
    virtual void SpTRSV(T *in, T *out) = 0;
    virtual ~TriSpMat() {}
};

template <Floating T>
class TriLCSR : public TriSpMat<T> {
private:
    T *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
public:
    TriLCSR(CSRMat *Mat) {
        Mat->CopyL<T>(&val, &cind, &rptr);
        TriSpMat<T>::N = Mat->N;
        TriSpMat<T>::M = Mat->M;
    }
    ~TriLCSR() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
    }
    void SpTRSV(T *in, T *out) {
        for(int i=0; i<TriSpMat<T>::N; i++) {
            T t = in[i];
            for(int j=rptr[i]; j<rptr[i+1]; j++) {
                t -= val[j] * out[cind[j]];
            }
            out[i] = t;
        }
    }
};

template <Floating T>
class TriDinvUCSR : public TriSpMat<T> {
private:
    T *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
public:
    TriDinvUCSR(CSRMat *Mat) {
        Mat->CopyDinvU<T>(&val, &cind, &rptr);
        TriSpMat<T>::N = Mat->N;
        TriSpMat<T>::M = Mat->M;
    }
    ~TriDinvUCSR() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
    }
    void SpTRSV(T *in, T *out) {
        for(int i=TriSpMat<T>::N-1; i>=0; i--) {
            T t = in[i];
            int j;
            for(j=rptr[i+1]-1; j>=rptr[i]+1; j--) {
                t -= val[j] * out[cind[j]];
            }
            out[i] = t * val[j];
        }
    }
};

template <Floating T>
class TriLCSR_BJ : public TriSpMat<T> {
private:
    T *val;
    int *cind, *rptr;
    int *bj_size, bj_num;
public:
    TriLCSR_BJ(CSRMat *Mat, int *_bj_size, int _bj_num) {
        Mat->CopyL<T>(&val, &cind, &rptr);
        TriSpMat<T>::N = Mat->N;
        TriSpMat<T>::M = Mat->M;
        bj_size = _bj_size;
        bj_num = _bj_num;
    }
    ~TriLCSR_BJ() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
    }
    void SpTRSV(T *in, T *out) {
        #pragma omp parallel for
        for(int id=0; id<bj_num; id++) {
            int s = bj_size[id];
            int e = bj_size[id+1];
            for(int i=s; i<e; i++) {
                T t = in[i];
                for(int j=rptr[i]; j<rptr[i+1]; j++) {
                    t -= val[j] * out[cind[j]];
                }
                out[i] = t;
            }
        }
    }
};

template <Floating T>
class TriDinvUCSR_BJ : public TriSpMat<T> {
private:
    T *val;
    int *cind, *rptr;
    int *bj_size, bj_num;
public:
    TriDinvUCSR_BJ(CSRMat *Mat, int *_bj_size, int _bj_num)
    {
        Mat->CopyDinvU<T>(&val, &cind, &rptr);
        TriSpMat<T>::N = Mat->N;
        TriSpMat<T>::M = Mat->M;
        bj_size = _bj_size; bj_num = _bj_num;
    }
    ~TriDinvUCSR_BJ() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
    }
    void SpTRSV(T *in, T *out) {
        #pragma omp parallel for
        for(int id=0; id<bj_num; id++) {
            int s = bj_size[id];
            int e = bj_size[id+1];
            for(int i=e-1; i>=s; i--) {
                T t = in[i];
                int j;
                for(j=rptr[i+1]-1; j>=rptr[i]+1; j--) {
                    t -= val[j] * out[cind[j]];
                }
                out[i] = t * val[j];
            }
        }
    }
};

// --------------- //

template <typename T1, typename T2>
class TriSpMat2 {
public:
    int N, M; // N denots #rows; M denots #columns.
    virtual void SpTRSV(T1 *in, T1 *out) = 0;
    virtual void SpTRSV2(T2 *in, T2 *out) = 0;
    virtual ~TriSpMat2() {}
};


template <int bit>
class TriLCSR_DI : public TriSpMat2<double, int> {
private:
    double *val = nullptr;
    int32_t *val32 = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
public:
    TriLCSR_DI(CSRMat *Mat) {
        Mat->CopyL<double>(&val, &cind, &rptr);
        TriSpMat2<double, int>::N = Mat->N;
        TriSpMat2<double, int>::M = Mat->M;
        val32 = utils::SafeMalloc<int>(rptr[N]);
        utils::Convert_DI<int32_t, bit>(val, val32, rptr[N]);
    }
    void Free() {
        utils::SafeFree(&val);
        utils::SafeFree(&val32);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
    }
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

template <int bit>
class TriDinvUCSR_DI : public TriSpMat2<double, int> {
private:
    double *val = nullptr;
    int32_t *val32 = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
public:
    TriDinvUCSR_DI(CSRMat *Mat) {
        Mat->CopyDinvU<double>(&val, &cind, &rptr);
        TriSpMat2<double, int>::N = Mat->N;
        TriSpMat2<double, int>::M = Mat->M;
        val32 = utils::SafeMalloc<int>(rptr[N]);
        utils::Convert_DI<int32_t, bit>(val, val32, rptr[N]);
    }
    void Free() {
        utils::SafeFree(&val);
        utils::SafeFree(&val32);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
    }
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

/*
template <typename T>
class TriCSR_MC : public TriSpMat<T> {
private:
    T *val;
    int *cind, *rptr;
    int *c_size, c_num;
public:
    TriCSR_MC(T *_val, int *_cind, int *_rptr,
        int _N, int _M, int *_c_size, int _c_num)
    {
        val = _val; cind = _cind; rptr = _rptr;
        TriSpMat<T>::N = _N; TriSpMat<T>::M = _M;
        c_size = _c_size; c_num = _c_num;
    }
    ~TriCSR_MC() {}
    void Free() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
    }
    void SpTRSV_L(T *in, T *out) {
        #pragma omp parallel
        {
            for(int id=0; id<c_num; id++) {
                int s = c_size[id];
                int e = c_size[id+1];
                #pragma omp for
                for(int i=s; i<e; i++) {
                    T t = in[i];
                    for(int j=rptr[i]; j<rptr[i+1]; j++) {
                        t -= val[j] * out[cind[j]];
                    }
                    out[i] = t;
                }
            }
        }
    }
    void SpTRSV_U(T *in, T *out) {
        #pragma omp parallel
        {
            for(int id=c_num-1; id>=0; id--) {
                int s = c_size[id];
                int e = c_size[id+1];
                #pragma omp for
                for(int i=e-1; i>=s; i--) {
                    T t = in[i];
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

template <typename T>
class TriCSR_BMC : public TriSpMat<T> {
private:
    T *val;
    int *cind, *rptr;
    int *c_size, c_num, b_size;
public:
    TriCSR_BMC(T *_val, int *_cind, int *_rptr,
        int _N, int _M, int *_c_size, int _c_num,
        int _b_size)
    {
        val = _val; cind = _cind; rptr = _rptr;
        TriSpMat<T>::N = _N; TriSpMat<T>::M = _M;
        c_size = _c_size; c_num = _c_num; b_size = _b_size;
    }
    ~TriCSR_BMC() {}
    void Free() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
    }
    void SpTRSV_L(T *in, T *out) {
        #pragma omp parallel
        {
            for(int id=0; id<c_num; id++) {
                int s = c_size[id];
                int e = c_size[id+1];
                #pragma omp for
                for(int bid=s; bid<e; bid++) {
                    int off = bid*b_size;
                    for(int i=off; i<off+b_size; i++) {
                        T t = in[i];
                        for(int j=rptr[i]; j<rptr[i+1]; j++) {
                            t -= val[j] * out[cind[j]];
                        }
                        out[i] = t;
                    }
                }
            }
        }
    }
    void SpTRSV_U(T *in, T *out) {
        #pragma omp parallel
        {
            for(int id=c_num-1; id>=0; id--) {
                int s = c_size[id];
                int e = c_size[id+1];
                #pragma omp for
                for(int bid=e-1; bid>=s; bid--) {
                    int off = bid*b_size;
                    for(int i=off+b_size-1; i>=off; i--) {
                        T t = in[i];
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

template <typename T, int Bnl, int Bnw>
class TriBCSR : public TriSpMat<T> {
private:
    T *val;
    int *cind, *rptr;
public:
    TriBCSR(T *_val, int *_cind, int *_rptr,
        int _N, int _M)
    {
        val = _val; cind = _cind; rptr = _rptr;
        TriSpMat<T>::N = _N; TriSpMat<T>::M = _M;
    }
    ~TriBCSR() {}
    void Free() {
        utils::SafeFree<T>(&val);
        utils::SafeFree<int>(&cind);
        utils::SafeFree<int>(&rptr);
    }
    void SpTRSV_L(T *in, T *out) {
        const int l_off = (Bnw == 1)? Bnl-1 : Bnl / Bnw;
        const int b_size = Bnl * Bnw;
        for(int i=0; i<TriSpMat<T>::N; i+=Bnl) {
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
            int off = j*b_size;
            for(int k=0; k<Bnl-1; k++) {
                for(int l=k+1; l<Bnl; l++) {
                    out[i+l] -= val[off+l] * out[i+k];
                }
                off += Bnl;
            }
        }
    }
    void SpTRSV_U(T *in, T *out) {
        const int u_off = Bnl / Bnw;
        const int b_size = Bnl * Bnw;
        for(int i=TriSpMat<T>::N-Bnl; i>=0; i-=Bnl) {
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
            int off = (j+1)*b_size-Bnl;
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

template <typename T, int Bnl, int Bnw>
class TriBCSR_BJ : public TriSpMat<T> {
private:
    T *val;
    int *cind, *rptr;
    int *bj_size, bj_num;
public:
    TriBCSR_BJ(T *_val, int *_cind, int *_rptr,
        int _N, int _M, int *_bj_size, int _bj_num)
    {
        val = _val; cind = _cind; rptr = _rptr;
        TriSpMat<T>::N = _N; TriSpMat<T>::M = _M;
        bj_size = _bj_size; bj_num = _bj_num;
    }
    ~TriBCSR_BJ() {}
    void Free() {
        utils::SafeFree<T>(&val);
        utils::SafeFree<int>(&cind);
        utils::SafeFree<int>(&rptr);
    }
    void SpTRSV_L(T *in, T *out) {
        const int l_off = (Bnw == 1)? Bnl-1 : Bnl / Bnw;
        const int b_size = Bnl * Bnw;
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
                int off = j*b_size;
                for(int k=0; k<Bnl-1; k++) {
                    for(int l=k+1; l<Bnl; l++) {
                        out[i+l] -= val[off+l] * out[i+k];
                    }
                    off += Bnl;
                }
            }
        }
    }
    void SpTRSV_U(T *in, T *out) {
        const int u_off = Bnl / Bnw;
        const int b_size = Bnl * Bnw;
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
                int off = (j+1)*b_size-Bnl;
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

template <typename T, int Bnl, int Bnw>
class TriBCSR_BMC : public TriSpMat<T> {
private:
    T *val;
    int *cind, *rptr;
    int *c_size, c_num, g_size;
public:
    TriBCSR_BMC(T *_val, int *_cind, int *_rptr,
        int _N, int _M, int *_c_size, int _c_num,
        int _g_size)
    {
        val = _val; cind = _cind; rptr = _rptr;
        TriSpMat<T>::N = _N; TriSpMat<T>::M = _M;
        c_size = _c_size; c_num = _c_num; g_size = _g_size;
    }
    ~TriBCSR_BMC() {}
    void Free() {
        utils::SafeFree<T>(&val);
        utils::SafeFree<int>(&cind);
        utils::SafeFree<int>(&rptr);
    }
    void SpTRSV_L(T *in, T *out) {
        const int l_off = (Bnw == 1)? Bnl-1 : Bnl / Bnw;
        const int b_size = Bnl * Bnw;
        #pragma omp parallel
        {
            for(int cid=0; cid<c_num; cid++) {
                int s = c_size[cid];
                int e = c_size[cid+1];
                #pragma omp for
                for(int bid=s; bid<e; bid++) {
                    int off = bid*g_size;
                    for(int i=off; i<off+g_size; i+=Bnl) {
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
                        int off = j*b_size;
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
    void SpTRSV_U(T *in, T *out) {
        const int u_off = Bnl / Bnw;
        const int b_size = Bnl * Bnw;
        #pragma omp parallel
        {
            for(int cid=c_num-1; cid>=0; cid--) {
                int s = c_size[cid];
                int e = c_size[cid+1];
                #pragma omp for
                for(int bid=e-1; bid>=s; bid--) {
                    int off = bid*g_size;
                    for(int i=off+g_size-Bnl; i>=off; i-=Bnl) {
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
                        int off = (j+1)*b_size-Bnl;
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

*/

} // senk

#endif
