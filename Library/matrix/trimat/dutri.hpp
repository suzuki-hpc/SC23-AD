#ifndef SENKPP_MATRIX_TRIMAT_DUTRI_HPP
#define SENKPP_MATRIX_TRIMAT_DUTRI_HPP

namespace senk {

namespace dutri {

template <typename T>
class SpMat {
public:
    virtual int GetN() = 0;
    virtual void SpTRSV(T *in, T *out) = 0;
    virtual ~SpMat() {}
};

/**
 * Sequential CSR triangular solver
 **/
template <typename T>
class CSR : public dutri::SpMat<T> {
    int N, M; // N denots #rows; M denots #columns.
    T *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
public:
    CSR(CSRMat *Mat) {
        N = Mat->N; M = Mat->M;
        if constexpr(std::is_same_v<double, T>) {
            Mat->CopyDinvU(&val, &cind, &rptr);
        }else {
            double *tval;
            Mat->CopyDinvU(&tval, &cind, &rptr);
            int nnz = rptr[N];
            val = utils::SafeMalloc<T>(nnz);
            utils::Convert<double, T>(tval, val, nnz);
            free(tval);
        }
    }
    ~CSR() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
    }
    int GetN() { return N; }
    void SpTRSV(T *in, T *out) {
        for(int i=N-1; i>=0; i--) {
            T t = in[i];
            int j;
            for(j=rptr[i+1]-1; j>=rptr[i]+1; j--) {
                t -= val[j] * out[cind[j]];
            }
            out[i] = t * val[j];
        }
    }
};

template <int bit>
class CSR<Fixed<bit>> : public dutri::SpMat<int> {
    int N, M; // N denots #rows; M denots #columns.
    int *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    const int dbit = 24;
public:
    CSR(CSRMat *Mat) {
        N = Mat->N; M = Mat->M;
        double *tval;
        Mat->CopyDinvU(&tval, &cind, &rptr);
        val = utils::SafeMalloc<int>(rptr[N]);
        // utils::Convert_DI<int, bit>(tval, val, rptr[N]);
        for(int i=0; i<N; i++) {
            for(int j=rptr[i]; j<rptr[i+1]; j++) {
                if(cind[j] == i) val[j] = (int)(tval[j] * (1 << dbit));
                else val[j] = (int)(tval[j] * (1 << bit));
            }
        }
        free(tval);
    }
    ~CSR() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
    }
    int GetN() { return N; }
    void SpTRSV(int *in, int *out) {
        for(int i=N-1; i>=0; i--) {
            long t = (long)in[i] << bit;
            int j;
            for(j=rptr[i+1]-1; j>=rptr[i]+1; j--) {
                t -= (long)val[j] * (long)out[cind[j]];
            }
            // out[i] = (int)((t >> bit) * (long)val[j] >> bit);
            out[i] = (int)((t >> bit) * (long)val[j] >> dbit);
        }
    }
};

class CSRflex : public dutri::SpMat<int> {
    int N, M; // N denots #rows; M denots #columns.
    int *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    int bit = 0, dbit = 0;
public:
    CSRflex(CSRMat *Mat) {
        N = Mat->N; M = Mat->M;
        double *tval;
        Mat->CopyDinvU(&tval, &cind, &rptr);
        val = utils::SafeMalloc<int>(rptr[N]);
        double max = 0, dmax = 0;
        for(int i=0; i<N; i++) {
            for(int j=rptr[i]; j<rptr[i+1]; j++) {
                if(cind[j] == i) {
                    if(std::fabs(tval[j]) > dmax) dmax = std::fabs(tval[j]);
                }else {
                    if(std::fabs(tval[j]) > max) max = std::fabs(tval[j]);
                }
            }
        }
        while( dbit < TRIMAT_FLEX_PARAM && dmax * (1 << dbit) < (1 << TRIMAT_FLEX_PARAM) ) {
            dbit++;
        }
        while( bit < TRIMAT_FLEX_PARAM && max * (1 << bit) < (1 << TRIMAT_FLEX_PARAM) ) {
            bit++;
        }
#if TRIMAT_FLEX_PRINT
        printf("# dutri %d %d\n", bit, dbit);
#endif
        for(int i=0; i<N; i++) {
            for(int j=rptr[i]; j<rptr[i+1]; j++) {
                if(cind[j] == i) {
                    val[j] = (int)(tval[j] * (1 << dbit));
                }else {
                    val[j] = (int)(tval[j] * (1 << bit));
                }
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
    void SpTRSV(int *in, int *out) {
        for(int i=N-1; i>=0; i--) {
            long t = (long)in[i] << bit;
            int j;
            for(j=rptr[i+1]-1; j>=rptr[i]+1; j--) {
                t -= (long)val[j] * (long)out[cind[j]];
            }
            // out[i] = (int)((t >> bit) * (long)val[j] >> bit);
            out[i] = (int)((t >> bit) * (long)val[j] >> dbit);
        }
    }
};

/**
 * Syncronus Jacobi CSR triangular solver
 **/
template <typename T, int Iter>
class CSRJa : public dutri::SpMat<T> {
    int N, M; // N denots #rows; M denots #columns.
    T *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    T *temp = nullptr;
public:
    CSRJa(CSRMat *Mat) {
        N = Mat->N; M = Mat->M;
        if constexpr(std::is_same_v<double, T>) {
            Mat->CopyDinvU(&val, &cind, &rptr);
        }else {
            double *tval;
            Mat->CopyDinvU(&tval, &cind, &rptr);
            int nnz = rptr[N];
            val = utils::SafeMalloc<T>(nnz);
            utils::Convert<double, T>(tval, val, nnz);
            free(tval);
        }
        temp = utils::SafeMalloc<T>(N);
    }
    ~CSRJa() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
        utils::SafeFree(&temp);
    }
    int GetN() { return N; }
    void SpTRSV(T *in, T *out) {
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
                        T t = in[i];
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

template <int bit, int Iter>
class CSRJa<Fixed<bit>, Iter> : public dutri::SpMat<int> {
    int N, M; // N denots #rows; M denots #columns.
    int *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    int *temp = nullptr;
public:
    CSRJa(CSRMat *Mat) {
        N = Mat->N; M = Mat->M;
        double *tval;
        Mat->CopyDinvU(&tval, &cind, &rptr);
        int nnz = rptr[N];
        val = utils::SafeMalloc<int>(nnz);
        utils::Convert_DI<int, bit>(tval, val, nnz);
        free(tval);
        temp = utils::SafeMalloc<int>(N);
    }
    ~CSRJa() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
        utils::SafeFree(&temp);
    }
    int GetN() { return N; }
    void SpTRSV(int *in, int *out) {
        #pragma omp parallel for
        for(int i=0; i<N; i++) {
            out[i] = (int)((long)in[i] * (long)val[rptr[i]] >> bit);
        }
        if constexpr(Iter > 1) {
            #pragma omp parallel
            {
                for(int k=1; k<Iter; k++) {
                    #pragma omp for
                    for(int i=N-1; i>=0; i--) {
                        long t = (long)in[i] << bit;
                        int j;
                        for(j=rptr[i+1]-1; j>=rptr[i]+1; j--) {
                            t -= (long)val[j] * (long)out[cind[j]];
                        }
                        temp[i] = (int)((t >> bit) * (long)val[j] >> bit);
                    }
                    #pragma omp for
                    for(int i=0; i<N; i++) { out[i] = temp[i]; }
                }
            }
        }
    }
};

/**
 * Sequential BCSR triangular solver
 **/
template <typename T, int Bnl, int Bnw>
class BCSR : public dutri::SpMat<T> {
    int N, M; // N denots #rows; M denots #columns.
    T *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
public:
    BCSR(CSRMat *Mat) {
        N = Mat->N; M = Mat->M;
        double *tval;
        int *tcind, *trptr;
        Mat->CopyDinvU(&tval, &tcind, &trptr);
        if constexpr(std::is_same_v<double, T>) {
            helper::matrix::csr_to_bcsr(
                tval, tcind, trptr, &val, &cind, &rptr, Mat->N, Bnl, Bnw);
            free(tval); free(tcind); free(trptr);
        }else {
            double *tbval;
            helper::matrix::csr_to_bcsr(
                tval, tcind, trptr, &tbval, &cind, &rptr, Mat->N, Bnl, Bnw);
            free(tval); free(tcind); free(trptr);
            int nnz = rptr[N/Bnl]*Bnl*Bnw;
            val = utils::SafeMalloc<T>(nnz);
            utils::Convert<double, T>(tbval, val, nnz);
            free(tbval);
        }
    }
    ~BCSR() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
    }
    int GetN() { return N; }
    void SpTRSV(T *in, T *out) {
        const int u_off = Bnl / Bnw;
        const int Bsize = Bnl * Bnw;
        for(int i=N-Bnl; i>=0; i-=Bnl) {
            int bidx = i / Bnl;
            #pragma omp simd simdlen(Bnl)
            for(int k=0; k<Bnl; k++) { out[i+k] = in[i+k]; }
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

/**
 * Block Jacobi parallel CSR triangular solver
 **/
template <typename T>
class BJCSR : public dutri::SpMat<T> {
    int N, M; // N denots #rows; M denots #columns.
    T *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    int *bj_size, bj_num;
public:
    BJCSR(CSRMat *Mat) {
        if(Mat->order != MMOrder::BJ) {
            printf("In dutri::BJCSR, the order of Mat must be BJ.\n");
            exit(1);
        }
        N = Mat->N; M = Mat->M;
        bj_size = Mat->bj_size;
        bj_num  = Mat->bj_num;
        if constexpr(std::is_same_v<double, T>) {
            Mat->CopyDinvU(&val, &cind, &rptr);
        }else {
            double *tval;
            Mat->CopyDinvU(&tval, &cind, &rptr);
            int nnz = rptr[N];
            val = utils::SafeMalloc<T>(nnz);
            utils::Convert<double, T>(tval, val, nnz);
            free(tval);
        }
    }
    ~BJCSR() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
    }
    int GetN() { return N; }
    void SpTRSV(T *in, T *out) {
        #pragma omp parallel for
        for(int id=0; id<bj_num; id++) {
            int s = bj_size[id];
            int e = bj_size[id+1];
            for(int i=e-1; i>=s; i--) {
                T temp = in[i];
                int j;
                for(j=rptr[i+1]-1; j>=rptr[i]+1; j--) {
                    temp -= val[j] * out[cind[j]];
                }
                out[i] = temp * val[j];
            }
        }
    }
};

template <int bit>
class BJCSR<Fixed<bit>> : public dutri::SpMat<int> {
    int N, M; // N denots #rows; M denots #columns.
    int *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    int *bj_size, bj_num;
public:
    BJCSR(CSRMat *Mat) {
        if(Mat->order != MMOrder::BJ) {
            printf("In dutri::BJCSR, the order of Mat must be BJ.\n");
            exit(1);
        }
        N = Mat->N; M = Mat->M;
        bj_size = Mat->bj_size;
        bj_num  = Mat->bj_num;
        double *tval;
        Mat->CopyDinvU(&tval, &cind, &rptr);
        int nnz = rptr[N];
        val = utils::SafeMalloc<int>(nnz);
        utils::Convert_DI<int, bit>(tval, val, nnz);
        free(tval);
    }
    ~BJCSR() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
    }
    int GetN() { return N; }
    void SpTRSV(int *in, int *out) {
        #pragma omp parallel for
        for(int id=0; id<bj_num; id++) {
            int s = bj_size[id];
            int e = bj_size[id+1];
            for(int i=e-1; i>=s; i--) {
                long temp = (long)in[i] << bit;
                int j;
                for(j=rptr[i+1]-1; j>=rptr[i]+1; j--) {
                    temp -= (long)val[j] * (long)out[cind[j]];
                }
                out[i] = (int)((temp >> bit) * (long)val[j] >> bit);
            }
        }
    }
};

/**
 * Block Jacobi parallel BCSR triangular solver
 **/
template <int Bnl, int Bnw>
class BJBCSR : public dutri::SpMat<double> {
    int N, M; // N denots #rows; M denots #columns.
    double *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    int *bj_size, bj_num;
public:
    BJBCSR(CSRMat *Mat) {
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

class MCCSR : public dutri::SpMat<double> {
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

class MCSELL32 : public dutri::SpMat<double> {
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

class BMCCSR : public dutri::SpMat<double> {
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
class BMCBCSR : public dutri::SpMat<double> {
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
