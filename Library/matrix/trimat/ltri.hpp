#ifndef SENKPP_MATRIX_TRIMAT_LTRI_HPP
#define SENKPP_MATRIX_TRIMAT_LTRI_HPP

namespace senk {

namespace ltri {

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
class CSR : public ltri::SpMat<T> {
    int N, M; // N denots #rows; M denots #columns.
    T *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
public:
    CSR(CSRMat *Mat) {
        N = Mat->N; M = Mat->M;
        if constexpr(std::is_same_v<double, T>) {
            Mat->CopyL(&val, &cind, &rptr);
        }else {
            double *tval;
            Mat->CopyL(&tval, &cind, &rptr);
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
        for(int i=0; i<N; i++) {
            T t = in[i];
            for(int j=rptr[i]; j<rptr[i+1]; j++) {
                t -= val[j] * out[cind[j]];
            }
            out[i] = t;
        }
    }
};

template <int bit>
class CSR<Fixed<bit>> : public ltri::SpMat<int> {
    int N, M; // N denots #rows; M denots #columns.
    int *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
public:
    CSR(CSRMat *Mat) {
        N = Mat->N; M = Mat->M;
        double *tval;
        Mat->CopyL(&tval, &cind, &rptr);
        val = utils::SafeMalloc<int>(rptr[N]);
        utils::Convert_DI<int, bit>(tval, val, rptr[N]);
        free(tval);
    }
    ~CSR() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
    }
    int GetN() { return N; }
    void SpTRSV(int *in, int *out) {
        for(int i=0; i<N; i++) {
            long t = (long)in[i] << bit;
            for(int j=rptr[i]; j<rptr[i+1]; j++) {
                t -= (long)val[j] * (long)out[cind[j]];
            }
            out[i] = (int)(t >> bit);
        }
    }
};

class CSRflex : public ltri::SpMat<int> {
    int N, M; // N denots #rows; M denots #columns.
    int *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    int bit = 0;
public:
    CSRflex(CSRMat *Mat) {
        N = Mat->N; M = Mat->M;
        double *tval;
        Mat->CopyL(&tval, &cind, &rptr);
        val = utils::SafeMalloc<int>(rptr[N]);
        double max = 0;
        for(int i=0; i<N; i++) {
            for(int j=rptr[i]; j<rptr[i+1]; j++) {
                if(std::fabs(tval[j]) > max) max = std::fabs(tval[j]);
            }
        }
        while( bit < TRIMAT_FLEX_PARAM && max * (1 << bit) < (1 << TRIMAT_FLEX_PARAM) ) {
            bit++;
        }
#if TRIMAT_FLEX_PRINT
        printf("# ltri %d\n", bit);
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
    void SpTRSV(int *in, int *out) {
        for(int i=0; i<N; i++) {
            long t = (long)in[i] << bit;
            for(int j=rptr[i]; j<rptr[i+1]; j++) {
                t -= (long)val[j] * (long)out[cind[j]];
            }
            out[i] = (int)(t >> bit);
        }
    }
};

/**
 * Syncronus Jacobi CSR triangular solver
 **/
template <typename T, int Iter>
class CSRJa : public ltri::SpMat<T> {
    int N, M; // N denots #rows; M denots #columns.
    T *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    T *temp = nullptr;
public:
    CSRJa(CSRMat *Mat) {
        N = Mat->N; M = Mat->M;
        if constexpr(std::is_same_v<double, T>) {
            Mat->CopyL(&val, &cind, &rptr);
        }else {
            double *tval;
            Mat->CopyL(&tval, &cind, &rptr);
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
        for(int i=0; i<N; i++) { out[i] = in[i]; }
        if constexpr(Iter > 1) {
            #pragma omp parallel
            {
                for(int k=1; k<Iter; k++) {
                    #pragma omp for
                    for(int i=0; i<N; i++) {
                        T t = in[i];
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

template <int bit, int Iter>
class CSRJa<Fixed<bit>, Iter> : public ltri::SpMat<int> {
    int N, M; // N denots #rows; M denots #columns.
    int *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    int *temp = nullptr;
public:
    CSRJa(CSRMat *Mat) {
        N = Mat->N; M = Mat->M;
        double *tval;
        Mat->CopyL(&tval, &cind, &rptr);
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
        for(int i=0; i<N; i++) { out[i] = in[i]; }
        if constexpr(Iter > 1) {
            #pragma omp parallel
            {
                for(int k=1; k<Iter; k++) {
                    #pragma omp for
                    for(int i=0; i<N; i++) {
                        long t = (long)in[i] << bit;
                        for(int j=rptr[i]; j<rptr[i+1]; j++) {
                            t -= (long)val[j] * (long)out[cind[j]];
                        }
                        temp[i] = (int)(t >> bit);
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
class BCSR : public ltri::SpMat<T> {
    int N, M; // N denots #rows; M denots #columns.
    T *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
public:
    BCSR(CSRMat *Mat) {
        N = Mat->N; M = Mat->M;
        double *tval;
        int *tcind, *trptr;
        Mat->CopyL(&tval, &tcind, &trptr);
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

/**
 * Block Jacobi parallel CSR triangular solver
 **/
template <typename T>
class BJCSR : public ltri::SpMat<T> {
    int N, M; // N denots #rows; M denots #columns.
    T *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    int *bj_size, bj_num;
public:
    BJCSR(CSRMat *Mat) {
        if(Mat->order != MMOrder::BJ) {
            printf("In ltri::BJCSR, the order of Mat must be BJ.\n");
            exit(1);
        }
        N = Mat->N; M = Mat->M;
        bj_size = Mat->bj_size;
        bj_num  = Mat->bj_num;
        if constexpr(std::is_same_v<double, T>) {
            Mat->CopyL(&val, &cind, &rptr);
        }else {
            double *tval;
            Mat->CopyL(&tval, &cind, &rptr);
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
            for(int i=s; i<e; i++) {
                T temp = in[i];
                for(int j=rptr[i]; j<rptr[i+1]; j++) {
                    temp -= val[j] * out[cind[j]];
                }
                out[i] = temp;
            }
        }
    }
};

template <int bit>
class BJCSR<Fixed<bit>> : public ltri::SpMat<int> {
    int N, M; // N denots #rows; M denots #columns.
    int *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    int *bj_size, bj_num;
public:
    BJCSR(CSRMat *Mat) {
        if(Mat->order != MMOrder::BJ) {
            printf("In ltri::BJCSR, the order of Mat must be BJ.\n");
            exit(1);
        }
        N = Mat->N; M = Mat->M;
        bj_size = Mat->bj_size;
        bj_num  = Mat->bj_num;
        double *tval;
        Mat->CopyL(&tval, &cind, &rptr);
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
            for(int i=s; i<e; i++) {
                long temp = (long)in[i] << bit;
                for(int j=rptr[i]; j<rptr[i+1]; j++) {
                    temp -= (long)val[j] * (long)out[cind[j]];
                }
                out[i] = (int)(temp >> bit);
            }
        }
    }
};

/**
 * Block Jacobi parallel BCSR triangular solver
 **/
template <int Bnl, int Bnw>
class BJBCSR : public ltri::SpMat<double> {
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

class MCCSR : public ltri::SpMat<double> {
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

class MCSELL32 : public ltri::SpMat<double> {
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

class BMCCSR : public ltri::SpMat<double> {
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
class BMCBCSR : public ltri::SpMat<double> {
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

} // senk

#endif
