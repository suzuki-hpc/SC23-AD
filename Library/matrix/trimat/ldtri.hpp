#ifndef SENKPP_MATRIX_TRIMAT_LDTRI_HPP
#define SENKPP_MATRIX_TRIMAT_LDTRI_HPP

namespace senk {

namespace ldtri {

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
class CSR : public ldtri::SpMat<T> {
    int N, M; // N denots #rows; M denots #columns.
    T *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
public:
    CSR(CSRMat *Mat) {
        N = Mat->N; M = Mat->M;
        if constexpr(std::is_same_v<double, T>) {
            helper::matrix::extractLDinv(
                Mat->val, Mat->cind, Mat->rptr,
                &val, &cind, &rptr, Mat->N);
        }else {
            double *tval;
            helper::matrix::extractLDinv(
                Mat->val, Mat->cind, Mat->rptr,
                &tval, &cind, &rptr, Mat->N);
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
            int j;
            for(j=rptr[i]; j<rptr[i+1]-1; j++) {
                t -= val[j] * out[cind[j]];
            }
            out[i] = t * val[j];
        }
    }
};

template <int bit>
class CSR<Fixed<bit>> : public ldtri::SpMat<int> {
    int N, M; // N denots #rows; M denots #columns.
    int *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    const int dbit = 24;
public:
    CSR(CSRMat *Mat) {
        N = Mat->N; M = Mat->M;
        double *tval;
        helper::matrix::extractLDinv(
            Mat->val, Mat->cind, Mat->rptr,
            &tval, &cind, &rptr, Mat->N);
        val = utils::SafeMalloc<int>(rptr[N]);
        // utils::Convert_DI<int, bit>(tval, val, rptr[N]);
        for(int i=0; i<N; i++) {
            for(int j=rptr[i]; j<rptr[i+1]; j++) {
                if(cind[j] == i) val[j] = (int)(tval[j] * (1 << dbit));
                else val[j] = (int)(tval[j] * (1 << bit));
            }
        }
/*
        for(int i=0; i<N; i++) {
            for(int j=rptr[i]; j<rptr[i+1]; j++) {
                if(cind[j] == i) {
                    if(std::fabs(tval[j] - (double)val[j]/(1 << dbit)) > 1.0e-4 ) {
                        printf("%e %e %e xx\n", tval[j], (double)val[j]/(1 << dbit), std::fabs(tval[j] - (double)val[j]/(1 << dbit)));
                    }
                }else {
                    if(std::fabs(tval[j] - (double)val[j]/(1 << bit)) > 1.0e-4 ) {
                        printf("%e %e %e\n", tval[j], (double)val[j]/(1 << bit), std::fabs(tval[j] - (double)val[j]/(1 << bit)));
                    }
                }
            }
        }
*/
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
            int j;
            for(j=rptr[i]; j<rptr[i+1]-1; j++) {
                t -= (long)val[j] * (long)out[cind[j]];
            }
            // out[i] = (int)((t >> bit) * (long)val[j] >> bit);
            out[i] = (int)((t >> bit) * (long)val[j] >> dbit);
            // out[i] = (int)(t / (long)val[j]);
        }
    }
};

class CSRflex : public ldtri::SpMat<int> {
    int N, M; // N denots #rows; M denots #columns.
    int *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    int bit = 0, dbit = 0;
public:
    CSRflex(CSRMat *Mat) {
        N = Mat->N; M = Mat->M;
        double *tval;
        helper::matrix::extractLDinv(
            Mat->val, Mat->cind, Mat->rptr,
            &tval, &cind, &rptr, Mat->N);
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
        dbit = (dmax < 1)? 30 : 32 - ((int)std::ceil(std::log2(int(dmax)+1)) + 1);
        bit = (max < 1)? 30 : 32 - ((int)std::ceil(std::log2(int(max)+1)) + 1);
        // while( dbit < TRIMAT_FLEX_PARAM && dmax * (1 << dbit) < (1 << TRIMAT_FLEX_PARAM) ) {
        //     dbit++;
        // }
        // while( bit < TRIMAT_FLEX_PARAM && max * (1 << bit) < (1 << TRIMAT_FLEX_PARAM) ) {
        //     bit++;
        // }
#if TRIMAT_FLEX_PRINT
        printf("# ldtri %d %d\n", bit, dbit);
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
        for(int i=0; i<N; i++) {
            long t = (long)in[i] << bit;
            int j;
            for(j=rptr[i]; j<rptr[i+1]-1; j++) {
                t -= (long)val[j] * (long)out[cind[j]];
            }
            // out[i] = (int)((t >> bit) * (long)val[j] >> bit);
            out[i] = (int)((t >> bit) * (long)val[j] >> dbit);
            // out[i] = (int)(t / (long)val[j]);
        }
    }
};

template <typename T>
class testCSR : public ldtri::SpMat<T> {
    int N, M; // N denots #rows; M denots #columns.
    T *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    T *temp = nullptr;
public:
    testCSR(CSRMat *Mat) {
        N = Mat->N; M = Mat->M;
        if constexpr(std::is_same_v<double, T>) {
            helper::matrix::extractLDinv(
                Mat->val, Mat->cind, Mat->rptr,
                &val, &cind, &rptr, Mat->N);
        }else {
            double *tval;
            helper::matrix::extractLDinv(
                Mat->val, Mat->cind, Mat->rptr,
                &tval, &cind, &rptr, Mat->N);
            int nnz = rptr[N];
            val = utils::SafeMalloc<T>(nnz);
            utils::Convert<double, T>(tval, val, nnz);
            free(tval);
        }
        temp = utils::SafeMalloc<T>(N);
    }
    ~testCSR() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
        utils::SafeFree(&temp);
    }
    int GetN() { return N; }
    void SpTRSV(T *in, T *out) {
        int size = N/10;
        #pragma omp parallel
        {
            for(int k=0; k<N; k+=size) {
                #pragma omp for
                for(int i=0; i<size; i++) {
                    if(k+i >= N) continue;
                    T t = in[k+i];
                    int j;
                    for(j=rptr[k+i]; j<rptr[k+i+1]-1; j++) {
                        t -= val[j] * out[cind[j]];
                    }
                    temp[k+i] = t * val[j];
                }
                #pragma omp for
                for(int i=0; i<size; i++) {
                    if(k+i >= N) continue;
                    out[k+i] = temp[k+i];
                }
            }
        }
    }
};

/**
 * Syncronus Jacobi CSR triangular solver
 **/
template <typename T, int Iter>
class CSRJa : public ldtri::SpMat<T> {
    int N, M; // N denots #rows; M denots #columns.
    T *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    T *temp = nullptr;
public:
    CSRJa(CSRMat *Mat) {
        N = Mat->N; M = Mat->M;
        if constexpr(std::is_same_v<double, T>) {
            helper::matrix::extractLDinv(
                Mat->val, Mat->cind, Mat->rptr,
                &val, &cind, &rptr, Mat->N);
        }else {
            double *tval;
            helper::matrix::extractLDinv(
                Mat->val, Mat->cind, Mat->rptr,
                &tval, &cind, &rptr, Mat->N);
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
            out[i] = (in[i]) * val[rptr[i+1]-1];
        }
        if constexpr(Iter > 1) {
            #pragma omp parallel
            {
                for(int k=1; k<Iter; k++) {
                    #pragma omp for
                    for(int i=0; i<N; i++) {
                        T t = in[i];
                        int j;
                        for(j=rptr[i]; j<rptr[i+1]-1; j++) {
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
class CSRJa<Fixed<bit>, Iter> : public ldtri::SpMat<int> {
    int N, M; // N denots #rows; M denots #columns.
    int *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    int *temp = nullptr;
public:
    CSRJa(CSRMat *Mat) {
        N = Mat->N; M = Mat->M;
        double *tval;
        helper::matrix::extractLDinv(
            Mat->val, Mat->cind, Mat->rptr,
            &tval, &cind, &rptr, Mat->N);
        val = utils::SafeMalloc<int>(rptr[N]);
        utils::Convert_DI<int, bit>(tval, val, rptr[N]);
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
            out[i] = (int)((long)in[i] * (long)val[rptr[i+1]-1] >> bit);
        }
        if constexpr(Iter > 1) {
            #pragma omp parallel
            {
                for(int k=1; k<Iter; k++) {
                    #pragma omp for
                    for(int i=0; i<N; i++) {
                        long t = (long)in[i] << bit;
                        int j;
                        for(j=rptr[i]; j<rptr[i+1]-1; j++) {
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
 * Block Jacobi parallel CSR triangular solver
 **/
template <typename T>
class BJCSR : public ldtri::SpMat<T> {
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
            helper::matrix::extractLDinv(
                Mat->val, Mat->cind, Mat->rptr,
                &val, &cind, &rptr, Mat->N);
        }else {
            double *tval;
            helper::matrix::extractLDinv(
                Mat->val, Mat->cind, Mat->rptr,
                &tval, &cind, &rptr, Mat->N);
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
                int j;
                for(j=rptr[i]; j<rptr[i+1]-1; j++) {
                    temp -= val[j] * out[cind[j]];
                }
                out[i] = temp * val[j];
            }
        }
    }
};

template <int bit>
class BJCSR<Fixed<bit>> : public ldtri::SpMat<int> {
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
        helper::matrix::extractLDinv(
            Mat->val, Mat->cind, Mat->rptr,
            &tval, &cind, &rptr, Mat->N);
        val = utils::SafeMalloc<int>(rptr[N]);
        utils::Convert_DI<int, bit>(tval, val, rptr[N]);
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
                int j;
                for(j=rptr[i]; j<rptr[i+1]-1; j++) {
                    temp -= (long)val[j] * (long)out[cind[j]];
                }
                out[i] = (int)((temp >> bit) * (long)val[j] >> bit);
            }
        }
    }
};

class BJCSRflex : public ldtri::SpMat<int> {
    int N, M; // N denots #rows; M denots #columns.
    int *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    int *bj_size, bj_num;
    int bit = 0, dbit = 0;
public:
    BJCSRflex(CSRMat *Mat) {
        N = Mat->N; M = Mat->M;
        bj_size = Mat->bj_size;
        bj_num  = Mat->bj_num;
        double *tval;
        helper::matrix::extractLDinv(
            Mat->val, Mat->cind, Mat->rptr,
            &tval, &cind, &rptr, Mat->N);
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
        dbit = (dmax < 1)? 30 : 32 - ((int)std::ceil(std::log2(int(dmax)+1)) + 1);
        bit = (max < 1)? 30 : 32 - ((int)std::ceil(std::log2(int(max)+1)) + 1);
#if TRIMAT_FLEX_PRINT
        printf("# ldtri %d %d\n", bit, dbit);
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
    ~BJCSRflex() {
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
                int j;
                for(j=rptr[i]; j<rptr[i+1]-1; j++) {
                    temp -= (long)val[j] * (long)out[cind[j]];
                }
                out[i] = (int)((temp >> bit) * (long)val[j] >> dbit);
            }
        }
    }
};

template <typename T>
class MCCSR : public ldtri::SpMat<T> {
    int N, M; // N denots #rows; M denots #columns.
    T *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    int *c_size, c_num;
public:
    MCCSR(CSRMat *Mat) {
        if(Mat->order != MMOrder::MC) {
            printf("In ldtri::MCCSR, the order of Mat must be MC.\n");
            exit(1);
        }
        N = Mat->N; M = Mat->M;
        c_size = Mat->c_size;
        c_num  = Mat->c_num;
        if constexpr(std::is_same_v<double, T>) {
            helper::matrix::extractLDinv(
                Mat->val, Mat->cind, Mat->rptr,
                &val, &cind, &rptr, Mat->N);
        }else {
            double *tval;
            helper::matrix::extractLDinv(
                Mat->val, Mat->cind, Mat->rptr,
                &tval, &cind, &rptr, Mat->N);
            int nnz = rptr[N];
            val = utils::SafeMalloc<T>(nnz);
            utils::Convert<double, T>(tval, val, nnz);
            free(tval);
        }
    }
    ~MCCSR() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
    }
    int GetN() { return N; }
    void SpTRSV(T *in, T *out) {
        #pragma omp parallel
        {
            for(int id=0; id<c_num; id++) {
                int s = c_size[id];
                int e = c_size[id+1];
                #pragma omp for
                for(int i=s; i<e; i++) {
                    T t = in[i];
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

template <int bit>
class MCCSR<Fixed<bit>> : public ldtri::SpMat<int> {
    int N, M; // N denots #rows; M denots #columns.
    int *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    int *c_size, c_num;
public:
    MCCSR(CSRMat *Mat) {
        if(Mat->order != MMOrder::MC) {
            printf("In ldtri::MCCSR, the order of Mat must be MC.\n");
            exit(1);
        }
        N = Mat->N; M = Mat->M;
        c_size = Mat->c_size;
        c_num  = Mat->c_num;
        double *tval;
        helper::matrix::extractLDinv(
            Mat->val, Mat->cind, Mat->rptr,
            &tval, &cind, &rptr, Mat->N);
        val = utils::SafeMalloc<int>(rptr[N]);
        utils::Convert_DI<int, bit>(tval, val, rptr[N]);
        free(tval);
    }
    ~MCCSR() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
    }
    int GetN() { return N; }
    void SpTRSV(int *in, int *out) {
        #pragma omp parallel
        {
            for(int id=0; id<c_num; id++) {
                int s = c_size[id];
                int e = c_size[id+1];
                #pragma omp for
                for(int i=s; i<e; i++) {
                    long t = (long)in[i] << bit;
                    int j;
                    for(j=rptr[i]; j<rptr[i+1]-1; j++) {
                        t -= (long)val[j] * (long)out[cind[j]];
                    }
                    out[i] = (int)((t >> bit) * (long)val[j] >> bit);
                }
            }
        }
    }
};

class MCCSRflex : public ldtri::SpMat<int> {
    int N, M; // N denots #rows; M denots #columns.
    int *val = nullptr;
    int *cind = nullptr;
    int *rptr = nullptr;
    int *c_size, c_num;
    int bit = 0, dbit = 0;
public:
    MCCSRflex(CSRMat *Mat) {
        N = Mat->N; M = Mat->M;
        c_size = Mat->c_size;
        c_num  = Mat->c_num;
        double *tval;
        helper::matrix::extractLDinv(
            Mat->val, Mat->cind, Mat->rptr,
            &tval, &cind, &rptr, Mat->N);
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
        dbit = (dmax < 1)? 30 : 32 - ((int)std::ceil(std::log2(int(dmax)+1)) + 1);
        bit = (max < 1)? 30 : 32 - ((int)std::ceil(std::log2(int(max)+1)) + 1);
#if TRIMAT_FLEX_PRINT
        printf("# ldtri %d %d\n", bit, dbit);
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
    ~MCCSRflex() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
    }
    int GetN() { return N; }
    void SpTRSV(int *in, int *out) {
        #pragma omp parallel
        {
            for(int id=0; id<c_num; id++) {
                int s = c_size[id];
                int e = c_size[id+1];
                #pragma omp for
                for(int i=s; i<e; i++) {
                    long t = (long)in[i] << bit;
                    int j;
                    for(j=rptr[i]; j<rptr[i+1]-1; j++) {
                        t -= (long)val[j] * (long)out[cind[j]];
                    }
                    out[i] = (int)((t >> bit) * (long)val[j] >> dbit);
                }
            }
        }
    }
};

class BMCCSR : public ldtri::SpMat<double> {
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

} // senk

#endif
