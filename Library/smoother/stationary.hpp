#ifndef SENKPP_SMOOTHER_STATIONARY_HPP
#define SENKPP_SMOOTHER_STATIONARY_HPP

#include "parameters.hpp"
#include "blas/blas1.hpp"
#include "matrix/spmat.hpp"
#include "matrix/trimat/spmat.hpp"
#include "smoother/smoother.hpp"

namespace senk {

template <class ClassLU>
class Jacobi {};

template <
    template <typename T, int...> class ClassLU,
    typename T, int... args>
class Jacobi<ClassLU<T, args...>> : public Smoother<T> {
private:
    SpMat<T> *LU;
    T *d_inv;
    int iter;
    T *temp;
public:
    Jacobi(
        CSRMat *Mat, SmootherParam* _param,
        SpMat<T> *dummy=nullptr)
    {
        JacobiParam *param = dynamic_cast<JacobiParam*>(_param);
        if(!param) {
            std::cout << "In Jacobi, the input parameter is wrong.\n";
            exit(EXIT_FAILURE);
        }
        iter = param->iter;
        temp = utils::SafeMalloc<T>(Mat->N);

        T omega = param->omega;
        d_inv = utils::SafeMalloc<T>(Mat->N);
        for(int i=0; i<Mat->N; i++) {
            for(int j=Mat->rptr[i]; j<Mat->rptr[i+1]; j++) {
                if(Mat->cind[j] == i) {
                    d_inv[i] = omega/Mat->val[j];
                    break;
                }
            }
        }
        LU = new ClassLU<T, args...>(Mat);
    }
    ~Jacobi() {
        delete LU;
        utils::SafeFree(&temp);
        utils::SafeFree(&d_inv);
    }
    void Smooth(T *in, T *out, bool isInited) {
        if (isInited) {
            for(int k=0; k<iter; k++) {
                LU->SpMV(out, temp);
                #pragma omp parallel for
                for(int i=0; i<LU->GetN(); i++) {
                    out[i] = out[i] + d_inv[i]*(in[i]-temp[i]);
                }
            }
        }else {
            #pragma omp parallel for
            for(int i=0; i<LU->GetN(); i++) {
                out[i] = d_inv[i] * in[i];
            }
            for(int k=1; k<iter; k++) {
                LU->SpMV(out, temp);
                #pragma omp parallel for
                for(int i=0; i<LU->GetN(); i++) {
                    out[i] = out[i] + d_inv[i]*(in[i]-temp[i]);
                }
            }
        }
    }
    void Precondition(T *in, T *out) {
        #pragma omp parallel for
        for(int i=0; i<LU->GetN(); i++) {
            out[i] = d_inv[i] * in[i];
        }
        for(int k=1; k<iter; k++) {
            LU->SpMV(out, temp);
            #pragma omp parallel for
            for(int i=0; i<LU->GetN(); i++) {
                out[i] = out[i] + d_inv[i]*(in[i]-temp[i]);
            }
        }
    }
};

template <
    template <typename T, int...> class ClassLU,
    int bit, int... args>
class Jacobi<ClassLU<Fixed<bit>, args...>> : public Smoother<int> {
private:
    SpMat<int> *LU;
    int *d_inv;
    int iter;
    int *temp;
public:
    Jacobi(
        CSRMat *Mat, SmootherParam* _param,
        SpMat<int> *dummy=nullptr)
    {
        JacobiParam *param = dynamic_cast<JacobiParam*>(_param);
        if(!param) {
            std::cout << "In Jacobi, the input parameter is wrong.\n";
            exit(EXIT_FAILURE);
        }
        iter = param->iter;
        temp = utils::SafeMalloc<int>(Mat->N);

        d_inv = utils::SafeMalloc<int>(Mat->N);
        double fact = (double)(1 << bit);
        for(int i=0; i<Mat->N; i++) {
            for(int j=Mat->rptr[i]; j<Mat->rptr[i+1]; j++) {
                if(Mat->cind[j] == i) {
                    d_inv[i] = (int)(param->omega/Mat->val[j] * fact);
                    break;
                }
            }
        }
        LU = new ClassLU<Fixed<bit>, args...>(Mat);
    }
    ~Jacobi() {
        delete LU;
        utils::SafeFree(&temp);
        utils::SafeFree(&d_inv);
    }
    void Smooth(int *in, int *out, bool isInited) {
        if (isInited) {
            for(int k=0; k<iter; k++) {
                LU->SpMV(out, temp);
                #pragma omp parallel for
                for(int i=0; i<LU->GetN(); i++) {
                    out[i] = out[i] + d_inv[i]*(in[i]-temp[i]);
                }
            }
        }else {
            #pragma omp parallel for
            for(int i=0; i<LU->GetN(); i++) {
                out[i] = d_inv[i] * in[i];
            }
            for(int k=1; k<iter; k++) {
                LU->SpMV(out, temp);
                #pragma omp parallel for
                for(int i=0; i<LU->GetN(); i++) {
                    out[i] += (int)((long)d_inv[i]*(long)(in[i]-temp[i]) >> bit);
                }
            }
        }
    }
    void Precondition(int *in, int *out) {
        #pragma omp parallel for
        for(int i=0; i<LU->GetN(); i++) {
            out[i] = (int)((long)d_inv[i] * (long)in[i] >> bit);
        }
        for(int k=1; k<iter; k++) {
            LU->SpMV(out, temp);
            #pragma omp parallel for
            for(int i=0; i<LU->GetN(); i++) {
                out[i] += (int)((long)d_inv[i]*(long)(in[i]-temp[i]) >> bit);
            }
        }
    }
};

/**
 * SOR
 **/
template <class ClassLD, class ClassU>
class SOR {};

template <
    template <typename, int...> class ClassLD,
    template <typename, int...> class ClassU,
    typename T, int... args1, int... args2>
class SOR<ClassLD<T, args1...>, ClassU<T, args2...>> : public Smoother<T> {
private:
    ldtri::SpMat<T> *LD;
    SpMat<T> *U;
    int iter;
    T *temp;
public:
    SOR(
        CSRMat *Mat, SmootherParam* _param,
        SpMat<T> *dummy=nullptr)
    {
        SORParam *param = dynamic_cast<SORParam*>(_param);
        if(!param) {
            std::cout << "In SOR, the input parameter is wrong.\n";
            exit(EXIT_FAILURE);
        }
        iter = param->iter;
        temp = utils::SafeMalloc<T>(Mat->N);

        double *lval, *uval;
        int *lcind, *lrptr, *ucind, *urptr;
        helper::matrix::split<double>(
            Mat->val, Mat->cind, Mat->rptr,
            &lval, &lcind, &lrptr, &uval, &ucind, &urptr, nullptr,
            Mat->N, "LD-U", false);

        uval  = utils::SafeRealloc<double>(uval, urptr[Mat->N]+Mat->N);
        ucind = utils::SafeRealloc<int>(ucind, urptr[Mat->N]+Mat->N);

        double omega = param->omega;
        for(int i=Mat->N-1; i>=0; i--) {
            int j;
            for(j=urptr[i+1]-1; j>=urptr[i]; j--) {
                uval[j+i+1] = uval[j];
                ucind[j+i+1] = ucind[j];
            }
            uval[j+i+1] = (1-1.0/omega)*lval[lrptr[i+1]-1];
            ucind[j+i+1] = lcind[lrptr[i+1]-1];
            lval[lrptr[i+1]-1] = 1.0/omega*lval[lrptr[i+1]-1];
            urptr[i+1] += i+1;
        }

        CSRMat *tLD = new CSRMat(
            lval, lcind, lrptr, Mat->N, Mat->M, Mat->shape, Mat->type, Mat->order,
            Mat->b_size, Mat->c_size, Mat->c_num, Mat->bj_size, Mat->bj_num);
        CSRMat *tU  = new CSRMat(
            uval, ucind, urptr, Mat->N, Mat->M, Mat->shape, Mat->type);

        LD = new ClassLD<T, args1...>(tLD);
        U  = new ClassU<T, args2...>(tU);
        
        tLD->Free(); delete tLD;
        tU->Free();  delete tU;
    }
    ~SOR() {
        delete LD;
        delete U;
        utils::SafeFree(&temp);
    }
    void Smooth(T *in, T *out, bool isInited) {
        if(isInited) {
            for(int k=0; k<iter; k++) {
                U->SpMV(out, temp);
                blas1::Axpby<T>(1, in, -1, temp, U->GetN());
                LD->SpTRSV(temp, out);
            }
        }else {
            // #pragma omp parallel for
            // for(int k=0; k<U->GetN(); k++) { out[k] = 0; }
            LD->SpTRSV(in, out);
            for(int k=1; k<iter; k++) {
                U->SpMV(out, temp);
                blas1::Axpby<T>(1, in, -1, temp, U->GetN());
                LD->SpTRSV(temp, out);
            }
        }
    }
    void Precondition(T *in, T *out) {
        LD->SpTRSV(in, out);
        for(int k=1; k<iter; k++) {
            U->SpMV(out, temp);
            blas1::Axpby<T>(1, in, -1, temp, U->GetN());
            LD->SpTRSV(temp, out);
        }
    }
};

template <
    template <typename, int...> class ClassLD,
    template <typename, int...> class ClassU,
    int bit, int... args1, int... args2>
class SOR<ClassLD<Fixed<bit>, args1...>, ClassU<Fixed<bit>, args2...>> : public Smoother<int> {
private:
    ldtri::SpMat<int> *LD;
    SpMat<int> *U;
    int iter;
    int *temp;
public:
    SOR(
        CSRMat *Mat, SmootherParam* _param,
        SpMat<int> *dummy=nullptr)
    {
        SORParam *param = dynamic_cast<SORParam*>(_param);
        if(!param) {
            std::cout << "In SOR, the input parameter is wrong.\n";
            exit(EXIT_FAILURE);
        }
        iter = param->iter;
        temp = utils::SafeMalloc<int>(Mat->N);

        double *lval, *uval;
        int *lcind, *lrptr, *ucind, *urptr;
        helper::matrix::split<double>(
            Mat->val, Mat->cind, Mat->rptr,
            &lval, &lcind, &lrptr, &uval, &ucind, &urptr, nullptr,
            Mat->N, "LD-U", false);

        uval  = utils::SafeRealloc<double>(uval, urptr[Mat->N]+Mat->N);
        ucind = utils::SafeRealloc<int>(ucind, urptr[Mat->N]+Mat->N);

        double omega = param->omega;
        for(int i=Mat->N-1; i>=0; i--) {
            int j;
            for(j=urptr[i+1]-1; j>=urptr[i]; j--) {
                uval[j+i+1] = uval[j];
                ucind[j+i+1] = ucind[j];
            }
            uval[j+i+1] = (1-1.0/omega)*lval[lrptr[i+1]-1];
            ucind[j+i+1] = lcind[lrptr[i+1]-1];
            lval[lrptr[i+1]-1] = 1.0/omega*lval[lrptr[i+1]-1];
            urptr[i+1] += i+1;
        }

        CSRMat *tLD = new CSRMat(
            lval, lcind, lrptr, Mat->N, Mat->M, Mat->shape, Mat->type, Mat->order,
            Mat->b_size, Mat->c_size, Mat->c_num, Mat->bj_size, Mat->bj_num);
        CSRMat *tU  = new CSRMat(
            uval, ucind, urptr, Mat->N, Mat->M, Mat->shape, Mat->type);

        LD = new ClassLD<Fixed<bit>, args1...>(tLD);
        U  = new ClassU<Fixed<bit>, args2...>(tU);
        
        tLD->Free(); delete tLD;
        tU->Free();  delete tU;
    }
    ~SOR() {
        delete LD;
        delete U;
        utils::SafeFree(&temp);
    }
    void Smooth(int *in, int *out, bool isInited) {
        if(isInited) {
            for(int k=0; k<iter; k++) {
                U->SpMV(out, temp);
                blas1::Axpby<int>(1, in, -1, temp, U->GetN());
                LD->SpTRSV(temp, out);
            }
        }else {
            LD->SpTRSV(in, out);
            for(int k=1; k<iter; k++) {
                U->SpMV(out, temp);
                blas1::Axpby<int>(1, in, -1, temp, U->GetN());
                LD->SpTRSV(temp, out);
            }
        }
    }
    void Precondition(int *in, int *out) {
        LD->SpTRSV(in, out);
        for(int k=1; k<iter; k++) {
            U->SpMV(out, temp);
            blas1::Axpby<int>(1, in, -1, temp, U->GetN());
            LD->SpTRSV(temp, out);
        }
    }
};

template <>
class SOR<ldtri::CSRflex, CSRflex> : public Smoother<int> {
private:
    ldtri::SpMat<int> *LD;
    SpMat<int> *U;
    int iter;
    int *temp;
public:
    SOR(
        CSRMat *Mat, SmootherParam* _param,
        SpMat<int> *dummy=nullptr)
    {
        SORParam *param = dynamic_cast<SORParam*>(_param);
        if(!param) {
            std::cout << "In SOR, the input parameter is wrong.\n";
            exit(EXIT_FAILURE);
        }
        iter = param->iter;
        temp = utils::SafeMalloc<int>(Mat->N);

        double *lval, *uval;
        int *lcind, *lrptr, *ucind, *urptr;
        helper::matrix::split<double>(
            Mat->val, Mat->cind, Mat->rptr,
            &lval, &lcind, &lrptr, &uval, &ucind, &urptr, nullptr,
            Mat->N, "LD-U", false);

        uval  = utils::SafeRealloc<double>(uval, urptr[Mat->N]+Mat->N);
        ucind = utils::SafeRealloc<int>(ucind, urptr[Mat->N]+Mat->N);

        double omega = param->omega;
        for(int i=Mat->N-1; i>=0; i--) {
            int j;
            for(j=urptr[i+1]-1; j>=urptr[i]; j--) {
                uval[j+i+1] = uval[j];
                ucind[j+i+1] = ucind[j];
            }
            uval[j+i+1] = (1-1.0/omega)*lval[lrptr[i+1]-1];
            ucind[j+i+1] = lcind[lrptr[i+1]-1];
            lval[lrptr[i+1]-1] = 1.0/omega*lval[lrptr[i+1]-1];
            urptr[i+1] += i+1;
        }

        CSRMat *tLD = new CSRMat(
            lval, lcind, lrptr, Mat->N, Mat->M, Mat->shape, Mat->type, Mat->order,
            Mat->b_size, Mat->c_size, Mat->c_num, Mat->bj_size, Mat->bj_num);
        CSRMat *tU  = new CSRMat(
            uval, ucind, urptr, Mat->N, Mat->M, Mat->shape, Mat->type);

        LD = new ldtri::CSRflex(tLD);
        U  = new CSRflex(tU);
        
        tLD->Free(); delete tLD;
        tU->Free();  delete tU;
    }
    ~SOR() {
        delete LD;
        delete U;
        utils::SafeFree(&temp);
    }
    void Smooth(int *in, int *out, bool isInited) {
        if(isInited) {
            for(int k=0; k<iter; k++) {
                U->SpMV(out, temp);
                blas1::Axpby<int>(1, in, -1, temp, U->GetN());
                LD->SpTRSV(temp, out);
            }
        }else {
            LD->SpTRSV(in, out);
            for(int k=1; k<iter; k++) {
                U->SpMV(out, temp);
                blas1::Axpby<int>(1, in, -1, temp, U->GetN());
                LD->SpTRSV(temp, out);
            }
        }
    }
    void Precondition(int *in, int *out) {
        LD->SpTRSV(in, out);
        for(int k=1; k<iter; k++) {
            U->SpMV(out, temp);
            blas1::Axpby<int>(1, in, -1, temp, U->GetN());
            LD->SpTRSV(temp, out);
        }
    }
};

/**
 * Jacobi SOR
 **/
template <class ClassLD, class ClassU>
class JacobiSOR {};

template <
    template <typename, int...> class ClassLD,
    template <typename, int...> class ClassU,
    typename T, int... args1, int... args2>
class JacobiSOR<ClassLD<T, args1...>, ClassU<T, args2...>> : public Smoother<T> {
private:
    ldtri::SpMat<T> *LD;
    SpMat<T> *U;
    int iter;
    T *temp;
    int *c_size, c_num;
public:
    JacobiSOR(
        CSRMat *Mat, SmootherParam* _param,
        SpMat<T> *dummy=nullptr)
    {
        SORParam *param = dynamic_cast<SORParam*>(_param);
        if(!param) {
            std::cout << "In SOR, the input parameter is wrong.\n";
            exit(EXIT_FAILURE);
        }
        int N = Mat->N;
        iter = param->iter;
        temp = utils::SafeMalloc<T>(N);

        int size = 1200;
        c_num = (N+size-1)/size;
        // size = (N+c_num-1)/c_num;

        c_size = utils::SafeMalloc<int>(c_num+1);
        c_size[0] = 0;
        for(int i=0; i<c_num; i++) {
            c_size[i+1] = c_size[i] + size;
        }
        c_size[c_num] = N;

        int lnnz = 0, unnz = 0;
        for(int i=0; i<N; i++) {
            int left = c_size[i/size];
            for(int j=Mat->rptr[i]; j<Mat->rptr[i+1]; j++) {
                if(Mat->cind[j] == i) { lnnz++; unnz++; }
                else if(Mat->cind[j] < left) { lnnz++; }
                else { unnz++; }
            }
        }
        double *lval = utils::SafeMalloc<double>(lnnz);
        int *lcind   = utils::SafeMalloc<int>(lnnz);
        int *lrptr   = utils::SafeMalloc<int>(N+1);
        double *uval = utils::SafeMalloc<double>(unnz);
        int *ucind   = utils::SafeMalloc<int>(unnz);
        int *urptr   = utils::SafeMalloc<int>(N+1);

        double omega = param->omega;
        lrptr[0] = 0; urptr[0] = 0;
        lnnz = 0; unnz = 0;
        for(int i=0; i<N; i++) {
            int left = c_size[i/size];
            for(int j=Mat->rptr[i]; j<Mat->rptr[i+1]; j++) {
                if(Mat->cind[j] == i) {
                    lval[lnnz] = (1.0/omega)*Mat->val[j]; lcind[lnnz] = Mat->cind[j];
                    lnnz++;
                    uval[unnz] = (1-1.0/omega)*Mat->val[j]; ucind[unnz] = Mat->cind[j];
                    unnz++;
                }else if(Mat->cind[j] < left) {
                    lval[lnnz] = Mat->val[j]; lcind[lnnz] = Mat->cind[j];
                    lnnz++;
                }else {
                    uval[unnz] = Mat->val[j]; ucind[unnz] = Mat->cind[j];
                    unnz++;
                }
            }
            lrptr[i+1] = lnnz; urptr[i+1] = unnz;
        }

        CSRMat *tLD = new CSRMat(
            lval, lcind, lrptr, Mat->N, Mat->M, Mat->shape, Mat->type, MMOrder::MC,
            1, c_size, c_num, nullptr, 0);
        CSRMat *tU  = new CSRMat(
            uval, ucind, urptr, Mat->N, Mat->M, Mat->shape, Mat->type);

        LD = new ClassLD<T, args1...>(tLD);
        U  = new ClassU<T, args2...>(tU);
        
        tLD->Free(); delete tLD;
        tU->Free();  delete tU;
    }
    ~JacobiSOR() {
        delete LD;
        delete U;
        utils::SafeFree(&temp);
    }
    void Smooth(T *in, T *out, bool isInited) {
        if(isInited) {
            for(int k=0; k<iter; k++) {
                U->SpMV(out, temp);
                blas1::Axpby<T>(1, in, -1, temp, U->GetN());
                LD->SpTRSV(temp, out);
            }
        }else {
            LD->SpTRSV(in, out);
            for(int k=1; k<iter; k++) {
                U->SpMV(out, temp);
                blas1::Axpby<T>(1, in, -1, temp, U->GetN());
                LD->SpTRSV(temp, out);
            }
        }
    }
    void Precondition(T *in, T *out) {
        LD->SpTRSV(in, out);
        for(int k=1; k<iter; k++) {
            U->SpMV(out, temp);
            blas1::Axpby<T>(1, in, -1, temp, U->GetN());
            LD->SpTRSV(temp, out);
        }
    }
};

template <class ClassLD, class ClassU>
class JacobiSORflex : public Smoother<int> {
private:
    ldtri::SpMat<int> *LD;
    SpMat<int> *U;
    int iter;
    int *temp;
    int *c_size, c_num;
public:
    JacobiSORflex(
        CSRMat *Mat, SmootherParam* _param,
        SpMat<int> *dummy=nullptr)
    {
        SORParam *param = dynamic_cast<SORParam*>(_param);
        if(!param) {
            std::cout << "In SOR, the input parameter is wrong.\n";
            exit(EXIT_FAILURE);
        }
        int N = Mat->N;
        iter = param->iter;
        temp = utils::SafeMalloc<int>(N);

        int size = 1200;
        c_num = (N+size-1)/size;
        // size = (N+c_num-1)/c_num;

        c_size = utils::SafeMalloc<int>(c_num+1);
        c_size[0] = 0;
        for(int i=0; i<c_num; i++) {
            c_size[i+1] = c_size[i] + size;
        }
        c_size[c_num] = N;

        int lnnz = 0, unnz = 0;
        for(int i=0; i<N; i++) {
            int left = c_size[i/size];
            for(int j=Mat->rptr[i]; j<Mat->rptr[i+1]; j++) {
                if(Mat->cind[j] == i) { lnnz++; unnz++; }
                else if(Mat->cind[j] < left) { lnnz++; }
                else { unnz++; }
            }
        }
        double *lval = utils::SafeMalloc<double>(lnnz);
        int *lcind   = utils::SafeMalloc<int>(lnnz);
        int *lrptr   = utils::SafeMalloc<int>(N+1);
        double *uval = utils::SafeMalloc<double>(unnz);
        int *ucind   = utils::SafeMalloc<int>(unnz);
        int *urptr   = utils::SafeMalloc<int>(N+1);

        double omega = param->omega;
        lrptr[0] = 0; urptr[0] = 0;
        lnnz = 0; unnz = 0;
        for(int i=0; i<N; i++) {
            int left = c_size[i/size];
            for(int j=Mat->rptr[i]; j<Mat->rptr[i+1]; j++) {
                if(Mat->cind[j] == i) {
                    lval[lnnz] = (1.0/omega)*Mat->val[j]; lcind[lnnz] = Mat->cind[j];
                    lnnz++;
                    uval[unnz] = (1-1.0/omega)*Mat->val[j]; ucind[unnz] = Mat->cind[j];
                    unnz++;
                }else if(Mat->cind[j] < left) {
                    lval[lnnz] = Mat->val[j]; lcind[lnnz] = Mat->cind[j];
                    lnnz++;
                }else {
                    uval[unnz] = Mat->val[j]; ucind[unnz] = Mat->cind[j];
                    unnz++;
                }
            }
            lrptr[i+1] = lnnz; urptr[i+1] = unnz;
        }

        CSRMat *tLD = new CSRMat(
            lval, lcind, lrptr, Mat->N, Mat->M, Mat->shape, Mat->type, MMOrder::MC,
            1, c_size, c_num, nullptr, 0);
        CSRMat *tU  = new CSRMat(
            uval, ucind, urptr, Mat->N, Mat->M, Mat->shape, Mat->type);

        LD = new ClassLD(tLD);
        U  = new ClassU(tU);
        
        tLD->Free(); delete tLD;
        tU->Free();  delete tU;
    }
    ~JacobiSORflex() {
        delete LD;
        delete U;
        utils::SafeFree(&temp);
    }
    void Smooth(int *in, int *out, bool isInited) {
        if(isInited) {
            for(int k=0; k<iter; k++) {
                U->SpMV(out, temp);
                blas1::Axpby<int>(1, in, -1, temp, U->GetN());
                LD->SpTRSV(temp, out);
            }
        }else {
            LD->SpTRSV(in, out);
            for(int k=1; k<iter; k++) {
                U->SpMV(out, temp);
                blas1::Axpby<int>(1, in, -1, temp, U->GetN());
                LD->SpTRSV(temp, out);
            }
        }
    }
    void Precondition(int *in, int *out) {
        LD->SpTRSV(in, out);
        for(int k=1; k<iter; k++) {
            U->SpMV(out, temp);
            blas1::Axpby<int>(1, in, -1, temp, U->GetN());
            LD->SpTRSV(temp, out);
        }
    }
};

/**
 * Blocked SOR
 **/
template <class ClassLD, class ClassU>
class BlockSOR {};

template <
    template <typename, int...> class ClassLD,
    template <typename, int...> class ClassU,
    typename T, int... args1, int... args2>
class BlockSOR<ClassLD<T, args1...>, ClassU<T, args2...>> : public Smoother<T> {
private:
    ldtri::SpMat<T> *LD;
    SpMat<T> *U;
    int iter;
    T *temp;
    int *bj_size, bj_num;
public:
    BlockSOR(
        CSRMat *Mat, SmootherParam* _param,
        SpMat<T> *dummy=nullptr)
    {
        BlockSORParam *param = dynamic_cast<BlockSORParam*>(_param);
        if(!param) {
            std::cout << "In BlockSOR, the input parameter is wrong.\n";
            exit(EXIT_FAILURE);
        }
        int N = Mat->N;
        iter = param->iter;
        temp = utils::SafeMalloc<T>(N);

        bj_num = param->b_num;
        bj_size = utils::SafeMalloc<int>(bj_num+1);
        int size = (N+bj_num-1)/bj_num;
        bj_size[0] = 0;
        for(int i=0; i<bj_num; i++) { bj_size[i+1] = bj_size[i] + size; }
        bj_size[bj_num] = N;

        int lnnz = 0, unnz = 0;
        for(int i=0; i<N; i++) {
            int left = bj_size[i/size];
            for(int j=Mat->rptr[i]; j<Mat->rptr[i+1]; j++) {
                if(Mat->cind[j] == i) { lnnz++; unnz++; }
                else if(left <= Mat->cind[j] && Mat->cind[j] < i) { lnnz++; }
                else { unnz++; }
            }
        }

        double *lval = utils::SafeMalloc<double>(lnnz);
        int *lcind   = utils::SafeMalloc<int>(lnnz);
        int *lrptr   = utils::SafeMalloc<int>(N+1);
        double *uval = utils::SafeMalloc<double>(unnz);
        int *ucind   = utils::SafeMalloc<int>(unnz);
        int *urptr   = utils::SafeMalloc<int>(N+1);

        double omega = param->omega;
        lrptr[0] = 0; urptr[0] = 0;
        lnnz = 0; unnz = 0;
        for(int i=0; i<N; i++) {
            int left = bj_size[i/size];
            for(int j=Mat->rptr[i]; j<Mat->rptr[i+1]; j++) {
                if(Mat->cind[j] == i) {
                    lval[lnnz] = (1.0/omega)*Mat->val[j]; lcind[lnnz] = Mat->cind[j];
                    lnnz++;
                    uval[unnz] = (1-1.0/omega)*Mat->val[j]; ucind[unnz] = Mat->cind[j];
                    unnz++;
                }else if(left <= Mat->cind[j] && Mat->cind[j] < i) {
                    lval[lnnz] = Mat->val[j]; lcind[lnnz] = Mat->cind[j];
                    lnnz++;
                }else {
                    uval[unnz] = Mat->val[j]; ucind[unnz] = Mat->cind[j];
                    unnz++;
                }
            }
            lrptr[i+1] = lnnz; urptr[i+1] = unnz;
        }

        CSRMat *tLD = new CSRMat(
            lval, lcind, lrptr, Mat->N, Mat->M, Mat->shape, Mat->type, MMOrder::BJ,
            Mat->b_size, Mat->c_size, Mat->c_num, bj_size, bj_num);
        CSRMat *tU  = new CSRMat(
            uval, ucind, urptr, Mat->N, Mat->M, Mat->shape, Mat->type);

        LD = new ClassLD<T, args1...>(tLD);
        U  = new ClassU<T, args2...>(tU);
        
        tLD->Free(); delete tLD;
        tU->Free();  delete tU;
    }
    ~BlockSOR() {
        delete LD;
        delete U;
        utils::SafeFree(&temp);
    }
    void Smooth(T *in, T *out, bool isInited) {
        if(isInited) {
            for(int k=0; k<iter; k++) {
                U->SpMV(out, temp);
                blas1::Axpby<T>(1, in, -1, temp, U->GetN());
                LD->SpTRSV(temp, out);
            }
        }else {
            LD->SpTRSV(in, out);
            for(int k=1; k<iter; k++) {
                U->SpMV(out, temp);
                blas1::Axpby<T>(1, in, -1, temp, U->GetN());
                LD->SpTRSV(temp, out);
            }
        }
    }
    void Precondition(T *in, T *out) {
        LD->SpTRSV(in, out);
        for(int k=1; k<iter; k++) {
            U->SpMV(out, temp);
            blas1::Axpby<T>(1, in, -1, temp, U->GetN());
            LD->SpTRSV(temp, out);
        }
    }
};

template <
    template <typename, int...> class ClassLD,
    template <typename, int...> class ClassU,
    int bit, int... args1, int... args2>
class BlockSOR<ClassLD<Fixed<bit>, args1...>, ClassU<Fixed<bit>, args2...>> : public Smoother<int> {
private:
    ldtri::SpMat<int> *LD;
    SpMat<int> *U;
    int iter;
    int *temp;
    int *bj_size, bj_num;
public:
    BlockSOR(
        CSRMat *Mat, SmootherParam* _param,
        SpMat<int> *dummy=nullptr)
    {
        BlockSORParam *param = dynamic_cast<BlockSORParam*>(_param);
        if(!param) {
            std::cout << "In BlockSOR, the input parameter is wrong.\n";
            exit(EXIT_FAILURE);
        }
        int N = Mat->N;
        iter = param->iter;
        temp = utils::SafeMalloc<int>(N);

        bj_num = param->b_num;
        bj_size = utils::SafeMalloc<int>(bj_num+1);
        int size = (N+bj_num-1)/bj_num;
        bj_size[0] = 0;
        for(int i=0; i<bj_num; i++) { bj_size[i+1] = bj_size[i] + size; }
        bj_size[bj_num] = N;

        int lnnz = 0, unnz = 0;
        for(int i=0; i<N; i++) {
            int left = bj_size[i/size];
            for(int j=Mat->rptr[i]; j<Mat->rptr[i+1]; j++) {
                if(Mat->cind[j] == i) { lnnz++; unnz++; }
                else if(left <= Mat->cind[j] && Mat->cind[j] < i) { lnnz++; }
                else { unnz++; }
            }
        }

        double *lval = utils::SafeMalloc<double>(lnnz);
        int *lcind   = utils::SafeMalloc<int>(lnnz);
        int *lrptr   = utils::SafeMalloc<int>(N+1);
        double *uval = utils::SafeMalloc<double>(unnz);
        int *ucind   = utils::SafeMalloc<int>(unnz);
        int *urptr   = utils::SafeMalloc<int>(N+1);

        double omega = param->omega;
        lrptr[0] = 0; urptr[0] = 0;
        lnnz = 0; unnz = 0;
        for(int i=0; i<N; i++) {
            int left = bj_size[i/size];
            for(int j=Mat->rptr[i]; j<Mat->rptr[i+1]; j++) {
                if(Mat->cind[j] == i) {
                    lval[lnnz] = (1.0/omega)*Mat->val[j]; lcind[lnnz] = Mat->cind[j];
                    lnnz++;
                    uval[unnz] = (1-1.0/omega)*Mat->val[j]; ucind[unnz] = Mat->cind[j];
                    unnz++;
                }else if(left <= Mat->cind[j] && Mat->cind[j] < i) {
                    lval[lnnz] = Mat->val[j]; lcind[lnnz] = Mat->cind[j];
                    lnnz++;
                }else {
                    uval[unnz] = Mat->val[j]; ucind[unnz] = Mat->cind[j];
                    unnz++;
                }
            }
            lrptr[i+1] = lnnz; urptr[i+1] = unnz;
        }

        CSRMat *tLD = new CSRMat(
            lval, lcind, lrptr, Mat->N, Mat->M, Mat->shape, Mat->type, MMOrder::BJ,
            Mat->b_size, Mat->c_size, Mat->c_num, bj_size, bj_num);
        CSRMat *tU  = new CSRMat(
            uval, ucind, urptr, Mat->N, Mat->M, Mat->shape, Mat->type);

        LD = new ClassLD<Fixed<bit>, args1...>(tLD);
        U  = new ClassU<Fixed<bit>, args2...>(tU);
        
        tLD->Free(); delete tLD;
        tU->Free();  delete tU;
    }
    ~BlockSOR() {
        delete LD;
        delete U;
        utils::SafeFree(&temp);
    }
    void Smooth(int *in, int *out, bool isInited) {
        if(isInited) {
            for(int k=0; k<iter; k++) {
                U->SpMV(out, temp);
                blas1::Axpby<int>(1, in, -1, temp, U->GetN());
                LD->SpTRSV(temp, out);
            }
        }else {
            LD->SpTRSV(in, out);
            for(int k=1; k<iter; k++) {
                U->SpMV(out, temp);
                blas1::Axpby<int>(1, in, -1, temp, U->GetN());
                LD->SpTRSV(temp, out);
            }
        }
    }
    void Precondition(int *in, int *out) {
        LD->SpTRSV(in, out);
        for(int k=1; k<iter; k++) {
            U->SpMV(out, temp);
            blas1::Axpby<int>(1, in, -1, temp, U->GetN());
            LD->SpTRSV(temp, out);
        }
    }
};

template <class ClassLD, class ClassU>
class BlockSORflex : public Smoother<int> {
private:
    ldtri::SpMat<int> *LD;
    SpMat<int> *U;
    int iter;
    int *temp;
    int *bj_size, bj_num;
public:
    BlockSORflex(
        CSRMat *Mat, SmootherParam* _param,
        SpMat<int> *dummy=nullptr)
    {
        BlockSORParam *param = dynamic_cast<BlockSORParam*>(_param);
        if(!param) {
            std::cout << "In BlockSORflex, the input parameter is wrong.\n";
            exit(EXIT_FAILURE);
        }
        int N = Mat->N;
        iter = param->iter;
        temp = utils::SafeMalloc<int>(N);

        bj_num = param->b_num;
        bj_size = utils::SafeMalloc<int>(bj_num+1);
        int size = (N+bj_num-1)/bj_num;
        bj_size[0] = 0;
        for(int i=0; i<bj_num; i++) { bj_size[i+1] = bj_size[i] + size; }
        bj_size[bj_num] = N;

        int lnnz = 0, unnz = 0;
        for(int i=0; i<N; i++) {
            int left = bj_size[i/size];
            for(int j=Mat->rptr[i]; j<Mat->rptr[i+1]; j++) {
                if(Mat->cind[j] == i) { lnnz++; unnz++; }
                else if(left <= Mat->cind[j] && Mat->cind[j] < i) { lnnz++; }
                else { unnz++; }
            }
        }

        double *lval = utils::SafeMalloc<double>(lnnz);
        int *lcind   = utils::SafeMalloc<int>(lnnz);
        int *lrptr   = utils::SafeMalloc<int>(N+1);
        double *uval = utils::SafeMalloc<double>(unnz);
        int *ucind   = utils::SafeMalloc<int>(unnz);
        int *urptr   = utils::SafeMalloc<int>(N+1);

        double omega = param->omega;
        lrptr[0] = 0; urptr[0] = 0;
        lnnz = 0; unnz = 0;
        for(int i=0; i<N; i++) {
            int left = bj_size[i/size];
            for(int j=Mat->rptr[i]; j<Mat->rptr[i+1]; j++) {
                if(Mat->cind[j] == i) {
                    lval[lnnz] = (1.0/omega)*Mat->val[j]; lcind[lnnz] = Mat->cind[j];
                    lnnz++;
                    uval[unnz] = (1-1.0/omega)*Mat->val[j]; ucind[unnz] = Mat->cind[j];
                    unnz++;
                }else if(left <= Mat->cind[j] && Mat->cind[j] < i) {
                    lval[lnnz] = Mat->val[j]; lcind[lnnz] = Mat->cind[j];
                    lnnz++;
                }else {
                    uval[unnz] = Mat->val[j]; ucind[unnz] = Mat->cind[j];
                    unnz++;
                }
            }
            lrptr[i+1] = lnnz; urptr[i+1] = unnz;
        }

        CSRMat *tLD = new CSRMat(
            lval, lcind, lrptr, Mat->N, Mat->M, Mat->shape, Mat->type, MMOrder::BJ,
            Mat->b_size, Mat->c_size, Mat->c_num, bj_size, bj_num);
        CSRMat *tU  = new CSRMat(
            uval, ucind, urptr, Mat->N, Mat->M, Mat->shape, Mat->type);

        LD = new ClassLD(tLD);
        U  = new ClassU(tU);
        
        tLD->Free(); delete tLD;
        tU->Free();  delete tU;
    }
    ~BlockSORflex() {
        delete LD;
        delete U;
        utils::SafeFree(&temp);
    }
    void Smooth(int *in, int *out, bool isInited) {
        if(isInited) {
            for(int k=0; k<iter; k++) {
                U->SpMV(out, temp);
                blas1::Axpby<int>(1, in, -1, temp, U->GetN());
                LD->SpTRSV(temp, out);
            }
        }else {
            LD->SpTRSV(in, out);
            for(int k=1; k<iter; k++) {
                U->SpMV(out, temp);
                blas1::Axpby<int>(1, in, -1, temp, U->GetN());
                LD->SpTRSV(temp, out);
            }
        }
    }
    void Precondition(int *in, int *out) {
        LD->SpTRSV(in, out);
        for(int k=1; k<iter; k++) {
            U->SpMV(out, temp);
            blas1::Axpby<int>(1, in, -1, temp, U->GetN());
            LD->SpTRSV(temp, out);
        }
    }
};

} // senk

#endif
