#ifndef SENKPP_SMOOTHER_RICHARDSON_HPP
#define SENKPP_SMOOTHER_RICHARDSON_HPP

#include <iostream>
#include "parameters.hpp"
#include "utils/alloc.hpp"
#include "matrix/spmat.hpp"
#include "preconditioner/preconditioner.hpp"

namespace senk {

template <class PreClass, typename T>
class Richardson : public Smoother<T> {
private:
    Preconditioner<T> *pre;
    SpMat<T> *mat;
    int iter;
    T *r, *temp;
public:
    Richardson(
        CSRMat *Mat, SmootherParam *_param,
        SpMat<T> *A)
    {
        RichardsonParam *param = dynamic_cast<RichardsonParam*>(_param);
        pre = new PreClass(Mat, param->param);
        if(!pre) {
            std::cout << "In Richardson, nullptr was returned because preconditioner == nullptr.";
            std::cout << std::endl;
            exit(EXIT_FAILURE);
        }
        mat  = A;
        iter = param->iter;
        r    = utils::SafeMalloc<T>(Mat->N);
        temp = utils::SafeMalloc<T>(Mat->N);
    }
    ~Richardson() {
        delete pre;
        utils::SafeFree(&r);
        utils::SafeFree(&temp);
    }
    void Smooth(T *in, T *out, bool isInited) {
        if(isInited) {
            for(int k=0; k<iter; k++) {
                mat->SpMV(out, r);
                blas1::Axpby<T>(1, in, -1, r, mat->GetN());
                pre->Precondition(r, temp);
                blas1::Axpy<T>(1, temp, out, mat->GetN());
            }
        }else {
            pre->Precondition(in, out);
            for(int k=1; k<iter; k++) {
                mat->SpMV(out, r);
                blas1::Axpby<T>(1, in, -1, r, mat->GetN());
                pre->Precondition(r, temp);
                blas1::Axpy<T>(1, temp, out, mat->GetN());
            }
        }
    }
    void Precondition(T *in, T *out) {
        pre->Precondition(in, out);
        for(int k=1; k<iter; k++) {
            mat->SpMV(out, r);
            blas1::Axpby<T>(1, in, -1, r, mat->GetN());
            pre->Precondition(r, temp);
            blas1::Axpy<T>(1, temp, out, mat->GetN());
        }
    }
};
/*
template <class PreClass>
class Richardson_DI : public Smoother2<int> {
private:
    Preconditioner2<int> *pre;
    SpMat2<int> *mat;
    int iter;
    double *r, *temp;
    int *ir, *itemp;
public:
    Richardson_DI(
        CSRMat *Mat, SmootherParam *_param,
        SpMat2<int> *A)
    {
        RichardsonParam *param = dynamic_cast<RichardsonParam*>(_param);

        pre = new PreClass(Mat, param->param);
        if(!pre) {
            std::cout << "In Richardson, nullptr was returned because preconditioner == nullptr.";
            std::cout << std::endl;
            exit(EXIT_FAILURE);
        }
        mat  = A;
        iter = param->iter;
        r    = utils::SafeMalloc<double>(Mat->N);
        temp = utils::SafeMalloc<double>(Mat->N);
        ir    = utils::SafeMalloc<int>(Mat->N);
        itemp = utils::SafeMalloc<int>(Mat->N);
    }
    ~Richardson_DI() {
        delete pre;
        utils::SafeFree(&r);
        utils::SafeFree(&temp);
        utils::SafeFree(&ir);
        utils::SafeFree(&itemp);
    }
    void Smooth(double *in, double *out, bool isInited) {
        if(isInited) {
            for(int k=0; k<iter; k++) {
                mat->SpMV(out, r);
                blas1::Axpby<double>(1, in, -1, r, mat->GetN());
                pre->Precondition(r, temp);
                blas1::Axpy<double>(1, temp, out, mat->GetN());
            }
        }else {
            pre->Precondition(in, out);
            for(int k=1; k<iter; k++) {
                mat->SpMV(out, r);
                blas1::Axpby<double>(1, in, -1, r, mat->GetN());
                pre->Precondition(r, temp);
                blas1::Axpy<double>(1, temp, out, mat->GetN());
            }
        }
    }
    void Smooth2(int *in, int *out, bool isInited) {
        if(isInited) {
            for(int k=0; k<iter; k++) {
                mat->SpMV2(out, ir);
                blas1::Axpby<int>(1, in, -1, ir, mat->GetN());
                pre->Precondition2(ir, itemp);
                blas1::Axpy<int>(1, itemp, out, mat->GetN());
            }
        }else {
            pre->Precondition2(in, out);
            for(int k=1; k<iter; k++) {
                mat->SpMV2(out, ir);
                blas1::Axpby<int>(1, in, -1, ir, mat->GetN());
                pre->Precondition2(ir, itemp);
                blas1::Axpy<int>(1, itemp, out, mat->GetN());
            }
        }
    }
    void Precondition(double *in, double *out) {
        pre->Precondition(in, out);
        for(int k=1; k<iter; k++) {
            mat->SpMV(out, r);
            blas1::Axpby<double>(1, in, -1, r, mat->GetN());
            pre->Precondition(r, temp);
            blas1::Axpy<double>(1, temp, out, mat->GetN());
        }
    }
    void Precondition2(int *in, int *out) {
        pre->Precondition2(in, out);
        for(int k=1; k<iter; k++) {
            mat->SpMV2(out, ir);
            blas1::Axpby<int>(1, in, -1, ir, mat->GetN());
            pre->Precondition2(ir, itemp);
            blas1::Axpy<int>(1, itemp, out, mat->GetN());
        }
    }
};
*/
} // senk

#endif
