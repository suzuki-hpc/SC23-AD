#ifndef SENKPP_SMOOTHER_DUAL_HPP
#define SENKPP_SMOOTHER_DUAL_HPP

#include "matrix/spmat.hpp"
#include "smoother/smoother.hpp"
#include "smoother/richardson.hpp"

namespace senk {

struct DualParam : public SmootherParam {
public:
    SmootherParam *param1, *param2;
    int iter;
    DualParam(
        SmootherParam *_param1, SmootherParam *_param2, int _iter)
    {
        param1 = _param1;
        param2 = _param2;
        iter  = _iter;
    }
    ~DualParam() {}
};

struct Dual2Param : public SmootherParam {
public:
    int iter;
    Dual2Param(int _iter) {
        iter  = _iter;
    }
    ~Dual2Param() {}
};

template <class PreClass1, class PreClass2>
class Dual : public Smoother {
private:
    Smoother *smoother1, *smoother2;
    int iter;
public:
    Dual(CSRMat *Mat, SmootherParam *_param, SpMat<double> *A)
    {
        DualParam *param = dynamic_cast<DualParam*>(_param);
        if(!param) exit(1);
        smoother1 = new Richardson<PreClass1>(Mat, param->param1, A);
        smoother2 = new Richardson<PreClass2>(Mat, param->param2, A);
        iter = param->iter;
    }
    ~Dual() {
        delete smoother1;
        delete smoother2;
    }
    void Smooth(double *in, double *out, bool isInited) {
        if(isInited) {
            smoother1->Smooth(in, out, true);
            smoother2->Smooth(in, out, true);
        }else {
            smoother1->Smooth(in, out, false);
            smoother2->Smooth(in, out, true);
        }
    }
    void Precondition(double *in, double *out) {
        smoother1->Smooth(in, out, false);
        smoother2->Smooth(in, out, true);
    }
};

class Dual2 : public Smoother {
private:
    Smoother *smoother1, *smoother2;
    int iter;
public:
    Dual2(Smoother *_smoother1, Smoother *_smoother2, SmootherParam *_param)
    {
        Dual2Param *param = dynamic_cast<Dual2Param*>(_param);
        if(!param) exit(1);
        smoother1 = _smoother1;
        smoother2 = _smoother2;
        iter = param->iter;
    }
    ~Dual2() {
        // delete smoother1;
        // delete smoother2;
    }
    void Smooth(double *in, double *out, bool isInited) {
        if(isInited) {
            smoother1->Smooth(in, out, true);
            smoother2->Smooth(in, out, true);
        }else {
            smoother1->Smooth(in, out, false);
            smoother2->Smooth(in, out, true);
        }
    }
    void Precondition(double *in, double *out) {
        smoother1->Smooth(in, out, false);
        smoother2->Smooth(in, out, true);
    }
};

/*
Smoother *GetDual(
    Smoother *smoother1, Smoother *smoother2)
{
    // RichardsonParam *param1 = static_cast<RichardsonParam*>(_param1);
    Smoother *smoother1 = GetSmoother<SmootherClass1>(Mat, _param1);
    Smoother *smoother2 = GetRichardson<SmootherClass2>(Mat, A, _param2);
    if(!smoother1 || !smoother2) {
        std::cout << "In GetDual, nullptr was returned because smoother == nullptr.";
        std::cout << std::endl;
        return nullptr;
    }

    Dual *res = new Dual(smoother1, smoother2);
    return res;
}
*/
/*
template <typename T>
Dual<T> *GetDual_BJ_ILU0(
    base::CSRMat *Mat, matrix::SpMat<T> *A, int iter, int b_num)
{
    BJ_ILU<T> *smooth1 = GetBJ_ILU0<T>(Mat, A, iter, b_num);
    BJ_ILU<T> *smooth2 = GetBJ_ILU0_shifted<T>(Mat, A, iter, b_num);

    Dual<T> *res = new Dual<T>(smooth1, smooth2);
    return res;
}

template <typename T>
Dual<T> *GetDual_BJ_ILU0_test(
    base::CSRMat *Mat, matrix::SpMat<T> *A, int iter, int b_num)
{
    BJ_ILU<T> *smooth1 = GetBJ_ILU0_test<T>(Mat, A, iter, b_num);
    BJ_ILU<T> *smooth2 = GetBJ_ILU0_shifted_test<T>(Mat, A, iter, b_num);

    Dual<T> *res = new Dual<T>(smooth1, smooth2);
    return res;
}

template <typename T>
Dual<T> *GetDual_SDAINV_ILU0(
    base::CSRMat *Mat, matrix::SpMat<T> *A, int iter, double tol, double acc)
{
    AINV<T> *smooth1 = GetSDAINV<T>(Mat, A, iter, tol, acc);
    ILU<T> *smooth2 = GetILU0<T>(Mat, A, iter);

    Dual<T> *res = new Dual<T>(smooth1, smooth2);
    return res;
}

template <typename T>
Dual<T> *GetDual_SDAINV_Jacobi(
    base::CSRMat *Mat, matrix::SpMat<T> *A, int iter1, int iter2, double tol, double acc)
{
    AINV<T> *smooth2 = GetSDAINV<T>(Mat, A, iter1, tol, acc);
    Jacobi<T> *smooth1 = GetJacobi<T>(Mat, iter2);

    Dual<T> *res = new Dual<T>(smooth1, smooth2);
    return res;
}
*/

} // senk

#endif
