#ifndef SENKPP_SOLVER_DIRECT_LU_HPP
#define SENKPP_SOLVER_DIRECT_LU_HPP

#include "matrix/trimat/spmat.hpp"
#include "preconditioner/preconditioner.hpp"

namespace senk {

namespace direct {

template <typename T>
class LU {
    ltri::CSR<T> *L;
    dutri::CSR<T> *U;
    T *temp;
public:
    LU(CSRMat *Mat) {
        CSRMat *Temp = Mat->Copy();
        // helper::matrix::ilup<double>(
        //     &Temp->val, &Temp->cind, &Temp->rptr,
        //     Temp->N, Temp->N);
        helper::matrix::ilu(
            &Temp->val, &Temp->cind, &Temp->rptr,
            Temp->N);
        L = new ltri::CSR<T>(Temp);
        U = new dutri::CSR<T>(Temp);
        Temp->Free(); delete Temp;
        temp = utils::SafeMalloc<T>(Mat->N);
    }
    ~LU() {
        delete L;
        delete U;
        utils::SafeFree(&temp);
    };
    void Solve(T* r, T *x) {
        L->SpTRSV(r, temp);
        U->SpTRSV(temp, x);
    }
};

template <int bit>
class LU<Fixed<bit>> {
    ltri::CSR<Fixed<bit>> *L;
    dutri::CSR<Fixed<bit>> *U;
    int *temp;
public:
    LU(CSRMat *Mat) {
        CSRMat *Temp = Mat->Copy();
        // helper::matrix::ilup<double>(
        //     &Temp->val, &Temp->cind, &Temp->rptr,
        //     Temp->N, Temp->N);
        helper::matrix::ilu(
            &Temp->val, &Temp->cind, &Temp->rptr,
            Temp->N);
        L = new ltri::CSR<Fixed<bit>>(Temp);
        U = new dutri::CSR<Fixed<bit>>(Temp);
        Temp->Free(); delete Temp;
        temp = utils::SafeMalloc<int>(Mat->N);
    }
    ~LU() {
        delete L;
        delete U;
        utils::SafeFree(&temp);
    };
    void Solve(int* r, int *x) {
        L->SpTRSV(r, temp);
        U->SpTRSV(temp, x);
    }
};

class LUflex {
    ltri::CSRflex *L;
    dutri::CSRflex *U;
    int *temp;
public:
    LUflex(CSRMat *Mat) {
        CSRMat *Temp = Mat->Copy();
        helper::matrix::ilu(
            &Temp->val, &Temp->cind, &Temp->rptr,
            Temp->N);
        L = new ltri::CSRflex(Temp);
        U = new dutri::CSRflex(Temp);
        Temp->Free(); delete Temp;
        temp = utils::SafeMalloc<int>(Mat->N);
    }
    ~LUflex() {
        delete L;
        delete U;
        utils::SafeFree(&temp);
    };
    void Solve(int* r, int *x) {
        L->SpTRSV(r, temp);
        U->SpTRSV(temp, x);
    }
};

} // direct

} // senk

#endif
