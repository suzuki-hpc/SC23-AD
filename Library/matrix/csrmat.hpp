#ifndef SENKPP_MATRIX_CSRMAT_HPP
#define SENKPP_MATRIX_CSRMAT_HPP

#include "enums.hpp"
#include "matrix/csrmat.hpp"
#include "printer.hpp"
#include "utils/utils.hpp"
#include "helper/helper_matrix.hpp"
#include "helper/helper_ainv.hpp"

namespace senk {

struct CSRMat {
    double *val;
    int *cind, *rptr;
    int N, M;
    MMShape shape;
    MMType type;
    MMOrder order;
    int b_size = 0;
    int *c_size = nullptr, c_num = 0;
    int *bj_size = nullptr, bj_num = 0;
    CSRMat(double *_val, int *_cind, int *_rptr, int _N, int _M,
        MMShape _shape, MMType _type, MMOrder _order,
        int _b_size, int *_c_size, int _c_num,
        int *_bj_size, int _bj_num)
    {
        val  = _val;
        cind = _cind;
        rptr = _rptr;
        N = _N;
        M = _M;
        shape = _shape;
        type  = _type;
        order = _order;
        b_size  = _b_size;
        c_size  = _c_size;
        c_num   = _c_num;
        bj_size = _bj_size;
        bj_num  = _bj_num;
    }
    CSRMat(double *_val, int *_cind, int *_rptr, int _N, int _M,
        MMShape _shape, MMType _type)
    {
        val  = _val;
        cind = _cind;
        rptr = _rptr;
        N = _N;
        M = _M;
        shape = _shape;
        type  = _type;
        order = MMOrder::General;
    }
    CSRMat(double *_val, int *_cind, int *_rptr, int _N, int _M)
    {
        val  = _val;
        cind = _cind;
        rptr = _rptr;
        N = _N;
        M = _M;
        shape = MMShape::General;
        type  = MMType::Real;
        order = MMOrder::General;
    }
    void Free() {
        utils::SafeFree(&val);
        utils::SafeFree(&cind);
        utils::SafeFree(&rptr);
    }
    void LMax1Scaling(double *b) {
        for(int i=0; i<N; i++) {
            double max = 0;
            for(int j=rptr[i]; j<rptr[i+1]; j++) {
                if(std::abs(val[j]) > max) { max = std::abs(val[j]); }
            }
            for(int j=rptr[i]; j<rptr[i+1]; j++) { val[j] /= max; }
            b[i] /= max;
        }
        shape = MMShape::General;
    }
    CSRMat *Copy() {
        double *r_val;
        int *r_cind, *r_rptr;
        helper::matrix::copy(
            val, cind, rptr, &r_val, &r_cind, &r_rptr, N);
        CSRMat *res = new CSRMat(
            r_val, r_cind, r_rptr, N, M, shape, type, order,
            b_size, c_size, c_num, bj_size, bj_num);
        return res;
    }
    CSRMat *CopyAsTransposedCSRMat() {
        double *r_val;
        int *r_cind, *r_rptr;
        helper::matrix::csr_to_csc(
            val, cind, rptr,
            &r_val, &r_cind, &r_rptr, N, M);
        CSRMat *res = new CSRMat(
            r_val, r_cind, r_rptr, M, N, shape, type, order,
            b_size, c_size, c_num, bj_size, bj_num);
        return res;
    }
    void Copy(double **r_val, int **r_cind, int **r_rptr) {
        helper::matrix::copy(
            val, cind, rptr, r_val, r_cind, r_rptr, N);
    }
    void CopyAsSell(double **r_val, int **r_cind, int **r_rptr, int C) {
        if(N % C != 0) {
            std::cerr << "N must be a multiple of "<< C <<"." << std::endl;
            exit(1);
        }
        helper::matrix::csr_to_sell(
            val, cind, rptr, r_val, r_cind, r_rptr, C, N);
    }
    void CopyAsBCSR(double **r_val, int **r_cind, int **r_rptr, int Bnl, int Bnw) {
        if (!(Bnl == 2 && Bnw == 1) && !(Bnl == 2 && Bnw == 2) &&
            !(Bnl == 4 && Bnw == 1) && !(Bnl == 4 && Bnw == 4) &&
            !(Bnl == 8 && Bnw == 1) && !(Bnl == 8 && Bnw == 8) ) { exit(EXIT_FAILURE); }
        helper::matrix::csr_to_bcsr(
            val, cind, rptr, r_val, r_cind, r_rptr, N, Bnl, Bnw);
    }
    void CopyL(double **r_val, int **r_cind, int **r_rptr) {
        helper::matrix::extractL(
            val, cind, rptr, r_val, r_cind, r_rptr, N);
    }
    void CopyDinvU(double **r_val, int **r_cind, int **r_rptr) {
        helper::matrix::extractDinvU(
            val, cind, rptr, r_val, r_cind, r_rptr, N);
    }
    void GetILUp(int level, double alpha) {
        if(N != M) exit(EXIT_FAILURE);
        if(level == 0) {
            helper::matrix::ilu0<double>(val, cind, rptr, N, alpha);
        }else {
            printf("Coming soon.\n");
            exit(EXIT_FAILURE);
        }
    }

};

} // senk


#endif
