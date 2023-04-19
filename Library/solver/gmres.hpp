#ifndef SENKPP_SOLVER_KRYLOV_GMRES_HPP
#define SENKPP_SOLVER_KRYLOV_GMRES_HPP

#include "timer.hpp"
#include "printer.hpp"
#include "matrix/spmat.hpp"
#include "matrix/spmat2.hpp"
#include "solver/solver.hpp"
#include "preconditioner/preconditioner.hpp"
#include "blas/blas1.hpp"
#include "blas/blas2.hpp"

namespace senk {

template <typename T>
class FGMRES { };

template <>
class FGMRES<double> : public Solver {
private:
    SpMat<double> *A;
    int m;
    double epsilon;
    Preconditioner<double> *M;
    int N;
    double *c, *s, *e, *H, *V, *Z, *y;
public:
    FGMRES(SpMat<double> *_A, int _max_iter, double _epsilon,
        Preconditioner<double> *_M)
    {
        A = _A;
        m = _max_iter;
        epsilon = _epsilon;
        M = _M;
        N = A->GetN();
        c = utils::SafeMalloc<double>(m);
        s = utils::SafeMalloc<double>(m);
        e = utils::SafeMalloc<double>(m+1);
        H = utils::SafeMalloc<double>((m+1)*m);
        V = utils::SafeMalloc<double>(N*(m+1));
        Z = utils::SafeMalloc<double>(N*(m+1));
        y = utils::SafeMalloc<double>(m);
        // Printer *printer = new Printer("#", "FGMRESm<double> constructor");
        // printer->PrintNameValue("max_iter", m);
        // printer->PrintNameValue("epsilon", epsilon);
        // delete printer;
    }
    Converge Solve(double* r, double nrm_b, double *x) {
        bool flag = false;
        e[0] = blas1::Nrm2<double>(r, N);
        double e_inv = 1.0 / e[0];
        #pragma omp parallel for simd
        for(int k=0; k<N; k++) { V[k] = r[k] * e_inv; }
        int j;
        for(j=0; j<m; j++) {
            M->Precondition(&V[j*N], &Z[j*N]);
            A->SpMV(&Z[j*N], &V[(j+1)*N]);
            for(int k=0; k<=j; k++) {
                H[j*(m+1)+k] = blas1::Dot<double>(&V[k*N], &V[(j+1)*N], N);
                blas1::Axpy<double>(-H[j*(m+1)+k], &V[k*N], &V[(j+1)*N], N);
            }
            H[j*(m+1)+j+1] = blas1::Nrm2<double>(&V[(j+1)*N], N);
            blas1::Scal<double>(1/H[j*(m+1)+j+1], &V[(j+1)*N], N);
            for(int k=0; k<j; k++) {
                blas1::Grot<double>(c[k], s[k], &H[j*(m+1)+k], &H[j*(m+1)+k+1]);
            }
            H[j*(m+1)+j] = blas1::Ggen<double>(H[j*(m+1)+j], H[j*(m+1)+j+1], &c[j], &s[j]);
            H[j*(m+1)+j+1] = 0;
            e[j+1] = s[j] * e[j];
            e[j] = c[j] * e[j];
        #if PRINT_RES
            printf("# [%d] %e\n", j+1, std::abs(e[j+1]/nrm_b));
        #endif
            if(std::abs(e[j+1]) <= nrm_b*epsilon) {
                // printf("%s %d\n", ITER_SYMBOL, j+1);
                printf("%s %e\n", RES_SYMBOL, std::abs(e[j+1])/nrm_b);
                j++;
                flag = true;
                break;
            }
        }
        blas2::Trsv<double>(H, e, y, m+1, j);
        for(int k=0; k<j; k++) {
            blas1::Axpy<double>(y[k], &Z[k*N], x, N);
        }
        if(flag) return Converge(true, j);
        return Converge(false, m);
    }
    ~FGMRES() {
        free(c);
        free(s);
        free(e);
        free(H);
        free(V);
        free(Z);
        free(y);
    }
};

template <>
class FGMRES<float> : public Solver {
private:
    SpMat<float> *A;
    int m;
    double epsilon;
    Preconditioner<float> *M;
    int N;
    float *c, *s, *e, *H, *V, *Z, *y;
public:
    FGMRES(SpMat<float> *_A, int _max_iter, double _epsilon,
        Preconditioner<float> *_M)
    {
        A = _A;
        m = _max_iter;
        epsilon = _epsilon;
        M = _M;
        N = A->GetN();
        c = utils::SafeMalloc<float>(m);
        s = utils::SafeMalloc<float>(m);
        e = utils::SafeMalloc<float>(m+1);
        H = utils::SafeMalloc<float>((m+1)*m);
        V = utils::SafeMalloc<float>(N*(m+1));
        Z = utils::SafeMalloc<float>(N*(m+1));
        y = utils::SafeMalloc<float>(m);
        // Printer *printer = new Printer("#", "FGMRESm<float> constructor");
        // printer->PrintNameValue("max_iter", m);
        // printer->PrintNameValue("epsilon", epsilon);
        // delete printer;
    }
    Converge Solve(double* r, double nrm_b, double *x) {
        bool flag = false;
        e[0] = blas1::Nrm2<double>(r, N);
        double e_inv = 1.0 / e[0];
        #pragma omp parallel for simd
        for(int k=0; k<N; k++) { V[k] = (float)(r[k] * e_inv); }
        int j;
        for(j=0; j<m; j++) {
            M->Precondition(&V[j*N], &Z[j*N]);
            A->SpMV(&Z[j*N], &V[(j+1)*N]);
            for(int k=0; k<=j; k++) {
                H[j*(m+1)+k] = blas1::Dot<float>(&V[k*N], &V[(j+1)*N], N);
                blas1::Axpy<float>(-H[j*(m+1)+k], &V[k*N], &V[(j+1)*N], N);
            }
            H[j*(m+1)+j+1] = blas1::Nrm2<float>(&V[(j+1)*N], N);
            blas1::Scal<float>(1/H[j*(m+1)+j+1], &V[(j+1)*N], N);
            for(int k=0; k<j; k++) {
                blas1::Grot<float>(c[k], s[k], &H[j*(m+1)+k], &H[j*(m+1)+k+1]);
            }
            H[j*(m+1)+j] = blas1::Ggen<float>(H[j*(m+1)+j], H[j*(m+1)+j+1], &c[j], &s[j]);
            H[j*(m+1)+j+1] = 0;
            e[j+1] = s[j] * e[j];
            e[j] = c[j] * e[j];
        #if PRINT_RES
            printf("# [%d] %e\n", j+1, std::abs(e[j+1]/nrm_b));
        #endif
            if(std::abs(e[j+1]) <= nrm_b*epsilon) {
                // printf("%s %d\n", ITER_SYMBOL, j+1);
                printf("%s %e\n", RES_SYMBOL, std::abs(e[j+1])/nrm_b);
                j++;
                flag = true;
                break;
            }
        }
        blas2::Trsv<float>(H, e, y, m+1, j);
        for(int k=0; k<j; k++) {
            blas1::AxpyFD(y[k], &Z[k*N], x, N);
        }
        if(flag) return Converge(true, j);
        return Converge(false, m);
    }
    ~FGMRES() {
        free(c);
        free(s);
        free(e);
        free(H);
        free(V);
        free(Z);
        free(y);
    }
};

class FGMRESflex : public Solver {
private:
    SpMat<int> *A;
    int m;
    double epsilon;
    Preconditioner<int> *M;
    int N;
    long *c, *s, *e, *H;
    int *V, *Z;
    double *y;
    int8_t bit;
public:
    FGMRESflex(SpMat<int> *_A, int _max_iter, double _epsilon,
        Preconditioner<int> *_M, int8_t _bit)
    {
        A = _A;
        m = _max_iter;
        epsilon = _epsilon;
        M = _M;
        N = A->GetN();
        bit = _bit;
        c = utils::SafeMalloc<long>(m);
        s = utils::SafeMalloc<long>(m);
        e = utils::SafeMalloc<long>(m+1);
        H = utils::SafeMalloc<long>((m+1)*m);
        V = utils::SafeMalloc<int>(N*(m+1));
        Z = utils::SafeMalloc<int>(N*(m+1));
        y = utils::SafeMalloc<double>(m);
        // Printer *printer = new Printer("#", "FGMRESm constructor");
        // printer->PrintNameValue("bit", (int)bit);
        // printer->PrintNameValue("m", m);
        // printer->PrintNameValue("epsilon", epsilon);
        // delete printer;
    }
    ~FGMRESflex() {
        free(c);
        free(s);
        free(e);
        free(H);
        free(V);
        free(Z);
        free(y);
    }
    Converge Solve(double* r, double nrm_b, double *x) {
        const double fact = (double)((long)1 << bit);
        double e0;
        int flag = 0;
        e0 = blas1::Nrm2<double>(r, N);
        double e_inv = 1.0 / e0;
        long criteria = (long)(nrm_b*epsilon*e_inv*fact);
        #pragma omp parallel for simd
        for(int k=0; k<N; k++) { V[k] = (int)(r[k] * e_inv * fact); }
        e[0] = (long)1 << bit;
        int j;
        for(j=0; j<m; j++) {
            M->Precondition(&V[j*N], &Z[j*N]);
            A->SpMV(&Z[j*N], &V[(j+1)*N]);
            for(int k=0; k<=j; k++) {
                H[j*(m+1)+k] = blas1::int32::Dot(&V[k*N], &V[(j+1)*N], N, bit);
                blas1::int32::Axpy(-H[j*(m+1)+k], &V[k*N], &V[(j+1)*N], N, bit);
            }
            H[j*(m+1)+j+1] = blas1::int32::Nrm2(&V[(j+1)*N], N, bit);
            long H_inv = ((long)1 << 62) / H[j*(m+1)+j+1] >> (62-bit-bit);
            blas1::int32::Scal(H_inv, &V[(j+1)*N], N, bit);
            for(int k=0; k<j; k++) {
                blas1::int32::Grot(c[k], s[k], &H[j*(m+1)+k], &H[j*(m+1)+k+1], bit);
            }
            H[j*(m+1)+j] = blas1::int32::Ggen(H[j*(m+1)+j], H[j*(m+1)+j+1], &c[j], &s[j], bit);
            H[j*(m+1)+j+1] = 0;
            e[j+1] = s[j] * e[j] >> bit;
            e[j] = c[j] * e[j] >> bit;
        #if PRINT_RES
            printf("# [%d] %e\n", j+1, std::fabs(e0*(double)e[j+1]/fact)/nrm_b);
        #endif
            // if(std::fabs(e0*(double)e[j+1]/fact) <= nrm_b*epsilon) {
            if(std::abs(e[j+1]) <= criteria) {
                // printf("%s %d\n", ITER_SYMBOL, j+1);
                printf("%s %e\n", RES_SYMBOL, std::abs(e0*(double)e[j+1]/fact)/nrm_b);
                j++;
                flag = 1;
                break;
            }
        }
        blas2::int32::Trsv(H, e0, e, y, m+1, j, bit);
        for(int k=0; k<j; k++) {
            blas1::int32::Axpy_D(y[k], &Z[k*N], x, N, bit);
        }
        if(flag == 1) return Converge(true, j);
        return Converge(false, m);
    }
};

} // senk

#endif
