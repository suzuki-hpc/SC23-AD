#ifndef SENKPP_SOLVER_IRSOLVER_HPP
#define SENKPP_SOLVER_IRSOLVER_HPP

#include "solver/solver.hpp"

namespace senk {

class Restarted : public Solver {
    Solver *solver;
    SpMat<double> *A;
    int max_iter;
    double epsilon;
    int N;
    double *r;
public:
    Restarted(SpMat<double> *_A, Solver *_solver, int _max_iter, double _epsilon) {
        A = _A;
        solver = _solver;
        max_iter = _max_iter;
        epsilon = _epsilon;
        N = A->GetN();
        r = new double[N];
    }
    ~Restarted() {
        delete r;
    }
    Converge Solve(double* b, double nrm_b, double *x) {
        double nrm_r, prev = 0;
        Converge inner(false, 0);
        int iter = 0;
        for(int i=0; i<max_iter; i++) {
            A->SpMV(x, r);
            blas1::Axpby<double>(1, b, -1, r, N);
            nrm_r = blas1::Nrm2<double>(r, N);
            if(std::fabs(nrm_r - prev)/nrm_r < 1.0e-8) return Converge(false, 0);
            prev = nrm_r;
            // printf("# [%d] %e\n", i, nrm_r/nrm_b);
            inner = solver->Solve(r, nrm_b, x);
            iter += inner.iter;
            if(inner.isConverged) return Converge(true, iter);
        }
        return Converge(false, 0);
    }
};

} // senk

#endif
