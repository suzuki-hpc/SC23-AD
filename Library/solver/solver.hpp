#ifndef SENKPP_SOLVER_BASE_HPP
#define SENKPP_SOLVER_BASE_HPP

#include "preconditioner/preconditioner.hpp"

namespace senk {

#define PRINT_RES true
#define ITER_SYMBOL "[Iter] :"
#define RES_SYMBOL "[Res] :"

struct Converge {
    bool isConverged;
    int iter;
    Converge() { }
    Converge(bool _isConverged, int _iter) {
        isConverged = _isConverged;
        iter = _iter;
    }
};

class Solver {
public:
    /**
     * Return 0 if it does not reach convergence
     * Return # of iterations if it do reach convergence
     **/
    virtual Converge Solve(double* r, double nrm_b, double *x) = 0;
    virtual ~Solver() {};
};

} // senk

#endif
