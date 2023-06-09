#include <iostream>
#include <random>

#include "timer.hpp"
#include "utils/utils.hpp"
#include "io/mm.hpp"
#include "matrix/spmat.hpp"
#include "matrix/trimat/spmat.hpp"
#include "blas/blas1.hpp"
#include "solver/gmres.hpp"
#include "solver/irsolver.hpp"
#include "preconditioner/INTAMG.hpp"
#include "smoother/stationary.hpp"

// Namespace of our own library.
using namespace senk;

// Enum for selecting the used smoother
enum class E_Smoother {
    IS_SOR,
    IS_HYBRID_GS,
};

// Enum for selecting the used solver
enum class E_Solver {
    IS_DOUBLE,
    IS_FLOAT,
    IS_INT,
};

// The value of theta of the classical AMG method.
#define AMG_THETA 0.8

int main(int argc, char **argv) {
    if(argc != 5) return 1;
    /**
     * Reading a matrix of the Matrix Market format and
     * Storing the read matrix in the CSR format.
     **/
    std::string path = "../Matrix/";
    path += (argv[1]);
    const char *filename = path.c_str();
    CSRMat *Abase = io::ReadMM(filename, true, false);

    E_Smoother smoother_type = static_cast<E_Smoother>(atoi(argv[2]));
    E_Solver solver_type = static_cast<E_Solver>(atoi(argv[3]));
    int res_period = atoi(argv[4]);

    int N = Abase->N;

    /**
     * Setting the right-hand side vector.
     * The vector elements are generated by the Mersenne Twister.
     **/
    double *b = utils::SafeMalloc<double>(N);
    std::mt19937_64 engine(0);
    std::uniform_real_distribution<double> dist(0, 1);
    for(int i=0; i<N; i++) { b[i] = dist(engine); }

    // Initial scaling
    Abase->LMax1Scaling(b);

    double nrm_b = blas1::Nrm2<double>(b, N);

    auto timer = new Timer();

    /** 
     * Generate a coefficient matrix used for SpMV.
     **/
    auto dA = new CSR<double>(Abase);
    
    const int max = 1000;
    const int m = res_period;
    const int outer = max / m;
    const double epsilon = 1.0e-10;
    double *x = utils::SafeMalloc<double>(N);

    /**
     * Constructing a preconditioner and 
     * executing a solver.
     **/
    auto sor_param = new SORParam(1.0, 1);
    auto hybrid_param = new BlockSORParam(1.0, 1, 40);

    if(solver_type == E_Solver::IS_DOUBLE) {
        Preconditioner<double> *M;
        if(smoother_type == E_Smoother::IS_SOR) {
            M  = new dfAMG<CSR<double>,SOR<ldtri::CSR<double>, CSR<double>>, double>(
                Abase, dA, new AMGParam(20, AMG_THETA, 1, sor_param));
        }else {
            M  = new dfAMG<CSR<double>,BlockSOR<ldtri::BJCSR<double>, CSR<double>>, double>(
                Abase, dA, new AMGParam(20, AMG_THETA, 1, hybrid_param));
        }
        // Constructing the solver and solving the problem.
        auto d_inner = new FGMRES<double>(dA, m, epsilon, M);
        auto d_solver = new Restarted(dA, d_inner, outer, epsilon);
        utils::Set<double>(0, x, N);
        timer->Restart();
        auto d_converge = d_solver->Solve(b, nrm_b, x);
        timer->Elapsed();
        printf("[Iter] : %d\n", d_converge.iter);
    }

    if(solver_type == E_Solver::IS_FLOAT) {
        auto *fA = new CSR<float>(Abase);
        Preconditioner<float> *fM;
        if(smoother_type == E_Smoother::IS_SOR) {
            fM = new dfAMG<CSR<float>,SOR<ldtri::CSR<float>, CSR<float>>, float>(
                Abase, fA, new AMGParam(20, AMG_THETA, 1, sor_param));
        }else {
            fM  = new dfAMG<CSR<float>,BlockSOR<ldtri::BJCSR<float>, CSR<float>>, float>(
                Abase, fA, new AMGParam(20, AMG_THETA, 1, hybrid_param));
        }
        // Constructing the solver and solving the problem.
        auto f_inner = new FGMRES<float>(fA, m, epsilon, fM);
        auto f_solver = new Restarted(dA, f_inner, outer, epsilon);
        utils::Set<double>(0, x, N);
        timer->Restart();
        auto f_converge = f_solver->Solve(b, nrm_b, x);
        timer->Elapsed();
        printf("[Iter] : %d\n", f_converge.iter);
    }

    if(solver_type == E_Solver::IS_INT) {
        /**
         * NOTE
         * In this source file, we generate both integer and double type
         * preconditioners at the same time for simplicity.
         * However, constructing an integer-type one if sufficient
         * if we perform the following bit-length determination method
         * inside the constructor of it.
         **/
        auto iA = new CSR<Fixed<30>>(Abase);
        Preconditioner<double> *M;
        Preconditioner<int> *iM;
        if(smoother_type == E_Smoother::IS_SOR) {
            M  = new dfAMG<CSR<double>,SOR<ldtri::CSR<double>, CSR<double>>, double>(
                Abase, dA, new AMGParam(20, AMG_THETA, 1, sor_param));
            iM = new intAMGflex<CSRflex, CSRflex, SOR<ldtri::CSRflex, CSRflex>>(
                Abase, iA, new AMGParam(20, AMG_THETA, 1, sor_param));
        }else {
            M  = new dfAMG<CSR<double>,BlockSOR<ldtri::BJCSR<double>, CSR<double>>, double>(
                Abase, dA, new AMGParam(20, AMG_THETA, 1, hybrid_param));
            iM = new intAMGflex<CSRflex, CSRflex, BlockSORflex<ldtri::BJCSRflex, CSRflex>>(
                Abase, iA, new AMGParam(20, AMG_THETA, 1, hybrid_param));
        }
        /**
         * Determining the fractional bit length (``bit'')
         * of the Fix<32, F_G> format.
         * If you want to set the fractional bit length to you preference,
         * you can manually change the ''bit`` value.
         **/
        double *c = utils::SafeMalloc<double>(N);
        M->Precondition(b, c);
        double c_max = 0;
        for(int i=0; i<N; i++) {
            if(std::fabs(c[i]) > c_max)
                c_max = std::fabs(c[i]);
        }
        int8_t bit = (c_max/nrm_b < 1)? 29 : 31 - ((int)std::ceil(std::log2(int(c_max/nrm_b)+1)) + 1);
        free(c);
        delete M;
        // Constructing the solver and solving the problem.
        auto i_inner = new FGMRESflex(iA, m, epsilon, iM, bit);
        auto i_solver = new Restarted(dA, i_inner, outer, epsilon);
        utils::Set<double>(0, x, N);
        timer->Restart();
        auto i_converge = i_solver->Solve(b, nrm_b, x);
        timer->Elapsed();
        printf("[Iter] : %d\n", i_converge.iter);
    }
    
    delete timer;

    // Checking the value of the explicit relative residual norm.
    double *r = senk::utils::SafeMalloc<double>(N);
    dA->SpMV(x, r);
    senk::blas1::Axpby<double>(1, b, -1, r, N);
    double nrm_r = senk::blas1::Nrm2<double>(r, N);
    printf("# [Res] : %e\n", nrm_r/nrm_b);

    return 0;
}
