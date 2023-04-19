#ifndef SENKPP_PRECONDITIONER_INTAMG_HPP
#define SENKPP_PRECONDITIONER_INTAMG_HPP

#include "parameters.hpp"
#include "blas/blas1.hpp"
#include "matrix/spmat.hpp"
#include "helper/helper_multigrid.hpp"
#include "smoother/stationary.hpp"
#include "solver/lu.hpp"

namespace senk {

template <class SpMatClass, class SmootherClass, typename T>
class dfAMG : public Preconditioner<T> {
private:
    SpMat<T> *A;
    SpMatClass **C_A, **Intrpl, **Intrpl_t;
    SmootherClass **smoother;
    direct::LU<T> *lu_solver;
    int depth;
    int cycle;
    T *r, *e, **rr, **ee, **xx;
    int last_N;
public:
    dfAMG(CSRMat *Mat, SpMat<T> *_A, PreconditionerParam *_param) {
        AMGParam *param = dynamic_cast<AMGParam*>(_param);
        if(!param) {
            std::cout << "In dfAMG, the input parameter is wrong.\n";
            exit(EXIT_FAILURE);
        }
        A = _A;
        cycle = param->cycle;

        CSRMat **CA = utils::SafeMalloc<CSRMat*>(1);
        CSRMat **I  = utils::SafeMalloc<CSRMat*>(1);
        CSRMat **It = utils::SafeMalloc<CSRMat*>(1);
        
        helper::multigrid::get_amg_set(Mat, &CA[0], &I[0], &It[0], param->theta);

        // printf("# %d %d\n", CA[0]->N, CA[0]->rptr[CA[0]->N]);
        depth = 1;
        while( param->level > depth && CA[depth-1]->N > 2000 )
        {
            depth++;
            CA = utils::SafeRealloc<CSRMat*>(CA, depth);
            I  = utils::SafeRealloc<CSRMat*>(I, depth);
            It = utils::SafeRealloc<CSRMat*>(It, depth);
            helper::multigrid::get_amg_set(
                CA[depth-2], &CA[depth-1], &I[depth-1], &It[depth-1], param->theta);
            // printf("# %d %d\n", CA[depth-1]->N, CA[depth-1]->rptr[CA[depth-1]->N]);
        }
        C_A      = utils::SafeMalloc<SpMatClass*>(depth);
        Intrpl   = utils::SafeMalloc<SpMatClass*>(depth);
        Intrpl_t = utils::SafeMalloc<SpMatClass*>(depth);
        
        smoother = utils::SafeMalloc<SmootherClass*>(depth+1);
        smoother[0] = new SmootherClass(Mat, param->param, A);
        for(int i=0; i<depth; i++) {
            if(i == depth-1) {
                lu_solver = new direct::LU<T>(CA[i]);
                last_N = CA[i]->N;
            }else {
                C_A[i] = new SpMatClass(CA[i]);
                smoother[i+1] = new SmootherClass(CA[i], param->param, C_A[i]);
            }
            CA[i]->Free(); delete CA[i];
            Intrpl[i]   = new SpMatClass(I[i]);
            I[i]->Free();  delete I[i];
            Intrpl_t[i] = new SpMatClass(It[i]);
            It[i]->Free(); delete It[i];
        }

        r = utils::SafeMalloc<T>(Mat->N);
        e = utils::SafeMalloc<T>(Mat->N);
        rr = utils::SafeMalloc<T*>(depth);
        ee = utils::SafeMalloc<T*>(depth);
        xx = utils::SafeMalloc<T*>(depth);
        for(int i=0; i<depth; i++) {
            int lenght = (i == depth-1)? last_N : C_A[i]->GetN();
            rr[i] = utils::SafeMalloc<T>(lenght);
            ee[i] = utils::SafeMalloc<T>(lenght);
            xx[i] = utils::SafeMalloc<T>(lenght);
        }
    }
    ~dfAMG() {
        utils::SafeFree(&r);
        utils::SafeFree(&e);
        delete smoother;
        for(int i=0; i<depth; i++) {
            utils::SafeFree(&rr[i]);
            utils::SafeFree(&ee[i]);
            utils::SafeFree(&xx[i]);
            if(i == depth-1) {
                delete lu_solver;
            }else {
                delete C_A[i];
            }
            delete Intrpl[i];
            delete Intrpl_t[i];
        }
        utils::SafeFree(&rr);
        utils::SafeFree(&ee);
        utils::SafeFree(&xx);
        utils::SafeFree(&C_A);
        utils::SafeFree(&Intrpl);
        utils::SafeFree(&Intrpl_t);
    }
    void Precondition(T *in, T *out) {
        smoother[0]->Smooth(in, out, false);
        A->SpMV(out, r);
        blas1::Sub<T>(in, r, A->GetN());
        Intrpl_t[0]->SpMV(r, ee[0]);

        for(int i=0; i<depth-1; i++) {
            smoother[i+1]->Smooth(ee[i], xx[i], false);
            C_A[i]->SpMV(xx[i], rr[i]);
            blas1::Sub<T>(ee[i], rr[i], C_A[i]->GetN());
            Intrpl_t[i+1]->SpMV(rr[i], ee[i+1]);
        }
        
        lu_solver->Solve(ee[depth-1], xx[depth-1]);

        for(int i=depth-2; i>=0; i--) {
            Intrpl[i+1]->SpMV(xx[i+1], rr[i]);
            blas1::Add<T>(rr[i], xx[i], C_A[i]->GetN());
            smoother[i+1]->Smooth(ee[i], xx[i], true);
        }

        Intrpl[0]->SpMV(xx[0], r);
        blas1::Add<T>(r, out, A->GetN());

        smoother[0]->Smooth(in, out, true);
    }
};

template <class SpMatClassC, class SpMatClassI, class SmootherClass>
class intAMGflex : public Preconditioner<int> {
private:
    SpMat<int> *A; // Do not change.
    SpMatClassC **C_A;
    SpMatClassI **Intrpl, **Intrpl_t;
    SmootherClass **smoother;
    direct::LUflex *lu_solver;
    int depth;
    int cycle;
    int *r, *e, **rr, **ee, **xx;
    int last_N;
public:
    intAMGflex(CSRMat *Mat, SpMat<int> *_A, PreconditionerParam *_param)
    {
        AMGParam *param = dynamic_cast<AMGParam*>(_param);
        if(!param) {
            std::cout << "In AMG, the input parameter is wrong.\n";
            exit(EXIT_FAILURE);
        }
        A = _A;
        cycle = param->cycle;

        CSRMat **CA = utils::SafeMalloc<CSRMat*>(1);
        CSRMat **I  = utils::SafeMalloc<CSRMat*>(1);
        CSRMat **It = utils::SafeMalloc<CSRMat*>(1);
        helper::multigrid::get_amg_set(Mat, &CA[0], &I[0], &It[0], param->theta);
        // printf("# %d %d\n", CA[0]->N, CA[0]->rptr[CA[0]->N]);
        depth = 1;
        double *prev = utils::SafeCalloc<double>(Mat->N);
        while( param->level > depth && CA[depth-1]->N > 2000 )
        {
            depth++;
            CA = utils::SafeRealloc<CSRMat*>(CA, depth);
            I  = utils::SafeRealloc<CSRMat*>(I, depth);
            It = utils::SafeRealloc<CSRMat*>(It, depth);
            helper::multigrid::get_amg_set(
                CA[depth-2], &CA[depth-1], &I[depth-1], &It[depth-1], param->theta);
            // printf("# %d %d\n", CA[depth-1]->N, CA[depth-1]->rptr[CA[depth-1]->N]);

            double *max = utils::SafeCalloc<double>(CA[depth-2]->N);
            for(int i=0; i<CA[depth-2]->N; i++) {
                for(int j=CA[depth-2]->rptr[i]; j<CA[depth-2]->rptr[i+1]; j++) {
                    if(std::abs(CA[depth-2]->val[j]) > max[i])
                        max[i] = std::abs(CA[depth-2]->val[j]);
                }
                for(int j=CA[depth-2]->rptr[i]; j<CA[depth-2]->rptr[i+1]; j++) {
                    CA[depth-2]->val[j] /= max[i];
                }
                for(int j=It[depth-2]->rptr[i]; j<It[depth-2]->rptr[i+1]; j++) {
                    It[depth-2]->val[j] /= max[i];
                }
                if(depth == 2) continue;
                for(int j=It[depth-2]->rptr[i]; j<It[depth-2]->rptr[i+1]; j++) {
                    It[depth-2]->val[j] *= prev[It[depth-2]->cind[j]];
                }
            }
            for(int i=0; i<CA[depth-2]->N; i++) { prev[i] = max[i]; }
            free(max);
        }

        for(int i=0; i<CA[depth-1]->N; i++) {
            double max = 0;
            for(int j=CA[depth-1]->rptr[i]; j<CA[depth-1]->rptr[i+1]; j++) {
                if(std::abs(CA[depth-1]->val[j]) > max) max = std::abs(CA[depth-1]->val[j]);
            }
            for(int j=CA[depth-1]->rptr[i]; j<CA[depth-1]->rptr[i+1]; j++) {
                CA[depth-1]->val[j] /= max;
            }
            for(int j=It[depth-1]->rptr[i]; j<It[depth-1]->rptr[i+1]; j++) {
                It[depth-1]->val[j] /= max;
            }
            for(int j=It[depth-1]->rptr[i]; j<It[depth-1]->rptr[i+1]; j++) {
                It[depth-1]->val[j] *= prev[It[depth-1]->cind[j]];
            }
        }

        C_A      = utils::SafeMalloc<SpMatClassC*>(depth);
        Intrpl   = utils::SafeMalloc<SpMatClassI*>(depth);
        Intrpl_t = utils::SafeMalloc<SpMatClassI*>(depth);
        
        smoother = utils::SafeMalloc<SmootherClass*>(depth+1);
        smoother[0] = new SmootherClass(Mat, param->param, A);
        for(int i=0; i<depth; i++) {
            if(i == depth-1) {
                lu_solver = new direct::LUflex(CA[i]);
                last_N = CA[i]->N;
            }else {
                C_A[i] = new SpMatClassC(CA[i]);
                smoother[i+1] = new SmootherClass(CA[i], param->param, C_A[i]);
            }
            CA[i]->Free(); delete CA[i];
            Intrpl[i]   = new SpMatClassI(I[i]);
            I[i]->Free();  delete I[i];
            Intrpl_t[i] = new SpMatClassI(It[i]);
            It[i]->Free(); delete It[i];
        }

        r = utils::SafeMalloc<int>(Mat->N);
        e = utils::SafeMalloc<int>(Mat->N);
        rr = utils::SafeMalloc<int*>(depth);
        ee = utils::SafeMalloc<int*>(depth);
        xx = utils::SafeMalloc<int*>(depth);
        for(int i=0; i<depth; i++) {
            int lenght = (i == depth-1)? last_N : C_A[i]->GetN();
            rr[i] = utils::SafeMalloc<int>(lenght);
            ee[i] = utils::SafeMalloc<int>(lenght);
            xx[i] = utils::SafeMalloc<int>(lenght);
        }
    }
    ~intAMGflex() {
        utils::SafeFree(&r);
        utils::SafeFree(&e);
        delete smoother;
        for(int i=0; i<depth; i++) {
            utils::SafeFree(&rr[i]);
            utils::SafeFree(&ee[i]);
            utils::SafeFree(&xx[i]);
            if(i == depth-1) {
                delete lu_solver;
            }else {
                delete C_A[i];
            }
            delete Intrpl[i];
            delete Intrpl_t[i];
        }
        utils::SafeFree(&rr);
        utils::SafeFree(&ee);
        utils::SafeFree(&xx);
        utils::SafeFree(&C_A);
        utils::SafeFree(&Intrpl);
        utils::SafeFree(&Intrpl_t);
    }
    void Precondition(int *in, int *out) {
        smoother[0]->Smooth(in, out, false);
        A->SpMV(out, r);
        blas1::Sub<int>(in, r, A->GetN());
        Intrpl_t[0]->SpMV(r, ee[0]);

        for(int i=0; i<depth-1; i++) {
            smoother[i+1]->Smooth(ee[i], xx[i], false);
            C_A[i]->SpMV(xx[i], rr[i]);
            blas1::Sub<int>(ee[i], rr[i], C_A[i]->GetN());
            Intrpl_t[i+1]->SpMV(rr[i], ee[i+1]);
        }
        
        lu_solver->Solve(ee[depth-1], xx[depth-1]);

        for(int i=depth-2; i>=0; i--) {
            Intrpl[i+1]->SpMV(xx[i+1], rr[i]);
            blas1::Add<int>(rr[i], xx[i], C_A[i]->GetN());
            smoother[i+1]->Smooth(ee[i], xx[i], true);
        }
        Intrpl[0]->SpMV(xx[0], r);
        blas1::Add<int>(r, out, A->GetN());
        
        smoother[0]->Smooth(in, out, true);
    }
};

} // senk

#endif
