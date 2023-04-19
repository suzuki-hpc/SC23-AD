#ifndef SENKPP_HELPER_SA_MULTIGRID_HPP
#define SENKPP_HELPER_SA_MULTIGRID_HPP

#include <cmath>
#include "enums.hpp"
#include "utils/utils.hpp"
#include "helper/helper_sparse.hpp"

namespace senk {

namespace helper {

namespace multigrid {

/** Prototype declaration **/
CSRMat *filtering(CSRMat *A, double **diag, double theta);
int aggregating(CSRMat *FilteredA, double *diag, int **aggregation);
CSRMat *smoothing(CSRMat *FilteredA, double *diag, int *aggregation, int size, double omega);
CSRMat *get_coarse_matrix(
    CSRMat *A, CSRMat *Intrpl, CSRMat *Intrpl_t);

void get_sa_amg_set(
    CSRMat *A,
    CSRMat **C_A, CSRMat **Intrpl, CSRMat **Intrpl_t,
    double theta)
{
    if(theta < 0 || 1 <= theta) {
        printf("Error: SA_AMG theta must be in 0 <= theta < 1\n");
        exit(EXIT_FAILURE);
    }

    double *diag;
    CSRMat *FilteredA = filtering(A, &diag, theta);
    int *aggregation, coarseSize;
    coarseSize = aggregating(FilteredA, diag, &aggregation);
    double omega = 1.0;
    
    *Intrpl = smoothing(FilteredA, diag, aggregation, coarseSize, omega);
    *Intrpl_t = (*Intrpl)->CopyAsTransposedCSRMat();
    *C_A = get_coarse_matrix(A, *Intrpl, *Intrpl_t);
    
    FilteredA->Free();
    delete FilteredA;
}

/** Suplimental functions **/
CSRMat *filtering(CSRMat *A, double **diag, double theta)
{
    double *val = utils::SafeMalloc<double>(A->rptr[A->N]);
    int *cind   = utils::SafeMalloc<int>(A->rptr[A->N]);
    int *rptr   = utils::SafeMalloc<int>(A->N);
    *diag = utils::SafeMalloc<double>(A->N);
    for(int i=0; i<A->N; i++) {
        for(int j=A->rptr[i]; j<A->rptr[i+1]; j++) {
            if(A->cind[j] == i) {
                (*diag)[i] = A->val[j]; break;
            }
        }
    }
    double theta_2 = theta * theta;
    int cnt = 0;
    rptr[0] = cnt;
    for(int i=0; i<A->N; i++) {
        double a_ii = fabs((*diag)[i]);
        double sum = 0;
        int diag_ptr = 0;
        for(int j=A->rptr[i]; j<A->rptr[i+1]; j++) {
            double a_jj = fabs((*diag)[A->cind[j]]);
            if(A->val[j]*A->val[j] <= theta_2*a_ii*a_jj) {
                sum += A->val[j]; continue;
            }
            if(A->cind[j] == i) diag_ptr = cnt;
            val[cnt] = A->val[j];
            cind[cnt] = A->cind[j];
            cnt++;
        }
        val[diag_ptr] -= sum;
        rptr[i+1] = cnt;
    }
    val  = utils::SafeRealloc<double>(val, cnt);
    cind = utils::SafeRealloc<int>(cind, cnt);
    CSRMat *res = new CSRMat(
        val, cind, rptr, A->N, A->M, A->shape, A->type);
    return res;
}

int aggregating(CSRMat *FilteredA, double *diag, int **aggregation)
{
    int N = FilteredA->N;
    *aggregation = utils::SafeCalloc<int>(N);
    // Step 1
    int now = 1;
    for(int i=0; i<N; i++) {
        if(FilteredA->rptr[i+1]-FilteredA->rptr[i] <= 1) continue;
        if((*aggregation)[i] != 0) continue;
        int flag = 0;
        for(int j=FilteredA->rptr[i]; j<FilteredA->rptr[i+1]; j++) {
            if((*aggregation)[FilteredA->cind[j]] == 0) continue;
            flag = 1;
            break;
        }
        if(flag) continue;
        for(int j=FilteredA->rptr[i]; j<FilteredA->rptr[i+1]; j++) {
            (*aggregation)[FilteredA->cind[j]] = now;
        }
        now++;
    }
    // Step 2
    int *copy = utils::SafeCalloc<int>(N);
    for(int i=0; i<N; i++) { copy[i] = (*aggregation)[i]; }
    for(int i=0; i<N; i++) {
        if(FilteredA->rptr[i+1]-FilteredA->rptr[i] <= 1) continue;
        if((*aggregation)[i] != 0) continue;
        int id = 0;
        double max = 0;
        for(int j=FilteredA->rptr[i]; j<FilteredA->rptr[i+1]; j++) {
            if(copy[FilteredA->cind[j]] == 0) continue;
            if(FilteredA->cind[j] == i) continue;
            if(fabs(FilteredA->val[j]/diag[i]/diag[FilteredA->cind[j]]) > max){
                id = copy[FilteredA->cind[j]];
                max = fabs(FilteredA->val[j]/diag[i]/diag[FilteredA->cind[j]]);
            }
        }
        (*aggregation)[i] = id;
    }
    // Step 3
    for(int i=0; i<FilteredA->N; i++) {
        if(FilteredA->rptr[i+1]-FilteredA->rptr[i] <= 1) continue;
        if((*aggregation)[i] != 0) continue;
        (*aggregation)[i] = now;
        for(int j=FilteredA->rptr[i]; j<FilteredA->rptr[i+1]; j++) {
            if((*aggregation)[FilteredA->cind[j]] != 0) continue;
            (*aggregation)[FilteredA->cind[j]] = now;
        }
        now++;
    }
    /*
    for(int i=0; i<fA->N; i++) {
        if((*aggregation)[i] != 0) continue;
        if(fA->rptr[i+1]-fA->rptr[i] <= 1) continue;
        printf("hoge\n");
    }
    */
    free(copy);
    return now-1;
}

CSRMat *smoothing(
    CSRMat *FilteredA, double *diag, int *aggregation,
    int coarseSize, double omega)
{
    int N = FilteredA->N;
    // Creating a temporal interpolater.
    double *val  = utils::SafeMalloc<double>(N);
    int *cind = utils::SafeMalloc<int>(N);
    int *rptr = utils::SafeMalloc<int>(coarseSize+1);

    int *temp = utils::SafeCalloc<int>(coarseSize);
    for(int i=0; i<N; i++) {
        if(aggregation[i] == 0) { continue; }
        temp[aggregation[i]-1]++;
    }
    rptr[0] = 0;
    for(int i=0; i<coarseSize; i++) {
        rptr[i+1] = rptr[i] + temp[i];
        temp[i] = 0;
    }
    for(int i=0; i<N; i++) {
        if(aggregation[i] == 0) continue;
        int start = rptr[aggregation[i]-1];
        int off = temp[aggregation[i]-1];
        val[start+off] = 1;
        cind[start+off] = i;
        temp[aggregation[i]-1]++;
    }
    for(int i=0; i<N; i++) {
        for(int j=FilteredA->rptr[i]; j<FilteredA->rptr[i+1]; j++) {
            double t = omega / FilteredA->val[j] * diag[i];
            FilteredA->val[j] = (FilteredA->cind[j] == i)? 1-t: -t;
        }
    }

    double *t_val;
    int *t_cind, *t_rptr;
    helper::matrix::csr_to_csc(
        val, cind, rptr,
        &t_val, &t_cind, &t_rptr, coarseSize, N);

    double *r_val;
    int *r_cind, *r_rptr;
    helper::sparse::cscSpMM(
        t_val, t_cind, t_rptr,
        FilteredA->val, FilteredA->cind, FilteredA->rptr,
        &r_val, &r_cind, &r_rptr,
        coarseSize, N, N);
    free(t_val);
    free(t_cind);
    free(t_rptr);

    CSRMat *Intrpl = new CSRMat(
        r_val, r_cind, r_rptr,
        N, coarseSize, MMShape::General, MMType::Real);

    free(temp);
    free(val);
    free(cind);
    free(rptr);

    return Intrpl;
}

CSRMat *get_coarse_matrix(
    CSRMat *A, CSRMat *Intrpl, CSRMat *Intrpl_t)
{
    int N = Intrpl->N, M = Intrpl->M;
    double *t_val;
    int *t_rind;
    int *t_cptr;
    double *c_val;
    int *c_cind;
    int *c_rptr;
    helper::sparse::cscSpMM(
        A->val, A->cind, A->rptr,
        Intrpl_t->val, Intrpl_t->cind, Intrpl_t->rptr,
        &t_val, &t_rind, &t_cptr,
        N, N, M);
    helper::sparse::cscSpMM(
        Intrpl->val, Intrpl->cind, Intrpl->rptr,
        t_val, t_rind, t_cptr,
        &c_val, &c_cind, &c_rptr,
        M, N, M);
    free(t_val);
    free(t_rind);
    free(t_cptr);
    CSRMat *C_A = new CSRMat(
        c_val, c_cind, c_rptr, M, M, MMShape::General, MMType::Real);
    return C_A;
}

} // multigrid

} // helper

} // senk

#endif