#ifndef SENKPP_HELPER_MULTIGRID_HPP
#define SENKPP_HELPER_MULTIGRID_HPP

#include <cmath>
#include <vector>
#include "utils/utils.hpp"
#include "helper/helper_sparse.hpp"

namespace senk {

namespace helper {

namespace multigrid {

/** Prototype declaration **/
void get_strong_weak_set( // Week contains the diagonal elements.
    CSRMat *A, CSRMat **Stg, CSRMat **Week, double theta);
void get_influence_set(   // Transpose the Stg matrix.
    CSRMat *Stg, CSRMat **Stg_t);
void first_pass(
    CSRMat *Stg, CSRMat *Stg_t, int **is_coarse);
int second_pass(
    CSRMat *Stg, CSRMat *Stg_t, int *is_coarse);
void sepalate_strong_to_coarse_fine( // C_Stg for coarse-grid; DS_Stg for fine-grid.
    CSRMat *Stg, CSRMat **C_Stg, CSRMat **DS_Stg, int *is_coarse);
void get_interpolation_matrix(
    CSRMat *A, CSRMat *C_Stg, CSRMat *DS_Stg, CSRMat *Week,
    CSRMat **Intrpl, int *is_coarse);
void transpose_interpolation_matrix( // Intrpl_t equals the Restriction matrix.
    CSRMat *Intrpl, CSRMat **Intrpl_t);
void get_coarse_matrix(
    CSRMat *A, CSRMat *Intrpl, CSRMat *Intrpl_t, CSRMat **C_A);

void get_amg_set(
    CSRMat *A,
    CSRMat **C_A, CSRMat **Intrpl, CSRMat **Intrpl_t,
    double theta)
{
    if(theta <= 0 || 1 < theta) {
        printf("Error: Theta must be in 0 < theta <= 1\n");
        exit(EXIT_FAILURE);
    }
    CSRMat *Stg, *Week;
    CSRMat *Stg_t;
    CSRMat *C_Stg, *DS_Stg;
    get_strong_weak_set(A, &Stg, &Week, theta);
    get_influence_set(Stg, &Stg_t);
    int *is_coarse;
    first_pass(Stg, Stg_t, &is_coarse);
    // int C_N = second_pass(Stg, Stg_t, is_coarse);
    // printf("%d\n", C_N);
    second_pass(Stg, Stg_t, is_coarse);
    sepalate_strong_to_coarse_fine(Stg, &C_Stg, &DS_Stg, is_coarse);
    get_interpolation_matrix(A, C_Stg, DS_Stg, Week, Intrpl, is_coarse);
    transpose_interpolation_matrix(*Intrpl, Intrpl_t);
    get_coarse_matrix(A, *Intrpl, *Intrpl_t, C_A); 
    
    Stg->Free(); delete Stg;
    Stg_t->Free(); delete Stg_t;
    C_Stg->Free(); delete C_Stg;
    DS_Stg->Free(); delete DS_Stg;
    free(is_coarse);
}

/** Suplimental functions **/

void get_strong_weak_set(
    CSRMat *A, CSRMat **Stg, CSRMat **Week, double theta)
{
    int N = A->N;
    int stgNNZ = 0;
    int weekNNZ = 0;
    double *max = utils::SafeCalloc<double>(N);
    for(int i=0; i<N; i++) {
        for(int j=A->rptr[i]; j<A->rptr[i+1]; j++) {
            if(A->cind[j] == i) continue;
            if(std::fabs(A->val[j]) > max[i]) max[i] = std::fabs(A->val[j]);
        }
        for(int j=A->rptr[i]; j<A->rptr[i+1]; j++) {
            if(A->cind[j] == i) continue;
            if(std::fabs(A->val[j]) >= theta * max[i]) { stgNNZ++; }
        }
    }
    weekNNZ = A->rptr[N]-stgNNZ;
    double *s_val  = utils::SafeMalloc<double>(stgNNZ);
    int *s_cind    = utils::SafeMalloc<int>(stgNNZ);
    int *s_rptr    = utils::SafeMalloc<int>(N+1);
    double *wd_val = utils::SafeMalloc<double>(weekNNZ);
    int *wd_cind   = utils::SafeMalloc<int>(weekNNZ);
    int *wd_rptr   = utils::SafeMalloc<int>(N+1);
    stgNNZ = 0; weekNNZ = 0;
    s_rptr[0] = stgNNZ; wd_rptr[0] = weekNNZ;
    for(int i=0; i<N; i++) {
        for(int j=A->rptr[i]; j<A->rptr[i+1]; j++) {
            if(A->cind[j] != i && std::fabs(A->val[j]) >= theta * max[i]) {
                s_val[stgNNZ] = A->val[j]; s_cind[stgNNZ] = A->cind[j];
                stgNNZ++;
            }else {
                wd_val[weekNNZ] = A->val[j]; wd_cind[weekNNZ] = A->cind[j];
                weekNNZ++;
            }
        }
        s_rptr[i+1] = stgNNZ; wd_rptr[i+1] = weekNNZ;
    }
    *Stg = new CSRMat(s_val, s_cind, s_rptr, N, N, MMShape::General, MMType::Real);
    *Week = new CSRMat(wd_val, wd_cind, wd_rptr, N, N, MMShape::General, MMType::Real);
    free(max);
}

void get_influence_set(
    CSRMat *Stg, CSRMat **Stg_t)
{
    int N = Stg->N;
    int *num = utils::SafeCalloc<int>(N);
    for(int i=0; i<N; i++) {
        for(int j=Stg->rptr[i]; j<Stg->rptr[i+1]; j++) {
            num[Stg->cind[j]]++;
        }
    }
    int *st_cind = utils::SafeMalloc<int>(Stg->rptr[N]);
    int *st_rptr = utils::SafeMalloc<int>(N+1);
    st_rptr[0] = 0;
    for(int i=0; i<N; i++) {
        st_rptr[i+1] = st_rptr[i] + num[i];
        num[i] = 0;
    }
    for(int i=0; i<N; i++) {
        for(int j=Stg->rptr[i]; j<Stg->rptr[i+1]; j++) {
            int ind = Stg->cind[j];
            st_cind[st_rptr[ind]+num[ind]] = i;
            num[ind]++;
        }
    }
    *Stg_t = new CSRMat(nullptr, st_cind, st_rptr, N, N, MMShape::General, MMType::Real);
    free(num);
}

void first_pass(CSRMat *Stg, CSRMat *Stg_t, int **is_coarse) {
    int N = Stg->N;
    (*is_coarse)     = utils::SafeMalloc<int>(N);
    int *is_visited  = utils::SafeCalloc<int>(N);

    int *st_cardinal = utils::SafeCalloc<int>(N);
    int *s_cardinal  = utils::SafeCalloc<int>(N);
    int *influence   = utils::SafeCalloc<int>(N);
    int *index       = utils::SafeCalloc<int>(N);
    int *where       = utils::SafeCalloc<int>(N);
    for(int i=0; i<N; i++) {
        st_cardinal[i] = Stg_t->rptr[i+1] - Stg_t->rptr[i];
        s_cardinal[i]  = Stg->rptr[i+1] - Stg->rptr[i];
        index[i] = i;
        if(st_cardinal[i] == 0 && s_cardinal[i] == 0) {
            (*is_coarse)[i] = 0; is_visited[i] = 1;
        }else {
            (*is_coarse)[i] = -38;
        }
        for(int j=Stg_t->rptr[i]; j<Stg_t->rptr[i+1]; j++) {
            int idx = Stg_t->cind[j];
            influence[i] += Stg->rptr[idx+1] - Stg->rptr[idx];
        }
    }
    utils::QuickSort<int, int*, int*, int*>(0, N-1, false, st_cardinal, s_cardinal, influence, index);

    int max_crdnl = st_cardinal[0];
    std::vector<int> crdnl_ptr(max_crdnl+2);
    crdnl_ptr[max_crdnl+1] = 0;
    int strt = 0;
    for(int i=max_crdnl; i>=0; i--) {
        int num = 0;
        bool flag = false;
        for(int j=strt; j<N; j++) {
            if(st_cardinal[j] == i) {num++;}
            else {
                crdnl_ptr[i] = crdnl_ptr[i+1] + num;
                strt = j;
                flag = true;
                break;
            }
        }
        if(flag) continue;
        crdnl_ptr[i] = crdnl_ptr[i+1] + num;
    }
    for(int i=max_crdnl; i>=0; i--) {
        if(crdnl_ptr[i] - crdnl_ptr[i+1] > 1) {
            // utils::QuickSort<int, int*>(crdnl_ptr[i+1], crdnl_ptr[i]-1, false, s_cardinal, index);
            utils::QuickSort<int, int*>(crdnl_ptr[i+1], crdnl_ptr[i]-1, true, influence, index);
        }
    }

    for(int i=0; i<N; i++) { where[index[i]] = i; }
    int pos = 0;
    while(pos<N) {
        while(pos<N && is_visited[index[pos]]) pos++;
        if(pos >= N) break;
        int c_idx = index[pos];
        is_visited[c_idx] = 1;
        (*is_coarse)[c_idx] = 1; // 1 = Coarse
        for(int i=Stg_t->rptr[c_idx]; i<Stg_t->rptr[c_idx+1]; i++) {
            int f_idx = Stg_t->cind[i];
            if(is_visited[f_idx]) continue;
            is_visited[f_idx] = 1;
            (*is_coarse)[f_idx] = 0; // 0 = Fine;

            for(int j=Stg->rptr[f_idx]; j<Stg->rptr[f_idx+1]; j++) {
                if(is_visited[Stg->cind[j]]) continue;
                int idx = where[Stg->cind[j]];
                if(idx <= pos) {
                    printf("Warning 1: %d %d\n", idx, pos);
                }
                st_cardinal[idx]++;
                int obj;
                if(st_cardinal[idx] > max_crdnl) {
                    max_crdnl = st_cardinal[idx];
                    if(crdnl_ptr[st_cardinal[idx]] > pos+1) {
                        printf("Warning 2: %d %d\n", crdnl_ptr[st_cardinal[idx]], pos+1);
                    }
                    obj = pos+1;
                    crdnl_ptr.push_back(pos+1);
                    crdnl_ptr[max_crdnl] = pos+2;
                }else {
                    if(crdnl_ptr[st_cardinal[idx]] > pos+1) {
                        obj = crdnl_ptr[st_cardinal[idx]];
                        crdnl_ptr[st_cardinal[idx]] += 1;
                    }else {
                        obj = pos+1;
                        crdnl_ptr[st_cardinal[idx]] = pos+2;
                    }
                }
                utils::swap<int>(&st_cardinal[idx], &st_cardinal[obj]);
                utils::swap<int>(&index[idx], &index[obj]);
                utils::swap<int>(&where[index[idx]], &where[index[obj]]);
            }
        }
        for(int i=Stg->rptr[c_idx]; i<Stg->rptr[c_idx+1]; i++) {
            if(is_visited[Stg->cind[i]]) continue;
            int idx = where[Stg->cind[i]];
            if(st_cardinal[idx] == 0) { continue; }
            st_cardinal[idx]--;
            int obj;
            obj = crdnl_ptr[st_cardinal[idx]+1]-1;
            crdnl_ptr[st_cardinal[idx]+1] -= 1;
            utils::swap<int>(&st_cardinal[idx], &st_cardinal[obj]);
            utils::swap<int>(&index[idx], &index[obj]);
            utils::swap<int>(&where[index[idx]], &where[index[obj]]);
        }
        
        pos++;
        // if(pos % 1000 == 0) {
        //     // printf("yes\n");
        //     for(int l=pos+1; l<N-1; l++) {
        //         if(st_cardinal[l] < st_cardinal[l+1]) {
        //             for(int o=pos+1; o<N; o++) {
        //                 printf("%d %d\n", o, st_cardinal[o]);
        //             }
        //             exit(1);
        //         }
        //     }
        // }
    }
    free(st_cardinal);
    free(s_cardinal);
    free(influence);
    free(is_visited);
    free(index);
    free(where);
}

/*
void first_pass(CSRMat *Stg, CSRMat *Stg_t, int **is_coarse) {
    int N = Stg->N;
    (*is_coarse)     = utils::SafeMalloc<int>(N);
    int *is_visited  = utils::SafeCalloc<int>(N);

    int *st_cardinal = utils::SafeCalloc<int>(N);
    int *s_cardinal  = utils::SafeCalloc<int>(N);
    int *index       = utils::SafeCalloc<int>(N);
    int *where       = utils::SafeCalloc<int>(N);
    for(int i=0; i<N; i++) {
        st_cardinal[i] = Stg_t->rptr[i+1] - Stg_t->rptr[i];
        s_cardinal[i] = Stg->rptr[i+1] - Stg->rptr[i];
        index[i] = i;
        if(st_cardinal[i] == 0 && st_cardinal[i] == 0) {
            (*is_coarse)[i] = 0;
            is_visited[i] = 1;
        }
    }
    utils::QuickSort<int, int*, int*>(0, N-1, false, st_cardinal, s_cardinal, index);
    
    int max_crdnl = st_cardinal[0];
    std::vector<int> crdnl_ptr(max_crdnl+1, 0);
    int strt = 0;
    for(int i=max_crdnl; i>=0; i--) {
        int num = 0;
        bool flag = false;
        for(int j=strt; j<N; j++) {
            if(st_cardinal[j] == i) {num++;}
            else {
                crdnl_ptr[i-1] = crdnl_ptr[i] + num;
                strt = j;
                flag = true;
                break;
            }
        }
        if(flag) continue;
        crdnl_ptr[i-1] = crdnl_ptr[i] + num;
    }

    for(int i=max_crdnl; i>=0; i--) {
        utils::QuickSort<int, int*>(crdnl_ptr[i], crdnl_ptr[i-1]-1, false, s_cardinal, index);
    }

    for(int i=0; i<N; i++) { where[index[i]] = i; }
    int pos = 0;
    while(pos<N) {
        while(pos<N && is_visited[index[pos]]) pos++;
        if(pos >= N) break;
        int c_idx = index[pos];
        is_visited[c_idx] = 1;
        (*is_coarse)[c_idx] = 1; // 1 = Coarse
        for(int i=Stg_t->rptr[c_idx]; i<Stg_t->rptr[c_idx+1]; i++) {
            int f_idx = Stg_t->cind[i];
            if(is_visited[f_idx]) continue;
            is_visited[f_idx] = 1;
            (*is_coarse)[f_idx] = 0; // 0 = Fine;

            for(int j=Stg->rptr[f_idx]; j<Stg->rptr[f_idx+1]; j++) {
                if(is_visited[Stg->cind[j]]) continue;
                int idx = where[Stg->cind[j]];
                st_cardinal[idx]++;
                int obj;
                if(st_cardinal[idx] > max_crdnl) {
                    max_crdnl = st_cardinal[idx];
                    if(crdnl_ptr[st_cardinal[idx]-1] > pos+1) {
                        printf("Warning: %d %d\n", crdnl_ptr[st_cardinal[idx]-1], pos+1);
                    }
                    obj = pos+1;
                    crdnl_ptr.push_back(pos+1);
                    crdnl_ptr[max_crdnl-1] = pos+2;
                }else {
                    if(crdnl_ptr[st_cardinal[idx]-1] > pos+1) {
                        obj = crdnl_ptr[st_cardinal[idx]-1];
                        crdnl_ptr[st_cardinal[idx]-1] += 1;
                    }else {
                        obj = pos+1;
                        crdnl_ptr[st_cardinal[idx]-1] = pos+2;
                    }
                }
                utils::swap<int>(&st_cardinal[idx], &st_cardinal[obj]);
                utils::swap<int>(&index[idx], &index[obj]);
                utils::swap<int>(&where[index[idx]], &where[index[obj]]);
            }
        }
        for(int i=Stg->rptr[c_idx]; i<Stg->rptr[c_idx+1]; i++) {
            if(is_visited[Stg->cind[i]]) continue;
            int idx = where[Stg->cind[i]];
            st_cardinal[idx]--;
            if(st_cardinal[idx] < 0) {
                printf("This has not been supported yet.\n");
                exit(1);
            }
            int obj;
            obj = crdnl_ptr[st_cardinal[idx]]-1;
            crdnl_ptr[st_cardinal[idx]] -= 1;
            utils::swap<int>(&st_cardinal[idx], &st_cardinal[obj]);
            utils::swap<int>(&index[idx], &index[obj]);
            utils::swap<int>(&where[index[idx]], &where[index[obj]]);
        }
        pos++;
    }
    free(st_cardinal);
    free(s_cardinal);
    free(is_visited);
    free(index);
    free(where);
}
*/

int second_pass(
    CSRMat *Stg, CSRMat *Stg_t, int *is_coarse)
{
    int N = Stg->N;
    int num_coarse = 0;
    for(int i=0; i<N; i++) {
        if(is_coarse[i] == -38) { exit(1); }
        if(is_coarse[i]) continue;
        for(int j=Stg->rptr[i]; j<Stg->rptr[i+1]; j++) { // i is a fine point.
            if(is_coarse[Stg->cind[j]]) continue;
            int f_idx = Stg->cind[j];
            int flag = 0;
            int main_ptr = Stg->rptr[i];
            int sub_ptr = Stg->rptr[f_idx];
            while(main_ptr < Stg->rptr[i+1] && sub_ptr < Stg->rptr[f_idx+1]) {
                if(Stg->cind[sub_ptr] < Stg->cind[main_ptr]) {
                    sub_ptr++;
                }else if(Stg->cind[sub_ptr] == Stg->cind[main_ptr]) {
                    if(is_coarse[Stg->cind[main_ptr]]) {
                        flag = 1;
                        break;
                    }
                    sub_ptr++;
                    main_ptr++;
                }else {
                    main_ptr++;
                }
            }
            if(flag) continue;
            is_coarse[i] = 1; // Change into a C-point
            break;
            // is_coarse[f_idx] = 1; // Change into a C-point
        }
    }
    for(int i=0; i<N; i++) {
        if(is_coarse[i]) num_coarse++;
    }
    // printf("%d\n", num_coarse);
    return num_coarse;
}

void sepalate_strong_to_coarse_fine(
    CSRMat *Stg, CSRMat **C_Stg, CSRMat **DS_Stg, int *is_coarse)
{
    int N = Stg->N;
    int cNNZ = 0, dsNNZ = 0;
    for(int i=0; i<N; i++) {
        for(int j=Stg->rptr[i]; j<Stg->rptr[i+1]; j++) {
            if(is_coarse[Stg->cind[j]]) cNNZ++;
            else dsNNZ++;
        }
    }
    double *c_val  = utils::SafeMalloc<double>(cNNZ);
    int *c_cind    = utils::SafeMalloc<int>(cNNZ);
    int *c_rptr    = utils::SafeMalloc<int>(N+1);
    double *ds_val = utils::SafeMalloc<double>(dsNNZ);
    int *ds_cind   = utils::SafeMalloc<int>(dsNNZ);
    int *ds_rptr   = utils::SafeMalloc<int>(N+1);
    cNNZ = 0; dsNNZ = 0;
    c_rptr[0] = cNNZ; ds_rptr[0] = dsNNZ;
    for(int i=0; i<N; i++) {
        for(int j=Stg->rptr[i]; j<Stg->rptr[i+1]; j++) {
            if(is_coarse[Stg->cind[j]]) {
                c_val[cNNZ]  = Stg->val[j];
                c_cind[cNNZ] = Stg->cind[j];
                cNNZ++;
            }else {
                ds_val[dsNNZ]  = Stg->val[j];
                ds_cind[dsNNZ] = Stg->cind[j];
                dsNNZ++;
            }
        }
        c_rptr[i+1]  = cNNZ;
        ds_rptr[i+1] = dsNNZ;
    }
    *C_Stg = new CSRMat(c_val, c_cind, c_rptr, N, N, MMShape::General, MMType::Real);
    *DS_Stg = new CSRMat(ds_val, ds_cind, ds_rptr, N, N, MMShape::General, MMType::Real);
}

void get_interpolation_matrix(
    CSRMat *A, CSRMat *C_Stg, CSRMat *DS_Stg, CSRMat *Week,
    CSRMat **Intrpl, int *is_coarse)
{
    int N = A->N;
    int *col = utils::SafeMalloc<int>(N);
    int count = 0;
    for(int i=0; i<N; i++) {
        if(is_coarse[i]) { col[i] = count; count++; }
    }
    int *i_rptr = utils::SafeMalloc<int>(N+1);
    int nnz = 0;
    i_rptr[0] = nnz;
    for(int i=0; i<N; i++) {
        if(is_coarse[i]) { nnz++; }
        else {
            for(int j=C_Stg->rptr[i]; j<C_Stg->rptr[i+1]; j++) nnz++;
        }
        i_rptr[i+1] = nnz;
    }
    double *i_val = utils::SafeMalloc<double>(nnz);
    int *i_cind   = utils::SafeMalloc<int>(nnz);
    nnz = 0;
    for(int i=0; i<N; i++) {
        if(is_coarse[i]) {
            i_val[nnz] = 1;
            i_cind[nnz] = col[i];
            nnz++;
        }else {
            double deno = 0;
            for(int j=Week->rptr[i]; j<Week->rptr[i+1]; j++) deno += Week->val[j];
            for(int j=C_Stg->rptr[i]; j<C_Stg->rptr[i+1]; j++) {
                double aij = C_Stg->val[j];
                for(int m=DS_Stg->rptr[i]; m<DS_Stg->rptr[i+1]; m++) { 
                    double amj = 0;
                    for(int l=A->rptr[DS_Stg->cind[m]]; l<A->rptr[DS_Stg->cind[m]+1]; l++) {
                        if(A->cind[l] == C_Stg->cind[j]) {
                            amj = A->val[l]; break;
                        }
                    }
                    bool flag = false;
                    double amk = 0;
                    int m_ptr = A->rptr[DS_Stg->cind[m]];
                    int k_ptr = C_Stg->rptr[i];
                    while(m_ptr < A->rptr[DS_Stg->cind[m]+1] && k_ptr < C_Stg->rptr[i+1]) {
                        if(A->cind[m_ptr] < C_Stg->cind[k_ptr]) {
                            m_ptr++;
                        }else if(A->cind[m_ptr] == C_Stg->cind[k_ptr]) {
                            amk += A->val[m_ptr];
                            flag = true;
                            m_ptr++;
                            k_ptr++;
                        }else {
                            k_ptr++;
                        }
                    }
                    if(amk==0) {
                        if(flag) {
                            printf("1 amk = 0\n"); // exit(1);
                            amk = 0.00001;
                        }else {
                            printf("2 amk = 0\n"); exit(1); 
                        }
                    }
                    aij += DS_Stg->val[m] * amj / amk;
                }
                i_val[nnz] = -aij / deno;
                i_cind[nnz] = col[C_Stg->cind[j]];
                nnz++;
            }
        }
        i_rptr[i+1] = nnz;
    }
    free(col);
    *Intrpl = new CSRMat(i_val, i_cind, i_rptr, N, count, MMShape::General, MMType::Real);
}

void transpose_interpolation_matrix(
    CSRMat *Intrpl, CSRMat **Intrpl_t)
{
    int N = Intrpl->N, M = Intrpl->M;
    int *num = utils::SafeCalloc<int>(M);
    for(int i=0; i<N; i++) {
        for(int j=Intrpl->rptr[i]; j<Intrpl->rptr[i+1]; j++) {
            num[Intrpl->cind[j]]++;
        }
    }
    int *it_rptr = utils::SafeMalloc<int>(M+1);
    it_rptr[0] = 0;
    for(int i=0; i<M; i++) {
        it_rptr[i+1] = it_rptr[i] + num[i];
        num[i] = 0;
    }
    double *it_val = utils::SafeMalloc<double>(it_rptr[M]);
    int *it_cind   = utils::SafeMalloc<int>(it_rptr[M]);
    for(int i=0; i<N; i++) {
        for(int j=Intrpl->rptr[i]; j<Intrpl->rptr[i+1]; j++) {
            int st = it_rptr[Intrpl->cind[j]];
            int off = num[Intrpl->cind[j]];
            it_val[st+off] = Intrpl->val[j];
            it_cind[st+off] = i;
            num[Intrpl->cind[j]]++;
        }
    }
    *Intrpl_t = new CSRMat(it_val, it_cind, it_rptr, M, N, MMShape::General, MMType::Real);
}

void get_coarse_matrix(
    CSRMat *A, CSRMat *Intrpl, CSRMat *Intrpl_t, CSRMat **C_A)
{
    int N = Intrpl->N, M = Intrpl->M;
    double *t_val;
    int *t_rind, *t_cptr;
    double *c_val;
    int *c_cind, *c_rptr;
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
    *C_A = new CSRMat(c_val, c_cind, c_rptr, M, M, MMShape::General, MMType::Real);
}

} // multigrid

} // helper

} // senk

#endif