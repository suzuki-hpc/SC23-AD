#ifndef SENKPP_HELPER_MATRIX_HPP
#define SENKPP_HELPER_MATRIX_HPP

#include <cstdio>
#include <cstring>

#include "utils/utils.hpp"
#include "helper/helper_sparse.hpp"

namespace senk {

namespace helper {

namespace matrix {

template <typename T>
void csr_to_csc(
    T *val, int *cind, int *rptr,
    T **cval, int **crind, int **ccptr,
    int N, int M)
{
    // The size of the input matrix is N * M
    int nnz = rptr[N];
    *cval  = utils::SafeMalloc<T>(nnz);
    *crind = utils::SafeMalloc<int>(nnz);
    *ccptr = utils::SafeMalloc<int>(M+1);
    int *num = utils::SafeCalloc<int>(M);
    for(int i=0; i<N; i++) {
        for(int j=rptr[i]; j<rptr[i+1]; j++) { num[cind[j]]++; }
    }
    (*ccptr)[0] = 0;
    for(int i=0; i<M; i++) {
        (*ccptr)[i+1] = (*ccptr)[i] + num[i];
        num[i] = 0;
    }
    for(int i=0; i<N; i++) {
        for(int j=rptr[i]; j<rptr[i+1]; j++) {
            int off = (*ccptr)[cind[j]];
            int pos = num[cind[j]];
            (*crind)[off+pos] = i;
            (*cval)[off+pos] = val[j];
            num[cind[j]]++;
        }
    }
    free(num);
}

template <typename T>
void csr_to_sell32(
    T *val, int *cind, int *rptr,
    T **sval, int **scind, int **srptr, int N)
{
    if(N%32 != 0) {
        std::cerr << "N must be a multiple of 32." << std::endl;
        exit(1);
    }
    int N_32 = N / 32;
    *srptr = utils::SafeMalloc<int>(N_32+1);
    int row_id;
    int row_max;
    int row_count;
    int nnz = 0;
    (*srptr)[0] = 0;
    for(int i=0; i<N_32; i++) {
        row_max = 0;
        for(int j=0; j<32; j++) {
            row_id = i*32+j;
            if(row_max < rptr[row_id+1] - rptr[row_id]) {
                row_max = rptr[row_id+1] - rptr[row_id];
            }
        }
        (*srptr)[i+1] = (*srptr)[i] + row_max;
        nnz += row_max * 32;
    }
    *sval  = utils::SafeMalloc<double>(nnz);
    *scind = utils::SafeMalloc<int>(nnz);
    for(int i=0; i<N_32; i++) {
        for(int j=0; j<32; j++) {
            row_id = i*32+j;
            row_count = 0;
            for(int k=rptr[row_id]; k<rptr[row_id+1]; k++) {
                (*sval)[(*srptr)[i]*32+row_count*32+j] = val[k];
                (*scind)[(*srptr)[i]*32+row_count*32+j] = cind[k];
                row_count++;
            }
            if(row_count < (*srptr)[i+1] - (*srptr)[i]) {
                int temp = (*srptr)[i+1] - (*srptr)[i] - row_count;
                for(int k=0; k<temp; k++) {
                    (*sval)[(*srptr)[i]*32+row_count*32+j] = 0;
                    (*scind)[(*srptr)[i]*32+row_count*32+j] = 0;
                    row_count++;
                }
            }
        }
    }
}

inline void csr_to_mcsell32(
    double *val, int *cind, int *rptr,
    double **sval, int **scind, int **srptr, int **snnz,
    int **c_s_num,
    int N, int *c_size, int c_num)
{
    *c_s_num = utils::SafeMalloc<int>(c_num+1);
    (*c_s_num)[0] = 0;
    int s_num = 0; // The number of the slices.
    for(int cid=0; cid<c_num; cid++) {
        int size = c_size[cid+1] - c_size[cid];
        s_num += (size+32-1) / 32;
        (*c_s_num)[cid+1] = s_num;
    }
    *srptr = utils::SafeMalloc<int>(s_num+1);
    *snnz  = utils::SafeMalloc<int>(s_num+1);
 
    *sval  = nullptr;
    *scind = nullptr;
    int nnz = 0;   
    //Count the row length of each slice
    s_num = 0;
    (*srptr)[0] = 0;
    (*snnz)[0] = 0;
    for(int cid=0; cid<c_num; cid++) {
        int offset = c_size[cid];
        int size = c_size[cid+1] - c_size[cid];
        int s_num_per_c = (size+32-1)/32;
        for(int j=0; j<s_num_per_c; j++) {
            int slice = (j==s_num_per_c-1 && size%32!=0)? size%32 : 32;
            int row_max = 0;
            for(int k=0; k<slice; k++) {
                int row_id = offset + j*32 + k;
                int row_len = rptr[row_id+1] - rptr[row_id];
                if(row_max < row_len) { row_max = row_len; }
            }
            (*srptr)[s_num+j+1] = (*srptr)[s_num+j] + row_max;
            nnz += row_max * slice;
            (*snnz)[s_num+j+1] = nnz;
        }
        s_num += s_num_per_c;
    }
    *sval  = utils::SafeMalloc<double>(nnz);
    *scind = utils::SafeMalloc<int>(nnz);

    nnz = 0;
    s_num = 0;
    for(int cid=0; cid<c_num; cid++) {
        int offset = c_size[cid];
        int size = c_size[cid+1] - c_size[cid];
        int s_num_per_c = (size+32-1)/32;
        for(int j=0; j<s_num_per_c; j++) {
            int slice = (j==s_num_per_c-1 && size%32!=0)? size%32 : 32;
            for(int k=0; k<slice; k++) {
                int row_id = offset + j*32+k;
                int row_count = 0;
                for(int l=rptr[row_id]; l<rptr[row_id+1]; l++) {
                    (*sval)[nnz+row_count*slice+k] = val[l];
                    (*scind)[nnz+row_count*slice+k] = cind[l];
                    row_count++;
                }
                if(row_count < (*srptr)[s_num+j+1]-(*srptr)[s_num+j]) {
                    int remainder = (*srptr)[s_num+j+1]-(*srptr)[s_num+j] - row_count;
                    for(int l=0; l<remainder; l++) {
                        (*sval)[nnz+row_count*slice+k] = 0;
                        (*scind)[nnz+row_count*slice+k] = 0;
                        row_count++;
                    }
                }
            }
            nnz += ( (*srptr)[s_num+j+1]-(*srptr)[s_num+j]) * slice;
        }
        s_num += s_num_per_c;
    }
    // printf("# MCsell nnz %d\n", nnz);
}

template <typename T>
void copy(
    T *val, int *cind, int *rptr,
    T **val2, int **cind2, int **rptr2, int N)
{
    *val2  = utils::SafeMalloc<T>(rptr[N]);
    *cind2 = utils::SafeMalloc<int>(rptr[N]);
    *rptr2 = utils::SafeMalloc<int>(N+1);
    utils::Copy<T>(val, *val2, rptr[N]);
    utils::Copy<int>(cind, *cind2, rptr[N]);
    utils::Copy<int>(rptr, *rptr2, N+1);
}

template <typename T>
void block_copy(
    T *val, int *cind, int *rptr,
    T **val2, int **cind2, int **rptr2,
    int **b_size, int b_num, int N, int unit)
{
    if(b_num * unit > N) {
        printf("Error: b_num * unit is larger than N\n");
        exit(EXIT_FAILURE);
    }
    if(N % unit != 0) {
        printf("Error: N %% unit != 0\n");
        exit(EXIT_FAILURE);
    }
    /* Calculate the size of each block. */
    int N_unit = N / unit;
    int size = (N_unit+b_num-1)/b_num * unit;
    *b_size = utils::SafeMalloc<int>(b_num+1);
    (*b_size)[0] = 0;
    for(int i=0; i<b_num; i++) {
        (*b_size)[i+1] = (i != b_num-1) ? (*b_size)[i] + size : N;
    }
    /* Count the number of elements in each block. */
    *rptr2 = utils::SafeMalloc<int>(N+1);
    int nnz = 0;
    (*rptr2)[0] = nnz;
    for(int i=0; i<N; i++) {
        int left = i / size * size;
        int right = (i != N-1) ? left + size : N;
        for(int j=rptr[i]; j<rptr[i+1]; j++) {
            if(left <= cind[j] && cind[j] < right) nnz++;
        }
        (*rptr2)[i+1] = nnz;
    }
    /* Store the nonzero elements into each block. */
    *val2 = utils::SafeMalloc<T>(nnz);
    *cind2 = utils::SafeMalloc<int>(nnz);
    int count = 0;
    for(int i=0; i<N; i++) {
        int left = i / size * size;
        int right = (i != N-1) ? left + size : N;
        for(int j=rptr[i]; j<rptr[i+1]; j++) {
            if(left <= cind[j] && cind[j] < right) {
                (*val2)[count] = val[j];
                (*cind2)[count] = cind[j];
                count++;
            }
        }
    }
}

template <typename T>
void block_copy_offset(
    T *val, int *cind, int *rptr,
    T **val2, int **cind2, int **rptr2,
    int **b_size, int b_num, int N)
{
    /* Calculate the size of each block. */
    int size = (N+b_num-1)/b_num;
    *b_size = utils::SafeMalloc<int>(b_num+2);
    (*b_size)[0] = 0;
    (*b_size)[1] = size / 2;
    for(int i=1; i<b_num+1; i++) {
        (*b_size)[i+1] = (i != b_num) ? (*b_size)[i] + size : N;
    }
    /* Count the number of elements in each block. */
    *rptr2 = utils::SafeMalloc<int>(N+1);
    int nnz = 0;
    (*rptr2)[0] = nnz;
    for(int k=0; k<b_num+1; k++) {
        int start = (*b_size)[k];
        int end = (*b_size)[k+1];
        for(int i=start; i<end; i++) {
            for(int j=rptr[i]; j<rptr[i+1]; j++) {
                /*
                if(k==0 || k==b_num) {
                    if(cind[j] == i) nnz++;
                }else {
                    if(start <= cind[j] && cind[j] < end) nnz++;
                }
                */
                if(start <= cind[j] && cind[j] < end) nnz++;
            }
            (*rptr2)[i+1] = nnz;
        }
    }
    /* Store the nonzero elements into each block. */
    *val2  = utils::SafeMalloc<T>(nnz);
    *cind2 = utils::SafeMalloc<int>(nnz);
    int count = 0;
    for(int k=0; k<b_num+1; k++) {
        int start = (*b_size)[k];
        int end = (*b_size)[k+1];
        for(int i=start; i<end; i++) {
            for(int j=rptr[i]; j<rptr[i+1]; j++) {
                /*
                if(k==0 || k==b_num) {
                    if(cind[j] == i) {
                        (*val2)[count] = val[j];
                        (*cind2)[count] = cind[j];
                        count++;
                    }
                }else {
                    if(start <= cind[j] && cind[j] < end) {
                        (*val2)[count] = val[j];
                        (*cind2)[count] = cind[j];
                        count++;
                    }
                }
                */
                if(start <= cind[j] && cind[j] < end) {
                    (*val2)[count] = val[j];
                    (*cind2)[count] = cind[j];
                    count++;
                }
            }
        }
    }
}

template <typename T>
int csr_to_sell(
    T *val, int *cind, int *rptr,
    T **s_val, int **s_cind, int **s_len,
    int size, int N)
{
    int num_slice = (N+size-1)/size;
    *s_len = utils::SafeMalloc<int>(num_slice+1);
    int temp_slice;
    int row_id;
    int row_max;
    int row_count;
    int nnz = 0;
    
    (*s_len)[0] = 0;
    for(int i=0; i<num_slice; i++) {
        temp_slice = size;
        if(i==num_slice-1 && N%size!=0) temp_slice = N % size;
        row_max = 0;
        for(int j=0; j<temp_slice; j++) {
            row_id = i*size+j;
            if(row_max < rptr[row_id+1] - rptr[row_id]) {
                row_max = rptr[row_id+1] - rptr[row_id];
            }
        }
        (*s_len)[i+1] = (*s_len)[i] + row_max;
        nnz += row_max * temp_slice;
    }
    *s_val  = utils::SafeMalloc<T>(nnz);
    *s_cind = utils::SafeMalloc<int>(nnz);
    for(int i=0; i<num_slice; i++) {
        temp_slice = size;
        if(i==num_slice-1 && N%size!=0) temp_slice = N % size;
        for(int j=0; j<temp_slice; j++) {
            row_id = i*size+j;
            row_count = 0;
            for(int k=rptr[row_id]; k<rptr[row_id+1]; k++) {
                (*s_val)[(*s_len)[i]*size+row_count*temp_slice+j] = val[k];
                (*s_cind)[(*s_len)[i]*size+row_count*temp_slice+j] = cind[k];
                row_count++;
            }
            if(row_count < (*s_len)[i+1] - (*s_len)[i]) {
                int temp = (*s_len)[i+1] - (*s_len)[i] - row_count;
                for(int k=0; k<temp; k++) {
                    (*s_val)[(*s_len)[i]*size+row_count*temp_slice+j] = 0;
                    (*s_cind)[(*s_len)[i]*size+row_count*temp_slice+j] = 0;
                    row_count++;
                }
            }
        }
    }
    return nnz;
}

template <typename T>
void csr_to_bcsr(
    T *val, int *cind, int *rptr,
    T **bval, int **bcind, int **brptr,
    int N, int bnl, int bnw)
{
    if(N % bnl || N % bnw) {
        printf("Error: Csr2Bcsr\n");
        exit(EXIT_FAILURE);
    }
    *brptr = utils::SafeMalloc<int>(N/bnl+1);
    int cnt = 0;
    (*brptr)[0] = 0;
    int *ptr = utils::SafeMalloc<int>(bnl);
// Count the number of block
    for(int i=0; i<N; i+=bnl) {
        // Initialize "ptr"
        for(int j=0; j<bnl; j++) { ptr[j] = rptr[i+j]; }
        while(true) {
            // Find minimum col value
            int min = N;
            for(int j=0; j<bnl; j++) {
                if(ptr[j] != -1 && cind[ptr[j]] < min)
                    min = cind[ptr[j]];
            }
            if(min == N) break;
            // Increment ptr[j] whose col-idx is in min block
            for(int j=0; j<bnl; j++) {
                if(ptr[j] == -1) continue;
                while(cind[ptr[j]]/bnw == min/bnw) {
                    ptr[j]++;
                    if(ptr[j] >= rptr[i+j+1]) {ptr[j] = -1; break;}
                }
            }
            cnt++;
        }
        (*brptr)[i/bnl+1] = cnt;
    }
    *bcind = utils::SafeMalloc<int>(cnt);
    *bval  = utils::SafeCalloc<T>(cnt*bnl*bnw);
// Assign val to bval
    cnt = 0;
    for(int i=0; i<N; i+=bnl) {
        // Initialize "ptr"
        for(int j=0; j<bnl; j++) { ptr[j] = rptr[i+j]; }
        while(true) {
            int min = N;
            for(int j=0; j<bnl; j++) {
                if(ptr[j] != -1 && cind[ptr[j]] < min)
                    min = cind[ptr[j]];
            }
            if(min == N) break;
            for(int j=0; j<bnl; j++) {
                if(ptr[j] == -1) continue;
                while(cind[ptr[j]]/bnw == min/bnw) {
                    int off = cind[ptr[j]] % bnw;
                    (*bval)[cnt*bnl*bnw+off*bnl+j] = val[ptr[j]];
                    //printf("%e\n", (*bval)[cnt*bnl*bnw+off*bnl+j]);
                    ptr[j]++;
                    if(ptr[j] >= rptr[i+j+1]) {ptr[j] = -1; break;}
                }
            }
            (*bcind)[cnt] = min/bnw;
            cnt++;
        }
    }
    free(ptr);
}

template <typename T>
int bcsr_to_csr(
    T *bval, int *bcind, int *brptr,
    T **val, int **cind, int **rptr,
    int N, int bnl, int bnw)
{
    int bsize = bnl * bnw;
    int num_block = brptr[N/bnl];
    int nnz;
    *val = senk::utils::SafeMalloc<T>(num_block*bnl*bnw);
    *cind = senk::utils::SafeMalloc<int>(num_block*bnl*bnw);
    *rptr = senk::utils::SafeMalloc<int>(N+1);
    (*rptr)[0] = 0;
    int count = 0;
    for(int i=0; i<N; i++) {
        int bid = i / bnl;
        int id = i % bnl; // 0 to bnl-1
        for(int bj=brptr[bid]; bj<brptr[bid+1]; bj++) {
            for(int j=0; j<bnw; j++) {
                (*cind)[count] = (bcind[bj])*bnw+j;
                (*val)[count] = bval[bj*bsize+j*bnl+id];
                count++;
            }
        }
        (*rptr)[i+1] = count;
    }
    nnz = (*rptr)[N];
    return nnz;
}

template <typename T>
void padding(T **val, int **cind, int **rptr, int size, int *N)
{
    int remain = (N[0] % size == 0) ? 0 : size - N[0] % size;
    int NNZ = rptr[0][N[0]];
    *val  = utils::SafeRealloc<T>(*val, NNZ+remain);
    *cind = utils::SafeRealloc<int>(*cind, NNZ+remain);
    *rptr = utils::SafeRealloc<int>(*rptr, N[0]+1+remain);
    for(int i=0; i<remain; i++) val[0][NNZ+i] = 1;
    for(int i=0; i<remain; i++) cind[0][NNZ+i] = N[0]+i;
    for(int i=0; i<remain; i++) rptr[0][N[0]+1+i] = rptr[0][N[0]+i]+1;
    N[0] += remain;
}

inline void remove_zero(double **val, int **cind, int **rptr, int N) {
    int i, j;
    int NNZ = (*rptr)[N];
    double *temp_val = utils::SafeMalloc<double>(NNZ);
    int *temp_cind   = utils::SafeMalloc<int>(NNZ);
    int *temp_rptr   = utils::SafeMalloc<int>(N+1);
    int nnz = 0;
    temp_rptr[0] = 0;
    for(i=0; i<N; i++) {
        for(j=(*rptr)[i]; j<(*rptr)[i+1]; j++) {
            if((*val)[j] != 0 || (*cind)[j] == i) {
                nnz++;
            }
        }
        if(nnz == temp_rptr[i]) {
            printf("hoge\n");
            nnz++;
        }
        temp_rptr[i+1] = nnz;
    }
    // #pragma omp parallel for private(j)
    for(i=0; i<N; i++) {
        int st = temp_rptr[i];
        int num = 0;
        for(j=(*rptr)[i]; j<(*rptr)[i+1]; j++) {
            if((*val)[j] != 0 || (*cind)[j] == i) {
                temp_val[st+num] = (*val)[j];
                temp_cind[st+num] = (*cind)[j];
                num++;
            }
        }
        if(num == temp_rptr[i]) {
            temp_val[st+num] = 1;
            temp_cind[st+num] = i;
            num++;
            printf("hoge\n");
        }
    }
    free(*val);
    free(*cind);
    free(*rptr);
    *val  = utils::SafeRealloc<double>(temp_val, nnz);
    *cind = utils::SafeRealloc<int>(temp_cind, nnz);
    *rptr = utils::SafeRealloc<int>(temp_rptr, N+1);
}

inline void base_one_zero(int *cind, int *rptr, int N) {
    int i;
    if(rptr[0] == 0) return;
    #pragma omp parallel for
    for(i=0; i<N+1; i++) {rptr[i]--;}
    #pragma omp parallel for
    for(i=0; i<rptr[N]; i++) {cind[i]--;}
}

template <typename T>
void split(
    T *val, int *cind, int *rptr,
    T **lval, int **lcind, int **lrptr,
    T **uval, int **ucind, int **urptr,
    T **diag, int N, const char *key, bool invDiag)
{
    int lNNZ = 0, uNNZ = 0, dNNZ = 0;
    int *l_ptr, *u_ptr, *d_ptr;
    T *lv_ptr, *uv_ptr, *dv_ptr;
    int *lc_ptr, *uc_ptr, *dc_ptr;
    int *lr_ptr, *ur_ptr;
    if(std::strcmp(key, "LD-U") == 0) {
        l_ptr=&lNNZ; u_ptr=&uNNZ; d_ptr=&lNNZ;
    }else if(std::strcmp(key, "L-DU") == 0) {
        l_ptr=&lNNZ; u_ptr=&uNNZ; d_ptr=&uNNZ;
    }else if(std::strcmp(key, "L-D-U") == 0) {
        l_ptr=&lNNZ; u_ptr=&uNNZ; d_ptr=&dNNZ;
    }else if(std::strcmp(key, "LU-D") == 0) {
        l_ptr=&lNNZ; u_ptr=&lNNZ; d_ptr=&dNNZ;
    }else { printf("Split: Keyword is not valid."); exit(1); }
    for(int i=0; i<N; i++) {
        for(int j=rptr[i]; j<rptr[i+1]; j++) {
            if(cind[j] < i) { (*l_ptr)++; }
            else if(cind[j] > i) { (*u_ptr)++; }
            else { (*d_ptr)++; }
        }
    }
    *lval  = utils::SafeMalloc<T>(lNNZ);
    *lcind = utils::SafeMalloc<int>(lNNZ);
    *lrptr = utils::SafeMalloc<int>(N+1);
    if(uNNZ != 0) {
        *uval  = utils::SafeMalloc<T>(uNNZ);
        *ucind = utils::SafeMalloc<int>(uNNZ);
        *urptr = utils::SafeMalloc<int>(N+1);
    }
    if(dNNZ == N) { *diag = utils::SafeMalloc<T>(dNNZ); }
    lNNZ = 0; uNNZ = 0; dNNZ = 0;
    if(std::strcmp(key, "LD-U") == 0) {
        lv_ptr = *lval;  uv_ptr = *uval;  dv_ptr = *lval;
        lc_ptr = *lcind; uc_ptr = *ucind; dc_ptr = *lcind;
        lr_ptr = *lrptr; ur_ptr = *urptr;
    }else if(std::strcmp(key, "L-DU") == 0) {
        lv_ptr = *lval;  uv_ptr = *uval;  dv_ptr = *uval;
        lc_ptr = *lcind; uc_ptr = *ucind; dc_ptr = *ucind;
        lr_ptr = *lrptr; ur_ptr = *urptr;
    }else if(std::strcmp(key, "L-D-U") == 0) {
        lv_ptr = *lval;  uv_ptr = *uval;  dv_ptr = *diag;
        lc_ptr = *lcind; uc_ptr = *ucind; dc_ptr = nullptr;
        lr_ptr = *lrptr; ur_ptr = *urptr;
    }else if(std::strcmp(key, "LU-D") == 0) {
        lv_ptr = *lval;  uv_ptr = *lval;  dv_ptr = *diag;
        lc_ptr = *lcind; uc_ptr = *lcind; dc_ptr = nullptr;
        lr_ptr = *lrptr; ur_ptr = *lrptr;
    }else { return; }
    lr_ptr[0] = 0;
    ur_ptr[0] = 0;
    for(int i=0; i<N; i++) {
        for(int j=rptr[i]; j<rptr[i+1]; j++) {
            if(cind[j] < i) {
                lv_ptr[*l_ptr] = val[j]; lc_ptr[*l_ptr] = cind[j];
                (*l_ptr)++;
            }else if(cind[j] > i) {
                uv_ptr[*u_ptr] = val[j]; uc_ptr[*u_ptr] = cind[j];
                (*u_ptr)++;
            }else {
                dv_ptr[*d_ptr] = (invDiag) ? 1/val[j] : val[j];
                if(dc_ptr) { dc_ptr[*d_ptr] = cind[j]; }
                (*d_ptr)++;
            }
        }
        lr_ptr[i+1] = *l_ptr;
        ur_ptr[i+1] = *u_ptr;
    }
}

template <typename T>
void extractL(
    T *val, int *cind, int *rptr,
    T **lval, int **lcind, int **lrptr, int N)
{
    int nnz = 0;
    for(int i=0; i<N; i++) {
        for(int j=rptr[i]; j<rptr[i+1]; j++) {
            if(cind[j] < i) { nnz++; }
        }
    }
    *lval  = utils::SafeMalloc<T>(nnz);
    *lcind = utils::SafeMalloc<int>(nnz);
    *lrptr = utils::SafeMalloc<int>(N+1);
    nnz = 0;
    (*lrptr)[0] = nnz;
    for(int i=0; i<N; i++) {
        for(int j=rptr[i]; j<rptr[i+1]; j++) {
            if(cind[j] < i) {
                (*lval)[nnz] = val[j];
                (*lcind)[nnz] = cind[j];
                nnz++;
            }
        }
        (*lrptr)[i+1] = nnz;
    }
}

template <typename T>
void extractLDinv(
    T *val, int *cind, int *rptr,
    T **lval, int **lcind, int **lrptr, int N)
{
    int nnz = 0;
    for(int i=0; i<N; i++) {
        for(int j=rptr[i]; j<rptr[i+1]; j++) {
            if(cind[j] <= i) { nnz++; }
        }
    }
    *lval  = utils::SafeMalloc<T>(nnz);
    *lcind = utils::SafeMalloc<int>(nnz);
    *lrptr = utils::SafeMalloc<int>(N+1);
    nnz = 0;
    (*lrptr)[0] = nnz;
    for(int i=0; i<N; i++) {
        for(int j=rptr[i]; j<rptr[i+1]; j++) {
            if(cind[j] == i) {
                (*lval)[nnz] = 1 / val[j];
                (*lcind)[nnz] = cind[j];
                nnz++;
            }else if(cind[j] < i) {
                (*lval)[nnz] = val[j];
                (*lcind)[nnz] = cind[j];
                nnz++;
            }
        }
        (*lrptr)[i+1] = nnz;
    }
}

template <typename T>
void extractLD(
    T *val, int *cind, int *rptr,
    T **lval, int **lcind, int **lrptr, int N)
{
    int nnz = 0;
    for(int i=0; i<N; i++) {
        for(int j=rptr[i]; j<rptr[i+1]; j++) {
            if(cind[j] <= i) { nnz++; }
        }
    }
    *lval  = utils::SafeMalloc<T>(nnz);
    *lcind = utils::SafeMalloc<int>(nnz);
    *lrptr = utils::SafeMalloc<int>(N+1);
    nnz = 0;
    (*lrptr)[0] = nnz;
    for(int i=0; i<N; i++) {
        for(int j=rptr[i]; j<rptr[i+1]; j++) {
            if(cind[j] == i) {
                (*lval)[nnz] = val[j];
                (*lcind)[nnz] = cind[j];
                nnz++;
            }else if(cind[j] < i) {
                (*lval)[nnz] = val[j];
                (*lcind)[nnz] = cind[j];
                nnz++;
            }
        }
        (*lrptr)[i+1] = nnz;
    }
}

template <typename T>
void extractDinvU(
    T *val, int *cind, int *rptr,
    T **uval, int **ucind, int **urptr, int N)
{
    int nnz = 0;
    for(int i=0; i<N; i++) {
        for(int j=rptr[i]; j<rptr[i+1]; j++) {
            if(cind[j] >= i) { nnz++; }
        }
    }
    *uval  = utils::SafeMalloc<T>(nnz);
    *ucind = utils::SafeMalloc<int>(nnz);
    *urptr = utils::SafeMalloc<int>(N+1);
    nnz = 0;
    (*urptr)[0] = nnz;
    for(int i=0; i<N; i++) {
        for(int j=rptr[i]; j<rptr[i+1]; j++) {
            if(cind[j] == i) {
                (*uval)[nnz] = 1 / val[j];
                (*ucind)[nnz] = cind[j];
                nnz++;
            }else if(cind[j] > i) {
                (*uval)[nnz] = val[j];
                (*ucind)[nnz] = cind[j];
                nnz++;
            }
        }
        (*urptr)[i+1] = nnz;
    }
}

template <typename T>
void expand(
    T *tval, int *tcind, int *trptr,
    T **val, int **cind, int **rptr,
    int N, const char *key)
{
    T   *lval,  *uval,  *tval2;
    int *lcind, *ucind, *tcind2;
    int *lrptr, *urptr, *trptr2;
    int nnz = trptr[N]*2 - N;
    csr_to_csc<T>(tval, tcind, trptr, &tval2, &tcind2, &trptr2, N, N);
    if(std::strcmp(key, "L") == 0) {
        lval = tval;  lcind = tcind;  lrptr = trptr;
        uval = tval2; ucind = tcind2; urptr = trptr2;
    }else if(std::strcmp(key, "U") == 0) {
        lval = tval2; lcind = tcind2; lrptr = trptr2;
        uval = tval;  ucind = tcind;  urptr = trptr;
    }else { printf("Expand: Keyword is not valid."); exit(1); }
    *val  = utils::SafeMalloc<T>(nnz);
    *cind = utils::SafeMalloc<int>(nnz);
    *rptr = utils::SafeMalloc<int>(N+1);
    int cnt = 0; (*rptr)[0] = cnt;
    for(int i=0; i<N; i++) {
        for(int j=lrptr[i]; j<lrptr[i+1]; j++) {
            (*val)[cnt] = lval[j]; (*cind)[cnt] = lcind[j];
            cnt++;
        }
        for(int j=urptr[i]+1; j<urptr[i+1]; j++) {
            (*val)[cnt] = uval[j]; (*cind)[cnt] = ucind[j];
            cnt++;
        }
        (*rptr)[i+1] = cnt;
    }
    free(tval2);
    free(tcind2);
    free(trptr2);
}

template <typename T>
void duplicate(
    T *tval, int *tcind, int *trptr,
    T **val, int **cind, int **rptr,
    int N)
{
    *val  = utils::SafeMalloc<T>(trptr[N]);
    *cind = utils::SafeMalloc<int>(trptr[N]);
    *rptr = utils::SafeMalloc<int>(N+1);
    (*rptr)[0] = trptr[0];
    for(int i=0; i<N; i++) {
        (*rptr)[i+1] = trptr[i+1];
        for(int j=trptr[i]; j<trptr[i+1]; j++) {
            (*val)[j] = tval[j]; (*cind)[j] = tcind[j];
        }
    }
}

template <typename T>
void get_diag(T *val, int *cind, int *rptr, T **diag, int N)
{
    *diag = utils::SafeMalloc<T>(N);
    #pragma omp parallel for
    for(int i=0; i<N; i++) {
        for(int j=rptr[i]; j<rptr[i+1]; j++) {
            if(cind[j] == i) {
                (*diag)[i] = val[j];
                break;
            }
        }
    }
}

template <typename T>
void scaling(T *val, int *cind, int *rptr, T *ld, T *rd, int N)
{
    #pragma omp parallel for
    for(int i=0; i<N; i++) {
        T left = ld[i];
        for(int j=rptr[i]; j<rptr[i+1]; j++) {
            val[j] = left * val[j] * rd[cind[j]];
        }
    }
}

template <typename T>
void ilu0(T *val, int *cind, int *rptr, int N, T alpha)
{
    val[0] *= alpha;
    for(int i=1; i<N; i++) {
        for(int k=rptr[i]; k<rptr[i+1]; k++) {
            if(cind[k] == i) val[k] *= alpha;
        }
        for(int k=rptr[i]; k<rptr[i+1]; k++) {
            if(cind[k] >= i) break;
            for(int l=rptr[cind[k]]; l<rptr[cind[k]+1]; l++) {
                if(cind[l] == cind[k]) {
                    if(val[l] == 0) {
                        printf("Error: Ilu0, 0 pivot\n");
                        exit(1);
                    }
                    val[k] = val[k] / val[l];
                    break;
                }
            }
            int pos = rptr[cind[k]];
            for(int j=k+1; j<rptr[i+1]; j++) {
                for(int l=pos; l<rptr[cind[k]+1]; l++) {
                    if(cind[l] < cind[j]) continue;
                    pos = l;
                    if(cind[l] == cind[j]) {
                        val[j] -= val[k] * val[l];
                        pos++;
                    }
                    break;
                }
            }
        }
    }
}

template <typename T>
void ic0(T *val, int *cind, int *rptr, int N)
{
    int i, j, k, l;
    for(i=1; i<N; i++) {
        for(j=rptr[i]; j<rptr[i+1]; j++) {
            if(cind[j] > i) break;
            if(cind[j] < i) {
                for(k=rptr[i]; k<rptr[i+1]; k++) {
                    if(cind[k] >= cind[j]) break;
                    for(l=rptr[cind[j]]; l<rptr[cind[j]+1]; l++) {
                        if(cind[l] > cind[k]) break;
                        if(cind[l] == cind[k]) {
                            val[j] -= val[k] * val[rptr[cind[k]+1]-1] * val[l];
                        }
                    }
                }
                val[j] /= val[rptr[cind[j]+1]-1];
            }else if(cind[j] == i) {
                for(k=rptr[i]; k<rptr[i+1]; k++) {
                    if(cind[k] >= cind[j]) break;
                    val[j] -= val[k] * val[rptr[cind[k]+1]-1] * val[k];
                }
            }
        }
    }
}

template <typename T>
void ilup(T **val, int **cind, int **rptr, int N, int p)
{
    int i, j;
    //int NNZ = (*rptr)[N];
    sparse::SpVecLev<T, int> temp;
    sparse::SpVecLev<T, int> temp2;
    int *zeros = utils::SafeCalloc<int>(N);

    T pivot = 0;
    int pivot_lev = 0;

    int first_len = (*rptr)[1] - (*rptr)[0];
    T   *new_val  = utils::SafeMalloc<T>(first_len);
    int *new_cind = utils::SafeMalloc<int>(first_len);
    int *new_lev  = utils::SafeCalloc<int>(first_len);
    int *new_rptr = utils::SafeMalloc<int>(N+1);
    int new_len = first_len;
    //一行目
    memcpy(new_val, *val, sizeof(T)*first_len);
    memcpy(new_cind, *cind, sizeof(int)*first_len);
    new_rptr[0] = 0;
    new_rptr[1] = first_len;

    for(i=1; i<N; i++) {
        int now_len = (*rptr)[i+1] - (*rptr)[i];
        temp.Clear();
        temp.Append(&(*val)[(*rptr)[i]], zeros, &(*cind)[(*rptr)[i]], now_len);
        int count = 0;
        while(count < temp.GetLen() && temp.GetIdx(count) < i) {
            int k_ptr = count;
            if(temp.GetLev(k_ptr) > p) { count++; continue; }
            int k = temp.GetIdx(k_ptr);
            temp2.Clear();
            for(j=0; j<count; j++) {
                temp2.Append(temp.GetVal(j), temp.GetLev(j), temp.GetIdx(j));
            }
            for(j=new_rptr[k]; j<new_rptr[k+1]; j++) {
                if(new_cind[j] == k) {
                    pivot = temp.GetVal(k_ptr) / new_val[j];
                    pivot_lev = temp.GetLev(k_ptr);
                    break;
                }
            }
            j++;
            k_ptr++;
            temp2.Append(pivot, 0, k);
            while(k_ptr<temp.GetLen() || j<new_rptr[k+1]) {
                int t_col1 = N+1;
                int t_col2 = N+1;
                if(k_ptr < temp.GetLen()) t_col1 = temp.GetIdx(k_ptr);
                if(j < new_rptr[k+1]) t_col2 = new_cind[j];
                if(t_col1<t_col2) {
                    T t_val = temp.GetVal(k_ptr);
                    int t_lev    = temp.GetLev(k_ptr);
                    int t_col    = temp.GetIdx(k_ptr);
                    temp2.Append(t_val, t_lev, t_col);
                    k_ptr++;
                }else if(t_col1==t_col2) {
                    T t_val = temp.GetVal(k_ptr) - pivot*new_val[j];
                    int t_lev    = temp.GetLev(k_ptr);
                    int t_col    = temp.GetIdx(k_ptr);
                    if(t_lev > pivot_lev+new_lev[j]+1) {
                        t_lev = pivot_lev+new_lev[j]+1;
                    }
                    temp2.Append(t_val, t_lev, t_col);
                    k_ptr++;
                    j++;
                }else { // (k>j) fill-in
                    T t_val = -pivot*new_val[j];
                    int t_lev    = pivot_lev+new_lev[j]+1;
                    int t_col    = new_cind[j];
                    //printf("lev %d\n", t_lev);
                    temp2.Append(t_val, t_lev, t_col);
                    j++;
                }
            }
            count++;
            temp.Clear();
            for(j=0; j<temp2.GetLen(); j++) {
                temp.Append(temp2.GetVal(j), temp2.GetLev(j), temp2.GetIdx(j));
            }
        }
        new_val  = utils::SafeRealloc<T>(new_val, new_len+temp.GetLen());
        new_cind = utils::SafeRealloc<int>(new_cind, new_len+temp.GetLen());
        new_lev  = utils::SafeRealloc<int>(new_lev, new_len+temp.GetLen());
        for(j=0; j<temp.GetLen(); j++) {
            if(temp.GetLev(j) <= p) {
                new_val[new_len]  = temp.GetVal(j);
                new_cind[new_len] = temp.GetIdx(j);
                new_lev[new_len]  = temp.GetLev(j);
                new_len++;
            }
        }
        new_rptr[i+1] = new_len;
    }
    (*val)  = utils::SafeRealloc<T>(new_val, new_len);
    (*cind) = utils::SafeRealloc<int>(new_cind, new_len);
    (*rptr) = utils::SafeRealloc<int>(new_rptr, N+1);
}

inline void ilu(double **val, int **cind, int **rptr, int N)
{
    int i, j;
    //int NNZ = (*rptr)[N];
    sparse::SpVec temp(8);
    sparse::SpVec temp2(8);
    
    double pivot = 0;

    int first_len = (*rptr)[1] - (*rptr)[0];
    double *new_val  = utils::SafeMalloc<double>(first_len);
    int *new_cind = utils::SafeMalloc<int>(first_len);
    int *new_rptr = utils::SafeMalloc<int>(N+1);
    int new_len = first_len;
    //一行目
    memcpy(new_val, *val, sizeof(double)*first_len);
    memcpy(new_cind, *cind, sizeof(int)*first_len);
    new_rptr[0] = 0;
    new_rptr[1] = first_len;

    for(i=1; i<N; i++) {
        int now_len = (*rptr)[i+1] - (*rptr)[i];
        temp.len = 0;
        temp.Append(&(*val)[(*rptr)[i]], &(*cind)[(*rptr)[i]], now_len);
        int count = 0;
        while(count < temp.len && temp.row[count] < i) {
            int k_ptr = count;
            int k = temp.row[k_ptr];
            temp2.len = 0;
            for(j=0; j<count; j++) {
                temp2.Append(temp.val[j], temp.row[j]);
            }
            for(j=new_rptr[k]; j<new_rptr[k+1]; j++) {
                if(new_cind[j] == k) {
                    pivot = temp.val[k_ptr] / new_val[j];
                    break;
                }
            }
            j++;
            k_ptr++;
            temp2.Append(pivot, k);
            while(k_ptr<temp.len || j<new_rptr[k+1]) {
                int t_col1 = (k_ptr < temp.len)?  temp.row[k_ptr] : N+1;
                int t_col2 = (j < new_rptr[k+1])? new_cind[j] : N+1;
                double t_val;
                int t_col;
                if(t_col1<t_col2) {
                    t_val = temp.val[k_ptr];
                    t_col = temp.row[k_ptr];
                    k_ptr++;
                }else if(t_col1==t_col2) {
                    t_val = temp.val[k_ptr] - pivot*new_val[j];
                    t_col = temp.row[k_ptr];
                    k_ptr++; j++;
                }else {
                    t_val = -pivot*new_val[j];
                    t_col = new_cind[j];
                    j++;
                }
                temp2.Append(t_val, t_col);
            }
            count++;
            temp.len = 0;
            for(j=0; j<temp2.len; j++) {
                temp.Append(temp2.val[j], temp2.row[j]);
            }
        }
        new_val  = utils::SafeRealloc<double>(new_val, new_len+temp.len);
        new_cind = utils::SafeRealloc<int>(new_cind, new_len+temp.len);
        for(j=0; j<temp.len; j++) {
            new_val[new_len]  = temp.val[j];
            new_cind[new_len] = temp.row[j];
            new_len++;
        }
        new_rptr[i+1] = new_len;
    }
    (*val)  = utils::SafeRealloc<double>(new_val, new_len);
    (*cind) = utils::SafeRealloc<int>(new_cind, new_len);
    (*rptr) = utils::SafeRealloc<int>(new_rptr, N+1);

    temp.Free();
    temp2.Free();
}

/*
template <typename T>
void AllocLevelZero(T **val, int **cind, int **rptr, int N, int p)
{
    int i, j;
    //int NNZ = (*rptr)[N];
    sparse::SpVec<T, int> temp;
    sparse::SpVec<T, int> temp2;
    int *zeros = utils::SafeCalloc<int>(N);

    T pivot = 0;
    int pivot_lev = 0;

    int first_len = (*rptr)[1] - (*rptr)[0];
    T   *new_val  = utils::SafeMalloc<T>(first_len);
    int *new_cind = utils::SafeMalloc<int>(first_len);
    int *new_lev  = utils::SafeCalloc<int>(first_len);
    int *new_rptr = utils::SafeMalloc<int>(N+1);
    int new_len = first_len;
    //一行目
    memcpy(new_val, *val, sizeof(T)*first_len);
    memcpy(new_cind, *cind, sizeof(int)*first_len);
    new_rptr[0] = 0;
    new_rptr[1] = first_len;

    for(i=1; i<N; i++) {
        int now_len = (*rptr)[i+1] - (*rptr)[i];
        temp.Clear();
        temp.Append(&(*val)[(*rptr)[i]], zeros, &(*cind)[(*rptr)[i]], now_len);
        int count = 0;
        while(count < temp.GetLen() && temp.GetIdx(count) < i) {
            int k_ptr = count;
            if(temp.GetLev(k_ptr) > p) { count++; continue; }
            int k = temp.GetIdx(k_ptr);
            temp2.Clear();
            for(j=0; j<count; j++) {
                temp2.Append(temp.GetVal(j), temp.GetLev(j), temp.GetIdx(j));
            }
            for(j=new_rptr[k]; j<new_rptr[k+1]; j++) {
                if(new_cind[j] == k) {
                    pivot = temp.GetVal(k_ptr);// / new_val[j];
                    pivot_lev = temp.GetLev(k_ptr);
                    break;
                }
            }
            j++;
            k_ptr++;
            temp2.Append(pivot, 0, k);
            while(k_ptr<temp.GetLen() || j<new_rptr[k+1]) {
                int t_col1 = N+1;
                int t_col2 = N+1;
                if(k_ptr < temp.GetLen()) t_col1 = temp.GetIdx(k_ptr);
                if(j < new_rptr[k+1]) t_col2 = new_cind[j];
                if(t_col1<t_col2) {
                    T t_val = temp.GetVal(k_ptr);
                    int t_lev    = temp.GetLev(k_ptr);
                    int t_col    = temp.GetIdx(k_ptr);
                    temp2.Append(t_val, t_lev, t_col);
                    k_ptr++;
                }else if(t_col1==t_col2) {
                    T t_val = temp.GetVal(k_ptr);
                    int t_lev    = temp.GetLev(k_ptr);
                    int t_col    = temp.GetIdx(k_ptr);
                    if(t_lev > pivot_lev+new_lev[j]+1) {
                        t_lev = pivot_lev+new_lev[j]+1;
                    }
                    temp2.Append(t_val, t_lev, t_col);
                    k_ptr++;
                    j++;
                }else { // (k>j) fill-in
                    T t_val = 0;
                    int t_lev    = pivot_lev+new_lev[j]+1;
                    int t_col    = new_cind[j];
                    //printf("lev %d\n", t_lev);
                    temp2.Append(t_val, t_lev, t_col);
                    j++;
                }
            }
            count++;
            temp.Clear();
            for(j=0; j<temp2.GetLen(); j++) {
                temp.Append(temp2.GetVal(j), temp2.GetLev(j), temp2.GetIdx(j));
            }
        }
        new_val  = utils::SafeRealloc<T>(new_val, new_len+temp.GetLen());
        new_cind = utils::SafeRealloc<int>(new_cind, new_len+temp.GetLen());
        new_lev  = utils::SafeRealloc<int>(new_lev, new_len+temp.GetLen());
        for(j=0; j<temp.GetLen(); j++) {
            if(temp.GetLev(j) <= p) {
                new_val[new_len]  = temp.GetVal(j);
                new_cind[new_len] = temp.GetIdx(j);
                new_lev[new_len]  = temp.GetLev(j);
                new_len++;
            }
        }
        new_rptr[i+1] = new_len;
    }
    (*val)  = utils::SafeRealloc<T>(new_val, new_len);
    (*cind) = utils::SafeRealloc<int>(new_cind, new_len);
    (*rptr) = utils::SafeRealloc<int>(new_rptr, N+1);
    free(zeros);
}

template <typename T>
void AllocBlockZero(T **val, int **cind, int **rptr, int N, int bnl, int bnw)
{
    T *bval;
    int *bcind;
    int *brptr;
    Csr2Bcsr<T>(*val, *cind, *rptr, &bval, &bcind, &brptr, N, bnl, bnw);
    free(*val);
    free(*cind);
    free(*rptr);
    Bcsr2Csr<T>(bval, bcind, brptr, val, cind, rptr, N, bnl, bnw);
    free(bval);
    free(bcind);
    free(brptr);
}

template <typename T>
void AllocPowerZero(T **val, int **cind, int **rptr, int N, int p)
{
    int len2 = 0;
    int mlen2 = 8;
    T *val2 = utils::SafeMalloc<T>(mlen2);
    int *cind2 = utils::SafeMalloc<int>(mlen2);
    int *rptr2 = utils::SafeMalloc<int>(N+1);
    rptr2[0] = 0;
    for(int i=0; i<N; i++) {
        int len = (*rptr)[i+1]-(*rptr)[i];
        int *ptr = utils::SafeMalloc<int>(len);
        int *t = utils::SafeMalloc<int>(8);
        int t_len = 0;
        int t_mlen = 8;
        // Initialize
        for(int j=0; j<len; j++) {
            ptr[j] = (*rptr)[(*cind)[(*rptr)[i]+j]];
        }
        //
        while(true) {
            int min = N;
            for(int j=0; j<len; j++) {
                if(ptr[j] != -1 && (*cind)[ptr[j]] < min)
                    min = (*cind)[ptr[j]];
            }
            if(min == N) break;
            for(int j=0; j<len; j++) {
                if(ptr[j] == -1) continue;
                if((*cind)[ptr[j]] == min) {
                    ptr[j]++;
                    if(ptr[j] >= (*rptr)[(*cind)[(*rptr)[i]+j]+1]) {
                        ptr[j] = -1;
                    }
                }
            }
            if(t_len+1 == t_mlen) {
                t_mlen *= 2;
                t = utils::SafeRealloc<int>(t, t_mlen);
            }
            t[t_len] = min; t_len++;
        }
        int a_ptr = (*rptr)[i];
        int t_ptr = 0;
        while(true) {
            int a_ind = (a_ptr != -1)? (*cind)[a_ptr]: N;
            int t_ind = (t_ptr != -1)? t[t_ptr]: N;
            if(a_ind == N && t_ind == N) break;
            T v_elem;
            int c_elem;
            if(a_ind < t_ind) {
                v_elem = (*val)[a_ptr]; c_elem = a_ind;
                a_ptr = (a_ptr != (*rptr)[i+1]-1)? a_ptr+1: -1;
            }else if(a_ind == t_ind) {
                v_elem = (*val)[a_ptr]; c_elem = a_ind;
                a_ptr = (a_ptr != (*rptr)[i+1]-1)? a_ptr+1: -1;
                t_ptr = (t_ptr != t_len-1)? t_ptr+1: -1;
            }else {
                v_elem = 0; c_elem = t_ind;
                t_ptr = (t_ptr != t_len-1)? t_ptr+1: -1;
            }
            if(len2+1 == mlen2) {
                mlen2 *= 2;
                val2 = utils::SafeRealloc<T>(val2, mlen2);
                cind2 = utils::SafeRealloc<int>(cind2, mlen2);
            }
            val2[len2] = v_elem;
            cind2[len2] = c_elem;
            len2++;
        }
        rptr2[i+1] = len2;
        free(t);
        free(ptr);
    }
    free(*val);
    free(*cind);
    free(*rptr);
    *val = utils::SafeRealloc<T>(val2, rptr2[N]);
    *cind = utils::SafeRealloc<int>(cind2, rptr2[N]);
    *rptr = utils::SafeRealloc<int>(rptr2, N+1);
}

template <typename T>
void RemoveOffDiagonal(T **val, int **cind, int **rptr, int N, int bnum)
{
    int bsize = N / bnum;
    int off = 0;
    for(int i=0; i<N; i++) {
        int start = (*rptr)[i]+off;
        int min = i / bsize * bsize;
        int max = min + bsize;
        for(int j=start; j<(*rptr)[i+1]; j++) {
            if((*cind)[j] < min || max <= (*cind)[j]) {
                off++;
                continue;
            }
            (*val)[j-off] = (*val)[j];
            (*cind)[j-off] = (*cind)[j];
        }
        (*rptr)[i+1] -= off;
    }
    *val = utils::SafeRealloc<double>(*val, (*rptr)[N]);
    *cind = utils::SafeRealloc<int>(*cind, (*rptr)[N]);
}

template <typename T>
void RemoveABMC(
    T **val, int **cind, int **rptr, int N,
    int num_color, int *size_color, int size_block)
{
    int off = 0;
    for(int cidx=0; cidx<num_color; cidx++) {
        int cstart = size_color[cidx] * size_block;
        int cend = size_color[cidx+1] * size_block;
        for(int i=cstart; i<cend; i+=size_block) {
            for(int j=0; j<size_block; j++) {
                int start = (*rptr)[i+j]+off;
                for(int k=start; k<(*rptr)[i+j+1]; k++) {
                    if(cstart <= (*cind)[k] && (*cind)[k] < cend) {
                        if((*cind)[k] < i || i + size_block <= (*cind)[k]) {
                            off++;
                            continue;
                        }
                    }
                    (*val)[k-off] = (*val)[k];
                    (*cind)[k-off] = (*cind)[k];
                }
                (*rptr)[i+j+1] -= off;
            }
        }
    }
}

template <typename T>
void Reordering(T *val, int *cind, int *rptr, int *RP, int N)
{
    #pragma omp parallel for
    for(int i=0; i<N; i++) {
        for(int j=rptr[i]; j<rptr[i+1]; j++) {
            cind[j] = RP[cind[j]];
        }
        helper::QuickSort<int, double>(cind, val, rptr[i], rptr[i+1]-1);
    }
}

template <typename T>
void Reordering(T *val, int *cind, int *rptr, int *LP, int *RP, int N)
{
    int NNZ = rptr[N];
    double *tval = utils::SafeMalloc<double>(NNZ);
    int *tcind = utils::SafeMalloc<int>(NNZ);
    int *trptr = utils::SafeMalloc<int>(N+1);
    trptr[0] = 0;
    for(int i=0; i<N; i++) {
        int id = LP[i];
        trptr[i+1] = trptr[i] + (rptr[id+1] - rptr[id]);
    }
    #pragma omp parallel for
    for(int i=0; i<N; i++) {
        int id = LP[i];
        for(int j=0; j<rptr[id+1]-rptr[id]; j++) {
            tval[trptr[i]+j] = val[rptr[id]+j];
            tcind[trptr[i]+j] = RP[cind[rptr[id]+j]];
        }
        helper::QuickSort<int, double>(tcind, tval, trptr[i], trptr[i+1]-1);
    }
    utils::Copy<double>(tval, val, NNZ);
    utils::Copy<int>(tcind, cind, NNZ);
    utils::Copy<int>(trptr, rptr, N+1);
    free(tval);
    free(tcind);
    free(trptr);
}
*/
/*
int Csr2PaddedCsr(
    double **val, int **cind, int **rptr,
    int size, int N)
{
    int nnz = 0;
    int *t_rptr = new int[N+1];
    t_rptr[0] = 0;
    for(int i=0; i<N; i++) {
        int num = rptr[0][i+1] - rptr[0][i];
        nnz += (num+size-1)/size * size;
        t_rptr[i+1] = nnz;
    }
    double *t_val = new double[nnz];
    int *t_cind = new int[nnz];
    for(int i=0; i<N; i++) {
        int num = rptr[0][i+1] - rptr[0][i];
        int cnt = 0;
        for(int j=t_rptr[i]; j<t_rptr[i+1]; j++) {
            if(cnt < num) {
                t_val[j] = val[0][rptr[0][i]+cnt];
                t_cind[j] = cind[0][rptr[0][i]+cnt];
            }else {
                t_val[j] = 0;
                t_cind[j] = t_cind[j-1];
            }
            cnt++;
        }
    }
    *val = (double*)realloc(*val, sizeof(double)*nnz);
    *cind = (int*)realloc(*cind, sizeof(int)*nnz);
    for(int i=0; i<nnz; i++) {val[0][i] = t_val[i];}
    for(int i=0; i<nnz; i++) {cind[0][i] = t_cind[i];}
    for(int i=0; i<N+1; i++) {rptr[0][i] = t_rptr[i];}
    
    return nnz;
}
*/

} // namespace matrix

} // namespace helper

} // namespace senk

#endif
