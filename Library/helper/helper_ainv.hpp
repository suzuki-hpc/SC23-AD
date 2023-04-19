#ifndef SENK_HELPER_AINV_HPP
#define SENK_HELPER_AINV_HPP

#include "helper/helper_sparse.hpp"

namespace senk {

namespace helper {

namespace ainv {

#define LEN 8
#define FREE_NOW 1

static void convert_to_csc(sparse::SpVec_V_R **z, double **val, int **rind, int **cptr, int N)
{
    int i, j;
    *cptr = utils::SafeMalloc<int>(N+1);
    (*cptr)[0] = 0;
    for(i=0; i<N; i++) {
        (*cptr)[i+1] = (*cptr)[i] + z[i]->len;
    }
    *val  = utils::SafeMalloc<double>((*cptr)[N]);
    *rind = utils::SafeMalloc<int>((*cptr)[N]);
    for(i=0; i<N; i++) {
        int off = (*cptr)[i];
        int len = z[i]->len;
        for(j=0; j<len; j++) {
            (*val)[off+j]  = z[i]->elems[j].val;
            (*rind)[off+j] = z[i]->elems[j].row;
        }
    }
}

inline void left_ainv_sym(
    double *val, int *cind, int *rptr,
    double **z_val, int **z_rind, int **z_cptr,
    double **d, int N, double tol, double acc)
{
    *d = utils::SafeMalloc<double>(N);
    sparse::SpVec_V_R **z = utils::SafeMalloc<sparse::SpVec_V_R*>(N);
    for(int i=0; i<N; i++) {
        z[i] = new sparse::SpVec_V_R(LEN);
        z[i]->elems[0].val = 1.0;
        z[i]->elems[0].row = i;
        z[i]->len = 1;
    }
    sparse::SpVec_R **h = utils::SafeMalloc<sparse::SpVec_R*>(N);
    for(int i=0; i<N; i++) {
        h[i] = new sparse::SpVec_R(LEN);
        h[i]->row[0] = i;
        h[i]->len = 1;
    }
    sparse::SpVec_V_R *temp = new sparse::SpVec_V_R(LEN);
    
    int *flag = utils::SafeCalloc<int>(N);
    for(int i=0; i<N; i++) {
        int a_len = rptr[i+1] - rptr[i];
        int off = rptr[i];
        flag[i] = 0;
        for(int j=0; j<a_len; j++) {
            int row = cind[off+j];
            if(row >= i) break;
            for(int k=0; k<h[row]->len; k++) {
                int col = h[row]->row[k];
                if(flag[col] == i) continue;
                double valu = z[col]->Dot(&val[off], &cind[off], a_len);
                double alpha = -valu/(*d)[col];
                sparse::ainv_axpy_tol(alpha, z[col], z[i], temp, tol, h, col);
                flag[col] = i;
            }
        }
        int end = z[i]->len-1;
        z[i]->elems[end].val *= acc;
        (*d)[i] = z[i]->Dot(&val[off], &cind[off], a_len);
        z[i]->elems[end].val /= acc;
        if((*d)[i] == 0) { (*d)[i] = 1; }
    }
    convert_to_csc(z, z_val, z_rind, z_cptr, N);
    temp->Free(); delete temp;
#if FREE_NOW
    for(int i=0; i<N; i++) {
        z[i]->Free(); delete z[i];
    }
    utils::SafeFree(&z);
    for(int i=0; i<N; i++) {
        h[i]->Free(); delete h[i];
    }
    utils::SafeFree(&h);
#endif
}

inline void left_ainv(
    double *val, int *cind, int *rptr,
    double *c_val, int *c_rind, int *c_cptr,
    double **z_val, int **z_rind, int **z_cptr,
    double **wt_val, int **wt_cind, int **wt_rptr,
    double **d, int N, double tol, double acc)
{
    *d = utils::SafeMalloc<double>(N);
    sparse::SpVec_V_R **z = utils::SafeMalloc<sparse::SpVec_V_R*>(N);
    for(int i=0; i<N; i++) {
        z[i] = new sparse::SpVec_V_R(LEN);
        z[i]->elems[0].val = 1.0;
        z[i]->elems[0].row = i;
        z[i]->len = 1;
    }
    sparse::SpVec_V_R **w = utils::SafeMalloc<sparse::SpVec_V_R*>(N);
    for(int i=0; i<N; i++) {
        w[i] = new sparse::SpVec_V_R(LEN);
        w[i]->elems[0].val = 1.0;
        w[i]->elems[0].row = i;
        w[i]->len = 1;
    }
    sparse::SpVec_R **h_z = utils::SafeMalloc<sparse::SpVec_R*>(N);
    for(int i=0; i<N; i++) {
        h_z[i] = new sparse::SpVec_R(LEN);
        h_z[i]->row[0] = i;
        h_z[i]->len = 1;
    }
    sparse::SpVec_R **h_w = utils::SafeMalloc<sparse::SpVec_R*>(N);
    for(int i=0; i<N; i++) {
        h_w[i] = new sparse::SpVec_R(LEN);
        h_w[i]->row[0] = i;
        h_w[i]->len = 1;
    }
    sparse::SpVec_V_R *temp = new sparse::SpVec_V_R(LEN);
    int *flag_z = utils::SafeCalloc<int>(N);
    int *flag_w = utils::SafeCalloc<int>(N);
    for(int i=0; i<N; i++) {
        int a_len = rptr[i+1] - rptr[i];
        int off = rptr[i];
        int at_len = c_cptr[i+1] - c_cptr[i];
        int c_off = c_cptr[i];
        // Update w[i]
        flag_w[i] = 0;
        for(int j=0; j<at_len; j++) {
            int row = c_rind[c_off+j];
            if(row >= i) break;
            for(int k=0; k<h_z[row]->len; k++) {
                int col = h_z[row]->row[k];
                if(flag_w[col] == i) continue;
                double valu = z[col]->Dot(&val[off], &cind[off], a_len);
                double alpha = -valu/(*d)[col];
                sparse::ainv_axpy_tol(alpha, w[col], w[i], temp, tol, h_w, col);
                flag_w[col] = i;
            }
        }
        // Update z[i]
        flag_z[i] = 0;
        for(int j=0; j<a_len; j++) {
            int row = cind[off+j];
            if(row >= i) break;
            for(int k=0; k<h_w[row]->len; k++) {
                int col = h_w[row]->row[k];
                if(flag_z[col] == i) continue;
                double valu = w[col]->Dot(&c_val[c_off], &c_rind[c_off], at_len);
                double alpha = -valu/(*d)[col];
                sparse::ainv_axpy_tol(alpha, z[col], z[i], temp, tol, h_z, col);
                flag_z[col] = i;
            }
        }
        int end = w[i]->len-1;
        w[i]->elems[end].val *= acc;
        (*d)[i] = w[i]->Dot(&c_val[c_off], &c_rind[c_off], at_len);
        w[i]->elems[end].val /= acc;
        if((*d)[i] == 0) { (*d)[i] = 1; }
    }
    convert_to_csc(z, z_val, z_rind, z_cptr, N);
    convert_to_csc(w, wt_val, wt_cind, wt_rptr, N);
    temp->Free(); delete temp;
#if FREE_NOW
    for(int i=0; i<N; i++) {
        z[i]->Free(); delete z[i];
    }
    utils::SafeFree<sparse::SpVec_V_R*>(&z);
    for(int i=0; i<N; i++) {
        w[i]->Free(); delete w[i];
    }
    utils::SafeFree<sparse::SpVec_V_R*>(&w);
    for(int i=0; i<N; i++) {
        h_z[i]->Free(); delete h_z[i];
    }
    utils::SafeFree(&h_z);
    for(int i=0; i<N; i++) {
        h_w[i]->Free(); delete h_w[i];
    }
    utils::SafeFree(&h_w);
#endif
}

inline void left_sdainv_sym(
    double *val, int *cind, int *rptr,
    double **z_val, int **z_rind, int **z_cptr,
    double **d, int N, double tol, double acc)
{
    *d = utils::SafeMalloc<double>(N);
    sparse::SpVec_V_R **z = utils::SafeMalloc<sparse::SpVec_V_R*>(N);
    for(int i=0; i<N; i++) {
        z[i] = new sparse::SpVec_V_R(LEN);
        z[i]->elems[0].val = 1.0;
        z[i]->elems[0].row = i;
        z[i]->len = 1;
    }
    sparse::SpVec_V_R *temp = new sparse::SpVec_V_R(LEN);
    for(int i=0; i<N; i++) {
        int a_len = rptr[i+1] - rptr[i];
        int off = rptr[i];
        int len = 0;
        while(cind[off+len] < i) len++;
        for(int j=0; j<len; j++) {
            int col = cind[off+j];
            double valu = z[col]->Dot(&val[off], &cind[off], a_len);
            double alpha = -valu/(*d)[col];
            sparse::axpy_tol(alpha, z[col], z[i], temp, tol);
        }
        int end = z[i]->len-1;
        z[i]->elems[end].val *= acc;
        (*d)[i] = z[i]->Dot(&val[off], &cind[off], a_len);
        z[i]->elems[end].val /= acc;
        if((*d)[i] == 0) { (*d)[i] = 1; }
    }
    convert_to_csc(z, z_val, z_rind, z_cptr, N);
    temp->Free(); delete temp;
#if FREE_NOW    
    for(int i=0; i<N; i++) {
        z[i]->Free(); delete z[i];
    }
    utils::SafeFree<sparse::SpVec_V_R*>(&z);
#endif
}

inline void left_sdainv(
    double *val, int *cind, int *rptr,
    double *c_val, int *c_rind, int *c_cptr,
    double **z_val, int **z_rind, int **z_cptr,
    double **wt_val, int **wt_cind, int **wt_rptr,
    double **d, int N, double tol, double acc)
{
    *d = utils::SafeMalloc<double>(N);
    sparse::SpVec_V_R **z = utils::SafeMalloc<sparse::SpVec_V_R*>(N);
    for(int i=0; i<N; i++) {
        z[i] = new sparse::SpVec_V_R(LEN);
        z[i]->elems[0].val = 1.0;
        z[i]->elems[0].row = i;
        z[i]->len = 1;
    }
    sparse::SpVec_V_R **w = utils::SafeMalloc<sparse::SpVec_V_R*>(N);
    for(int i=0; i<N; i++) {
        w[i] = new sparse::SpVec_V_R(LEN);
        w[i]->elems[0].val = 1.0;
        w[i]->elems[0].row = i;
        w[i]->len = 1;
    }
    sparse::SpVec_V_R *temp = new sparse::SpVec_V_R(LEN);
    for(int i=0; i<N; i++) {
        int a_len = rptr[i+1] - rptr[i];
        int off = rptr[i];
        int at_len = c_cptr[i+1] - c_cptr[i];
        int c_off = c_cptr[i];
        int z_len = 0, w_len = 0;
        while(cind[off+w_len] < i) w_len++;
        while(c_rind[c_off+z_len] < i) z_len++;
        // Update w[i]
        for(int j=0; j<w_len; j++) {
            int col = cind[off+j];
            double valu = z[col]->Dot(&val[off], &cind[off], a_len);
            double alpha = -valu/(*d)[col];
            sparse::axpy_tol(alpha, w[col], w[i], temp, tol);
        }
        // Update z[i]
        for(int j=0; j<z_len; j++) {
            int col = c_rind[c_off+j];
            double valu = w[col]->Dot(&c_val[c_off], &c_rind[c_off], at_len);
            double alpha = -valu/(*d)[col];
            sparse::axpy_tol(alpha, z[col], z[i], temp, tol);
        }
        int end = w[i]->len-1;
        w[i]->elems[end].val *= acc;
        (*d)[i] = w[i]->Dot(&c_val[c_off], &c_rind[c_off], at_len);
        w[i]->elems[end].val /= acc;
        if((*d)[i] == 0) { (*d)[i] = 1; }
    }
    convert_to_csc(z, z_val, z_rind, z_cptr, N);
    convert_to_csc(w, wt_val, wt_cind, wt_rptr, N);
    temp->Free(); delete temp;
#if FREE_NOW
    for(int i=0; i<N; i++) {
        z[i]->Free();
        delete z[i];
    }
    utils::SafeFree<sparse::SpVec_V_R*>(&z);
    for(int i=0; i<N; i++) {
        w[i]->Free();
        delete w[i];
    }
    utils::SafeFree<sparse::SpVec_V_R*>(&w);
#endif
}

} // ainv

} // helper

} // senk

#endif