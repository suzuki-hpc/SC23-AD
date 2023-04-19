#ifndef SENKPP_HELPER_SPARSE_HPP
#define SENKPP_HELPER_SPARSE_HPP

#include <cmath>
#include <climits>

namespace senk {

namespace helper {

namespace sparse {

/** For SpVec_V_R **/

struct V_R {
    double val;
    int row;
};

class SpVec_V_R {
public:
    V_R *elems;
    int len;
    int mlen;
    SpVec_V_R(int _mlen) {
        elems = utils::SafeMalloc<V_R>(_mlen);
        len = 0;
        mlen = _mlen;
    }
    double Dot(SpVec_V_R *v) {
        double res = 0;
        int pos = 0, v_pos = 0;
        while(pos < len && v_pos < v->len) {
            int temp   = elems[pos].row;
            int v_temp = v->elems[v_pos].row;
            if(temp < v_temp) {
                pos++;
            }else if(temp == v_temp) {
                res += elems[pos].val * v->elems[v_pos].val;
                pos++; v_pos++;
            }else {
                v_pos++;
            }
        }
        return res;
    }
    double Dot(double *v_val, int *v_row, int v_len) {
        double res = 0;
        int pos = 0, v_pos = 0;
        while(pos < len && v_pos < v_len) {
            int temp   = elems[pos].row;
            int v_temp = v_row[v_pos];
            if(temp < v_temp) {
                pos++;
            }else if(temp == v_temp) {
                res += elems[pos].val * v_val[v_pos];
                pos++; v_pos++;
            }else {
                v_pos++;
            }
        }
        return res;
    }
    void Append(V_R _elem) {
        if(mlen == len) {
            mlen *= 2;
            elems = utils::SafeRealloc<V_R>(elems, mlen);
        }
        elems[len] = _elem;
        len++;
    }
    void Append(double _val, int _row) {
        if(mlen == len) {
            mlen *= 2;
            elems = utils::SafeRealloc<V_R>(elems, mlen);
        }
        elems[len].val = _val;
        elems[len].row = _row;
        len++;
    }
    void Free() {
        utils::SafeFree<V_R>(&elems);
    }
};

inline void axpy(double alpha, SpVec_V_R *x, SpVec_V_R *y, SpVec_V_R *temp)
{
    V_R elem;
    int x_pos = 0, y_pos = 0;
    temp->len = 0;
    while(x_pos < x->len || y_pos < y->len) {
        int x_temp = (x_pos < x->len) ?
            x->elems[x_pos].row : INT_MAX;
        int y_temp = (y_pos < y->len) ?
            y->elems[y_pos].row : INT_MAX;
        if(x_temp < y_temp) {
            elem = x->elems[x_pos];
            elem.val *= alpha;
            x_pos++;
        }else if(x_temp == y_temp) {
            elem = y->elems[y_pos];
            elem.val += alpha * x->elems[x_pos].val;
            x_pos++; y_pos++;
        }else {
            elem = y->elems[y_pos];
            y_pos++;
        }
        if(std::fabs(elem.val) == 0) continue;
        //if(fabs(val) < tol) continue;
        temp->Append(elem);
    }
    y->len = 0;
    for(int i=0; i<temp->len; i++) {
        y->Append(temp->elems[i]);
    }
}

inline void axpy_tol(double alpha, SpVec_V_R *x, SpVec_V_R *y, SpVec_V_R *temp, double tol)
{
    V_R elem;
    int x_pos = 0, y_pos = 0;
    temp->len = 0;
    while(x_pos < x->len || y_pos < y->len) {
        int x_temp = (x_pos < x->len) ?
            x->elems[x_pos].row : INT_MAX;
        int y_temp = (y_pos < y->len) ?
            y->elems[y_pos].row : INT_MAX;
        if(x_temp < y_temp) {
            elem = x->elems[x_pos];
            elem.val *= alpha;
            x_pos++;
        }else if(x_temp == y_temp) {
            elem = y->elems[y_pos];
            elem.val += alpha * x->elems[x_pos].val;
            x_pos++; y_pos++;
        }else {
            elem = y->elems[y_pos];
            y_pos++;
        }
        if(std::fabs(elem.val) < tol) continue;
        temp->Append(elem);
    }
    y->len = 0;
    for(int i=0; i<temp->len; i++) {
        y->Append(temp->elems[i]);
    }
}

/** For SpVec **/

class SpVec {
public:
    double *val;
    int *row;
    int len;
    int mlen;
    SpVec(int _mlen) {
        val = utils::SafeMalloc<double>(_mlen);
        row = utils::SafeMalloc<int>(_mlen);
        len = 0;
        mlen = _mlen;
    }
    SpVec(double *_val, int *_row, int _len, int _mlen) {
        val = _val; row = _row; len = _len; mlen = _mlen;
    }
    double Dot(double *v_val, int *v_row, int v_len) {
        double res = 0;
        int pos = 0, v_pos = 0;
        while(pos < len && v_pos < v_len) {
            int temp   = row[pos];
            int v_temp = v_row[v_pos];
            if(temp < v_temp) {
                pos++;
            }else if(temp == v_temp) {
                res += val[pos] * v_val[v_pos];
                pos++; v_pos++;
            }else {
                v_pos++;
            }
        }
        return res;
    }
    void Append(double _val, int _row) {
        if(mlen == len) {
            mlen = (int)((double)mlen * 1.5);
            val = utils::SafeRealloc<double>(val, mlen);
            row = utils::SafeRealloc<int>(row, mlen);
        }
        val[len] = _val;
        row[len] = _row;
        len++;
    }
    void Append(double *_val, int *_row, int _len) {
        if(len+_len > mlen) {
            while(len+_len > mlen) {
                mlen = (int)((double)mlen * 1.5);
            }
            val = utils::SafeRealloc<double>(val, mlen);
            row = utils::SafeRealloc<int>(row, mlen);
        }
        for(int i=0; i<_len; i++) {
            val[len+i] = _val[i];
            row[len+i] = _row[i];
        }
        len += _len;
    }
    void Free() {
        utils::SafeFree<double>(&val);
        utils::SafeFree<int>(&row);
    }
};

inline void axpy(double alpha, SpVec *x, SpVec *y, SpVec *temp)
{
    double val;
    int row;
    int x_pos = 0, y_pos = 0;
    temp->len = 0;
    while(x_pos < x->len || y_pos < y->len) {
        int x_temp = (x_pos < x->len) ? x->row[x_pos] : INT_MAX;
        int y_temp = (y_pos < y->len) ? y->row[y_pos] : INT_MAX;
        if(x_temp < y_temp) {
            row = x->row[x_pos];
            val = alpha * x->val[x_pos];
            x_pos++;
        }else if(x_temp == y_temp) {
            row = x->row[x_pos];
            val = y->val[y_pos] + alpha * x->val[x_pos];
            x_pos++; y_pos++;
        }else {
            row = y->row[y_pos];
            val = y->val[y_pos];
            y_pos++;
        }
        if(std::fabs(val) == 0.0) continue;
        temp->Append(val, row);
    }
    y->len = 0;
    for(int i=0; i<temp->len; i++) {
        y->Append(temp->val[i], temp->row[i]);
    }
}

inline void cscSpMSpV(
    double *val, int *rind, int *cptr,
    SpVec *in, SpVec *out, SpVec *temp)
{
    out->len = 0;
    for(int i=0; i<in->len; i++) {
        int off = cptr[in->row[i]];
        int len = cptr[in->row[i]+1] - cptr[in->row[i]];
        SpVec mat = SpVec(&val[off], &rind[off], len, len);
        axpy(in->val[i], &mat, out, temp);
    }
}

inline void cscSpMM(
    double *l_val, int *l_rind, int *l_cptr,
    double *r_val, int *r_rind, int *r_cptr,
    double **res_val, int **res_rind, int **res_cptr,
    int L, int N, int R)
{
    SpVec *temp = new SpVec(8);
    SpVec *res  = new SpVec(8);
    *res_val = NULL;
    *res_rind = NULL;
    *res_cptr = utils::SafeMalloc<int>(R+1);
    (*res_cptr)[0] = 0;
    for(int i=0; i<R; i++) {
        res->len = 0;
        int off = r_cptr[i];
        int len = r_cptr[i+1] - r_cptr[i];
        SpVec vec = SpVec(&r_val[off], &r_rind[off], len, len);
        cscSpMSpV(l_val, l_rind, l_cptr, &vec, res, temp);
        (*res_cptr)[i+1] = (*res_cptr)[i] + res->len;
        *res_val  = utils::SafeRealloc<double>(*res_val, (*res_cptr)[i+1]);
        *res_rind = utils::SafeRealloc<int>(*res_rind, (*res_cptr)[i+1]);
        for(int j=0; j<res->len; j++) {
            (*res_val)[(*res_cptr)[i]+j] = res->val[j];
            (*res_rind)[(*res_cptr)[i]+j] = res->row[j];
        }
    }
    temp->Free(); delete temp;
    res->Free(); delete res;
}

class SpVec_R {
public:
    int *row;
    int len;
    int mlen;
    SpVec_R(int _mlen) {
        row = utils::SafeMalloc<int>(_mlen);
        len = 0;
        mlen = _mlen;
    }
    SpVec_R(double *_val, int *_row, int _len, int _mlen) {
        row = _row; len = _len; mlen = _mlen;
    }
    void Append(int _row) {
        if(mlen == len) {
            mlen *= 2;
            row = utils::SafeRealloc<int>(row, mlen);
        }
        row[len] = _row;
        len++;
    }
    void Free() {
        utils::SafeFree<int>(&row);
    }
};

inline void ainv_axpy_tol(double alpha, SpVec_V_R *x, SpVec_V_R *y, SpVec_V_R *temp, double tol, SpVec_R **h, int idx)
{
    V_R elem;
    int x_pos = 0, y_pos = 0;
    temp->len = 0;
    bool flag;
    while(x_pos < x->len || y_pos < y->len) {
        flag = false;
        int x_temp = (x_pos < x->len) ?
            x->elems[x_pos].row : INT_MAX;
        int y_temp = (y_pos < y->len) ?
            y->elems[y_pos].row : INT_MAX;
        if(x_temp < y_temp) {
            elem = x->elems[x_pos];
            elem.val *= alpha;
            x_pos++;
            flag = true;
        }else if(x_temp == y_temp) {
            elem = y->elems[y_pos];
            elem.val += alpha * x->elems[x_pos].val;
            x_pos++; y_pos++;
        }else {
            elem = y->elems[y_pos];
            y_pos++;
        }
        if(std::fabs(elem.val) < tol) continue;
        if(flag) h[elem.row]->Append(idx);
        temp->Append(elem);
    }
    y->len = 0;
    for(int i=0; i<temp->len; i++) {
        y->Append(temp->elems[i]);
    }
}

template <typename T1, typename T2>
class SpVecLev {
private:
    //! An array that stores the values of the nonzero elements.
    T1 *val;
    //! An array that stores the supplemental values of the nonzero elements.
    T2 *lev;
    //! An array that stores the indices of the nonzero elements.
    int *idx;
    //! The number of the nonzero elements.
    int len;
    //! The size of allocated memories.
    int mlen;
public:
    /**
     * @brief Constructor.
     * @details Allocate memory of size 8.
     */
    SpVecLev() {
        val = utils::SafeMalloc<T1>(8);
        lev = utils::SafeMalloc<T2>(8);
        idx = utils::SafeMalloc<int>(8);
        len = 0;
        mlen = 8;
    }
    /**
     * @brief Destructor.
     * @details Free all memories for the arrays. 
     */
    ~SpVecLev() {
        utils::SafeFree(&val);
        utils::SafeFree(&lev);
        utils::SafeFree(&idx);
    }
    /**
     * @brief Append argument values to the arrays.
     * @param t_val A value to be appended to val.
     * @param t_lev A value to be appended to lev.
     * @param t_idx A value to be appended to idx.
     */
    void Append(T1 t_val, T2 t_lev, int t_idx) {
        if(len == mlen) {
            mlen *= 2;
            val = utils::SafeRealloc(val, mlen);
            lev = utils::SafeRealloc(lev, mlen);
            idx = utils::SafeRealloc(idx, mlen);
        }
        val[len] = t_val;
        lev[len] = t_lev;
        idx[len] = t_idx;
        len++;
    }
    /**
     * @brief Append argument values to the arrays.
     * @param t_val Values to be appended to val.
     * @param t_lev Values to be appended to lev.
     * @param t_idx Values to be appended to idx.
     * @param t_len Length of these argument values.
     */
    void Append(T1 *t_val, T2 *t_lev, int *t_idx, int t_len) {
        if(len+t_len > mlen) {
            while(len+t_len > mlen) mlen *= 2;
            val = utils::SafeRealloc(val, mlen);
            lev = utils::SafeRealloc(lev, mlen);
            idx = utils::SafeRealloc(idx, mlen);
        }
        for(int i=0; i<t_len; i++) {
            val[len+i] = t_val[i];
            lev[len+i] = t_lev[i];
            idx[len+i] = t_idx[i];
        }
        len += t_len;
    }
    /**
     * @brief Return len.
     */
    int GetLen() { return len; }
    /**
     * @brief Return i-th value of val.
     */
    T1 GetVal(int i) { return val[i]; }
    /**
     * @brief Return i-th value of lev.
     */
    T2 GetLev(int i) { return lev[i]; }
    /**
     * @brief Return i-th value of idx.
     */
    int GetIdx(int i) { return idx[i]; }
    /**
     * @brief Return the tuple of i-th values of val, lev, and idx.
     */
    std::tuple<T1, T2, int> Get(int i) {
        //return std::make_tuple(val[i], idx[i]);
        return {val[i], lev[i], idx[i]};
    }
    /**
     * @brief Set the value of len to 0.
     */
    void Clear() { len = 0; }
};

} // sparse

} // helper

} // senl

#endif
