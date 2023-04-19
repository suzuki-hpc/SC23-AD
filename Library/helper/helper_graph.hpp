#ifndef HELPER_HELPER_GRAPH_HPP
#define HELPER_HELPER_GRAPH_HPP

#include <climits>

#include "utils/alloc.hpp"
#include "utils/sort.hpp"

namespace senk {

namespace helper {

namespace graph {

// For unsymmetric
inline void get_adjacency_matrix(
    int *cind, int *rptr, int *csc_rind, int *csc_cptr,
    int **am_cind, int **am_rptr, int N)
{
    *am_cind = utils::SafeMalloc<int>(rptr[N]*2);
    *am_rptr = utils::SafeCalloc<int>(N+1);
    for(int i=0; i<N; i++) {
        int count = 0;
        int r_ptr = rptr[i];
        int c_ptr = csc_cptr[i];
        while(r_ptr < rptr[i+1] || c_ptr < csc_cptr[i+1]) {
            int r_pos = (r_ptr < rptr[i+1]) ? cind[r_ptr] : INT_MAX;
            int c_pos = (c_ptr < csc_cptr[i+1]) ? csc_rind[c_ptr] : INT_MAX;
            if(r_pos < c_pos)       { r_ptr++; }
            else if(r_pos == c_pos) { r_ptr++; c_ptr++; }
            else                    { c_ptr++; }
            count++;
        }
        (*am_rptr)[i+1] = count;
    }
    for(int i=0; i<N; i++) { (*am_rptr)[i+1] += (*am_rptr)[i]; }
    for(int i=0; i<N; i++) {
        int idx = (*am_rptr)[i];
        int r_ptr = rptr[i];
        int c_ptr = csc_cptr[i];
        while(r_ptr < rptr[i+1] || c_ptr < csc_cptr[i+1]) {
            int r_pos = 
                (r_ptr < rptr[i+1]) ? cind[r_ptr] : INT_MAX;
            int c_pos =
                (c_ptr < csc_cptr[i+1]) ? csc_rind[c_ptr] : INT_MAX;
            if(r_pos < c_pos)       {
                (*am_cind)[idx] = r_pos;
                r_ptr++;
            } else if(r_pos == c_pos) {
                (*am_cind)[idx] = r_pos;
                r_ptr++; c_ptr++;
            } else {
                (*am_cind)[idx] = c_pos;
                c_ptr++;
            }
            idx++;
        }
    }
    *am_cind = utils::SafeRealloc<int>(*am_cind, (*am_rptr)[N]);
}

inline void csr_to_csc(
    int *cind, int *rptr,
    int **csc_rind, int **csc_cptr, int N)
{
    *csc_rind = utils::SafeMalloc<int>(rptr[N]);
    *csc_cptr = utils::SafeCalloc<int>(N+1);
    for(int i=0; i<N; i++) {
        for(int j=rptr[i]; j<rptr[i+1]; j++) {
            #pragma omp atomic
            (*csc_cptr)[cind[j]+1]++;
        }
    }
    (*csc_cptr)[0] = 0;
    for(int i=0; i<N; i++) {
        (*csc_cptr)[i+1] += (*csc_cptr)[i];
    }
    for(int i=0; i<N; i++) {
        for(int j=rptr[i]; j<rptr[i+1]; j++) {
            (*csc_rind)[(*csc_cptr)[cind[j]]] = i;
            (*csc_cptr)[cind[j]]++;
        }
    }
    for(int i=N-1; i>=0; i--) {
        (*csc_cptr)[i+1] = (*csc_cptr)[i];
    }
    (*csc_cptr)[0] = 0;
}

inline void blocking_simple(int **b_map, int b_size, int N)
{
    *b_map = utils::SafeMalloc<int>(N);
    for(int i=0; i<N; i++) {
        (*b_map)[i] = i / b_size + 1;
    }
}

inline void blocking_simple_connect(
    int *am_cind, int *am_rptr,
    int **b_map, int b_size, int N)
{
    if(N % b_size != 0) {
        fprintf(stderr, "Error: N must be a multiple of b_size.\n");
        exit(EXIT_FAILURE);
    }
    *b_map = utils::SafeCalloc<int>(N);
    int b_num = N / b_size;
    int *seed_q = utils::SafeCalloc<int>(b_size);
    int head, tail;
    int seed = 0, prev_seed = 0;
    int count = 0;
    bool isSame = false;
    for(int i=0; i<b_num; i++) {
        if(!isSame) count = 0;
        seed = prev_seed; // Selecting a new seed node.
        while((*b_map)[seed] != 0) {seed++;}
        prev_seed = seed;
        head = 0; tail = 0; // Initializing the queue.
        // Assigning the current block ID to the seed node.
        (*b_map)[seed] = i+1; count++;
        if(count == b_size) { isSame = false; continue; }
        // Assigning the current block ID to
        // the nodes adjacent to the seed node
        // until 'b_size' nodes are assigned.
        while(count < b_size) {
            for(int j=am_rptr[seed]; j<am_rptr[seed+1]; j++) {
                int id = am_cind[j];
                if((*b_map)[id] == 0) {
                    (*b_map)[id] = i+1; count++;
                    seed_q[tail] = id; tail++; // Enqueue
                    // When 'b_size' nodes are assigned the current ID,
                    // go to the next ID assignment.
                    if(count == b_size) { isSame = false; break; }
                }
            }
            // If the queue is empty,
            // go back to the initial seed selection.
            if(head == tail) { isSame = true; i--; break; }
            seed = seed_q[head]; head++; // Dequeue
        }
    }
    free(seed_q);
}

inline void get_block_adjacency_matrix(
    int *ajcind, int *ajrptr,
    int *b_map, int *b_list, int b_size,
    int **bajcind, int **bajrptr, int N)
{
    int b_num = N / b_size;
    *bajcind = NULL; //(int*)safeMalloc(sizeof(int), NNZ*2);
    *bajrptr = utils::SafeMalloc<int>(b_num+1);
    
    int *t = utils::SafeMalloc<int>(8);
    int t_len = 0;
    int t_mlen = 8;
    
    // int am_NNZ = 0;
    (*bajrptr)[0] = 0;
    for(int bid=0; bid<b_num; bid++) {     
        t_len = 0;
        for(int j=0; j<b_size; j++) {
            int id = b_list[bid*b_size+j];
            for(int k=ajrptr[id]; k<ajrptr[id+1]; k++) {
                if(t_len == t_mlen) {
                    t_mlen *= 2;
                    t = utils::SafeRealloc<int>(t, t_mlen);
                }
                t[t_len] = b_map[ajcind[k]]-1;
                t_len++;
            }
        }
        utils::QuickSort<int>(0, t_len-1, true, t);
        int off = 0;
        for(int j=1; j<t_len; j++) {
            if(t[j] == t[j-off-1]) { 
                off++; continue;
            }
            t[j-off] = t[j];
        }
        t_len -= off;
        (*bajrptr)[bid+1] = (*bajrptr)[bid] + t_len;
        *bajcind = utils::SafeRealloc<int>(*bajcind, (*bajrptr)[bid+1]);
        for(int j=0; j<t_len; j++) {
            (*bajcind)[(*bajrptr)[bid]+j] = t[j];
        }
    }
    free(t);
}

inline void coloring_greedy(
    int *cind, int *rptr,
    int **c_map, int **c_size, int *c_num, int N)
{
    *c_map = utils::SafeCalloc<int>(N); 
    *c_num = 0;
    *c_size = utils::SafeMalloc<int>((*c_num)+2);
    (*c_size)[0] = 0;
    while(true) {
        int hasOccurred = 0;
        for(int i=0; i<N; i++) {
            if( (*c_map)[i] != 0 ) continue;
            bool isAdjacent = false;
            for(int j=rptr[i]; j<rptr[i+1]; j++) {
                if( (*c_map)[cind[j]] == (*c_num)+1 ) {
                    isAdjacent = true;
                    break;
                }
            }
            if( isAdjacent ) { continue; }
            (*c_map)[i] = (*c_num)+1;
            hasOccurred++;
        }
        if( !hasOccurred ) { break; }
        (*c_num)++;
        *c_size = utils::SafeRealloc<int>(*c_size, (*c_num)+2);
        (*c_size)[*c_num] = (*c_size)[(*c_num)-1] + hasOccurred;
    }
}

inline void coloring_cyclic(
    int *cind, int *rptr,
    int **c_map, int **c_size, int *c_num, int N)
{
    *c_num = rptr[1];
    for(int i=1; i<N; i++) {
        int num = rptr[i+1] - rptr[i];
        if(num > *c_num) *c_num = num;
    }
    *c_map = utils::SafeCalloc<int>(N);
    *c_size = utils::SafeCalloc<int>((*c_num)+1);
    int color_id = 0;
    for(int i=0; i<N; i++) {
        while(1) {
            bool isAdjacent = false;
            for(int j=rptr[i]; j<rptr[i+1]; j++) {
                if(cind[j] >= i) continue;
                if((*c_map)[cind[j]] == color_id+1) {
                   isAdjacent = true; break;
                }
            }
            if( !isAdjacent ) {
                (*c_size)[color_id+1]++;
                (*c_map)[i] = color_id+1;
                color_id = (color_id+1) % *c_num;
                break;
            }else {
                color_id = (color_id+1) % *c_num;
            }
        }  
    }
    for(int i=0; i<*c_num; i++) {
        (*c_size)[i+1] += (*c_size)[i];
    }
}

inline void coloring_greedy_cyclic(
    int *cind, int *rptr,
    int **c_map, int **c_size, int *c_num, int N)
{
    *c_map = utils::SafeCalloc<int>(N);
    *c_num = 0;
    while(true) {
        int hasOccurred = 0;
        for(int i=0; i<N; i++) {
            if( (*c_map)[i] != 0 ) continue;
            bool isAdjacent = false;
            for(int j=rptr[i]; j<rptr[i+1]; j++) {
                if( (*c_map)[cind[j]] == (*c_num)+1 ) {
                    isAdjacent = true;
                    break;
                }
            }
            if( isAdjacent ) { continue; }
            (*c_map)[i] = (*c_num)+1;
            hasOccurred++;
        }
        if( !hasOccurred ) { break; }
        (*c_num)++;
    }
    *c_size = utils::SafeCalloc<int>((*c_num)+30);
    while(true) {
        for(int i=0; i<N; i++) { (*c_map)[i] = 0; }
        for(int i=0; i<(*c_num)+1; i++) { (*c_size)[i] = 0; }
        int color_id = 0;
        int loop_count = 0;
        for(int i=0; i<N; i++) {
            loop_count = 0;
            while(1) {
                if(loop_count >= (*c_num)) {
                    loop_count = -1; break;
                }
                bool isAdjacent = false;
                for(int j=rptr[i]; j<rptr[i+1]; j++) {
                    if(cind[j] >= i) continue;
                    if((*c_map)[cind[j]] == color_id+1) {
                       isAdjacent = true; break;
                    }
                }
                if( !isAdjacent ) {
                    (*c_size)[color_id+1]++;
                    (*c_map)[i] = color_id+1;
                    color_id = (color_id+1) % *c_num;
                    break;
                }else {
                    color_id = (color_id+1) % *c_num;
                }
                loop_count++;
            }
            if(loop_count == -1) { break; }
        }
        if(loop_count == -1) {
            (*c_num)+=1; continue;
        }
        for(int i=0; i<*c_num; i++) {
            (*c_size)[i+1] += (*c_size)[i];
        }
        break;
    }
}

} // graph

} // helper

} // senk

#endif