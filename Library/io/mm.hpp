#ifndef SENKPP_IO_MM_HPP
#define SENKPP_IO_MM_HPP

#include <sys/mman.h> /* For mmap and munmap*/
#include <sys/stat.h> /* For struct stat */
#include <unistd.h> /* For close */
#include <fcntl.h> /* For open */
#include <cstring>

#include "printer.hpp"
#include "utils/alloc.hpp"
#include "utils/sort.hpp"
#include "helper/helper_matrix.hpp"
#include "matrix/csrmat.hpp"

namespace senk {

namespace io {

inline char *open_map(const char *filename, struct stat *fileInfo);
inline int read_mm_header(char *cptr, MMShape *shape, MMType *type);
inline int read_mm_line(char *cptr, int *row, int *col, double *val);

inline CSRMat *ReadMM(
    const char *filename, bool is_expanded, bool has_zeros)
{
    // auto printer = new Printer("#", "Readfile MatrixMarket");
    
    struct stat fileInfo = {0};
    char *map = open_map(filename, &fileInfo);
    char *map_leading = map;
    
    int offset;
    MMShape shape;
    MMType type;
    int N, M, PE;
    offset = read_mm_header(map, &shape, &type);
    map+=offset;
    sscanf(map, "%d %d %d%n", &N, &M, &PE, &offset);
    map += offset;

    // printer->PrintNameValue("N", N);
    // printer->PrintNameValue("M", M);
    // printer->PrintNameValue("PE", PE);

    double *val = utils::SafeMalloc<double>(PE);
    int *cind   = utils::SafeMalloc<int>(PE);
    int *rptr   = utils::SafeMalloc<int>(N+1);
    int *row    = utils::SafeMalloc<int>(PE);

    int nnz = 0, t_row, t_col;
    double t_val;
    for(int i=0; i<PE; i++) {
        offset = read_mm_line(map, &t_row, &t_col, &t_val);
        map += offset;
        if(!has_zeros && t_val == 0) continue;
        val[nnz]  = t_val;
        row[nnz]  = t_row-1;
        cind[nnz] = t_col-1;
        nnz++;
    }

    utils::QuickSort<int, int*, double*>(0, nnz-1, true, row, cind, val);
    int cnt = 0;
    rptr[0] = 0;
    for(int i=0; i<N; i++) {
        int num = 0;
        while(row[cnt] == i) {
            num++; cnt++;
            if(cnt == nnz) { break; }
        }
        rptr[i+1] = rptr[i] + num;
        utils::QuickSort<int, double*>(rptr[i], rptr[i+1]-1, true, cind, val);
    }
    if(shape == MMShape::Symmetric && is_expanded) {
        double *t_val;
        int *t_cind, *t_rptr;
        senk::helper::matrix::expand<double>(
            val, cind, rptr,
            &t_val, &t_cind, &t_rptr, N, "L");
        utils::SafeFree<double>(&val);
        utils::SafeFree<int>(&cind);
        utils::SafeFree<int>(&rptr);
        val = t_val;
        cind = t_cind;
        rptr = t_rptr;
    }
    // printer->PrintNameValue("NNZ", rptr[N]);

    free(row);
    if (munmap(map_leading, fileInfo.st_size) == -1) {
        perror("Error un-mmapping the file");
        exit(EXIT_FAILURE);
    }

    CSRMat *res = new CSRMat(val, cind, rptr, N, M, shape, type);
    return res;
}

inline char *open_map(const char *filename, struct stat *fileInfo)
{
    int fd = open(filename, O_RDONLY);
    if (fd == -1) {
        std::cerr << "Error opening file for reading" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (fstat(fd, fileInfo) == -1) {
        std::cerr << "Error getting the file size" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (fileInfo->st_size == 0) {
        std::cerr << "Error: File is empty, nothing to do" << std::endl;
        exit(EXIT_FAILURE);
    }
    char *map = (char*)mmap(
        NULL, fileInfo->st_size,
        PROT_READ, MAP_PRIVATE, fd, 0 );
    if (map == MAP_FAILED) {
        close(fd);
        std::cerr << "Error mmapping the file" << std::endl;
        exit(EXIT_FAILURE);
    }
    close(fd);
    return map;
}

inline int read_mm_header(
    char *cptr, MMShape *shape, MMType *type)
{
    int offset = 0;
    const char *header = "%%MatrixMarket matrix coordinate";
    while(cptr[offset] == header[offset]) { offset++; }
    if(header[offset] != '\0') {
        std::cerr << "Error: Invalid coordinate" << std::endl;
        exit(EXIT_FAILURE);
    }
    offset++;
    char temp[256];
    int idx = 0;
    while(cptr[offset] != ' ') {
        temp[idx] = cptr[offset]; idx++; offset++;
    } offset++; temp[idx] = '\0';
    if( strcmp(temp, "real") == 0 ) { *type = MMType::Real; }
    else if( strcmp(temp, "integer") == 0 ) { *type = MMType::Integer; }
    else {
        fprintf(stderr, "%s is not supported.\n", temp);
        exit(EXIT_FAILURE);
    }
    idx = 0;
    while(cptr[offset] != '\n') {
        temp[idx] = cptr[offset]; idx++; offset++;
    } offset++; temp[idx] = '\0';
    if( strcmp(temp, "general") == 0 ) { *shape = MMShape::General; }
    else if( strcmp(temp, "symmetric") == 0 ) { *shape = MMShape::Symmetric; }
    else {
        fprintf(stderr, "%s is not supported.\n", temp);
        exit(EXIT_FAILURE);
        exit(1);
    }
    while(cptr[offset] == '%') {
        while(cptr[offset] != '\n') { offset++; }
        offset++;
    }
    return offset;
}

inline int read_mm_line(
    char *cptr, int *row, int *col, double *val)
{
    int offset = 0;
    char temp[256];
    int idx = 0;
    while(cptr[offset] != ' ') {
        temp[idx] = cptr[offset]; idx++; offset++;
    } offset++; temp[idx] = '\0'; *row = atoi(temp);
    idx = 0;
    while(cptr[offset] != ' ') {
        temp[idx] = cptr[offset]; idx++; offset++;
    } offset++; temp[idx] = '\0'; *col = atoi(temp);
    idx = 0;
    while(cptr[offset] != '\n') {
        temp[idx] = cptr[offset]; idx++; offset++;
    } offset++; temp[idx] = '\0'; *val = atof(temp);
    return offset;
}

} // io

} // senk

#endif
