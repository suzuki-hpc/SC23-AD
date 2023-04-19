/**
 * @file helper.h
 * @brief helper.h contains some supplemental functions.
 * @author Kengo Suzuki
 * @data 23/12/2022
 */
#ifndef SENKPP_HELPER_HPP
#define SENKPP_HELPER_HPP

#include <iostream>

#include <sys/mman.h> /* For mmap and munmap*/
#include <sys/stat.h> /* For struct stat */
#include <unistd.h> /* For close */
#include <fcntl.h> /* For open */
#include <cstring>

#include "enums.hpp"

namespace senk {

namespace helper {

template <typename T>
void swap(T *a, T *b)
{
    T temp = a[0];
    a[0] = b[0];
    b[0] = temp;
}

void pack_swap(int left, int right) {}

template <typename Head, typename... Tail>
void pack_swap(int left, int right, Head list, Tail... tail)
{
    swap(&list[left], &list[right]);
    pack_swap(left, right, std::forward<Tail>(tail)...);
}

char *open_map(const char *filename, struct stat *fileInfo)
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
    // printf("# File size is %ji\n", (intmax_t)fileInfo->st_size);
    char *map = (char*)mmap(
        0, fileInfo->st_size,
        PROT_READ, MAP_PRIVATE, fd, 0 );
    if (map == MAP_FAILED) {
        close(fd);
        std::cerr << "Error mmapping the file" << std::endl;
        exit(EXIT_FAILURE);
    }
    close(fd);
    return map;
}

int read_mm_header(
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

int read_mm_line(
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

}

}

#endif
