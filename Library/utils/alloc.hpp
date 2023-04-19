/**
 * @file alloc.hpp
 * @brief Template functions related to memory allocation are defined.
 * @author Kengo Suzuki
 * @date 02/02/2023
 */
#ifndef SENKPP_UTILS_ALLOC_HPP
#define SENKPP_UTILS_ALLOC_HPP

#include <iostream>
#include <cstdlib>

namespace senk {

namespace utils {

template <typename T>
inline T *SafeMalloc(size_t size)
{
    T *res = (T*)std::malloc(sizeof(T)*size);
    if(!res) {
        std::cerr << "Error: SafeMalloc\n" << std::endl;
        exit(1);
    }
    return res;
}

template <typename T>
inline T *SafeCalloc(size_t size)
{
    T *res = (T*)std::calloc(size, sizeof(T));
    if(!res) { 
        std::cerr << "Error: SafeCalloc\n" << std::endl;
        exit(1);
    }
    return res;
}

template <typename T>
inline T *SafeRealloc(T *old, size_t size)
{
    T *res = (T*)std::realloc(old, sizeof(T)*size);
    if(!res) {
        std::cerr << "Error: SafeRealloc\n" << std::endl;
        exit(1);
    }
    return res;
}

template <typename T>
inline void SafeFree(T **ptr)
{
    if(*ptr) { free(*ptr); }
    *ptr = nullptr;
}

} // utils

} // senk

#endif
