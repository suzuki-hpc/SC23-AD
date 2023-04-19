/**
 * @file sort.hpp
 * @brief Template function that sorts arrays are defined.
 * @author Kengo Suzuki
 * @date 02/02/2023
 */
#ifndef SENKPP_UTILS_SORT_HPP
#define SENKPP_UTILS_SORT_HPP

namespace senk {

namespace utils {

template <typename T> inline
void swap(T *a, T *b)
{
    T temp = a[0];
    a[0] = b[0];
    b[0] = temp;
}

inline void pack_swap(int left, int right) {}

template <typename Head, typename... Tail> inline
void pack_swap(int left, int right, Head list, Tail... tail)
{
    swap(&list[left], &list[right]);
    pack_swap(left, right, std::forward<Tail>(tail)...);
}

template <typename T, typename... Args> inline
void QuickSort(int left, int right, bool isAccending, T *key, Args... args)
{
    int Left = left, Right = right;
    T pivot = key[(left + right) / 2];
    while(1) {
        if(isAccending) {
            while(key[Left] < pivot) Left++;
            while(pivot < key[Right]) Right--;
        }else {
            while(key[Left] > pivot) Left++;
            while(pivot > key[Right]) Right--;
        }
        if (Left >= Right) break;
        swap<T>(&key[Left], &key[Right]);
        pack_swap(Left, Right, args...);
        Left++; Right--;
    }
    if(left < Left-1) QuickSort<T, Args...>(left, Left-1, isAccending, key, args...);
    if(Right+1 < right) QuickSort<T, Args...>(Right+1, right, isAccending, key, args...);
}

} // utils

} // senk

#endif