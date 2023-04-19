#ifndef SENKPP_SMOOTHER_HPP
#define SENKPP_SMOOTHER_HPP

#include <iostream>
#include "preconditioner/preconditioner.hpp"

namespace senk {

template <typename T>
class Smoother : public Preconditioner<T> {
public:
    virtual void Smooth(T *in, T *out, bool isInited) = 0;
    Preconditioner<T> *CastToPreconditioner() {
        return dynamic_cast<Preconditioner<T>*>(this);
    }
};

} // senk

#endif
