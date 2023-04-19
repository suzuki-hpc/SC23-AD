#ifndef SENKPP_PRECONDITIONER_BASE_HPP
#define SENKPP_PRECONDITIONER_BASE_HPP

namespace senk {

struct PreconditionerParam {
public:
    virtual ~PreconditionerParam() = 0;
};
PreconditionerParam::~PreconditionerParam() {}

template <typename T>
class Preconditioner {
public:
    virtual void Precondition(T *in, T *out) = 0;
    virtual ~Preconditioner() {};
};

template <typename T>
class Preconditioner2 {
public:
    virtual void Precondition(double *in, double *out) = 0;
    virtual void Precondition2(T *in, T *out) = 0;
    virtual ~Preconditioner2() {};
};

} // senk

#endif
