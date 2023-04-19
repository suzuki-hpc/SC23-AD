#ifndef SENKPP_PARAMETERS_HPP
#define SENKPP_PARAMETERS_HPP

namespace senk {

/**
 * For Preconditioners
 **/
struct ILUpParam : public PreconditionerParam {
public:
    int level;
    double alpha;
    ILUpParam(int _level, double _alpha) {
        level = _level; alpha = _alpha;
    }
    ~ILUpParam() {}
};

struct BJILUpParam : public PreconditionerParam {
public:
    int level, bj_num, unit;
    double alpha;
    BJILUpParam(int _level, double _alpha, int _bj_num, int _unit) {
        level  = _level;  alpha = _alpha;
        bj_num = _bj_num; unit  = _unit;
    }
    ~BJILUpParam() {}
};

struct Shifted_BJILUpParam : public PreconditionerParam {
public:
    int level, bj_num;
    Shifted_BJILUpParam(int _level, int _bj_num) {
        level  = _level; bj_num = _bj_num;
    }
    ~Shifted_BJILUpParam() {}
};

struct ILUBpParam : public PreconditionerParam {
public:
    int level;
    int bnl, bnw;
    ILUBpParam(int _level, int _bnl, int _bnw) {
        level = _level; bnl = _bnl; bnw = _bnw;
    }
    ~ILUBpParam() {}
};

struct BJILUBpParam : public PreconditionerParam {
public:
    int level, bj_num, unit;
    int bnl, bnw;
    BJILUBpParam(int _level, int _bj_num, int _unit, int _bnl, int _bnw) {
        level  = _level; bj_num = _bj_num; unit = _unit;
        bnl = _bnl; bnw = _bnw;
    }
    ~BJILUBpParam() {}
};

/**
 * For Smoothers
 **/
struct SmootherParam {
public:
    virtual ~SmootherParam() = 0;
};
SmootherParam::~SmootherParam() {}

struct JacobiParam : public SmootherParam {
public:
    double omega;
    int iter;
    JacobiParam(double _omega, int _iter) {
        omega = _omega; iter  = _iter;
    }
    ~JacobiParam() {}
};

struct SORParam : public SmootherParam {
public:
    double omega;
    int iter;
    SORParam(double _omega, int _iter) {
        omega = _omega; iter  = _iter;
    }
    ~SORParam() {}
};

struct BlockSORParam : public SmootherParam {
public:
    double omega;
    int iter, b_num;
    BlockSORParam(double _omega, int _iter, int _b_num) {
        omega = _omega; iter  = _iter; b_num = _b_num;
    }
    ~BlockSORParam() {}
};

struct RichardsonParam : public SmootherParam {
public:
    PreconditionerParam *param;
    int iter;
    RichardsonParam(PreconditionerParam *_param, int _iter) {
        param = _param; iter  = _iter;
    }
    ~RichardsonParam() {}
};

struct AMGParam : public PreconditionerParam {
public:
    int level;
    double theta;
    int cycle;
    SmootherParam *param;
    AMGParam(int _level, double _theta, int _cycle, SmootherParam *_param) {
        level = _level; theta = _theta; cycle = _cycle;
        param = _param;
    }
    ~AMGParam() {}
};

struct SA_AMGParam : public PreconditionerParam {
public:
    double theta;
    int cycle;
    SmootherParam *param;
    SA_AMGParam(double _theta, int _cycle, SmootherParam *_param) {
        theta = _theta; cycle = _cycle; param = _param;
    }
    ~SA_AMGParam() {}
};

} // senk

#endif
