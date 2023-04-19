#ifndef SENKPP_HELPER_EXPANSION_HPP
#define SENKPP_HELPER_EXPANSION_HPP

namespace senk {

namespace helper {

template <int head, int... hoge> inline
constexpr int get_head() { return head; }

}

}

#endif
