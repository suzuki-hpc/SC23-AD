/**
 * @file enums.hpp
 * @brief Some enum classes are defined.
 * @author Kengo Suzuki
 * @date 02/02/2023
 */
#ifndef SENKPP_ENUMS_HPP
#define SENKPP_ENUMS_HPP

namespace senk {

template <int bit>
class Fixed { };

enum class MMType {Real, Integer, Complex};
enum class MMShape {Symmetric, General};
enum class MMOrder {General, MC, BMC, BJ};

enum class ColorMethod { Greedy, Cyclic, Hybrid };
enum class BlockMethod { Simple, Connect };

}

#endif
