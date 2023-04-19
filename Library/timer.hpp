/**
 * @file timer.hpp
 * @brief The Timer class is defined.
 * @author Kengo Suzuki
 * @date 02/02/2023
 */
#ifndef SENKPP_TIMER_HPP
#define SENKPP_TIMER_HPP

#include <iostream>
#include <iomanip>
#include <sys/time.h>

namespace senk {

double getTime() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (double)tv.tv_sec + (double)tv.tv_usec*1.0e-6;
}

class Timer {
	double start, end;
public:
	Timer() { start = getTime(); }
	void Restart() { start = getTime(); }
	void Elapsed() {
		end = getTime();
		std::cout << "[Time] : " << std::fixed << std::setprecision(15) << end - start << std::endl;
	}
};

} // senk

#endif
