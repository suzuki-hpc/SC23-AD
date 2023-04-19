/**
 * @file printer.hpp
 * @brief The Printer class is defined.
 * @author Kengo Suzuki
 * @date 02/02/2023
 */
#ifndef SENKPP_PRINTER_HPP
#define SENKPP_PRINTER_HPP

#include <iostream>
#include <iomanip>
#include <string>

namespace senk {

class Printer {
    std::string title;
    std::string marker;
    int len;
public:
    Printer(std::string _marker, std::string _title) {
        title = _title;
        marker = _marker;
        len = _title.size();
        std::cout << marker << " == [" << title << "] == " << std::endl;
    }
    template <typename T>
    void PrintNameValue(std::string name, T value) {
        std::cout << marker << " " << std::setw(10) << std::left 
            << name << " : " << value << std::endl;
    }
    ~Printer() {
        std::cout << marker << " ====";
        for(int i=0; i<len; i++) {std::cout << "=";}
        std::cout << "==== " << std::endl;
    }
};

} // senk

#endif