
#include <pybind11/embed.h>
#include <iostream>
#include <cstdio>
#include <ctime> 
namespace py = pybind11;
 
int main() {
    py::scoped_interpreter python;
 
    py::module sys = py::module::import("sys");
    py::print(sys.attr("path"));
 
    py::module t = py::module::import("tttt");
    t.attr("add")(1,2);

    std::clock_t start;
    double duration;

    start = std::clock(); // get current time
    for (int i = 0 ; i< 1000; i++) {
       auto res = t.attr("_preprocess")();
    }
    //std::cout << "return : "<< res.cast<int>() <<std::endl;
    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

    std::cout << "Operation took "<< duration << "seconds" << std::endl;
    
}
