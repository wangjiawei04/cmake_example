
#include <pybind11/embed.h>
#include <iostream>
 
namespace py = pybind11;
 
int main() {
    py::scoped_interpreter python;
 
    py::module sys = py::module::import("sys");
    py::print(sys.attr("path"));
 
    py::module t = py::module::import("tttt");
    t.attr("add")(1,2);
    auto res = t.attr("client")();
    std::cout << "return : "<< res.cast<int>() <<std::endl;
    return 0;
}
