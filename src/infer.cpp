
#include <pybind11/embed.h>
#include <iostream>
#include <cstdio>
#include <ctime> 
#include <pybind11/numpy.h>
#include "paddle_inference_api.h"
namespace py = pybind11;
using paddle_infer::PrecisionType;

int main() {
    py::scoped_interpreter python;
 
    py::module sys = py::module::import("sys");
    py::print(sys.attr("path"));
 
    py::module t = py::module::import("python_infer");
    t.attr("add")(1,2);

    std::clock_t start, end;
    double duration;

    start = std::clock(); // get current time
    //auto res = t.attr("_preprocess")();
    //py::dict dict_res = res.cast<py::dict>();

    paddle_infer::Config config;
    config.SetModel("serving_server/__model__", "serving_server/__params__");
    //config.SwitchIrOptim(true);
    config.EnableMemoryOptim();
    auto predictor = paddle_infer::CreatePredictor(config);
    auto input_names = predictor->GetInputNames();
    std::cout << "input0: " << input_names[0] << std::endl;
    std::cout << "input1: " << input_names[1] << std::endl;
    std::cout << "input2: " << input_names[2] << std::endl;
    //set input0 imshape

    for (int i = 0; i < 10; i++) {
        auto res = t.attr("_preprocess")();
        py::dict dict_res = res.cast<py::dict>();
        auto input0 = dict_res["im_shape"].cast<py::array_t<float,0>>();
        std::vector<int> input_shape0 = {1,2};
        auto input_t0 = predictor->GetInputHandle(input_names[0]);  
        py::buffer_info buf0 = input0.request();
        input_t0->Reshape(input_shape0);
        input_t0->CopyFromCpu(static_cast<float*>(buf0.ptr));

        //set input1 image
        auto input1 = dict_res["image"].cast<py::array_t<float,0>>();
        auto r = input1.mutable_unchecked<4>();
        std::vector<int> input_shape1 = {r.shape(0), r.shape(1), r.shape(2), r.shape(3)};
        auto input_t1 = predictor->GetInputHandle(input_names[1]);
        py::buffer_info buf1 = input1.request();
        input_t1->Reshape(input_shape1);
        input_t1->CopyFromCpu(static_cast<float*>(buf1.ptr));

        //set input2 scale_factor
        auto input2 = dict_res["scale_factor"].cast<py::array_t<float,0>>();
        std::vector<int> input_shape2 = {1,2};
        auto input_t2 = predictor->GetInputHandle(input_names[2]);
        py::buffer_info buf2 = input2.request();
        input_t2->Reshape(input_shape2);
        input_t2->CopyFromCpu(static_cast<float*>(buf2.ptr));

        //run
        predictor->Run();
 
        auto output_names = predictor->GetOutputNames();
        auto output_t = predictor->GetOutputHandle(output_names[0]);
        std::vector<int> output_shape = output_t->shape();
        int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                  std::multiplies<int>());
        std::vector<float> out_data;
        out_data.resize(out_num);
        output_t->CopyToCpu(out_data.data());
        std::cout << "output name0: " << output_names[0] <<std::endl;    
    }    
    end = std::clock();
    std::cout << (end - start)/(double)CLOCKS_PER_SEC << " s" << std::endl;
}
/*
py::array_t<double> add_arrays(py::array_t<double> input1, py::array_t<double> input2) {
    py::buffer_info buf1 = input1.request(), buf2 = input2.request();

    if (buf1.ndim != 1 || buf2.ndim != 1)
        throw std::runtime_error("Number of dimensions must be one");

    if (buf1.size != buf2.size)
        throw std::runtime_error("Input shapes must match");

    auto result = py::array_t<double>(buf1.size);

    py::buffer_info buf3 = result.request();

    double *ptr1 = static_cast<double *>(buf1.ptr);
    double *ptr2 = static_cast<double *>(buf2.ptr);
    double *ptr3 = static_cast<double *>(buf3.ptr);

    for (size_t idx = 0; idx < buf1.shape[0]; idx++)
        ptr3[idx] = ptr1[idx] + ptr2[idx];

    return result;
}
*/
