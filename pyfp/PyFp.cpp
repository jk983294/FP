#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <fp_opt.h>

namespace py = pybind11;


PYBIND11_MODULE(pyfp, m) {
    m.doc() = "pybind11 pyfp";

    py::class_<FP::FpOpt>(m, "PyOpt")
        .def(py::init<>())
        .def("solve", &FP::FpOpt::solve)
        .def("set_covariance_vec", &FP::FpOpt::set_covariance_vec)
        .def("set_expected_return_vec", &FP::FpOpt::set_expected_return_vec)
        .def("set_covariance", &FP::FpOpt::set_covariance)
        .def("set_expected_return", &FP::FpOpt::set_expected_return)
        .def("set_size", &FP::FpOpt::set_size)
        .def("set_riskAversion", &FP::FpOpt::set_riskAversion)
        .def("set_tvAversion", &FP::FpOpt::set_tvAversion)
        .def("set_cashWeight", &FP::FpOpt::set_cashWeight)
        .def("set_insMaxWeight", &FP::FpOpt::set_insMaxWeight)
        .def("set_insMinWeight", &FP::FpOpt::set_insMinWeight)
        .def("set_verbose", &FP::FpOpt::set_verbose)
        .def("set_LongOnly", &FP::FpOpt::set_LongOnly)
        .def("set_oldWeights", &FP::FpOpt::set_oldWeights)
        .def("add_sector_constrain", &FP::FpOpt::add_sector_constrain)
        .def("add_constrain", &FP::FpOpt::add_constrain)
        .def("set_benchWeights", &FP::FpOpt::set_benchWeights)
        .def("tidy_info", &FP::FpOpt::tidy_info)
        .def("get_type", &FP::FpOpt::get_type)
        .def("get_result", &FP::FpOpt::get_result)
        .def("get_status", &FP::FpOpt::get_status)
        .def("get_variance", &FP::FpOpt::get_variance)
        .def("get_expected_return", &FP::FpOpt::get_expected_return)
        .def("get_turnover", &FP::FpOpt::get_turnover);
}