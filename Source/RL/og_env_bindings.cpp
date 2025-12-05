#ifndef OG_RL_BUILD
#pragma message("OG_RL_BUILD not defined; RL interface will not be built")
#endif

// Only include pybind11 headers if Python.h and pybind11 are available to avoid editor/intellisense include errors
#if __has_include(<Python.h>) && __has_include(<pybind11/pybind11.h>)
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <iostream>

#include "overgrowth_env.hpp"

namespace py = pybind11;

// PyBind11 module binding
// PyBind11 module binding
PYBIND11_MODULE(og_env, m) {
    m.doc() = "PyBind11 bindings for Overgrowth RL Environment";

    // Debug logging
    std::cout << "[DEBUG] Initializing og_env module bindings" << std::endl;

    // Bind StepResult struct
    py::class_<tuple_t>(m, "StepResult")
        .def_readonly("observation", &tuple_t::observation)
        .def_readonly("reward", &tuple_t::reward)
        .def_readonly("terminated", &tuple_t::terminated)
        .def_readonly("truncated", &tuple_t::truncated)
        .def_readonly("info", &tuple_t::info);

    std::cout << "[DEBUG] Bound StepResult struct" << std::endl;

    // Bind the singleton OvergrowthEnv class - the actual implementation is in overgrowth_env.cpp
    py::class_<OvergrowthEnv, std::shared_ptr<OvergrowthEnv>>(m, "OvergrowthEnv")
        .def(py::init([](py::kwargs kwargs) {
            return OvergrowthEnv::getInstance(kwargs);
        }))
        .def_static("getInstance", &OvergrowthEnv::getInstance)  // Factory function
        .def("reset", &OvergrowthEnv::reset, py::arg("seed") = std::nullopt)
        .def("step", &OvergrowthEnv::step)
        .def("get_action_mask", &OvergrowthEnv::get_action_mask)
        .def("render", &OvergrowthEnv::render, py::arg("mode") = "rgb_array")
        .def("close", &OvergrowthEnv::close)
        .def("get_profiling_data", &OvergrowthEnv::get_profiling_data)
        .def("set_deterministic", &OvergrowthEnv::set_deterministic)
        .def("set_log_level", &OvergrowthEnv::set_log_level)
        .def_readonly("action_space", &OvergrowthEnv::action_space)
        .def_readonly("observation_space", &OvergrowthEnv::observation_space)
        .def_readonly("spec", &OvergrowthEnv::spec)
        .def_readonly("metadata", &OvergrowthEnv::metadata)
        .def_readonly("unwrapped", &OvergrowthEnv::unwrapped);

    std::cout << "[DEBUG] Bound OvergrowthEnv class" << std::endl;

    // Register custom exception
    py::register_exception<RLException>(m, "RLException");

    std::cout << "[DEBUG] Registered RLException" << std::endl;
    std::cout << "[DEBUG] og_env module initialization complete" << std::endl;
}
#else
#pragma message("Warning: Python.h or pybind11 not found in include paths; pybind11 bindings are disabled for editor/intellisense. Configure c_cpp_properties.json includePath to point to your Python include directory and pybind11 headers.")
#endif