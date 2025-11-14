#ifdef OG_RL_BUILD
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "overgrowth_env.hpp"

namespace py = pybind11;

PYBIND11_MODULE(og_rl_interface, m) {
    m.doc() = "Overgrowth Reinforcement Learning Environment Interface";

    // Enable thread safety annotations for PyBind11
    py::class_<OvergrowthEnv, std::shared_ptr<OvergrowthEnv>>(m, "OvergrowthEnv")
        .def(py::init([]() {
            return OvergrowthEnv::getInstance();
        }))
        .def("reset", &OvergrowthEnv::reset,
             py::arg("seed") = py::none(),
             "Reset the environment and return initial observation and info")
        .def("step", &OvergrowthEnv::step,
             py::arg("action_id"),
             "Execute action and return next observation, reward, terminated, truncated, info")
        .def("render", &OvergrowthEnv::render,
             py::arg("mode") = "rgb_array",
             "Render the environment and return RGB array")
        .def("close", &OvergrowthEnv::close,
             "Close the environment and cleanup resources")
        .def_readonly("action_space", &OvergrowthEnv::action_space)
        .def_readonly("observation_space", &OvergrowthEnv::observation_space)
        .def_readonly("spec", &OvergrowthEnv::spec)
        .def_readonly("unwrapped", &OvergrowthEnv::unwrapped)
        .def_readonly("metadata", &OvergrowthEnv::metadata)
        .def("get_profiling_data", &OvergrowthEnv::get_profiling_data,
             "Get profiling data with timings, averages, and percentiles")
        .def("set_deterministic", &OvergrowthEnv::set_deterministic,
             py::arg("deterministic"),
             "Set deterministic mode for physics determinism")
        .def("set_log_level", &OvergrowthEnv::set_log_level,
             py::arg("level"),
             "Set logging level (DEBUG/INFO/WARN/ERROR)");

    // Enable GIL management for thread safety
    m.attr("PYBIND11_THREAD_SAFETY_ANNOTATION") = py::cast(true);
}

#endif // OG_RL_BUILD