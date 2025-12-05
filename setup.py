"""
Setup script for building og_env Python extension module.
Uses pybind11 to create Python bindings for the Overgrowth RL Environment.
"""
import os
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# Find Eigen from CMake's fetched location
script_dir = os.path.dirname(os.path.abspath(__file__))
eigen_include = os.path.join(script_dir, "build_rl", "_deps", "eigen-src")

# Define the extension module
source_dir = "overgrowth-main/Source/RL"
sources = [
    os.path.join(source_dir, "og_env_bindings.cpp"),
    os.path.join(source_dir, "overgrowth_env.cpp"),
]

include_dirs = [source_dir]
if os.path.exists(os.path.join(eigen_include, "Eigen", "Core")):
    include_dirs.append(eigen_include)
    print(f"Found Eigen at: {eigen_include}")
else:
    # Try alternate location
    alt_eigen = os.path.join(script_dir, "build2", "_deps", "eigen-src")
    if os.path.exists(os.path.join(alt_eigen, "Eigen", "Core")):
        include_dirs.append(alt_eigen)
        print(f"Found Eigen at: {alt_eigen}")
    else:
        print(f"Warning: Eigen not found at {eigen_include}")
        print("Please ensure CMake has been configured to fetch Eigen.")

ext_modules = [
    Pybind11Extension(
        "og_env",
        sources,
        include_dirs=include_dirs,
        define_macros=[
            ("VERSION_INFO", '"1.0.0"'),
            ("OG_RL_BUILD", "1"),  # Enable the build flag
        ],
        cxx_std=17,
    ),
]

setup(
    name="og_env",
    version="1.0.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
