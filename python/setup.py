#!/usr/bin/env python3
"""
Fallback setup.py for legacy builds.
"""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os

class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])

class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()

    def build_cmake(self, ext):
        self.announce("Building with CMake", level=3)

        # Set up build directory
        build_temp = os.path.abspath(self.build_temp)
        os.makedirs(build_temp, exist_ok=True)

        # Configure with cmake
        self.spawn([
            'cmake',
            '-S', os.path.dirname(os.path.abspath(__file__)),
            '-B', build_temp,
            '-DOG_RL=ON',
            '-DPYTHON_EXECUTABLE=' + sys.executable
        ])

        # Build
        self.spawn(['cmake', '--build', build_temp])

        # Copy built extension to package directory
        import shutil
        import glob

        ext_dir = os.path.abspath(self.build_lib)
        for pattern in ['*.so', '*.pyd', '*.dll']:
            for f in glob.glob(os.path.join(build_temp, '**', pattern), recursive=True):
                shutil.copy2(f, ext_dir)

setup(
    name="og-rl-env",
    version="0.1.0",
    description="Overgrowth Reinforcement Learning Environment",
    ext_modules=[CMakeExtension("og_rl_interface")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)