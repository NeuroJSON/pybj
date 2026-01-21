# Copyright (c) 2020-2025 Qianqian Fang <q.fang at neu.edu>. All rights reserved.
# Copyright (c) 2016-2019 Iotic Labs Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/NeuroJSON/pybj/blob/master/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import sys
import os
import warnings
from glob import glob
from platform import python_implementation
from site import getusersitepackages
from numpy import get_include as numpy_get_include

# Allow for environments without setuptools
try:
    from setuptools import setup
except ImportError:
    from ez_setup import use_setuptools

    use_setuptools()
    from setuptools import setup  # pylint: disable=ungrouped-imports

from distutils.core import Extension
from distutils.command.build_ext import build_ext
from distutils.errors import CCompilerError
from distutils.errors import DistutilsPlatformError, DistutilsExecError


def load_description(filename):
    script_dir = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(script_dir, filename), "r") as infile:
        return infile.read()


user_site = getusersitepackages()
user_numpy_include = os.path.join(user_site, "numpy", "_core", "include")


# Loosely based on https://github.com/mongodb/mongo-python-driver/blob/master/setup.py
class BuildExtWarnOnFail(build_ext):
    """Allow for extension building to fail."""

    def run(self):
        try:
            build_ext.run(self)
        except DistutilsPlatformError:
            ex = sys.exc_info()[1]
            sys.stdout.write("%s\n" % str(ex))
            warnings.warn(
                "Extension modules: There was an issue with your platform configuration - see above."
            )

    def build_extension(self, ext):
        try:
            build_ext.build_extension(self, ext)
        except (CCompilerError, DistutilsExecError, DistutilsPlatformError, IOError):
            ex = sys.exc_info()[1]
            sys.stdout.write("%s\n" % str(ex))
            warnings.warn(
                "Extension module %s: The output above this warning shows how the compilation failed."
                % ext.name
            )


BUILD_EXTENSIONS = (
    "PYBJDATA_NO_EXTENSION" not in os.environ and python_implementation() != "PyPy"
)

COMPILE_ARGS = ["-std=c99", "-DUSE__BJDATA"]
# For testing/debug only - some of these are GCC-specific
# COMPILE_ARGS += ['-Wall', '-Wextra', '-Wundef', '-Wshadow', '-Wcast-align', '-Wcast-qual', '-Wstrict-prototypes',
#                  '-pedantic']

setup(
    name="bjdata",
    version="0.6.6",
    description="Binary JData and UBJSON encoder/decoder",
    long_description=load_description("README.md"),
    long_description_content_type="text/markdown",
    author="Qianqian Fang",
    author_email="fangqq@gmail.com",
    maintainer="Qianqian Fang",
    maintainer_email="fangqq@gmail.com",
    url="https://github.com/NeuroJSON/pybj",
    license="Apache License 2.0",
    packages=["bjdata"],
    install_requires=["numpy>=1.8.0"],
    extras_require={"dev": ["Pympler>=0.7 ,<0.8", "coverage>=4.5.3,<4.6"]},
    zip_safe=False,
    ext_modules=(
        [
            Extension(
                "_bjdata",
                sorted(glob("src/*.c")),
                include_dirs=[user_numpy_include, numpy_get_include()],
                extra_compile_args=COMPILE_ARGS,
                # undef_macros=['NDEBUG']
            )
        ]
        if BUILD_EXTENSIONS
        else []
    ),
    cmdclass={"build_ext": BuildExtWarnOnFail},
    keywords=[
        "JSON",
        "JData",
        "UBJSON",
        "BJData",
        "OpenJData",
        "NeuroJSON",
        "JNIfTI",
        "Encoder",
        "Decoder",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Programming Language :: C",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.2",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
