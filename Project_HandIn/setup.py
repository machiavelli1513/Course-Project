#! /usr/bin/env pyhton3
#from setuptools import setup
#from Cython.Build import cythonize
#setup(ext_modules = cythonize("Project_cython_produce_FREYA_Standalone_data.pyx"))

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Define the extension module
extensions = [
    Extension(
        name="freya_fast",                 # name of the compiled module (import this!)
        sources=["freya_fast.pyx"],        # your .pyx file
        include_dirs=[np.get_include()],  # required for NumPy
        extra_compile_args=["-O3"],       # optimization flag (important for speed)
    )
]

# Run setup
setup(
    name="freya_fast",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",        # use Python 3 syntax
            "boundscheck": False,        # disable bounds checking (faster)
            "wraparound": False,         # disable negative indexing (faster)
            "cdivision": True,           # faster C division
        },
    ),
)