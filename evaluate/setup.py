"""
Setup box_overlaps calculator
"""
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

# Define the extension module for bounding box calculations
package = Extension(
    'bbox',
    ['box_overlaps.pyx'],
    include_dirs=[numpy.get_include()],
    define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]  # This line suppresses the deprecated NumPy API warnings
)

# Setup the package using cythonize to compile the Cython code
setup(
    ext_modules=cythonize(
        [package],
        compiler_directives={'language_level': "3"}  # This sets the language level to Python 3
    )
)

