from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

caalgo = Extension(
    "_caalgo", 
    ["_caalgo.pyx", 'src/evolve.c'],
    include_dirs=[numpy.get_include(), 'src'],
    libraries=["m"],
    extra_compile_args=['-std=c99']
)

setup(
    cmdclass = {'build_ext':build_ext},
    ext_modules = [caalgo,]
)
