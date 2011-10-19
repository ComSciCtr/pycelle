from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

caalgo = Extension(
    "_caalgo", 
    ["_caalgo.pyx"],
    include_dirs=[numpy.get_include()],
    libraries=["m"],
    extra_compile_args=['-std=c99']
)

lightcones = Extension(
    "_lightcones", 
    ["_lightcones.pyx"],
    include_dirs=[numpy.get_include()],
    libraries=["m"],
    extra_compile_args=['-std=c99']
)

ext_modules = [caalgo]
majorminor = tuple(numpy.__version__.split('.')[:2])
if majorminor > ('1', '7'):
    ext_modules.append(lightcones)



setup(
    cmdclass = {'build_ext':build_ext},
    ext_modules = ext_modules
)
