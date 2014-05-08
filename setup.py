#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Installation script for pycelle.

"""

from __future__ import print_function

import os
import sys
import warnings

try:
    from setuptools import setup
    has_setuptools = True
except ImportError:
    from distutils.core import setup
    has_setuptools = False

import distutils
from distutils.core import Extension
from distutils.command import install_data
from distutils.command.build_ext import build_ext

class my_install_data(install_data.install_data):
    # A custom install_data command, which will install it's files
    # into the standard directories (normally lib/site-packages).
    def finalize_options(self):
        if self.install_dir is None:
            installobj = self.distribution.get_command_obj('install')
            self.install_dir = installobj.install_lib
        print('Installing data files to {0}'.format(self.install_dir))
        install_data.install_data.finalize_options(self)

def has_cython():
    """Returns True if Cython is found on the system."""
    try:
        import Cython
        return True
    except ImportError:
        return False

def write_version():
    """Creates a file containing version information."""
    target = os.path.join(base, 'pycelle', 'version.py')
    fh = open(target, 'w')
    text = '''"""
Version information for pycelle, created during installation.
"""

__version__ = '%s'

'''
    fh.write(text % release.version)
    fh.close()

def check_opt(name):
    x = eval('has_{0}()'.format(name.lower()))
    msg = "%(name)s not found. %(name)s extensions will not be built."
    if not x:
        warnings.warn(msg % {'name':name})
    return x

def main():
    #write_version()

    # Handle optional extensions.
    opt = {}
    for name, option in [('Cython', 'nocython')]:
        lname = name.lower()

        # Determine if the Python module exists
        opt[lname] = check_opt(name)

        if not opt[lname]:
            continue
        else:
            # Disable installation of extensions, if user requested.
            try:
                idx = sys.argv.index("--{0}".format(option))
            except ValueError:
                pass
            else:
                opt[lname] = False
                del sys.argv[idx]

    cmdclass = {'install_data': my_install_data}

    cython_modules = []
    if opt['cython']:
        import Cython.Distutils
        try:
            import numpy as np
        except ImportError:
            msg = "Please install NumPy first."
            msg = "Alternatively, disable Cython extensions:\n\n"
            msg += "    python setup.py install --nocython\n"
            msg += "    pip install --install-option='--nocython pycelle'\n\n"
            print(msg)
            raise

        cmdclass['build_ext'] = Cython.Distutils.build_ext

        cython_modules = []

        caalgo = Extension(
            "pycelle._caalgo",
            ["pycelle/_caalgo.pyx"],
            include_dirs=[np.get_include()],
            libraries=["m"],
            extra_compile_args=['-std=c99']
        )

        lightcones = Extension(
            "pycelle._lightcones",
            ["pycelle/_lightcones.pyx"],
            include_dirs=[np.get_include()],
            libraries=["m"],
            extra_compile_args=['-std=c99']
        )

        # Active Cython modules
        cython_modules = [
            caalgo,
            lightcones
        ]

    other_modules = []

    ext_modules = cython_modules + \
                  other_modules

    data_files = ()

    install_requires = [
        'numpy >= 1.8',
        'six >= 1.4.0', # 1.4.0 includes six.moves.range.
    ]

    packages = [
        'pycelle',
    ]

    classifiers = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ]

    # Tests
    package_data = dict(zip(packages, [['tests/*.py']]*len(packages)))

    kwds = {
        'name':                 "pycelle",
        'version':              "0.0.1",
        'url':                  "https://github.com/ComSciCtr/pycelle",
        'packages':             packages,
        'package_data':         package_data,
        'provides':             ['pycelle'],
        'install_requires':     install_requires,
        'ext_modules':          ext_modules,
        'cmdclass':             cmdclass,
        'data_files':           data_files,
        'include_package_data': True,

        'author':               "pycelle developers",
        'author_email':         "",
        'description':          "Python package for cellular automata.",
        'long_description':     "",
        'license':              "BSD",
        'classifiers':          classifiers,
    }

    # Automatic dependency resolution is supported only by setuptools.
    if not has_setuptools:
        del kwds['install_requires']

    setup(**kwds)

if __name__ == '__main__':
    if sys.argv[-1] == 'setup.py':
        print("To install, run 'python setup.py install'\n")

    v = sys.version_info[:2]
    if v < (2, 6):
        msg = "pycelle requires Python version >2.6"
        print(msg.format(v))
        sys.exit(-1)

    main()
