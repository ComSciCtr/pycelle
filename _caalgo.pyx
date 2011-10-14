# encoding: utf-8
# cython: profile=False

"""
    Cython module for cellular automata algorithms.
"""

from __future__ import division

import cython
cimport cython

import numpy as np
cimport numpy as np
from numpy cimport PyArray_DATA

ITYPE   = np.int32
ctypedef np.int32_t ITYPE_t

cdef extern from "src/evolve.h":
    void evolve(ITYPE_t* lookup, ITYPE_t *sta, ITYPE_t iterations, ITYPE_t nCols)

__all__ = ['eca_cyevolve']

# Required before using any NumPy C-API
np.import_array()

def eca_cyevolve(object lookup, np.ndarray[ITYPE_t, ndim=2] sta not None, int iterations):
    # We reverse the lookup array so that 000 is at index 0.
    revlookup = list(reversed(lookup))
    cdef np.ndarray[ITYPE_t, ndim=1] rule = np.array(revlookup, dtype=ITYPE)

    # Prepare for passing to C function
    cdef ITYPE_t *rulePtr = <ITYPE_t *>PyArray_DATA(rule)
    cdef ITYPE_t *staPtr = <ITYPE_t *>PyArray_DATA(sta)
    cdef ITYPE_t nCols = sta.shape[1]
    
    evolve(rulePtr, staPtr, iterations, nCols)
    
