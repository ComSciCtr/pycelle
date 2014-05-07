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

__all__ = ['eca_cyevolve']

# Required before using any NumPy C-API
np.import_array()

ITYPE = np.uint64
ctypedef np.uint64_t ITYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
@cython.cdivision(True)
def eca_cyevolve(object eca, object iterations):
    """
    Evolve the ECA from row `rowIdx` by `t` time steps.

    Parameters
    ----------

    """
    cdef np.ndarray[ITYPE_t, ndim=1] bases = eca._bases
    cdef np.ndarray[ITYPE_t, ndim=2] sta = eca._sta
    cdef size_t rowIdx = eca.t
    cdef size_t rowIdx_final = eca.t + iterations

    cdef size_t nRows = sta.shape[0]
    cdef size_t nCols = sta.shape[1]

    # We reverse the lookup array so that 000 is at index 0 and 111 at index 7.
    revlookup = list(reversed(eca.lookup))
    cdef np.ndarray[ITYPE_t, ndim=1] lookup = np.array(revlookup, dtype=ITYPE)

    cdef ITYPE_t *lookupPtr = <ITYPE_t *>PyArray_DATA(lookup)
    cdef ITYPE_t *basePtr = <ITYPE_t *>PyArray_DATA(bases)
    cdef size_t i, j, k, idx, head, nexthead
    cdef int l

    cdef size_t radius = eca.radius
    cdef size_t maxValue = eca.base - 1
    cdef size_t nParents = 2 * radius + 1

    # Ensure the initial row consists of 0s and 1s. Without this, bad
    # initial values could create parent indexes larger than 7.
    head = rowIdx % nRows
    for i in range(nCols):
        if sta[head,i] > maxValue:
            # We clip down to the maxValue.
            sta[head,i] = maxValue

    # Evolve it

    for i in range(rowIdx, rowIdx_final):
        head = i % nRows
        nexthead = (i+1) % nRows

        # Special case: The first r cells have parents that wrap.
        for j in range(radius):
            idx = 0
            for k in range(nParents):
                l = j - radius + k
                if l < 0:
                    l += nCols
                idx += basePtr[k] * sta[head, l]
            sta[nexthead, j] = lookupPtr[idx]

        # Typical case: All parents have regular indexes.
        for j in range(radius, nCols - radius):
            idx = 0
            for k in range(nParents):
                idx += basePtr[k] * sta[head, j - radius + k]
            sta[nexthead, j] = lookupPtr[idx]

        # Special case: The last r cells have parents that wrap.
        for j in range(nCols - radius, nCols):
            idx = 0
            for k in range(nParents):
                l = j - radius + k
                if l >= nCols:
                    l -= nCols
                idx += basePtr[k] * sta[head, l]
            sta[nexthead, j] = lookupPtr[idx]

