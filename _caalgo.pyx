# encoding: utf-8
# cython: profile=False

"""
    Cython module for cellular automata algorithms.
"""

from __future__ import division

from collections import defaultdict
from cmpy.infotheory import ConditionalDistribution, Distribution, Event

import cython
cimport cython

from libc.stdlib cimport malloc, free

import numpy as np
cimport numpy as np
from numpy cimport PyArray_DATA

__all__ = ['eca_cyevolve']

# Required before using any NumPy C-API
np.import_array()

cdef inline size_t bindex(np.uint8_t x, np.uint8_t y, np.uint8_t z):
    """
    Given the parents of a cell, return the binary number it represents.
    This number will be used as an index in the (reversed) lookup array.
    Ex:  bindex(1,1,1) -> 7
    """
    return x * 4 + y * 2 + z


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
def eca_cyevolve(object lookup, np.ndarray[np.uint8_t, ndim=2] sta not None, size_t iterations, size_t rowIdx):
    """
    Evolve the ECA from row `rowIdx` by `t` time steps.

    Parameters
    ----------

    """
    # We reverse the lookup array so that 000 is at index 0.
    revlookup = list(reversed(lookup))
    cdef np.ndarray[np.uint8_t, ndim=1] lookup_ = np.array(revlookup, dtype=np.uint8)

    cdef np.uint8_t *lookupPtr = <np.uint8_t *>PyArray_DATA(lookup_)
    cdef np.uint8_t x, y, z
    cdef size_t i, j, head, nexthead

    cdef size_t nRows = sta.shape[0]
    cdef size_t nCols = sta.shape[1]

    # Ensure the initial row consists of 0s and 1s. Without this, bad
    # initial values could create parent indexes larger than 7.
    head = rowIdx % nRows
    for i in range(nCols):
        if sta[head,i] > 0:
            sta[head,i] = 1

    # Evolve it
    for i in range(rowIdx, rowIdx + iterations):
        head = i % nRows
        nexthead = (i+1) % nRows

        # First column
        x = sta[head, nCols-1]
        y = sta[head, 0]
        z = sta[head, 1]
        sta[nexthead, 0] = lookupPtr[bindex(x,y,z)];

        # All other columns
        for j in range(1, nCols-1):
            x = y
            y = z
            z = sta[head, j+1]
            sta[nexthead, j] = lookupPtr[bindex(x,y,z)]

        # Last column
        x = y
        y = z
        z = sta[head, 0]
        sta[nexthead, nCols-1] = lookupPtr[bindex(x,y,z)]


