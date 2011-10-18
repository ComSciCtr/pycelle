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

ITYPE   = np.int32
ctypedef np.int32_t ITYPE_t

__all__ = ['lightcone_counts']

# Required before using any NumPy C-API
np.import_array()

def lightcone_counts(np.ndarray[ITYPE_t, ndim=2] sta, ITYPE_t hLength, ITYPE_t fLength):
    """Counts light cones from an array.

    Parameters
    ----------
    sta : NumPy array, ndim=2
        The spacetime array to be analyzed.
    hLength : int
        The history length of the light cone. A length of zero includes a single
        cell. A length of one additionally includes the cell's three parents.
    fLength : int
        The future length of the light cone. A length of zero includes a single
        cell. A length of one additionally includes the cell's three children.

    Returns
    -------
    fgh_counts
        The unnormalized conditional distribution representing future light
        cone counts given history light cones.  The history light cone counts
        are included in the marginal distribution.

    lca : NumPy array, ndim=3
        An NA-masked spacetime array where each cell is given the value of
        the light cone at the cell.  The array is three-dimensional, with
        the first dimension representing the different light cones.

        lca[0]  :  cells are labeled by the history light cones
        lca[1]  :  cells are labeled by the future light cones
        lca[2]  :  cells are labeled by the history-future light cones

    """
    cdef np.npy_intp nRows = sta.shape[0]
    cdef np.npy_intp nCols = sta.shape[1]

    if nRows - (hLength + fLength) <= 0:
        msg = 'hLength and fLength are too large for the spacetime array.'
        raise Exception(msg)

    cdef np.ndarray[ITYPE_t, ndim=3] lca
    lca = -1 * np.zeros((3, nRows, nCols), dtype=ITYPE, maskna=True)
    lca[:,:hLength,:] = np.NA
    lca[:,-fLength:,:] = np.NA

    # Each light cone has \sum_{t=0}^{T} ( 1 + 2t ) = (1 + T)^2
    # We store the neighborhoods as a char array.
    # For the history-future light cone, we need to store
    #   (1 + hLength)^2 + (1 + fLength^2) - 1
    # values, but saving space for the null character we add one more.
    cdef int total = (1 + hLength)**2 + (1 + fLength)**2
    cdef char *lcPtr = <char *>malloc(total * sizeof(char))
    if not lcPtr:
        raise MemoryError()
    else:
        lcPtr[total] = '\0'

    cdef np.npy_intp i,j, lcidx, h, f, k, ni, nj, hlc_id, flc_id
    cdef bytes lc

    cdef np.npy_intp hlcSize = (1+hLength)**2
    cdef np.npy_intp flcSize = (1+fLength)**2

    hlcones = {}
    flcones = {}
    hflcones = {}

    fgh_counts = defaultdict(lambda : defaultdict(int))
    h_counts = defaultdict(int)
    try:
        for i in range(hLength, nRows-fLength):
            for j in range(nCols):

                # Now we grab the cones at each point (i,j)

                ### Grab the history light cone.
                lcPtr[hlcSize] = '\0'
                lcPtr[flcSize] = 'A' # any non-null character works
                lcidx = 0
                for h in range(hLength, -1, -1):
                    ni = i - h
                    nj = j - h
                    for k in range(1 + 2*h):
                        # 48 is 0 and 49 is 1 in ASCII
                        lcPtr[lcidx] = <char>(48 + (sta[ni, (nj + k) % nCols]))
                        lcidx += 1
                # Convert the C char* array to a Python string
                lc = lcPtr
                # Count the unique history light cones (this is pure Python)
                if lc in hlcones:
                    hlc_id = hlcones[lc]
                else:
                    hlc_id = len(hlcones)
                    hlcones[lc] = hlc_id
                lca[0,i,j] = hlc_id

                ### Grab the future light cone

                lcPtr[hlcSize] = 'A' # any non-null character works
                lcPtr[flcSize] = '\0'
                lcidx = 0
                for f in range(0, fLength+1):
                    ni = i + f
                    nj = j - f
                    for k in range(1 + 2*f):
                        # 48 is 0 and 49 is 1 in ASCII
                        lcPtr[lcidx] = <char>(48 + (sta[ni, (nj + k) % nCols]))
                        lcidx += 1
                # Convert the C char* array to a Python string
                lc = lcPtr
                # Count the unique history light cones (this is pure Python)
                if lc in flcones:
                    flc_id = flcones[lc]
                else:
                    flc_id = len(flcones)
                    flcones[lc] = flc_id
                lca[1,i,j] = flc_id


                # Increment the counts...
                h_counts[hlc_id] += 1
                fgh_counts[hlc_id][flc_id] += 1


                # Set the history-future light cones

                hflc = (hlc_id, flc_id)
                if hflc in hflcones:
                    hflc_id = hflcones[hflc]
                else:
                    hflc_id = len(hflcones)
                    hflcones[hflc] = hflc_id
                lca[2,i,j] = hflc_id

    finally:
        free(lcPtr)

    keys = fgh_counts.keys()
    values = [Distribution(x, event_type=Event) for x in fgh_counts.values()]
    marg = Distribution(h_counts, event_type=Event)
    fgh_counts = dict(zip(keys,values))
    fgh_counts = ConditionalDistribution(fgh_counts, marginal=marg, event_type=Event)

    return fgh_counts, lca
