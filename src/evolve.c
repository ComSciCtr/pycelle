#include <stdio.h>
#include "evolve.h"

/** 
    Given the parents of a cell, return the binary number it represents.
    This number will be used as an index in the (reversed) lookup array.
    Ex:  bindex(1,1,1) -> 7
 **/
inline int bindex(int x, int y, int z) {
    return x * 4 + y * 2 + z;
}

/**
    Given a flat 2D array, evolve the ECA forward.
 **/
void evolve(int* lookup, int* sta, int iterations, int nCols) {

    /** 
        Ensure that first row consists of 0 and 1. Without this, bad
        initial values could create parent indexes larger than 7. 
     **/
    for (int i=0; i < nCols; i++) {
        if (sta[i] > 0) {
            sta[i] = 1;
        }
    }
    
    /** The parents of a cell. **/
    int x,y,z;
    
    for (int i=0; i < iterations; i++) {
    
        int head = i * nCols;
    
        /* First column */
        x = sta[head + nCols - 1];
        y = sta[head];
        z = sta[head + 1];
        sta[head + nCols] = lookup[bindex(x,y,z)];
        
        /* All other columns */
        for (int j=1; j < nCols - 1; j++) {
            x = sta[head + j - 1];
            y = sta[head + j];
            z = sta[head + j + 1];
            sta[head + j + nCols] = lookup[bindex(x,y,z)];
        }
        
        /* Last column */
        x = sta[head + nCols - 2];
        y = sta[head + nCols - 1];
        z = sta[head];
        sta[head + 2 * nCols - 1] = lookup[bindex(x,y,z)];
    }
}


