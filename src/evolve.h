#ifndef ECA_EVOLVE_HEADER
#define ECA_EVOLVE_HEADER

#include <stdint.h>

void evolve(uint8_t *lookup, uint8_t *sta, int iterations, size_t rowIdx, size_t nCols);

#endif
