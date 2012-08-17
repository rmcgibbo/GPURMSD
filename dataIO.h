#ifndef __DATA_IO_H__
#define __DATA_IO_H__

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

/*typedef struct {

    float x;
    float y;
    float z;

} float3;
*/

typedef unsigned int uint;

void print_ADP(float *X, uint numAtoms, uint numConfs);
void print_ADP(float *X, uint numAtoms, uint numConfs, uint ID);


float* read_fileADP(char const * const filename, uint &numConfs, uint &numAtoms); 

#endif
