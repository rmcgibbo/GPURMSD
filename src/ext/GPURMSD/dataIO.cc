#include "dataIO.h"

#ifndef numDims
#define numDims 3
#endif
// move to IO. 
void print_ADP( float *X, uint numAtoms, uint numConfs) {
    
   for(int i=0;i<numConfs;i++) {
       printf("Conformer %d: \n",i); 
       for(int j=0;j<numAtoms;j++) {
            printf("%.5f %.5f %.5f\n", X[j*numConfs*numDims + 0*numConfs + i],
                                       X[j*numConfs*numDims + 1*numConfs + i],
                                       X[j*numConfs*numDims + 2*numConfs + i]);
       }
   } 

}

void print_ADP( float *X, uint numAtoms, uint numConfs, uint ID) {
    
    uint i=ID;
    printf("Conformer %d: \n",i); 
    for(int j=0;j<numAtoms;j++) {
         printf("%.5f %.5f %.5f\n", X[j*numConfs*numDims + 0*numConfs + i],
                                    X[j*numConfs*numDims + 1*numConfs + i],
                                    X[j*numConfs*numDims + 2*numConfs + i]);
    }
   
}

// we do NOT store by dimensions first
// we store conf by conf
float* read_fileADP(char const * const filename, uint &numConfs, uint &numAtoms) {

    FILE *fp = fopen(filename, "r");

    if( fp == NULL ) {
        printf("Can't read file: %s ! Aborting ...\n", filename);
        exit(1);
    }
    else {
        printf("Loading file: %s ... \n ", filename);
    }

    fscanf(fp, "%d %d", &numAtoms, &numConfs);

    uint confID = 0;
    uint atomID = 0;

    printf("%d atoms and %d conformers.\n", numAtoms, numConfs);

    float *X = (float *) malloc( numAtoms * numDims * numConfs * sizeof(float) );

    // this will segfault if you give it an improper file
    // eg. # of conformers are greater than stated in numConfs
    // but we are only use this for now - eventually we use a binary xtc reader
    /** \brief This uses the ADP format for indexing:
     * X[WHICH_ATOM*TOTAL_CONFS*TOTAL_DIMS + WHICH_DIM*TOTAL_CONFS + WHICH_CONF]
     * This is absolutely critical for coalesced memory reads!
     */
    while(fscanf(fp, "%f %f %f", &(X[atomID*numConfs*numDims+0*numConfs+confID]),
                                 &(X[atomID*numConfs*numDims+1*numConfs+confID]),
                                 &(X[atomID*numConfs*numDims+2*numConfs+confID])) == 3) {
        atomID++;
        if(atomID == numAtoms) {
            atomID = 0;
            confID++;
        }
    }
                
    assert(confID == numConfs);
    printf("Finished loading: %s. \n", filename);
    fclose(fp);

    return X;
            
}

