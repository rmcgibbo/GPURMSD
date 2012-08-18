#include <iostream>
#include "dataIO.h"
#include "RMSD.cuh"

int main() {

    uint numConfs;
    uint numAtoms;

    float *h_X  = read_fileADP("rand_conf.dat", numConfs, numAtoms); // Read this input file into ADP format!

    Rmsd r(numAtoms, numConfs, h_X); // Read an RMSD object

    float *h_rmsds = (float *) malloc(sizeof(float)*numConfs);

    r.all_against_one(2, h_rmsds); // Calculate RMSD
    r.print_params();

    for(int i=0; i<numConfs; i++) {
        std::cout << h_rmsds[i] << std::endl;
    }

    free(h_rmsds);
    free(h_X); //yes this is quirky - but we want to use arrays for this because ADP is not amenable to vectorization!
}
