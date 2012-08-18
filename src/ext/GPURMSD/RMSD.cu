#include "kernel_rmsd.cu"
#include <assert.h>
#include <iostream>
#include "RMSD.hh"

using std::cout;

void Rmsd::print_params() {
    
    cout << "numAtoms: " << numAtoms_ << "\n";
    cout << "numConfs: " << numConfs_ << "\n";
    cout << "numBlocks: " << numBlocks_ << "\n";
    cout << "mem size: " << size_ << "\n";
    cout << "GPU mem capacity: " << capacity_ << "\n";
    cout << "GPU compute capability : " << compute_capability_ << "\n";

}

void Rmsd::set_rmsd_array(int numConfs, float *h_rmsds) {

    assert(numConfs == numConfs_);
    h_rmsds_ = h_rmsds;

    cudaError_t t;
    t=cudaMalloc((void **) &d_rmsds_, numConfs_*sizeof(float));
    assert(t == 0);
}

/*
void Rmsd::set_rot_mat_array(int numConfs, int RotMatSize, float *h_rot_mat) {

    assert(numConfs == numConfs_);
    assert(RotMatSize == 9);
    h_rot_mat_ = h_rot_mat;

    cudaError_t t;
    t=cudaMalloc((void **) &d_rot_mat_, numConfs_*sizeof(float)*9);
    assert(t == 0);

}
*/

/*
void Rmsd::set_only_device_rot_mat_array() {

    cudaError_t t;
    t=cudaMalloc((void **) &d_rot_mat_, numConfs_*sizeof(float)*9);
    assert(t == 0);

}
*/

void Rmsd::set_subset_flag_array(int numConfs, int *h_subset_flag) {

    assert(numConfs == numConfs_);
    h_subset_flag_ = h_subset_flag;

    cudaError_t t;
    t=cudaMalloc((void **) &d_subset_flag_, numConfs_*sizeof(float));
    assert(t == 0);

    cudaMemcpy(d_subset_flag_, h_subset_flag_, numConfs_*sizeof(float), cudaMemcpyHostToDevice);

}

void Rmsd::all_against_one_rmsd(int test_conf) {

    k_all_against_one_rmsd<<<numBlocks_,threadsperblock_>>>(numAtoms_, numConfs_, test_conf, d_X_, d_rmsds_, d_G_);

    cudaMemcpy(h_rmsds_, d_rmsds_, numConfs_ * sizeof(float), cudaMemcpyDeviceToHost);

}

void Rmsd::all_against_one_lprmsd(int test_conf) {

    assert(d_subset_flag_ != NULL);
    assert(h_subset_flag_ != NULL);

    k_all_against_one_lprmsd<<<numBlocks_,threadsperblock_>>>(numAtoms_, numConfs_, test_conf, d_X_, d_rmsds_, d_G_, d_subset_flag_);

    cudaMemcpy(h_rmsds_, d_rmsds_, numConfs_ * sizeof(float), cudaMemcpyDeviceToHost);

}

/*
void Rmsd::apply_rotation() {

    assert(d_rot_mat_ != NULL);

    k_rotate_all<<<numBlocks_, threadsperblock_>>>( numConfs_, numAtoms_, d_X_, d_rot_mat_);

    //cudaMemcpy(h_X_, d_X_, size_, cudaMemcpyDeviceToHost);

}
*/

Rmsd::Rmsd(int numAtoms, int numConfs, int numDimens, float* h_X) :
    numAtoms_(numAtoms), 
    numConfs_(numConfs), 
    numBlocks_(ceil( (float) numConfs_ / (float) threadsperblock_ )), 
    size_(numAtoms*numConfs*numDims*sizeof(float)),
    h_X_(h_X),
    d_subset_flag_(NULL),
    h_subset_flag_(NULL),
    compute_capability_(0),
    capacity_(0)
{
    
    assert(numDimens == 3);

    set_gpu_parameters();

    assert(h_X_ != NULL); 
    //allocate on the GPU
    cudaError_t t;
    t=cudaMalloc((void **) &d_X_, size_);
    assert(t == 0);

    t=cudaMalloc((void **) &d_G_, numConfs_*sizeof(float));
    assert(t == 0);
    

    t=cudaMemcpy(d_X_,h_X_,size_,cudaMemcpyHostToDevice);
    assert(t == 0);
    assert(d_X_ != NULL);

    //check memory requirements and compute capability
    assert(size_ <= capacity_ );
    assert(compute_capability_>= 2.0);

}

void Rmsd::center_and_precompute_G() {

    assert(h_rmsds_ != NULL);
    assert(d_rmsds_ != NULL);

    center_conformers();
    precompute_G();

}

void Rmsd::set_gpu_parameters() {

    // this also does rotation
 
     cudaDeviceProp prop;
     cudaGetDeviceProperties( &prop, 0 );
     capacity_ = prop.totalGlobalMem;
     compute_capability_ = prop.major + 0.1*(float)prop.minor;

}

void Rmsd::precompute_G() {

    // kernel automatically takes care of conditions when
    // d_subset_flag_ is false
    k_precompute_G<<<numBlocks_, threadsperblock_>>>(numConfs_,numAtoms_,d_X_, d_G_,d_subset_flag_);

    have_precomputed_G_ = true;

}
 
void Rmsd::center_conformers() {

    k_center_conformers<<<numBlocks_, threadsperblock_>>>(numConfs_,numAtoms_,d_X_,d_subset_flag_);

    have_centered_ = true;

}

Rmsd::~Rmsd() {
    cudaFree(d_X_);
    if( d_rmsds_ != NULL) {
    cudaFree(d_rmsds_);
    }  
}
