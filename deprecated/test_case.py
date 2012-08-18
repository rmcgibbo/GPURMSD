import rmsd_gpu_python
import numpy
from numpy import *

numConfs = 10000
numAtoms = 200
numDims = 3

numpy.random.seed(0)

confs = numpy.random.rand(numAtoms,numConfs,numDims).astype(float32)

obj = rmsd_gpu_python.Rmsd(confs)

h_rmsds = zeros(numConfs).astype(float32)
h_rot_mat = zeros((numConfs,9)).astype(float32)
h_subset_flag = ones((numConfs),dtype=int32)

obj.set_rmsd_array(h_rmsds)
obj.set_subset_flag_array(h_subset_flag)
#obj.set_rot_mat_array(h_rot_mat)

obj.center_and_precompute_G()

obj.all_against_one_rmsd(0);
print h_rmsds


obj.all_against_one_lprmsd(0);
print h_rmsds
#obj.all_against_one_lprmsd(x);
    #obj.all_against_one_no_rot(x)
    #obj.all_against_one_and_rot(x)
    #obj.apply_rotation();

#print h_subset_flag
#print h_rot_mat
