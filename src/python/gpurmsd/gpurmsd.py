import sys, os
import numpy as np
import hashlib
import IPython as ip
from msmbuilder.metrics import AbstractDistanceMetric
from msmbuilder.clustering import concatenate_trajectories

from msmbuilder.gpurmsd.rmsd_gpu_python import Rmsd as _RMSD

class GPURMSD(AbstractDistanceMetric):
    def __init__(self, trajectory):
        xyzlist = trajectory['XYZList']        

        # hash the data that we send to the GPU so we can tell clients
        # whether we can handle their request -- we can only compute
        # RMSDs between these points

        self.traj_length = len(xyzlist)
        self.traj_hash = hashlib.sha1(xyzlist.view(np.uint8)).hexdigest()

        # the gpurmsd code wants the memory layout
        # to be n_atoms x n_frames x 3
        # not n_frames x n_atoms x 3
        self._rmsd = _RMSD(np.array(xyzlist.swapaxes(0,1),
                                    dtype=np.float32,
                                    copy=True))
        
        self._is_centered = False
      

    def one_to_all(self, ptraj1, ptraj2, index1):
        """Compute the distance from the `index1th` frame to the rest of the
        frames

        Note, for GPURMSD, ptraj1 and ptraj2 must be the same, and 
        must be the trajectories that were passed to __init__ to be loaded
        onto the GPU
        """
        assert ptraj1 == ptraj1
        

        results = np.empty(len(ptraj1), dtype=np.float32)
        self._rmsd.set_rmsd_array(results)
        self._rmsd.set_subset_flag_array(np.ones(len(ptraj1), dtype=np.int32))
        self._rmsd.center_and_precompute_G()
        self._rmsd.all_against_one_rmsd(index1)

        return results
    
    def prepare_trajectory(self, trajectory):
        """
        Ensure that this trajectory is suitable for computation on the GPU

        We really don't need this method, since __init__ actually sends
        the data to the GPU, but we do it anyways to fulfill the 
        AbstractDistanceMetric API
        """
        

        if isinstance(trajectory, np.ndarray):
            xyzlist = trajectory
        else:
            xyzlist = trajectory['XYZList']
        
        msg = 'I can only compute the trajectory that was given to __init__'

        if len(trajectory) != self.traj_length:
            raise ValueError(msg)
        hash = hashlib.sha1(xyzlist.view(np.uint8)).hexdigest()
        if hash != self.traj_hash:
            raise ValueError(msg)

        return trajectory
