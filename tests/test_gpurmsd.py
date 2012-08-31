import os, sys
import inspect
import numpy as np
from msmbuilder import Trajectory
from msmbuilder.gpurmsd import GPURMSD
from msmbuilder.metrics import RMSD
import matplotlib.pyplot as pp

import numpy.testing as npt

def fixtures_dir():
    #http://stackoverflow.com/questions/50499/in-python-how-do-i-get-the-path-and-name-of-the-file-that-is-currently-executin
    return os.path.join(os.path.dirname(inspect.getfile(inspect.currentframe())), 'fixtures')

trj_path = os.path.join(fixtures_dir(), 'trj0.lh5')

def test_gpurmsd():
    traj = Trajectory.LoadTrajectoryFile(trj_path)    

    gpurmsd = GPURMSD(traj)
    ptraj = gpurmsd.prepare_trajectory(traj)
    gpu_distances = gpurmsd.one_to_all(ptraj, ptraj, 0)

    rmsd = RMSD()
    ptraj = rmsd.prepare_trajectory(traj)
    distances = rmsd.one_to_all(ptraj, ptraj, 0)
    
    #pp.plot(distances, gpu_distances)
    #pp.show()
    npt.assert_array_almost_equal(distances, gpu_distances)
    


if __name__ == '__main__':
    test_gpurmsd()
