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
    gpurmsd._rmsd.print_params()
    ptraj = gpurmsd.prepare_trajectory(traj)
    gpu_distances = gpurmsd.one_to_all(ptraj, ptraj, 0)

    cpurmsd = RMSD()
    ptraj = cpurmsd.prepare_trajectory(traj)
    cpu_distances = cpurmsd.one_to_all(ptraj, ptraj, 0)
    
    #pp.plot(distances, gpu_distances)
    #pp.show()
    npt.assert_array_almost_equal(cpu_distances, gpu_distances)

def plot_gpu_cmd_correlation():
    traj = Trajectory.LoadTrajectoryFile(trj_path)

    gpurmsd = GPURMSD(traj)
    gpurmsd._rmsd.print_params()
    ptraj = gpurmsd.prepare_trajectory(traj)
    gpu_distances = gpurmsd.one_to_all(ptraj, ptraj, 0)

    rmsd = RMSD()
    ptraj = rmsd.prepare_trajectory(traj)
    cpu_distances = rmsd.one_to_all(ptraj, ptraj, 0)
    
    pp.scatter(gpu_distances, cpu_distances)
    pp.xlabel('gpu rmsd')
    pp.ylabel('cpu rmsd')
    pp.savefig('gpucpu_correlation.png')
    

if __name__ == '__main__':
    plot_gpu_cmd_correlation()
