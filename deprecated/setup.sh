#dynamic
#nvcc -arch=sm_20 --ptxas-options=-v --shared --compiler-options '-fPIC -O3' -o _RMSD.so RMSD.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart;
#g++ ./main_test.cpp ./dataIO.cpp ./RMSD.so;

#static
nvcc -arch=sm_20 --ptxas-options=-v -c --compiler-options '-fPIC' -o RMSD.o RMSD.cu -I/usr/local/cuda/include ;

swig -c++ -python -o RMSD_GPU_Wrapped.cpp swig.i;
gcc -fPIC -c RMSD_GPU_Wrapped.cpp -o RMSD_GPU_Wrapped.o -I/usr/include/python2.7/;

#gcc -fPIC -c RMSD_GPU_Wrapped.cpp -o RMSD_GPU_Wrapped.o -I/home/robert/epd-7.3-2-rh5-x86_64/include/python2.7

g++ -shared RMSD_GPU_Wrapped.o RMSD.o -L/usr/local/cuda/lib64 -lcudart -o _rmsd_gpu_python.so;

