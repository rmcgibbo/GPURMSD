import sys
from distutils.core import setup
from distutils.unixccompiler import UnixCCompiler
from distutils.extension import Extension
from distutils.command.build_ext import build_ext
import subprocess
from subprocess import CalledProcessError
import glob

class NVCC(UnixCCompiler):
    src_extensions = ['.cu']
    executables = {'preprocessor' : None,
                   'compiler'     : ["nvcc"],
                   'compiler_so'  : ["nvcc"],
                   'compiler_cxx' : ["dffd"],
                   'linker_so'    : ["echo", "-shared"],
                   'linker_exe'   : ["gcc"],
                   'archiver'     : ["ar", "-cr"],
                   'ranlib'       : None,
               }
    def __init__(self):
        # Check to ensure that nvcc can be located
        try:
            subprocess.check_output('nvcc --help', shell=True)
        except CalledProcessError:
            print >> sys.stderr, 'Could not find nvcc, the nvidia cuda compiler'
            sys.exit(1)
        UnixCCompiler.__init__(self)
        
kernel = Extension('_gpurmsd',
                   sources=['src/ext/gpurmsd/RMSD.cu'],
                   extra_compile_args=['-arch=sm_20', '--ptxas-options=-v', '-c', '--compiler-options', "'-fPIC'"],
                   include_dirs=['/usr/local/cuda/include'],                   
                   )
swig_wrapper = Extension('msmbuilder.gpurmsd._rmsd_gpu_python',
                         sources=['src/ext/gpurmsd/swig.i'],
                         swig_opts=['-c++'],
                         library_dirs=['/usr/local/cuda/lib64'],
                         libraries=['cudart'])

class custom_build_ext(build_ext):
    def build_extensions(self):

        # we're going to need to switch between compilers, so lets save both
        self.default_compiler = self.compiler
        self.nvcc = NVCC()
        build_ext.build_extensions(self)

    def build_extension(self, *args, **kwargs):

        # switch the compiler based on which thing we're compiling

        if args[0].name == '_gpurmsd':
            # for _GPURMSD, we use the nvcc compiler
            # note that we've DISABLED the linking (by setting the linker to be "echo")
            # in the nvcc compiler
            self.compiler = self.nvcc

        elif args[0].name == 'msmbuilder.gpurmsd._rmsd_gpu_python':
            # for _rmsd_gpu_python, we use regular gcc
            self.compiler = self.default_compiler

            # BUT, we need to also LINK the RMSD.o object file, so lets just
            # glob it onto the link line
            paths_to_other_o_file = glob.glob('build/*/src/ext/gpurmsd/RMSD.o')
            if len(paths_to_other_o_file) != 1:
                raise RuntimeError('RMSD.o not found in temp')
            
            self.compiler.linker_so.append(paths_to_other_o_file[0])

        else:
            self.compiler = self.default_compiler

        build_ext.build_extension(self, *args, **kwargs)

try:
    import msmbuilder
except ImportError:
    print 'MSMBuilder not found!'
    
setup(name='msmbuilder.gpurmsd',
      packages=['msmbuilder.gpurmsd'],
      package_dir={'msmbuilder.gpurmsd': 'src/python/gpurmsd'},
      ext_modules=[kernel, swig_wrapper],
      cmdclass={'build_ext': custom_build_ext})
