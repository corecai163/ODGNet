from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0;7.5;8.0"

setup(name='chamfer',
      version='2.0.0',
      ext_modules=[
          CUDAExtension('chamfer', [
              'chamfer_cuda.cpp',
              'chamfer.cu',
          ]),
      ],
      cmdclass={'build_ext': BuildExtension})
