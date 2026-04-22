from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='priority_streams',
    ext_modules=[
        CUDAExtension('priority_streams', [
            'stream_priority.cpp',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
