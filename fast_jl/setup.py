#!/usr/bin/env python

from os import environ
import torch

# Automatically detect CUDA architecture for current GPU only
def setup_cuda_architectures():
    # Check if already set by environment
    if 'TORCH_CUDA_ARCH_LIST' in environ:
        print(f"Using pre-set CUDA architectures: {environ['TORCH_CUDA_ARCH_LIST']}")
        return

    try:
        if torch.cuda.is_available():
            # Get CUDA capability of current GPU
            capability = torch.cuda.get_device_capability()
            major, minor = capability
            detected_arch = f"{major}.{minor}"
            print(f"Detected CUDA compute capability: {detected_arch}")
            print(f"Building for current GPU architecture only: {detected_arch}")
            
            # Use only the detected architecture for faster compilation
            environ['TORCH_CUDA_ARCH_LIST'] = detected_arch
        else:
            # Fallback to single common architecture if CUDA not available during setup
            print("CUDA not available during setup, using single fallback architecture")
            environ['TORCH_CUDA_ARCH_LIST'] = "8.0"  # Common modern architecture
    except Exception as e:
        # If detection fails, use single fallback architecture
        print(f"CUDA detection failed: {e}")
        print("Using single fallback architecture")
        environ['TORCH_CUDA_ARCH_LIST'] = "8.0"

# Set up CUDA architectures
setup_cuda_architectures()

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

long_description = open('README.rst').read()

setup(
    name='fast_jl',
    version="0.1.3",
    description="Fast JL: Compute JL projection fast on a GPU",
    author="MadryLab",
    author_email='trak@mit.edu',
    install_requires=["torch>=2.0.0"],
    long_description=long_description,
    ext_modules=[
        CUDAExtension('fast_jl', [
            'fast_jl.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    setup_requires=["torch>=2.0.0"])
