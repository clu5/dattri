#!/usr/bin/env python3

"""
Test script to verify CUDA version detection and compatibility checking
"""

import subprocess
import sys
import os

def test_cuda_detection():
    """Test CUDA version detection from the compilation script"""
    print("=== Testing CUDA Version Detection ===\n")
    
    try:
        # Run just the CUDA detection part of the script
        result = subprocess.run([
            'bash', '-c', '''
            source compile_fast_jl.sh
            echo "CUDA_HOME: $CUDA_HOME"
            echo "NVCC_VERSION: $NVCC_VERSION" 
            echo "PYTORCH_CUDA_VERSION: $PYTORCH_CUDA_VERSION"
            echo "NVCC_MAJOR: $NVCC_MAJOR"
            echo "PYTORCH_CUDA_MAJOR: $PYTORCH_CUDA_MAJOR"
            '''
        ], capture_output=True, text=True, cwd='.')
        
        print("Script output:")
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
            
    except Exception as e:
        print(f"Error running detection test: {e}")

def test_python_detection():
    """Test Python-based version detection"""
    print("\n=== Testing Python Version Detection ===\n")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"PyTorch CUDA version: {torch.version.cuda}")
            print(f"GPU compute capability: {torch.cuda.get_device_capability()}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            print(f"Current GPU: {torch.cuda.get_device_name()}")
            
            # Test the setup function
            print("\n--- Testing setup_cuda_architectures ---")
            sys.path.insert(0, '.')
            
            # Clear any existing environment variable
            if 'TORCH_CUDA_ARCH_LIST' in os.environ:
                del os.environ['TORCH_CUDA_ARCH_LIST']
                
            # Import and run the setup function
            from setup import setup_cuda_architectures
            setup_cuda_architectures()
            
            if 'TORCH_CUDA_ARCH_LIST' in os.environ:
                print(f"✓ CUDA architectures set: {os.environ['TORCH_CUDA_ARCH_LIST']}")
            else:
                print("✗ CUDA architectures not set")
                
    except ImportError as e:
        print(f"PyTorch not available: {e}")
    except Exception as e:
        print(f"Error in Python detection: {e}")

def test_compatibility_check():
    """Test version compatibility logic"""
    print("\n=== Testing Compatibility Logic ===\n")
    
    test_cases = [
        ("12.9", "12.6", True),   # Compatible major versions
        ("12.1", "12.6", True),   # Compatible major versions  
        ("11.8", "12.6", False),  # Incompatible major versions
        ("12.9", "11.8", False),  # Incompatible major versions
        ("11.7", "11.8", True),   # Compatible CUDA 11.x
    ]
    
    for nvcc_ver, pytorch_ver, expected in test_cases:
        nvcc_major = nvcc_ver.split('.')[0]
        pytorch_major = pytorch_ver.split('.')[0]
        is_compatible = (nvcc_major == pytorch_major)
        
        status = "✓" if is_compatible == expected else "✗"
        print(f"{status} CUDA {nvcc_ver} vs PyTorch CUDA {pytorch_ver}: {'Compatible' if is_compatible else 'Incompatible'}")

if __name__ == "__main__":
    print("Fast JL CUDA Version Detection Test")
    print("=" * 50)
    
    test_python_detection()
    test_compatibility_check()
    test_cuda_detection()
    
    print("\n" + "=" * 50)
    print("Version detection test complete!")