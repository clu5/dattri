# Fast JL CUDA Extension

A high-performance CUDA implementation of the Johnson-Lindenstrauss projection for fast random projections on GPU.

## Overview

This module provides fast random projections using CUDA for accelerated computation on NVIDIA GPUs. It supports:

- **Rademacher projections**: Random ±1 projections
- **Normal projections**: Gaussian random projections
- **GPU acceleration**: Optimized CUDA kernels for high throughput
- **Mixed precision**: Support for FP16 and FP32 computations

## Requirements

- NVIDIA GPU with compute capability 7.0+ (V100, RTX 20/30/40 series, A100, etc.)
- CUDA Toolkit 11.0+
- PyTorch 2.0+ with CUDA support
- Python 3.7+

## Quick Start

### 1. Compile the Extension

```bash
cd fast_jl/
./compile_fast_jl.sh
```

The script will:
- Check CUDA and PyTorch installation
- Automatically detect your GPU architecture
- Compile the CUDA extension
- Test the installation

### 2. Use in Python

```python
#!/usr/bin/env python3
import sys
import os
sys.path.append('path/to/dattri/fast_jl')

# Set PyTorch library path if needed
import torch
torch_lib_path = os.path.dirname(torch.__file__) + '/lib'
os.environ['LD_LIBRARY_PATH'] = f"{torch_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"

import fast_jl

# Create input tensor on GPU
input_tensor = torch.randn(100, 512, device='cuda', dtype=torch.float16)

# Perform random projection
projection_dim = 512  # Must be multiple of 512
seed = 42
result = fast_jl.project_rademacher_8(input_tensor, projection_dim, seed, 1)

print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {result.shape}")
```

## Installation Options

### Option 1: In-place Build (Recommended for development)
```bash
cd fast_jl/
python3 setup.py build_ext --inplace
```

### Option 2: User Installation
```bash
cd fast_jl/
python3 setup.py install --user
```

### Option 3: Development Installation
```bash
cd fast_jl/
python3 -m pip install -e .
```

## Architecture Support

The compilation script automatically detects your GPU's compute capability and builds for the appropriate architecture. Supported GPUs:

- **Tesla V100**: 7.0
- **RTX 20/30 series**: 7.5
- **A100**: 8.0
- **RTX 40 series**: 8.6/8.9
- **H100**: 9.0

**Optimized Compilation**: The build system detects your current GPU and compiles only for that specific architecture, resulting in faster compilation times.

**Smart Fallback**: If detection fails, uses a single common architecture (8.0) as fallback.

**Manual Override**: You can set architectures manually by setting the environment variable:
```bash
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"
python3 setup.py build_ext --inplace
```

## Available Functions

The module provides several projection functions:

- `project_rademacher_8(input, N, seed, num_feature_tiles)`: Rademacher projection with 8 batches
- `project_normal_8(input, N, seed, num_feature_tiles)`: Normal projection with 8 batches

Parameters:
- `input`: Input tensor (B, F) on CUDA device
- `N`: Output projection dimension
- `seed`: Random seed for reproducibility
- `num_feature_tiles`: Number of feature tiles for memory optimization

## Troubleshooting

### CUDA Compilation Issues

1. **Library not found errors**:
   ```bash
   export LD_LIBRARY_PATH="/path/to/torch/lib:$LD_LIBRARY_PATH"
   ```

2. **Architecture mismatch**:
   - Check your GPU: `nvidia-smi`
   - Verify compute capability matches compilation target

3. **Memory issues**:
   - Reduce `num_feature_tiles` parameter
   - Use smaller batch sizes

### Import Issues

1. **Module not found**:
   ```python
   import sys
   sys.path.append('/path/to/dattri/fast_jl')
   import fast_jl
   ```

2. **Symbol not found**:
   - Ensure PyTorch and CUDA versions are compatible
   - Reinstall PyTorch with matching CUDA version

## Performance Tips

1. **Use FP16 for better performance** on modern GPUs
2. **Optimize batch sizes** for your GPU memory
3. **Adjust feature tiles** based on input dimensions
4. **Keep tensors on GPU** to avoid CPU-GPU transfers

## Integration with Dattri

This Fast JL implementation is designed to work with the Dattri attribution library for accelerated influence computations:

```python
from dattri.func.projection import FastJLProjector

projector = FastJLProjector(
    output_dim=512,
    device='cuda',
    dtype=torch.float16
)
```