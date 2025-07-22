# Fast JL CUDA Setup for Dattri

This document describes the Fast JL CUDA integration with Dattri that works across different CUDA and Python versions.

## ✅ What's Included

The `fast_jl/` directory contains all files needed for CUDA compilation:

### Core Files
- `fast_jl.cu` - Main CUDA implementation
- `*.cuh` - CUDA header files (data_loading, error_handling, fast_jl_kernel, random, types)
- `setup.py` - Version-agnostic build configuration
- `setup.cfg` - Build metadata
- `MANIFEST.in` - Package manifest

### Tools & Documentation  
- `compile_fast_jl.sh` - Automated compilation script
- `README.md` - Complete usage guide
- This setup file

## 🚀 Quick Start

### Automatic Compilation (New!)
Fast JL now compiles automatically during package installation:

```bash
# Standard installation - automatically compiles for current CUDA version
pip install dattri

# Development installation - also triggers compilation
pip install -e .
python setup.py develop --user
```

The automatic compilation:
- Detects your CUDA version and Python installation
- Checks PyTorch CUDA compatibility  
- Finds your GPU compute capability
- Compiles for the appropriate architecture
- Tests the installation

### Manual Compilation
If automatic compilation fails or you want to recompile:

#### Option 1: Console Command
```bash
dattri_compile_cuda
```

#### Option 2: Setup Script  
```bash
python setup.py  # Runs compilation check and build
```

#### Option 3: Direct Script
```bash
cd fast_jl/
./compile_fast_jl.sh
```

### 2. Use with Dattri
```python
#!/usr/bin/env python3
from dattri.func.projection import CudaProjector, ProjectionType

# Create CUDA projector
projector = CudaProjector(
    feature_dim=2048,
    proj_dim=512, 
    seed=42,
    proj_type=ProjectionType.rademacher,
    device='cuda',
    max_batch_size=16
)

# Project features
projected = projector.project(features, ensemble_id=0)
```

## 🔧 Enhanced CUDA Version Compatibility

### Intelligent CUDA Detection
The build system provides robust CUDA version handling:

- **Auto-Discovery**: Searches common CUDA installation paths (`/usr/local/cuda*`, `/opt/cuda*`)
- **Version Detection**: Extracts CUDA toolkit version from `nvcc --version`
- **PyTorch Compatibility**: Compares CUDA toolkit vs PyTorch CUDA versions
- **Optimized Architecture Selection**: Builds only for your current GPU architecture for fastest compilation
- **Graceful Fallbacks**: Multiple retry strategies if initial compilation fails

### Version Mismatch Handling
- **Minor Version Differences**: CUDA 12.9 vs PyTorch CUDA 12.6 ✅ (Compatible)
- **Major Version Differences**: CUDA 11.x vs PyTorch CUDA 12.x ⚠️ (Warning + Guidance)
- **Automatic Fixes**: Retries with single architecture fallback if compilation fails

### Supported Hardware
- **Tesla V100**: Compute capability 7.0
- **RTX 20/30 series**: Compute capability 7.5
- **A100**: Compute capability 8.0  
- **RTX 40 series**: Compute capability 8.6/8.9
- **H100**: Compute capability 9.0

### Fallback Strategy
If automatic detection fails, the system builds for multiple architectures (7.0, 7.5, 8.0, 8.6, 8.9) to ensure broad compatibility.

## 🎯 Performance

Benchmarked performance on A100 80GB:
- **CudaProjector**: 0.14 ms average projection time
- **BasicProjector**: 13.7 ms average projection time  
- **Speedup**: 98x faster with Fast JL

## 🔍 Testing

Run the comprehensive test suite:
```bash
python3 test_dattri_cuda_projector.py
```

Tests verify:
- ✅ CudaProjector integration with fast_jl
- ✅ Different projection types (rademacher/normal)
- ✅ Various batch sizes and dimensions
- ✅ Ensemble consistency
- ✅ Performance benchmarks

## 🐛 Advanced Troubleshooting

### CUDA Version Issues
The enhanced build system handles most CUDA version problems automatically:

```bash
# Test CUDA detection
./compile_fast_jl.sh  # Will show detailed version info and compatibility

# Manual CUDA path override if needed
export CUDA_HOME=/usr/local/cuda-12.1
./compile_fast_jl.sh

# Force specific architectures
export TORCH_CUDA_ARCH_LIST="8.0;8.6"
python3 setup.py build_ext --inplace
```

### Version Compatibility Matrix
| CUDA Toolkit | PyTorch CUDA | Status | Action |
|--------------|--------------|---------|--------|
| 12.x | 12.x | ✅ Compatible | Auto-detected |
| 11.x | 11.x | ✅ Compatible | Auto-detected |
| 12.x | 11.x | ⚠️ Warning | Shows guidance |
| 11.x | 12.x | ⚠️ Warning | Shows guidance |

### Build Issues  
1. **CUDA not found**: Auto-searches common paths, shows installation guide
2. **Version mismatch**: Shows specific version guidance and retry options
3. **Architecture issues**: Automatic fallback to current GPU architecture
4. **Compilation failure**: Automatic retry with different settings

### Manual Architecture Override
```bash
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"
python3 setup.py build_ext --inplace
```

## 📁 File Structure
```
dattri/
├── fast_jl/                    # Fast JL CUDA library
│   ├── fast_jl.cu             # Main CUDA implementation
│   ├── *.cuh                  # CUDA headers
│   ├── setup.py               # Version-agnostic build
│   ├── compile_fast_jl.sh     # Compilation script
│   └── README.md              # Detailed documentation
├── dattri/func/projection.py  # Fixed enum handling
├── test_dattri_cuda_projector.py  # Integration tests
└── FAST_JL_SETUP.md           # This file
```

## 🔧 Integration Details

### Dattri CudaProjector Fix
Fixed `projection.py` to properly handle `ProjectionType` enum:
```python
# Convert ProjectionType enum to string value if needed
proj_type_str = self.proj_type.value if hasattr(self.proj_type, 'value') else str(self.proj_type)
function_name = f"project_{proj_type_str}_{effective_batch_size}"
```

### Dynamic Library Path
Test scripts automatically set the PyTorch library path:
```python
torch_lib_path = os.path.dirname(torch.__file__) + '/lib'
os.environ['LD_LIBRARY_PATH'] = f"{torch_lib_path}:{current_ld_path}"
```

## ✨ Usage Examples

### Basic Projection
```python
import torch
from dattri.func.projection import CudaProjector, ProjectionType

projector = CudaProjector(
    feature_dim=1024,
    proj_dim=512,
    seed=42,
    proj_type=ProjectionType.rademacher, 
    device='cuda',
    max_batch_size=8
)

features = torch.randn(32, 1024, device='cuda')
projected = projector.project(features, ensemble_id=0)
# Output: torch.Size([32, 512])
```

### Using make_random_projector
```python
from dattri.func.projection import make_random_projector

projector = make_random_projector(
    param_shape_list=[2048],  # Feature dimensions
    feature_batch_size=64,
    proj_dim=1024,
    proj_max_batch_size=16,
    device="cuda",
    proj_seed=42
)

features = torch.randn(64, 2048, device='cuda')
projected = projector.project(features, ensemble_id=0)
```

---

✅ **Setup Complete!** Fast JL is now ready for production use with Dattri across different CUDA and Python versions.