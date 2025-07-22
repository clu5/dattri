#!/bin/bash

# Fast JL CUDA Compilation Script
# This script compiles the Fast JL projection library for CUDA GPUs

set -e  # Exit on any error

echo "=== Fast JL CUDA Compilation Script ==="
echo ""

# Check if we're in the right directory
if [ ! -f "setup.py" ]; then
    echo "Error: setup.py not found. Make sure you're in the fast_jl directory."
    exit 1
fi

# Check CUDA availability and version
echo "Checking CUDA availability..."

# Find CUDA installation dynamically
CUDA_PATHS="/usr/local/cuda /usr/local/cuda-* /opt/cuda /opt/cuda-*"
CUDA_HOME=""

for path in $CUDA_PATHS; do
    if [ -d "$path" ] && [ -f "$path/bin/nvcc" ]; then
        CUDA_HOME="$path"
        break
    fi
done

# Add CUDA to PATH if found
if [ -n "$CUDA_HOME" ]; then
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
    echo "Found CUDA installation at: $CUDA_HOME"
else
    # Try system PATH
    if ! command -v nvcc &> /dev/null; then
        echo "Error: nvcc (CUDA compiler) not found."
        echo "Please ensure CUDA toolkit is installed and nvcc is in PATH."
        echo "Common installation paths:"
        echo "  /usr/local/cuda"
        echo "  /usr/local/cuda-XX.X"
        echo "  /opt/cuda"
        exit 1
    fi
    echo "Using system CUDA installation"
fi

# Get CUDA version info
NVCC_VERSION=$(nvcc --version 2>/dev/null | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
echo "CUDA compiler found: $(nvcc --version | head -n4)"
echo "CUDA version: $NVCC_VERSION"
echo ""

# Check PyTorch installation and CUDA compatibility
echo "Checking PyTorch installation..."
PYTORCH_INFO=$(python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'PyTorch CUDA version: {torch.version.cuda}')
    print(f'Number of GPUs: {torch.cuda.device_count()}')
else:
    print('CUDA not available in PyTorch')
" 2>&1)

if [ $? -ne 0 ]; then
    echo "Error: PyTorch not properly installed."
    echo "$PYTORCH_INFO"
    exit 1
fi

echo "$PYTORCH_INFO"

# Extract PyTorch CUDA version for compatibility check
PYTORCH_CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'None')" 2>/dev/null)

if [ "$PYTORCH_CUDA_VERSION" = "None" ]; then
    echo "Warning: PyTorch was not compiled with CUDA support."
    echo "Fast JL requires CUDA-enabled PyTorch."
    exit 1
fi

# Check version compatibility
if [ -n "$NVCC_VERSION" ] && [ -n "$PYTORCH_CUDA_VERSION" ]; then
    NVCC_MAJOR=$(echo $NVCC_VERSION | cut -d. -f1)
    PYTORCH_CUDA_MAJOR=$(echo $PYTORCH_CUDA_VERSION | cut -d. -f1)
    
    echo "CUDA toolkit version: $NVCC_VERSION"
    echo "PyTorch CUDA version: $PYTORCH_CUDA_VERSION"
    
    if [ "$NVCC_MAJOR" != "$PYTORCH_CUDA_MAJOR" ]; then
        echo "Warning: CUDA toolkit version ($NVCC_VERSION) differs from PyTorch CUDA version ($PYTORCH_CUDA_VERSION)"
        echo "This may cause compilation issues. Consider using compatible versions."
        echo "Continuing anyway..."
    else
        echo "✓ CUDA versions are compatible"
    fi
fi
echo ""

# Get GPU information
echo "GPU Information:"
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name()}' if torch.cuda.is_available() else 'No GPU available'); print(f'Compute capability: {torch.cuda.get_device_capability()}' if torch.cuda.is_available() else '')"
echo ""

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info/ fast_jl.cpython-*.so
echo "Previous builds cleaned."
echo ""

# Build the extension
echo "Building Fast JL CUDA extension..."

# Set up environment for compilation
if [ -n "$CUDA_HOME" ]; then
    export CUDA_HOME="$CUDA_HOME"
    export CUDA_ROOT="$CUDA_HOME"
fi

# Set CUDA library paths for compilation
if [ -n "$NVCC_VERSION" ]; then
    CUDA_LIB_PATH=""
    if [ -d "/usr/local/cuda-$NVCC_VERSION/lib64" ]; then
        CUDA_LIB_PATH="/usr/local/cuda-$NVCC_VERSION/lib64"
    elif [ -d "/usr/local/cuda/lib64" ]; then
        CUDA_LIB_PATH="/usr/local/cuda/lib64"
    elif [ -n "$CUDA_HOME" ] && [ -d "$CUDA_HOME/lib64" ]; then
        CUDA_LIB_PATH="$CUDA_HOME/lib64"
    fi
    
    if [ -n "$CUDA_LIB_PATH" ]; then
        export LD_LIBRARY_PATH="$CUDA_LIB_PATH:$LD_LIBRARY_PATH"
        echo "Using CUDA libraries from: $CUDA_LIB_PATH"
    fi
fi

# Log CUDA version for reference
if [ -n "$NVCC_VERSION" ]; then
    CUDA_MAJOR=$(echo $NVCC_VERSION | cut -d. -f1)
    echo "Detected CUDA $NVCC_VERSION - will build for current GPU only"
fi

# Build in-place with error handling
echo "Starting compilation..."
if python3 setup.py build_ext --inplace; then
    echo ""
    echo "✓ Compilation completed successfully!"
else
    echo ""
    echo "✗ Compilation failed. Attempting troubleshooting..."
    
    # Try single architecture fallback if detection-based build failed
    echo "Retrying with single architecture fallback..."
    
    # Get detected GPU architecture if available
    GPU_ARCH=$(python3 -c "
import torch
try:
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        print(f'{major}.{minor}')
    else:
        print('8.0')
except:
    print('8.0')
" 2>/dev/null)
    
    export TORCH_CUDA_ARCH_LIST="$GPU_ARCH"
    echo "Retrying with GPU architecture: $TORCH_CUDA_ARCH_LIST"
    
    if python3 setup.py build_ext --inplace --force; then
        echo "✓ Build succeeded with single architecture!"
    else
        echo "✗ Build still failed. Please check:"
        echo "  1. CUDA toolkit and PyTorch versions are compatible"
        echo "  2. CUDA development tools are installed"
        echo "  3. GPU compute capability is supported (>= 7.0)"
        echo ""
        echo "Current configuration:"
        echo "  CUDA toolkit: $NVCC_VERSION"
        echo "  PyTorch CUDA: $PYTORCH_CUDA_VERSION"
        echo "  GPU architecture: $GPU_ARCH"
        exit 1
    fi
fi

# Verify build artifacts
if ls fast_jl.cpython-*.so 1> /dev/null 2>&1; then
    echo "Built library: $(ls fast_jl.cpython-*.so)"
else
    echo "✗ No compiled library found after build"
    exit 1
fi

# Test the import
echo ""
echo "Testing Fast JL import..."
# Get torch library path dynamically and test import
TORCH_LIB_PATH=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__) + '/lib')")
export LD_LIBRARY_PATH="$TORCH_LIB_PATH:$LD_LIBRARY_PATH"
python3 -c "import fast_jl; print('✓ Fast JL module imported successfully!')" || {
    echo "✗ Failed to import Fast JL module"
    echo "Note: You may need to set LD_LIBRARY_PATH to include PyTorch libraries:"
    echo "export LD_LIBRARY_PATH=\"$TORCH_LIB_PATH:\$LD_LIBRARY_PATH\""
    exit 1
}

echo ""
echo "=== Compilation Complete ==="
echo "The Fast JL library is ready to use!"
echo ""
echo "Configuration Summary:"
echo "  CUDA Version: $NVCC_VERSION"
echo "  PyTorch CUDA Version: $PYTORCH_CUDA_VERSION"
if [ -n "$CUDA_HOME" ]; then
    echo "  CUDA Installation: $CUDA_HOME"
fi
echo ""
echo "To use in your Python code:"
echo "  import sys"
echo "  sys.path.append('$(pwd)')"
if [ "$NVCC_MAJOR" != "$PYTORCH_CUDA_MAJOR" ] 2>/dev/null; then
    echo "  # Note: Set library path due to version mismatch"
    echo "  import os, torch"
    echo "  torch_lib = os.path.dirname(torch.__file__) + '/lib'"
    echo "  os.environ['LD_LIBRARY_PATH'] = f'{torch_lib}:{os.environ.get(\"LD_LIBRARY_PATH\", \"\")}'"
fi
echo "  import fast_jl"
echo ""
echo "Or install it permanently with:"
echo "  python3 setup.py install --user"
echo ""
if [ -n "$NVCC_VERSION" ] && [ -n "$PYTORCH_CUDA_VERSION" ]; then
    if [ "$NVCC_MAJOR" != "$PYTORCH_CUDA_MAJOR" ] 2>/dev/null; then
        echo "⚠️  Version Mismatch Notice:"
        echo "   Your CUDA toolkit ($NVCC_VERSION) and PyTorch CUDA ($PYTORCH_CUDA_VERSION) versions differ."
        echo "   If you encounter runtime issues, consider installing compatible versions:"
        echo "   - For CUDA $NVCC_VERSION: Install PyTorch with CUDA $NVCC_MAJOR.x support"
        echo "   - Or install CUDA $PYTORCH_CUDA_MAJOR.x toolkit to match PyTorch"
        echo ""
    fi
fi