"""Setup file for dattri package with Fast JL CUDA compilation."""

import os
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.install import install
from setuptools.command.develop import develop


def should_compile_fast_jl():
    """Check if Fast JL compilation should be attempted."""
    # Only check during install/develop commands, not metadata generation
    if 'egg_info' in sys.argv or 'dist_info' in sys.argv:
        return False
        
    # Check if CUDA is available
    try:
        import torch
        if not torch.cuda.is_available():
            print("CUDA not available, skipping Fast JL compilation")
            return False
    except ImportError:
        print("PyTorch not found, skipping Fast JL compilation")
        return False
    
    # Check if nvcc is available (use same logic as compile_fast_jl.sh)
    cuda_paths = ["/usr/local/cuda", "/usr/local/cuda-*", "/opt/cuda", "/opt/cuda-*"]
    nvcc_found = False
    
    # First try PATH
    try:
        subprocess.run(['nvcc', '--version'], check=True, 
                     capture_output=True, text=True)
        nvcc_found = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Try common CUDA paths
        import glob
        for pattern in cuda_paths:
            for cuda_path in glob.glob(pattern):
                nvcc_path = Path(cuda_path) / "bin" / "nvcc"
                if nvcc_path.exists():
                    nvcc_found = True
                    break
            if nvcc_found:
                break
    
    if not nvcc_found:
        print("CUDA compiler (nvcc) not found, skipping Fast JL compilation")
        return False
    
    return True


def compile_fast_jl():
    """Compile Fast JL CUDA extension."""
    print("\n" + "=" * 50)
    print("COMPILING FAST JL CUDA EXTENSION FOR CURRENT GPU")
    print("=" * 50)
    
    # Change to fast_jl directory  
    setup_dir = Path(__file__).parent.resolve()
    fast_jl_dir = setup_dir / "fast_jl"
    
    if not fast_jl_dir.exists():
        print("Fast JL directory not found, skipping compilation")
        return
    
    # Run compilation script
    compile_script = fast_jl_dir / "compile_fast_jl.sh"
    if not compile_script.exists():
        print("Fast JL compilation script not found, skipping compilation")
        return
    
    try:
        # Change to fast_jl directory and run script
        original_cwd = os.getcwd()
        os.chdir(fast_jl_dir)
        
        # Make script executable
        os.chmod(compile_script, 0o755)
        
        # Run compilation with output shown
        print(f"Running compilation script: {compile_script}")
        subprocess.run([str(compile_script)], check=True)
        print("Fast JL compilation completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"Fast JL compilation failed: {e}")
        print("Continuing installation without Fast JL extension...")
        print("You can manually compile later by running:")
        print(f"  cd {fast_jl_dir}")
        print("  ./compile_fast_jl.sh")
    
    finally:
        os.chdir(original_cwd)
        print("=" * 50)


class CustomInstall(install):
    """Custom install command to compile Fast JL after installation."""
    
    def run(self):
        # Run normal install first
        install.run(self)
        
        # Check if we should compile Fast JL
        if should_compile_fast_jl():
            compile_fast_jl()


class CustomDevelop(develop):
    """Custom develop command to compile Fast JL after development installation."""
    
    def run(self):
        # Run normal develop first
        develop.run(self)
        
        # Check if we should compile Fast JL
        if should_compile_fast_jl():
            compile_fast_jl()


# Main entry point for compilation
def main():
    """Main entry point for standalone compilation."""
    if should_compile_fast_jl():
        compile_fast_jl()
    else:
        print("Fast JL compilation requirements not met")


if __name__ == "__main__":
    # If run directly, just do compilation
    main()

# Normal setup - always run this
setup(
    cmdclass={
        'install': CustomInstall,
        'develop': CustomDevelop,
    }
)
