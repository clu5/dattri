#!/usr/bin/env python3

"""
Fast JL Integration Test - Comprehensive verification of compiled Fast JL library
This test verifies that Dattri's CudaProjector works correctly with the optimized 
single-architecture Fast JL CUDA library we compiled.
"""

import sys
import os
import torch
import numpy as np

# Add fast_jl to path (go up to dattri root, then into fast_jl)
dattri_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
fast_jl_path = os.path.join(dattri_root, 'fast_jl')
sys.path.insert(0, fast_jl_path)

# Import dattri projector classes
from dattri.func.projection import CudaProjector, BasicProjector, ProjectionType, make_random_projector

def test_dattri_cuda_projector():
    """Test Dattri's CudaProjector with our compiled fast_jl"""
    print("=== Testing Dattri CudaProjector ===\n")
    
    if not torch.cuda.is_available():
        print("✗ CUDA not available, skipping test")
        return False
    
    try:
        # Test parameters
        feature_dim = 2048  # Must be compatible with fast_jl requirements
        proj_dim = 1024     # Must be multiple of 512
        seed = 42
        max_batch_size = 8
        
        print(f"Initializing CudaProjector:")
        print(f"  Feature dimension: {feature_dim}")
        print(f"  Projection dimension: {proj_dim}")
        print(f"  Max batch size: {max_batch_size}")
        print()
        
        # Create CudaProjector
        cuda_projector = CudaProjector(
            feature_dim=feature_dim,
            proj_dim=proj_dim,
            seed=seed,
            proj_type=ProjectionType.rademacher,
            device='cuda',
            max_batch_size=max_batch_size
        )
        
        print("✓ CudaProjector initialized successfully")
        
        # Test projection with different batch sizes
        batch_sizes = [4, 8, 16, 32]
        
        for batch_size in batch_sizes:
            print(f"\nTesting batch size: {batch_size}")
            
            # Create test features
            test_features = torch.randn(batch_size, feature_dim, device='cuda', dtype=torch.float16)
            print(f"  Input shape: {test_features.shape}")
            
            # Project features
            projected = cuda_projector.project(test_features, ensemble_id=0)
            print(f"  Output shape: {projected.shape}")
            print(f"  Output device: {projected.device}")
            print(f"  Output dtype: {projected.dtype}")
            
            # Verify output shape
            expected_shape = (batch_size, proj_dim)
            if projected.shape == expected_shape:
                print("  ✓ Output shape correct")
            else:
                print(f"  ✗ Output shape incorrect. Expected {expected_shape}, got {projected.shape}")
                return False
        
        print("\n✓ All batch size tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ CudaProjector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_projection_types():
    """Test both rademacher and normal projections"""
    print("\n=== Testing Different Projection Types ===\n")
    
    feature_dim = 1024
    proj_dim = 512
    batch_size = 16
    
    test_features = torch.randn(batch_size, feature_dim, device='cuda', dtype=torch.float16)
    
    projection_types = [ProjectionType.rademacher, ProjectionType.normal]
    
    for proj_type in projection_types:
        print(f"Testing {proj_type} projection...")
        try:
            projector = CudaProjector(
                feature_dim=feature_dim,
                proj_dim=proj_dim,
                seed=42,
                proj_type=proj_type,
                device='cuda',
                max_batch_size=8
            )
            
            result = projector.project(test_features, ensemble_id=0)
            print(f"  ✓ {proj_type} projection successful: {result.shape}")
            
        except Exception as e:
            print(f"  ✗ {proj_type} projection failed: {e}")
            return False
    
    return True

def test_make_random_projector():
    """Test the make_random_projector function with fast_jl"""
    print("\n=== Testing make_random_projector Function ===\n")
    
    try:
        # Test parameters
        param_shape_list = [1024, 512, 256]  # Different parameter group sizes
        feature_batch_size = 16
        proj_dim = 512
        proj_max_batch_size = 8
        device = "cuda"
        
        print(f"Parameter shapes: {param_shape_list}")
        print(f"Total feature dim: {sum(param_shape_list)}")
        print(f"Batch size: {feature_batch_size}")
        print(f"Projection dim: {proj_dim}")
        
        # Create projector using make_random_projector
        projector = make_random_projector(
            param_shape_list=param_shape_list,
            feature_batch_size=feature_batch_size,
            proj_dim=proj_dim,
            proj_max_batch_size=proj_max_batch_size,
            device=device,
            proj_seed=42,
            use_half_precision=True
        )
        
        print(f"✓ Projector created: {type(projector).__name__}")
        
        # Test projection
        total_feature_dim = sum(param_shape_list)
        test_features = torch.randn(feature_batch_size, total_feature_dim, device='cuda', dtype=torch.float16)
        
        projected = projector.project(test_features, ensemble_id=0)
        
        print(f"✓ Projection successful:")
        print(f"  Input shape: {test_features.shape}")
        print(f"  Output shape: {projected.shape}")
        print(f"  Output device: {projected.device}")
        print(f"  Output dtype: {projected.dtype}")
        
        return True
        
    except Exception as e:
        print(f"✗ make_random_projector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ensemble_consistency():
    """Test that different ensemble_ids produce different projections"""
    print("\n=== Testing Ensemble Consistency ===\n")
    
    try:
        feature_dim = 1024
        proj_dim = 512
        batch_size = 8
        
        projector = CudaProjector(
            feature_dim=feature_dim,
            proj_dim=proj_dim,
            seed=42,
            proj_type=ProjectionType.rademacher,
            device='cuda',
            max_batch_size=8
        )
        
        test_features = torch.randn(batch_size, feature_dim, device='cuda', dtype=torch.float16)
        
        # Project with different ensemble IDs
        proj1 = projector.project(test_features, ensemble_id=0)
        proj2 = projector.project(test_features, ensemble_id=1)
        proj3 = projector.project(test_features, ensemble_id=0)  # Same as first
        
        # Check that different ensemble IDs give different results
        diff_01 = torch.norm(proj1 - proj2).item()
        diff_02 = torch.norm(proj1 - proj3).item()
        
        print(f"Difference between ensemble 0 and 1: {diff_01:.6f}")
        print(f"Difference between ensemble 0 and 0: {diff_02:.6f}")
        
        if diff_01 > 1e-3 and diff_02 < 1e-6:
            print("✓ Ensemble consistency test passed")
            return True
        else:
            print("✗ Ensemble consistency test failed")
            return False
            
    except Exception as e:
        print(f"✗ Ensemble test failed: {e}")
        return False

def test_performance_comparison():
    """Compare performance of CudaProjector vs BasicProjector"""
    print("\n=== Performance Comparison ===\n")
    
    try:
        import time
        
        feature_dim = 2048
        proj_dim = 1024
        batch_size = 64
        num_trials = 10
        
        test_features = torch.randn(batch_size, feature_dim, device='cuda', dtype=torch.float16)
        
        # Test CudaProjector
        print("Benchmarking CudaProjector...")
        cuda_projector = CudaProjector(
            feature_dim=feature_dim,
            proj_dim=proj_dim,
            seed=42,
            proj_type=ProjectionType.rademacher,
            device='cuda',
            max_batch_size=16
        )
        
        # Warmup
        for _ in range(3):
            _ = cuda_projector.project(test_features, ensemble_id=0)
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(num_trials):
            result = cuda_projector.project(test_features, ensemble_id=0)
        
        torch.cuda.synchronize()
        cuda_time = (time.time() - start_time) / num_trials
        
        print(f"  CudaProjector average time: {cuda_time*1000:.2f} ms")
        
        # Test BasicProjector for comparison
        print("Benchmarking BasicProjector...")
        basic_projector = BasicProjector(
            feature_dim=feature_dim,
            proj_dim=proj_dim,
            seed=42,
            proj_type=ProjectionType.rademacher,
            device=torch.device('cuda'),
            block_size=100,
            dtype=torch.float16
        )
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(num_trials):
            result = basic_projector.project(test_features, ensemble_id=0)
        
        torch.cuda.synchronize()
        basic_time = (time.time() - start_time) / num_trials
        
        print(f"  BasicProjector average time: {basic_time*1000:.2f} ms")
        print(f"  Speedup: {basic_time/cuda_time:.1f}x")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance comparison failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Dattri CudaProjector Integration Test")
    print("=" * 50)
    
    # System info
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA capability: {torch.cuda.get_device_capability()}")
    print()
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - tests require GPU")
        return 1
    
    # Run tests
    tests = [
        test_dattri_cuda_projector,
        test_different_projection_types,
        test_make_random_projector,
        test_ensemble_consistency,
        test_performance_comparison
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "PASS" if result else "FAIL"
        print(f"{i+1}. {test.__name__:<30} [{status}]")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Dattri CudaProjector works with fast_jl.")
        print("\nUsage in your code:")
        print("  from dattri.func.projection import CudaProjector, ProjectionType")
        print("  projector = CudaProjector(feature_dim=2048, proj_dim=512, seed=42,")
        print("                            proj_type=ProjectionType.rademacher,")
        print("                            device='cuda', max_batch_size=16)")
        print("  projected = projector.project(features, ensemble_id=0)")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    # Set library path dynamically for PyTorch
    torch_lib_path = os.path.dirname(torch.__file__) + '/lib'
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    os.environ['LD_LIBRARY_PATH'] = f"{torch_lib_path}:{current_ld_path}"
    
    exit(main())