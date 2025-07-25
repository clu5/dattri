"""
Memory-efficient Grad-Cos implementation for LLM-scale attribution.
Handles large datasets through chunked computation and CPU offloading.
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

from dattri.algorithm.tracin import TracInAttributor
from torch.nn.functional import normalize

logger = logging.getLogger(__name__)


class MemoryEfficientGradCosAttributor(TracInAttributor):
    """
    Memory-efficient Grad-Cos attributor that processes training data in chunks
    and offloads gradients to CPU to handle large datasets with limited GPU memory.
    """
    
    def __init__(
        self,
        task,
        weight_list=None,
        projector_kwargs=None,
        chunk_size=None,
        cpu_offload=True,
        selective_layers=None,
        memory_threshold=0.8,
        enable_projections=True,
    ):
        """
        Initialize memory-efficient Grad-Cos attributor.
        
        Args:
            task: Attribution task containing model and checkpoints
            weight_list: Weights for checkpoints
            projector_kwargs: Random projection parameters (auto-enabled if None)
            chunk_size: Number of training samples to process at once (auto if None)
            cpu_offload: Whether to offload gradients to CPU
            selective_layers: List of layer names to compute gradients for (None for all)
            memory_threshold: Fraction of GPU memory to use for auto chunk sizing
            enable_projections: Whether to enable random projections by default
        """
        # Enable projections by default for memory efficiency
        if projector_kwargs is None and enable_projections:
            projector_kwargs = {
                "proj_dim": 512,
                "device": task.device if hasattr(task, 'device') else "cuda",
                "use_half_precision": False,
            }
            logger.info("Auto-enabled random projections for memory efficiency with proj_dim=512")
        
        super().__init__(task, weight_list, projector_kwargs)
        
        self.chunk_size = chunk_size
        self.cpu_offload = cpu_offload
        self.selective_layers = selective_layers
        self.memory_threshold = memory_threshold
        
        # Initialize memory monitoring
        if torch.cuda.is_available():
            self.gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
            logger.info(f"GPU memory total: {self.gpu_memory_total / 1e9:.1f} GB")
        else:
            self.gpu_memory_total = None
            
    def _get_optimal_chunk_size(self, model_params: int, available_memory: int) -> int:
        """
        Calculate optimal chunk size based on available GPU memory and model size.
        """
        # Estimate memory per gradient (float32 = 4 bytes)
        bytes_per_gradient = model_params * 4
        
        # Reserve memory for model, optimizer states, and intermediate computations
        usable_memory = available_memory * self.memory_threshold
        
        # Calculate max gradients that can fit in memory
        max_gradients = max(1, int(usable_memory // bytes_per_gradient))
        
        # Conservative estimate considering projection and intermediate tensors
        safe_chunk_size = max(1, max_gradients // 4)
        
        logger.info(f"Auto chunk size: {safe_chunk_size} (model params: {model_params}, "
                   f"available memory: {available_memory / 1e9:.1f} GB)")
        
        return safe_chunk_size
        
    def _get_selective_parameters(self, model):
        """
        Get parameters for selective layer computation.
        """
        if self.selective_layers is None:
            return list(model.parameters())
            
        target_params = []
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in self.selective_layers):
                target_params.append(param)
                
        logger.info(f"Selected {len(target_params)} parameters from layers: {self.selective_layers}")
        return target_params
        
    def _log_memory_usage(self, stage: str, sample_idx: int = None):
        """Log essential GPU memory usage only."""
        if torch.cuda.is_available() and stage in ["START_ATTRIBUTION", "FINAL_ATTRIBUTION"]:
            reserved = torch.cuda.memory_reserved() / 1e9
            free, total = torch.cuda.mem_get_info()
            free, total = free / 1e9, total / 1e9
            
            logger.info(f"[MEMORY {stage}]: Reserved={reserved:.1f}GB, Free={free:.1f}GB")

    def _compute_gradient_chunked(self, parameters, dataloader, chunk_size: int, 
                                is_test: bool = False) -> list:
        """
        Compute gradients one sample at a time with detailed memory logging and cleanup.
        """
        all_gradients = []
        
        # Process data in chunks
        data_list = list(dataloader)
        num_chunks = (len(data_list) + chunk_size - 1) // chunk_size
        
        desc = "computing test gradients" if is_test else "computing train gradients"
        
        for chunk_idx in tqdm(range(num_chunks), desc=desc):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(data_list))
            chunk_data = data_list[start_idx:end_idx]
            
            for i, batch_data in enumerate(chunk_data):
                sample_idx = start_idx + i
                
                # Move batch to device
                if isinstance(batch_data, (tuple, list)):
                    batch_data = tuple(x.to(self.device) for x in batch_data)
                
                # Compute gradient
                if is_test:
                    grad = self.grad_target_func(parameters, batch_data)
                else:
                    grad = self.grad_loss_func(parameters, batch_data)
                
                # Skip verbose logging
                
                # Apply projection if specified
                if self.projector_kwargs is not None:
                    if is_test:
                        if not hasattr(self, 'test_random_project'):
                            from dattri.metric.func.projection import random_project
                            self.test_random_project = random_project(
                                grad, batch_data[0].shape[0], **self.projector_kwargs
                            )
                        grad_proj = self.test_random_project(torch.nan_to_num(grad), ensemble_id=0)
                    else:
                        if not hasattr(self, 'train_random_project'):
                            from dattri.metric.func.projection import random_project
                            self.train_random_project = random_project(
                                grad, batch_data[0].shape[0], **self.projector_kwargs
                            )
                        grad_proj = self.train_random_project(torch.nan_to_num(grad), ensemble_id=0)
                else:
                    grad_proj = torch.nan_to_num(grad)
                
                # Skip verbose logging
                
                # Explicit cleanup of large gradient tensor
                del grad
                
                # Store gradient (optionally on CPU)
                if self.cpu_offload:
                    all_gradients.append(grad_proj.cpu())
                    # Free GPU copy immediately
                    del grad_proj
                else:
                    all_gradients.append(grad_proj)
                
                # Aggressive cleanup after each sample
                torch.cuda.empty_cache()
                # Cleanup completed
            
        return all_gradients
        
    def _compute_similarity_streaming(self, train_gradients: list, 
                                    test_gradients: list,
                                    batch_size: int = 50) -> torch.Tensor:
        """
        Compute cosine similarity matrix in streaming fashion to minimize memory usage.
        """
        n_train = len(train_gradients)
        n_test = len(test_gradients)
        
        # Initialize result tensor
        similarities = torch.zeros(n_train, n_test)
        
        # Process test gradients in batches
        for test_start in tqdm(range(0, n_test, batch_size), desc="computing cosine similarities"):
            test_end = min(test_start + batch_size, n_test)
            test_batch = test_gradients[test_start:test_end]
            
            # Move test batch to GPU and normalize
            test_tensor = torch.stack([g.to(self.device) if g.device.type == 'cpu' else g 
                                     for g in test_batch])
            test_normalized = normalize(test_tensor)
            
            # Process training gradients in batches
            for train_start in range(0, n_train, batch_size):
                train_end = min(train_start + batch_size, n_train)
                train_batch = train_gradients[train_start:train_end]
                
                # Move train batch to GPU and normalize
                train_tensor = torch.stack([g.to(self.device) if g.device.type == 'cpu' else g 
                                          for g in train_batch])
                train_normalized = normalize(train_tensor)
                
                # Compute cosine similarity
                batch_sim = torch.mm(train_normalized, test_normalized.T)
                
                # Store results
                similarities[train_start:train_end, test_start:test_end] = batch_sim.cpu()
                
                # Clear memory
                del train_tensor, train_normalized, batch_sim
                torch.cuda.empty_cache()
                
            # Clear test tensor
            del test_tensor, test_normalized
            torch.cuda.empty_cache()
            
        return similarities
        
    def attribute(self, train_dataloader: DataLoader, test_dataloader: DataLoader) -> torch.Tensor:
        """
        Memory-efficient attribution computation using cosine similarity.
        """
        logger.info("Starting memory-efficient Grad-Cos attribution")
        
        # Auto-determine chunk size if not specified
        if self.chunk_size is None:
            if self.gpu_memory_total is not None:
                available_memory = torch.cuda.mem_get_info()[0]  # Free memory
                model_params = sum(p.numel() for p in self.task.model.parameters())
                self.chunk_size = self._get_optimal_chunk_size(model_params, available_memory)
            else:
                self.chunk_size = 10  # Conservative default for CPU
                
        logger.info(f"Using chunk size: {self.chunk_size}")
        
        # Get model parameters (selective if specified)
        if hasattr(self.task, 'get_param'):
            # Check if we should use current model parameters directly
            if (len(self.task.checkpoints) == 1 and 
                isinstance(self.task.checkpoints[0], dict) and 
                self.task.checkpoints[0].get("use_current_model")):
                logger.info("Using current model parameters directly (no checkpoint loading)")
                model_params = self._get_selective_parameters(self.task.model)
                parameters = torch.cat([p.flatten() for p in model_params])
            else:
                parameters, _ = self.task.get_param(ckpt_idx=0)
        else:
            model_params = self._get_selective_parameters(self.task.model)
            parameters = torch.cat([p.flatten() for p in model_params])
            
        # Compute gradients for training data
        logger.info("Computing training gradients...")
        train_gradients = self._compute_gradient_chunked(
            parameters if hasattr(self.task, 'get_param') else model_params, 
            train_dataloader, 
            self.chunk_size, 
            is_test=False
        )
        
        # Compute gradients for test data  
        logger.info("Computing test gradients...")
        test_gradients = self._compute_gradient_chunked(
            parameters if hasattr(self.task, 'get_param') else model_params,
            test_dataloader, 
            self.chunk_size, 
            is_test=True
        )
        
        # Compute cosine similarities in streaming fashion
        logger.info("Computing cosine similarities...")
        similarities = self._compute_similarity_streaming(
            train_gradients, test_gradients, batch_size=min(50, self.chunk_size)
        )
        
        # Apply checkpoint weights if specified
        if self.weight_list is not None and len(self.weight_list) > 0:
            similarities = similarities * self.weight_list[0]
            
        logger.info(f"Attribution complete. Result shape: {similarities.shape}")
        return similarities