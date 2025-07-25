"""
Memory-efficient TracIn implementation for LLM-scale attribution.
Handles large datasets through chunked computation and CPU offloading.
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc
from typing import Optional, Dict, Any, List, Tuple
import logging

from dattri.algorithm.tracin import TracInAttributor
from torch.nn.functional import normalize

logger = logging.getLogger(__name__)


class MemoryEfficientTracInAttributor(TracInAttributor):
    """
    Memory-efficient TracIn attributor that processes training data in chunks
    and offloads gradients to CPU to handle large datasets with limited GPU memory.
    """
    
    def __init__(
        self,
        task,
        weight_list=None,
        normalized_grad=True,
        projector_kwargs=None,
        chunk_size=None,
        cpu_offload=True,
        selective_layers=None,
        memory_threshold=0.8,
        enable_projections=True,
    ):
        """
        Initialize memory-efficient TracIn attributor.
        
        Args:
            task: Attribution task containing model and checkpoints
            weight_list: Weights for checkpoints
            normalized_grad: Whether to normalize gradients
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
        
        super().__init__(task, weight_list, normalized_grad, projector_kwargs)
        
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
        
        # For LLM models, get parameters from specific layers only
        selective_params = []
        
        # Get all named parameters
        named_params = dict(model.named_parameters())
        
        # If selective_layers contains specific layer names
        if isinstance(self.selective_layers, list):
            for layer_name in self.selective_layers:
                for param_name, param in named_params.items():
                    if layer_name in param_name and param.requires_grad:
                        selective_params.append(param)
                        
        # If selective_layers is a number, take last N transformer layers + output
        elif isinstance(self.selective_layers, int):
            n_layers = self.selective_layers
            
            # Get transformer layers and output layers
            transformer_layers = []
            output_params = []
            
            for param_name, param in named_params.items():
                if param.requires_grad:
                    # Match transformer layer patterns for different models
                    if any(pattern in param_name for pattern in [
                        'layers.', 'layer.', 'blocks.', 'h.'  # Common transformer layer patterns
                    ]):
                        # Extract layer number
                        import re
                        layer_match = re.search(r'(?:layers?|blocks?|h)\.(\d+)', param_name)
                        if layer_match:
                            layer_num = int(layer_match.group(1))
                            transformer_layers.append((layer_num, param_name, param))
                    
                    # Collect output/final layers
                    elif any(pattern in param_name for pattern in [
                        'lm_head', 'output', 'classifier', 'score'
                    ]):
                        output_params.append(param)
            
            # Sort by layer number and take last N layers
            transformer_layers.sort(key=lambda x: x[0])
            if len(transformer_layers) > 0:
                total_layers = max(x[0] for x in transformer_layers) + 1
                start_layer = max(0, total_layers - n_layers)
                
                for layer_num, param_name, param in transformer_layers:
                    if layer_num >= start_layer:
                        selective_params.append(param)
            
            # Always include output parameters
            selective_params.extend(output_params)
            
        if len(selective_params) == 0:
            logger.warning("No parameters found for selective computation, falling back to all parameters")
            return list(model.parameters())
            
        total_params = sum(p.numel() for p in selective_params)
        all_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Using {n_layers} selective layers ({100 * total_params / all_params:.1f}% of parameters)")
        
        # Log moved to above to reduce redundancy
        
        return selective_params
    
    def _compute_selective_gradient(self, batch_data, is_test=False):
        """
        Compute gradients only for selective layers to reduce memory usage.
        """
        # Get selective parameters for gradient computation
        selective_params = self._get_selective_parameters(self.task.model)
        
        # Extract batch components and ensure they're on the right device
        input_ids, attention_mask, labels = batch_data
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0) 
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
            
        # Ensure all tensors are on the correct device
        device = next(self.task.model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        
        # Enable gradients and zero them for selective parameters
        for param in selective_params:
            param.requires_grad_(True)
            if param.grad is not None:
                param.grad.zero_()
        
        # Enable gradient computation explicitly
        with torch.enable_grad():
            # Forward pass
            outputs = self.task.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        if is_test:
            # For test gradients, use target function (e.g., correct probability)
            loss = outputs.loss  # Use the same loss for simplicity
        else:
            # For training gradients, use loss function
            loss = outputs.loss
        
        # Backward pass for selective parameters only
        with torch.enable_grad():
            loss.backward()
        
        # Collect gradients from selective parameters
        gradients = []
        for param in selective_params:
            if param.grad is not None:
                gradients.append(param.grad.clone().flatten())
            else:
                gradients.append(torch.zeros(param.numel(), device=param.device, dtype=param.dtype))
        
        # Concatenate all selective gradients and reshape for projection compatibility
        grad_vector = torch.cat(gradients)
        # Reshape to 2D for projection (batch_size=1, features=grad_dim)
        grad_vector = grad_vector.unsqueeze(0)
        
        # Clear gradients to free memory
        for param in selective_params:
            if param.grad is not None:
                param.grad = None
        
        # Memory cleanup done, gradients collected
                
        return grad_vector
        
    def _log_memory_usage(self, stage: str, sample_idx: int = None):
        """Log essential GPU memory usage only."""
        if torch.cuda.is_available() and stage in ["INIT_ATTRIBUTION", "FINAL_ATTRIBUTION"]:
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            free, total = torch.cuda.mem_get_info()
            free, total = free / 1e9, total / 1e9
            
            logger.info(f"[MEMORY {stage}]: Reserved={reserved:.1f}GB, Free={free:.1f}GB")

    def _compute_gradient_chunked(self, parameters, dataloader, chunk_size: int, 
                                is_test: bool = False) -> List[torch.Tensor]:
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
                
                # Compute gradient using selective parameters if specified
                if hasattr(self, '_selective_params') and self._selective_params is not None:
                    # Use custom selective gradient computation
                    if is_test:
                        grad = self._compute_selective_gradient(batch_data, is_test=True)
                    else:
                        grad = self._compute_selective_gradient(batch_data, is_test=False)
                else:
                    # Use full parameters
                    if is_test:
                        grad = self.grad_target_func(parameters, batch_data)
                    else:
                        grad = self.grad_loss_func(parameters, batch_data)
                
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
            
        return all_gradients
        
    def _compute_similarity_streaming(self, train_gradients: List[torch.Tensor], 
                                    test_gradients: List[torch.Tensor],
                                    batch_size: int = 50) -> torch.Tensor:
        """
        Compute similarity matrix in streaming fashion to minimize memory usage.
        """
        n_train = len(train_gradients)
        n_test = len(test_gradients)
        
        # Initialize result tensor
        similarities = torch.zeros(n_train, n_test)
        
        # Process test gradients in batches
        for test_start in tqdm(range(0, n_test, batch_size), desc="computing similarities"):
            test_end = min(test_start + batch_size, n_test)
            test_batch = test_gradients[test_start:test_end]
            
            # Move test batch to GPU and squeeze batch dimension
            test_tensor = torch.stack([g.squeeze(0).to(self.device) if g.device.type == 'cpu' else g.squeeze(0) 
                                     for g in test_batch])
            
            # Process training gradients in batches
            for train_start in range(0, n_train, batch_size):
                train_end = min(train_start + batch_size, n_train)
                train_batch = train_gradients[train_start:train_end]
                
                # Move train batch to GPU and squeeze batch dimension  
                train_tensor = torch.stack([g.squeeze(0).to(self.device) if g.device.type == 'cpu' else g.squeeze(0) 
                                          for g in train_batch])
                
                # Compute similarity
                if self.normalized_grad:
                    train_norm = normalize(train_tensor)
                    test_norm = normalize(test_tensor)
                    batch_sim = torch.mm(train_norm, test_norm.T)
                else:
                    batch_sim = torch.mm(train_tensor, test_tensor.T)
                
                # Store results
                similarities[train_start:train_end, test_start:test_end] = batch_sim.cpu()
                
                # Clear memory
                del train_tensor, batch_sim
                torch.cuda.empty_cache()
                
            # Clear test tensor
            del test_tensor
            torch.cuda.empty_cache()
            
        return similarities
        
    def attribute(self, train_dataloader: DataLoader, test_dataloader: DataLoader) -> torch.Tensor:
        """
        Memory-efficient attribution computation with proper memory defragmentation.
        """
        logger.info("Starting memory-efficient TracIn attribution")
        
        # Clear any existing memory fragmentation
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        self._log_memory_usage("INIT_ATTRIBUTION")
        
        # Basic model info
        total_params = sum(p.numel() for p in self.task.model.parameters())
        logger.info(f"Model: {total_params/1e6:.1f}M parameters")
        
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
        self._selective_params = None
        if self.selective_layers is not None:
            # Get selective parameters for memory efficiency
            model_params = self._get_selective_parameters(self.task.model)
            self._selective_params = torch.cat([p.flatten() for p in model_params])
            parameters = self._selective_params
        elif hasattr(self.task, 'get_param'):
            # Check if we should use current model parameters directly
            if (len(self.task.checkpoints) == 1 and 
                isinstance(self.task.checkpoints[0], dict) and 
                self.task.checkpoints[0].get("use_current_model")):
                logger.info("Using current model parameters directly (no checkpoint loading)")
                model_params = list(self.task.model.parameters())
                parameters = torch.cat([p.flatten() for p in model_params])
            else:
                parameters, _ = self.task.get_param(ckpt_idx=0)
        else:
            model_params = list(self.task.model.parameters())
            parameters = torch.cat([p.flatten() for p in model_params])
        
        # Force memory cleanup before gradient computation
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
            
        # Compute gradients for training data
        logger.info("Computing training gradients...")
        train_gradients = self._compute_gradient_chunked(
            parameters if hasattr(self.task, 'get_param') else model_params, 
            train_dataloader, 
            self.chunk_size, 
            is_test=False
        )
        
        # Force memory cleanup before test gradients
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Compute gradients for test data  
        logger.info("Computing test gradients...")
        test_gradients = self._compute_gradient_chunked(
            parameters if hasattr(self.task, 'get_param') else model_params,
            test_dataloader, 
            self.chunk_size, 
            is_test=True
        )
        
        # Force memory cleanup before similarities
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Compute similarities in streaming fashion
        logger.info("Computing attribution similarities...")
        similarities = self._compute_similarity_streaming(
            train_gradients, test_gradients, batch_size=min(50, self.chunk_size)
        )
        
        # Apply checkpoint weights if specified
        if self.weight_list is not None and len(self.weight_list) > 0:
            weight_tensor = torch.tensor(self.weight_list[0], device=similarities.device)
            similarities = similarities * weight_tensor
            
        self._log_memory_usage("FINAL_ATTRIBUTION")
        logger.info(f"Attribution complete. Result shape: {similarities.shape}")
        return similarities