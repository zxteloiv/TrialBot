"""
Multi-GPU training utilities for TrialBot.
Supports DataParallel, DistributedDataParallel, and DeepSpeed.
"""

import os
import torch
import torch.distributed as dist
import torch.nn as nn
from typing import List, Optional, Union
import logging

logger = logging.getLogger(__name__)


def get_available_gpus() -> List[int]:
    """Get list of available GPU IDs."""
    if torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return []


def parse_gpu_ids(gpu_str: str) -> List[int]:
    """
    Parse GPU IDs from string.
    
    Args:
        gpu_str: Comma-separated GPU IDs, e.g., "0,1,2,3" or "all" for all GPUs.
    
    Returns:
        List of GPU IDs.
    """
    if not gpu_str:
        return []
    
    if gpu_str.lower() == 'all':
        return get_available_gpus()
    
    try:
        gpu_ids = [int(x.strip()) for x in gpu_str.split(',') if x.strip()]
        return gpu_ids
    except ValueError:
        logger.warning(f"Invalid GPU string: {gpu_str}. Using all available GPUs.")
        return get_available_gpus()


def setup_distributed_training(args) -> bool:
    """
    Setup distributed training if needed.
    
    Args:
        args: Command line arguments with multi-GPU parameters.
    
    Returns:
        True if distributed training is enabled, False otherwise.
    """
    # Check if we should use distributed training
    gpu_ids = parse_gpu_ids(args.gpus)
    use_distributed = len(gpu_ids) > 1 and args.multiprocessing_distributed
    
    if not use_distributed:
        return False
    
    # Initialize distributed backend
    if args.world_size == 1:
        args.world_size = len(gpu_ids)
    
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    
    if args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank * len(gpu_ids) + args.local_rank
        )
        logger.info(f"Distributed training initialized: "
                   f"rank={dist.get_rank()}, world_size={dist.get_world_size()}")
        return True
    
    return False


def cleanup_distributed_training():
    """Cleanup distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group()


def prepare_model_for_multigpu(
    model: nn.Module,
    args,
    gpu_ids: Optional[List[int]] = None
) -> nn.Module:
    """
    Prepare model for multi-GPU training based on configuration.
    
    Args:
        model: PyTorch model to prepare.
        args: Command line arguments.
        gpu_ids: List of GPU IDs to use. If None, parsed from args.gpus.
    
    Returns:
        Prepared model (wrapped in DataParallel, DDP, or unchanged).
    """
    if gpu_ids is None:
        gpu_ids = parse_gpu_ids(args.gpus)
    
    if not gpu_ids:
        # No GPUs specified, use CPU or single GPU based on args.device
        if args.device >= 0 and torch.cuda.is_available():
            model = model.cuda(args.device)
        return model
    
    # Move model to first GPU
    model = model.cuda(gpu_ids[0])
    
    # Check for DeepSpeed
    if args.deepspeed:
        try:
            import deepspeed
            # DeepSpeed will handle model preparation
            return model
        except ImportError:
            logger.warning("DeepSpeed not installed. Falling back to PyTorch multi-GPU.")
    
    # Check for DistributedDataParallel
    if args.multiprocessing_distributed and len(gpu_ids) > 1:
        if dist.is_initialized():
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.local_rank] if args.local_rank >= 0 else None,
                output_device=args.local_rank if args.local_rank >= 0 else None
            )
            logger.info(f"Model wrapped in DistributedDataParallel")
            return model
    
    # Use DataParallel for multiple GPUs
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
        logger.info(f"Model wrapped in DataParallel using GPUs: {gpu_ids}")
    
    return model


def is_distributed() -> bool:
    """Check if distributed training is initialized."""
    return dist.is_initialized()


def get_world_size() -> int:
    """Get world size for distributed training."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Get rank for distributed training."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def is_main_process() -> bool:
    """Check if current process is the main process (rank 0)."""
    return get_rank() == 0


def synchronize():
    """Synchronize all processes in distributed training."""
    if dist.is_initialized():
        dist.barrier()


def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reduce tensor across all processes in distributed training.
    
    Args:
        tensor: Tensor to reduce.
    
    Returns:
        Reduced tensor.
    """
    if not dist.is_initialized():
        return tensor
    
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= get_world_size()
    return rt


def setup_deepspeed(args, model, optimizer=None):
    """
    Setup DeepSpeed if enabled.
    
    Args:
        args: Command line arguments.
        model: PyTorch model.
        optimizer: PyTorch optimizer (optional).
    
    Returns:
        Tuple of (model, optimizer, lr_scheduler) prepared by DeepSpeed.
    """
    if not args.deepspeed:
        return model, optimizer, None
    
    try:
        import deepspeed
        
        # DeepSpeed configuration
        if args.deepspeed_config and os.path.exists(args.deepspeed_config):
            with open(args.deepspeed_config, 'r') as f:
                import json
                ds_config = json.load(f)
        else:
            # Default DeepSpeed configuration
            ds_config = {
                "train_batch_size": 16,
                "gradient_accumulation_steps": 1,
                "optimizer": {
                    "type": "Adam",
                    "params": {
                        "lr": 0.001,
                        "betas": [0.9, 0.999],
                        "eps": 1e-8
                    }
                },
                "fp16": {
                    "enabled": False
                },
                "zero_optimization": {
                    "stage": 1,
                    "allgather_partitions": True,
                    "allgather_bucket_size": 2e8,
                    "overlap_comm": True,
                    "reduce_scatter": True,
                    "reduce_bucket_size": 2e8,
                    "contiguous_gradients": True
                }
            }
        
        # Initialize DeepSpeed
        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config=ds_config,
            model_parameters=model.parameters()
        )
        
        logger.info("DeepSpeed initialized successfully")
        return model, optimizer, lr_scheduler
    
    except ImportError:
        logger.error("DeepSpeed is not installed. Please install it with: pip install deepspeed")
        raise


def move_to_device_multigpu(obj, device_id: int = -1, gpu_ids: List[int] = None):
    """
    Move data to appropriate device for multi-GPU training.
    
    Args:
        obj: Data to move.
        device_id: Specific device ID (-1 for CPU).
        gpu_ids: List of GPU IDs for multi-GPU.
    
    Returns:
        Data moved to device.
    """
    from .move_to_device import move_to_device
    
    # For distributed training, use local rank
    if dist.is_initialized() and device_id < 0:
        device_id = dist.get_rank() % torch.cuda.device_count()
    
    # For DataParallel, move to first GPU
    elif gpu_ids and device_id < 0:
        device_id = gpu_ids[0]
    
    return move_to_device(obj, device_id)
