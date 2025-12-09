#!/usr/bin/env python
"""Test DDP training locally"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from pure_transformer.model import TransformerLM, TransformerConfig
from pure_transformer.configs import get_model_config


def test_ddp_training(rank, world_size):
    # Setup distributed
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    # Create model
    config = get_model_config('tiny')
    model = TransformerLM(config).cuda(rank)
    model = DDP(model, device_ids=[rank])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Run 5 training steps
    for step in range(5):
        optimizer.zero_grad()
        input_ids = torch.randint(0, 50304, (2, 256)).cuda(rank)
        labels = input_ids.clone()
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss, logits = model(input_ids, labels=labels)
        
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if rank == 0:
            print(f'Step {step+1}: loss={loss.item():.4f}, grad_norm={grad_norm.item():.4f}')
    
    if rank == 0:
        print('\nSUCCESS: DDP training works!')
    
    dist.destroy_process_group()


if __name__ == '__main__':
    world_size = 2
    mp.spawn(test_ddp_training, args=(world_size,), nprocs=world_size, join=True)
