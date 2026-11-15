"""All-reduce bandwidth for gradient-sized buffers (SPD 4L all-reduces ~2.4 GB fp32 per step)."""
import os
import time

import torch
import torch.distributed as dist

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
world = dist.get_world_size()
for numel in [150_000_000, 600_000_000]:
    x = torch.ones(numel, device="cuda")
    for _ in range(2):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    t, n = time.time(), 5
    for _ in range(n):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dt = (time.time() - t) / n
    if dist.get_rank() == 0:
        busbw = 2 * (world - 1) / world * numel * 4 / dt / 1e9
        print(f"{os.environ.get('TAG', '')}: {numel * 4 / 1e9:.1f} GB fp32 all-reduce {dt:.3f}s, busbw {busbw:.1f} GB/s", flush=True)
dist.destroy_process_group()
