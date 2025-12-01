#!/usr/bin/env python3
"""Test script to inspect PT file structure"""

import torch
import sys

if len(sys.argv) < 2:
    pt_file = "/workspace/EasySteer/backend/vectors/persona_vectors/Qwen2.5-7B-Instruct/happiness_prompt_avg_diff.pt"
else:
    pt_file = sys.argv[1]

print(f"Loading: {pt_file}")
data = torch.load(pt_file, map_location='cpu', weights_only=False)

print(f"\nType: {type(data)}")
if isinstance(data, torch.Tensor):
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
elif isinstance(data, dict):
    print(f"Keys: {list(data.keys())}")
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: type={type(v)}")
else:
    print(f"Unexpected type: {type(data)}")
