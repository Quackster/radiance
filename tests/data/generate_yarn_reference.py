"""Generate independent YaRN vectors using vLLM (Apache-2.0).

Run in the reference environment, redirect stdout to yarn_reference.json (CPU)
or pass cuda and redirect to yarn_reference_rocm.json (ROCm/gfx1201).
This script is not a runtime dependency and does not load model weights.
"""
import json
import sys
from types import SimpleNamespace
import torch
from vllm.model_executor.layers.rotary_embedding.yarn_scaling_rope import YaRNScalingRotaryEmbedding

positions = [0, 1, 4096, 262143, 262144, 500000, 524287]
cases = []
device = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
with torch.device(device):
    for rot in (64, 128, 256):
        cfg = SimpleNamespace(base=10000000.0, rotary_dim=rot, max_position_embeddings=262144,
                              beta_fast=32, beta_slow=1, truncate=True)
        inv = YaRNScalingRotaryEmbedding._compute_inv_freq(cfg, 2.0)
        phase = torch.tensor(positions, dtype=torch.float32)[:, None] * inv[None, :]
        magnitude = 1.0 + 0.1 * __import__('math').log(2.0)
        cases.append(dict(rot=rot, theta=cfg.base, factor=2.0, original=262144,
                          magnitude=magnitude, inv_freq=inv.tolist(),
                          cos=(phase.cos()*magnitude).tolist(), sin=(phase.sin()*magnitude).tolist()))
print(json.dumps({'reference': f'vLLM YaRNScalingRotaryEmbedding; torch {device} FP32; static factor=2',
                  'positions': positions, 'cases': cases}, indent=2))
