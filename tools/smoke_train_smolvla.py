#!/usr/bin/env python3

from __future__ import annotations

import math
import time

import torch
from huggingface_hub import snapshot_download
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    snapshot = snapshot_download(repo_id="lerobot/smolvla_base")
    policy = SmolVLAPolicy.from_pretrained(snapshot)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=snapshot,
    )

    sample: dict[str, torch.Tensor | str] = {"task": "Move the object to the target."}
    for key, feature in policy.config.input_features.items():
        sample[key] = torch.zeros(tuple(feature.shape), dtype=torch.float32)

    action_dim = policy.config.action_feature.shape[0]
    sample["action"] = torch.zeros(
        (1, policy.config.chunk_size, action_dim),
        dtype=torch.float32,
    )
    processed = preprocessor(sample)

    policy.train()
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-5)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    loss, metrics = policy.forward(processed)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=10.0)
    optimizer.step()
    torch.cuda.synchronize()

    loss_value = float(loss.detach().cpu())
    grad_norm_value = float(grad_norm.detach().cpu())
    print(f"loss={loss_value:.6f}")
    print(f"grad_norm={grad_norm_value:.6f}")
    print(f"trainable_parameters={sum(parameter.numel() for parameter in trainable)}")
    print(f"step_seconds={time.perf_counter() - started:.2f}")
    print(f"peak_cuda_mib={torch.cuda.max_memory_allocated() / 1048576:.1f}")
    print(f"metrics={metrics}")
    if not math.isfinite(loss_value) or not math.isfinite(grad_norm_value):
        raise RuntimeError("Training smoke test produced a non-finite value")


if __name__ == "__main__":
    main()

