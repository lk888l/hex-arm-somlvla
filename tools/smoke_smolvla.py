#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import time

import torch
from huggingface_hub import snapshot_download
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load SmolVLA and run one hardware-free action inference.")
    parser.add_argument("--model", default="lerobot/smolvla_base")
    parser.add_argument("--task", default="Move the object to the target.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    snapshot = snapshot_download(repo_id=args.model)
    started = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(snapshot)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=snapshot,
    )

    observation: dict[str, torch.Tensor | str] = {"task": args.task}
    for key, feature in policy.config.input_features.items():
        shape = tuple(feature.shape)
        if key.startswith("observation.images."):
            observation[key] = torch.zeros(shape, dtype=torch.float32)
        elif key == "observation.state":
            observation[key] = torch.zeros(shape, dtype=torch.float32)

    processed = preprocessor(observation)
    torch.cuda.reset_peak_memory_stats()
    infer_started = time.perf_counter()
    with torch.inference_mode():
        action = policy.select_action(processed)
        action = postprocessor(action)
    torch.cuda.synchronize()

    finite = all(math.isfinite(value) for value in action.detach().cpu().flatten().tolist())
    print(f"snapshot={snapshot}")
    print(f"load_seconds={time.perf_counter() - started:.2f}")
    print(f"inference_seconds={time.perf_counter() - infer_started:.2f}")
    print(f"action_shape={tuple(action.shape)}")
    print(f"action_dtype={action.dtype}")
    print(f"action_finite={finite}")
    print(f"peak_cuda_mib={torch.cuda.max_memory_allocated() / 1048576:.1f}")
    if not finite:
        raise RuntimeError("SmolVLA returned a non-finite action")


if __name__ == "__main__":
    main()

