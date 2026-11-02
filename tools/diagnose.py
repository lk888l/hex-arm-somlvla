#!/usr/bin/env python3

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess


def first_line(command: list[str]) -> str:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    output = result.stdout.strip() or result.stderr.strip()
    return output.splitlines()[0] if output else f"exit={result.returncode}"


def version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def main() -> None:
    import torch
    import torchcodec

    print(f"platform={platform.platform()}")
    print(f"python={platform.python_version()}")
    print(f"lerobot={version('lerobot')}")
    print(f"torch={torch.__version__}")
    print(f"torchvision={version('torchvision')}")
    print(f"torchcodec={torchcodec.__version__}")
    print(f"transformers={version('transformers')}")
    print(f"ffmpeg={first_line(['ffmpeg', '-version'])}")
    print(f"nvidia_smi={first_line(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'])}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cuda_runtime={torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")
        print(f"bf16_supported={torch.cuda.is_bf16_supported()}")
    for name in ("LEROBOT_WORKSPACE", "HF_HOME", "HF_HUB_CACHE", "HF_LEROBOT_HOME", "WANDB_MODE"):
        print(f"{name}={os.environ.get(name, '<unset>')}")


if __name__ == "__main__":
    main()

