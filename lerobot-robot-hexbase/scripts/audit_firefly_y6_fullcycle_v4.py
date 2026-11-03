#!/usr/bin/env python3
"""Audit the fixed full-cycle v4 start-motion window without touching hardware."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from lerobot_robot_hexbase.firefly_fullcycle_v4 import (  # noqa: E402
    FullcycleV4ProtocolError,
    audit_start_motion_windows,
)


DATASET_ROOT = Path(
    "/home/kk/kk_data/lerobot/datasets/local/firefly_y6_fullcycle_v4"
)


def _episode_count(dataset_root: Path) -> int:
    path = dataset_root / "meta/info.json"
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
        count = info["total_episodes"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise FullcycleV4ProtocolError(
            f"cannot read total_episodes from {path}: {exc}"
        ) from exc
    if type(count) is not int or count <= 0:
        raise FullcycleV4ProtocolError(
            f"dataset must contain at least one saved episode, got {count!r}"
        )
    return count


def main() -> int:
    try:
        report = audit_start_motion_windows(
            DATASET_ROOT,
            expected_episodes=_episode_count(DATASET_ROOT),
        )
    except FullcycleV4ProtocolError as exc:
        print(f"Full-cycle v4 protocol audit failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["safe_for_training"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
