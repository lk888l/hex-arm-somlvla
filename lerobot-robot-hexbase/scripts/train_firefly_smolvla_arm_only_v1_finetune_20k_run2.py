#!/usr/bin/env python3
"""Run the fixed offline 6-D Firefly SmolVLA continuation with attestation.

The production script self-executes under its pinned isolated Python.  Use
``--preflight`` before training and ``--verify-run`` for the fixed read-only
post-training verification; invoking it without a mode is the only training
entry point.
"""

from __future__ import annotations

import hashlib
import os
import sys


_BOOTSTRAP_TRAINING_PYTHON = (
    "/home/kk/kk_data/lerobot/miniforge3/envs/lerobot-train/bin/python3.12"
)
_BOOTSTRAP_TRAINING_PYTHON_SHA256 = (
    "aa584ad7b853904b0ae7fb5a8f44d9d27960e05de1f4838855f93fa4359df869"
)
_BOOTSTRAP_WORKING_DIRECTORY = (
    "/home/kk/kk_data/lerobot/miniforge3/envs/lerobot-train"
)
_BOOTSTRAP_SCRIPT = (
    "/home/kk/kk_data/lerobot/lerobot-robot-hexbase/scripts/"
    "train_firefly_smolvla_arm_only_v1_finetune_20k_run2.py"
)
_VERIFICATION_REEXEC_MARKER = "FIREFLY_ARM_ONLY_V1_VERIFIER_REEXEC"
_VERIFICATION_REEXEC_VALUE = "pinned-training-python-isolated-v1"
_BOOTSTRAP_OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "WANDB_MODE": "offline",
    "WANDB_DISABLED": "true",
}
_BOOTSTRAP_BASE_ENVIRONMENT = {
    "HOME": "/home/kk",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": f"{_BOOTSTRAP_WORKING_DIRECTORY}/bin:/usr/bin:/bin",
    "TMPDIR": "/tmp",
}
_BOOTSTRAP_INHERITED_ENVIRONMENT = frozenset(
    {
        "CUDA_DEVICE_ORDER",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "NVIDIA_VISIBLE_DEVICES",
    }
)


def _bootstrap_environment(source: dict[str, str]) -> dict[str, str]:
    environment = dict(_BOOTSTRAP_BASE_ENVIRONMENT)
    for name in _BOOTSTRAP_INHERITED_ENVIRONMENT:
        value = source.get(name)
        if value is not None:
            environment[name] = value
    environment.update(_BOOTSTRAP_OFFLINE_ENVIRONMENT)
    return environment


def _bootstrap_file_sha256(path: str) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    try:
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _bootstrap_runtime_snapshot() -> dict[str, object]:
    return {
        "executable": os.path.abspath(sys.executable),
        "executable_sha256": _bootstrap_file_sha256(os.path.realpath(sys.executable)),
        "isolated": sys.flags.isolated,
        "no_user_site": sys.flags.no_user_site,
        "ignore_environment": sys.flags.ignore_environment,
        "safe_path": getattr(sys.flags, "safe_path", False),
        "cwd": os.path.realpath(os.getcwd()),
        "script": os.path.realpath(__file__),
        "argv0": os.path.realpath(sys.argv[0]),
        "sitecustomize_loaded": "sitecustomize" in sys.modules,
        "usercustomize_loaded": "usercustomize" in sys.modules,
    }


def _bootstrap_runtime_violations(
    snapshot: dict[str, object], environment: dict[str, str]
) -> list[str]:
    expected_environment = _bootstrap_environment(environment)
    expected_environment[_VERIFICATION_REEXEC_MARKER] = _VERIFICATION_REEXEC_VALUE
    checks = {
        "executable": _BOOTSTRAP_TRAINING_PYTHON,
        "executable_sha256": _BOOTSTRAP_TRAINING_PYTHON_SHA256,
        "isolated": 1,
        "no_user_site": 1,
        "ignore_environment": 1,
        "safe_path": True,
        "cwd": _BOOTSTRAP_WORKING_DIRECTORY,
        "script": _BOOTSTRAP_SCRIPT,
        "argv0": _BOOTSTRAP_SCRIPT,
        "sitecustomize_loaded": False,
        "usercustomize_loaded": False,
    }
    violations = [
        name for name, expected in checks.items() if snapshot.get(name) != expected
    ]
    if environment != expected_environment:
        violations.append("environment")
    return violations


def _bootstrap_or_reexec(
    argv: list[str],
    *,
    execve=None,
    chdir=None,
) -> None:
    """Enter the fixed verifier runtime before importing any third-party code."""

    marker = os.environ.get(_VERIFICATION_REEXEC_MARKER)
    if marker is not None:
        snapshot = _bootstrap_runtime_snapshot()
        violations = _bootstrap_runtime_violations(snapshot, dict(os.environ))
        if marker != _VERIFICATION_REEXEC_VALUE:
            violations.insert(0, "internal_marker")
        if violations:
            raise SystemExit(
                "arm-only v1 verifier runtime refused: " + ", ".join(violations)
            )
        return

    environment = _bootstrap_environment(dict(os.environ))
    environment[_VERIFICATION_REEXEC_MARKER] = _VERIFICATION_REEXEC_VALUE
    execve = os.execve if execve is None else execve
    chdir = os.chdir if chdir is None else chdir
    try:
        chdir(_BOOTSTRAP_WORKING_DIRECTORY)
        execve(
            _BOOTSTRAP_TRAINING_PYTHON,
            [
                _BOOTSTRAP_TRAINING_PYTHON,
                "-I",
                "-s",
                _BOOTSTRAP_SCRIPT,
                *argv,
            ],
            environment,
        )
    except OSError as exc:
        raise SystemExit(f"cannot enter fixed verifier runtime: {exc}") from exc
    raise SystemExit("fixed verifier runtime exec unexpectedly returned")


if __name__ == "__main__":
    _bootstrap_or_reexec(sys.argv[1:])


import argparse
import copy
import ctypes
import importlib.machinery
import importlib.util
import json
import math
import re
import stat
import subprocess
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

DATASET_REPO_ID = "local/firefly_y6_fullcycle_v4_arm_only_v1"
DATASET_ROOT = Path(
    "/home/kk/kk_data/lerobot/datasets/local/"
    "firefly_y6_fullcycle_v4_arm_only_v1"
)
DATASET_MANIFEST_RELATIVE = Path("meta/firefly_arm_only_v1_manifest.json")
DATASET_MANIFEST_KIND = "firefly_y6_arm_only_dataset_projection"
DATASET_MANIFEST_SCHEMA_VERSION = 1
SOURCE_DATASET_REPO_ID = "local/firefly_y6_fullcycle_v4"
SOURCE_DATASET_ROOT = Path(
    "/home/kk/kk_data/lerobot/datasets/local/firefly_y6_fullcycle_v4"
)
SOURCE_FILES_MANIFEST_SHA256 = (
    "0d7fa29d13e9826b293c8f397fc4627a64dc5199e82e2bf09dedde8b9a5cfcf4"
)
SOURCE_DIRECTORIES_SHA256 = (
    "4dd16b2b62ef312e9a7542513f1ff43f8e587f70d4f71ea9fbee0fe9ea736e74"
)

PARENT_CHECKPOINT = Path(
    "/home/kk/kk_data/lerobot/outputs/train/"
    "firefly_y6_smolvla_fullcycle_v4_100k_attested_run1/"
    "checkpoints/100000/pretrained_model"
)
PARENT_ATTESTATION_FILENAME = "firefly_dataset_attestation.json"
PARENT_ATTESTATION_KIND = "firefly_smolvla_training_dataset"
PARENT_ATTESTATION_SCHEMA_VERSION = 1
PARENT_ATTESTATION_TOKEN_PREFIX = "firefly-dataset-attestation-v1:"
PARENT_INPUT_DIGEST = (
    "d07991b194092acb608ed1fbad7f73c1c4f6afc9b9a627bf5e23812f67921ca9"
)
PARENT_VLM_BUNDLE = Path(
    "/home/kk/kk_data/lerobot/assets/firefly/"
    "smolvlm2-500m-video-instruct-7b375e1b73b11138ff12fe22c8f2822d8fe03467"
)

# This is an allowlist, not merely a record of whatever happens to be present at
# launch time.  In particular, it pins the 1.2 GB learned weights and the old
# schema-1 training attestation independently.
PARENT_FILE_PINS: dict[str, tuple[int, str]] = {
    "config.json": (
        2453,
        "549fb990cbfd74bf932f3a35e95a4052b75e67c5228ffc9e27fae5e806cf6f6f",
    ),
    PARENT_ATTESTATION_FILENAME: (
        72010,
        "fef20c25818558a75ac974022766a2f305bcfd9ad9426896c1c54f96bb6e4e9a",
    ),
    "model.safetensors": (
        1197789224,
        "48bed305eacce05da7c3c1302b6b9bf84dc0ac3a42c88c7fb798dd5f718d7934",
    ),
    "policy_postprocessor.json": (
        660,
        "37d719a6584600988e6c343306d2eaee0575109f235ad500313577f73e47e8a1",
    ),
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors": (
        7624,
        "a7e4dca5b8049b8649009d3d04e638b4c82b7ca5bb2899bdc642a688bb36ff5a",
    ),
    "policy_preprocessor.json": (
        1784,
        "9633bd2433b3d4db207aa3ae60b73206db928544e31df6cec0b5aefbf75142dc",
    ),
    "policy_preprocessor_step_5_normalizer_processor.safetensors": (
        7632,
        "5918c7e30efaedb43896fd8dfceed4486bba1018f85448fffad7dbac8bbde7f3",
    ),
    "train_config.json": (
        7255,
        "1902b66faaaae222b6929a26b9229997def400d1017acb11c77f0d7876917ae0",
    ),
}

EXPECTED_TASK = (
    "Find the red block, pick it up, place it in the blue box, release it, "
    "and return the arm to its initial pose."
)
ARM_JOINT_NAMES = [f"joint_{index}.pos" for index in range(1, 7)]
EXPECTED_EPISODES = 50
EXPECTED_FRAMES = 91836
FPS = 30

CONTROLLER_CONTRACT = (
    REPOSITORY_ROOT
    / "outputs/firefly_diagnostics/"
    "controller_contract_validated_v2_arm0435_ee0165.json"
)
CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs/train_smolvla_firefly_y6_arm_only_v1_finetune_20k_run2.yaml"
)
JOB_NAME = (
    "firefly_y6_smolvla_fullcycle_v4_arm_only_v1_"
    "finetune_20k_attested_run2"
)
OUTPUT_DIR = Path("/home/kk/kk_data/lerobot/outputs/train") / JOB_NAME
TRAINING_PYTHON = Path(_BOOTSTRAP_TRAINING_PYTHON)
TRAINING_ENVIRONMENT_ROOT = TRAINING_PYTHON.parents[1]
TRAINING_WORKING_DIRECTORY = TRAINING_ENVIRONMENT_ROOT
TRAINING_PYTHON_SHA256 = _BOOTSTRAP_TRAINING_PYTHON_SHA256
EXPECTED_PACKAGE_VERSIONS = {
    "PyYAML": "6.0.3",
    "datasets": "4.8.5",
    "draccus": "0.10.0",
    "lerobot": "0.6.0",
    "numpy": "2.2.6",
    "pyarrow": "25.0.0",
    "safetensors": "0.8.0",
    "torch": "2.10.0+cu128",
    "transformers": "5.5.4",
}
LEROBOT_SITE_PACKAGES = (
    TRAINING_PYTHON.parents[1] / "lib/python3.12/site-packages"
)
LEROBOT_SEMANTIC_SOURCE_PINS = {
    "lerobot/configs/default.py": (
        "93ab849672a146123ca2f9e96a36123fbfd83e22f21e4353a468d7dca7d81712"
    ),
    "lerobot/configs/parser.py": (
        "69118561b33fbed68ad27066c26af894ad21d2caf4ed3577bad71bf32a05e036"
    ),
    "lerobot/configs/policies.py": (
        "94bfe65715b2d35ac0ddfd79e52b8f9c1cf8a28fc1d62d56cd6fb2c3908a8876"
    ),
    "lerobot/configs/types.py": (
        "1354e7d24b19d5d4970064104f76d8c85570c204bf2d09e2b4402f93a994ad99"
    ),
    "lerobot/datasets/dataset_metadata.py": (
        "57d5362571053bb67567ceb4e3fe2f879278fb4215577e9dd2d37b9bebe802c7"
    ),
    "lerobot/datasets/factory.py": (
        "8af7945ce4802e1aede4dda316f32cb57bc634482f12ee0eface4721cd748e05"
    ),
    "lerobot/configs/train.py": (
        "e947a6a0dc7006ef02bb1fb301007684c86930bc21e5a6abdfd4849a11727291"
    ),
    "lerobot/policies/factory.py": (
        "7d8a74352cb8691bd59de44a478b6f3a2bbe212583fdb04a5c19b669311b4ebe"
    ),
    "lerobot/policies/__init__.py": (
        "3b9ca3920fc536e8f69ee73cde7465501e84d940370882398611957e2f4b4b5d"
    ),
    "lerobot/policies/pretrained.py": (
        "39e30b8ff6216d71c59a1a798b5693cbb356659d9fc23597d83232c76eaf0e36"
    ),
    "lerobot/policies/smolvla/configuration_smolvla.py": (
        "2fb637cb428fa2fdf1d114646dcffaf4728216bfe5b7039d5d0cac4857ffc4e0"
    ),
    "lerobot/policies/smolvla/modeling_smolvla.py": (
        "5daaa297954acf9ae89397b571322f07bf30798e72f566e61a09393342cb1f99"
    ),
    "lerobot/policies/smolvla/__init__.py": (
        "0fcc03a2c5f20c656f26b34986a130e1a32899819e0cd68e4b4a5bd58781d0a4"
    ),
    "lerobot/policies/smolvla/processor_smolvla.py": (
        "d34594291b2ecf7801f8c27a940a998c52a7e308cbc89f2403c9cbecfb368f22"
    ),
    "lerobot/processor/__init__.py": (
        "6b7a6e1438c7492b73f68396e68fd557bf8b96568b40ae5499108811aaa07768"
    ),
    "lerobot/processor/batch_processor.py": (
        "3e5c1d21468b6852bfe5df6c0ab03a81731f580dc4f69623c0aa458c52ae88f6"
    ),
    "lerobot/processor/converters.py": (
        "e2569f4b421b8a747410cccc2a8db15dd1ef9e8baa05a6865dc4b06d780f4e30"
    ),
    "lerobot/processor/device_processor.py": (
        "69c4b0e4bed06607693a4ffbe5831bf30f13513de95973fdbde1fe09a8499553"
    ),
    "lerobot/processor/newline_task_processor.py": (
        "68ac6cabc262507bd1fd97471b07305d8129d2a826431293026cf83198ec0a5a"
    ),
    "lerobot/processor/normalize_processor.py": (
        "e9c4745d44f6f3808a84ed7ee9bc23b99ad930cc14222030e9167d4999da1012"
    ),
    "lerobot/processor/observation_processor.py": (
        "5604f209d9e0ab38c2550b6458b08bb42e3d88d57790f811960f0abb2fe84eda"
    ),
    "lerobot/scripts/lerobot_train.py": (
        "95f6d05d21205c03abbfbf4225ed1cb26b7af4826e3e8a84def3eb10ddd988dc"
    ),
    "lerobot/processor/pipeline.py": (
        "c735f552f17fd3247c07a3ab318c34fe7779339d69cb77590054122f6128e5a0"
    ),
    "lerobot/processor/rename_processor.py": (
        "10f1b190f0a1f8b4170114011dd2695308051618d6929c898062103e95499b02"
    ),
    "lerobot/processor/tokenizer_processor.py": (
        "7ba06e58a833a013421e67f74ffa992e6237df838c2b787711bf0c3156895b72"
    ),
    "lerobot/utils/feature_utils.py": (
        "53d503b8f0eda6e40be90e39504c2949f07b526a6e079d74390893be967aef19"
    ),
    "lerobot/utils/constants.py": (
        "966887445ea327a72724602b625650212a4c85148ab516e0784a41f167687e73"
    ),
}

TRAINING_STEPS = 20_000
SAVE_FREQ = 10_000
SEED = 1000
BATCH_SIZE = 1
OPTIMIZER_LR = 2e-5
SCHEDULER_WARMUP_STEPS = 500
SCHEDULER_DECAY_STEPS = 20_000
SCHEDULER_DECAY_LR = 2.5e-6
CHUNK_SIZE = 50
N_ACTION_STEPS = 50

OFFLINE_ENVIRONMENT = dict(_BOOTSTRAP_OFFLINE_ENVIRONMENT)
FIXED_BASE_ENVIRONMENT = dict(_BOOTSTRAP_BASE_ENVIRONMENT)
INHERITED_ENVIRONMENT_ALLOWLIST = _BOOTSTRAP_INHERITED_ENVIRONMENT
REMOVED_ENVIRONMENT_VARIABLES = frozenset(
    {
        "ALL_PROXY",
        "CUDA_INJECTION32_PATH",
        "CUDA_INJECTION64_PATH",
        "HF_TOKEN",
        "HF_ENDPOINT",
        "HF_INFERENCE_ENDPOINT",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_CO_STAGING",
        "LD_AUDIT",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "NO_PROXY",
        "PYTHONBREAKPOINT",
        "PYTHONHOME",
        "PYTHONINSPECT",
        "PYTHONNOUSERSITE",
        "PYTHONPATH",
        "PYTHONSAFEPATH",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
        "WANDB_API_KEY",
        "WANDB_BASE_URL",
        "all_proxy",
        "https_proxy",
        "http_proxy",
        "no_proxy",
    }
)

CHILD_ATTESTATION_FILENAME = "firefly_arm_only_v1_finetune_attestation.json"
CHILD_ATTESTATION_KIND = "firefly_smolvla_arm_only_v1_finetune"
CHILD_ATTESTATION_SCHEMA_VERSION = 1
CHILD_TOKEN_PREFIX = "firefly-arm-only-v1-finetune-inputs-v1:"
POLICY_SCHEMA_ROLE = "arm_only_6d"
AUTHORIZED_RUNTIME_ROLE = "arm6_seeking"
ARM_ONLY_OBSERVATION_LIMITATIONS = {
    "numeric_state_and_action_exclude_jaw": True,
    "source_videos_are_unchanged": True,
    "jaw_may_remain_visible_in_video_pixels": True,
    "not_a_claim_of_jaw_free_visual_observation": True,
}
RUN_COMMIT_FILENAME = "firefly_arm_only_v1_finetune_run_commit.json"
RUN_COMMIT_KIND = "firefly_smolvla_arm_only_v1_finetune_run_commit"
_STAGED_ATTESTATION_FILENAME = f".{CHILD_ATTESTATION_FILENAME}.staged"
_EXPECTED_CHECKPOINT_DIRECTORIES = {10_000: "010000", 20_000: "020000"}

_CHECKPOINT_NAME = re.compile(r"^[0-9]+$")
_HASH_CHUNK_SIZE = 1024 * 1024
_MAX_JSON_BYTES = 32 * 1024 * 1024
_REQUIRED_CHILD_FILES = {
    "config.json",
    "model.safetensors",
    "policy_postprocessor.json",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    "policy_preprocessor.json",
    "policy_preprocessor_step_5_normalizer_processor.safetensors",
    "train_config.json",
}
_VECTOR_STATISTICS = (
    "min",
    "max",
    "mean",
    "std",
    "q01",
    "q10",
    "q50",
    "q90",
    "q99",
)
PINNED_IMAGENET_PROCESSOR_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],
    "std": [[[0.229]], [[0.224]], [[0.225]]],
}
_VERIFICATION_IMPLEMENTATION_FILES = (
    Path("scripts/train_firefly_smolvla_arm_only_v1_finetune_20k_run2.py"),
    Path("scripts/materialize_firefly_y6_arm_only_v1.py"),
    Path("lerobot_robot_hexbase/firefly_training_attestation.py"),
    Path("lerobot_robot_hexbase/firefly_dataset_audit.py"),
    Path("lerobot_robot_hexbase/smolvla_base_bundle.py"),
    Path("lerobot_robot_hexbase/smolvlm_bundle.py"),
)
_VERIFICATION_DEPENDENCY_PINS = {
    Path("scripts/materialize_firefly_y6_arm_only_v1.py"): (
        "a6037746dd4087c11a2998104b7b5df00da3889cf5de157183c613a040ada2e7"
    ),
    Path("lerobot_robot_hexbase/firefly_training_attestation.py"): (
        "07b679a0e919a2badf076bbc440af3c5508fcdca72ff8d04d5b2d39184a65c44"
    ),
    Path("lerobot_robot_hexbase/firefly_dataset_audit.py"): (
        "57649ac3bfb8bbea7fd546ae3de3fa6a765a2c5677562be00b0a3d947c75ed27"
    ),
    Path("lerobot_robot_hexbase/smolvla_base_bundle.py"): (
        "10fbdf469c8d3a8cef102e07d62fe86bfcef797bde9ba2afd4cab7c9273a7a57"
    ),
    Path("lerobot_robot_hexbase/smolvlm_bundle.py"): (
        "6ab2e55bb24ebeddd9bf316bbc65e1c6b9a197f12a62478fa8d84e9054e0aab7"
    ),
}


class ArmOnlyFinetuneError(RuntimeError):
    """A controlled fine-tuning input or output failed closed validation."""


class ControlledInputChangedError(ArmOnlyFinetuneError):
    """A controlled input changed between or during captures."""


@dataclass(frozen=True)
class CoreInputCapture:
    """Inputs whose digest can safely be placed in the training command."""

    digest: str
    token: str
    payload: dict[str, Any]
    dataset_root: Path
    parent_checkpoint: Path
    controller_contract: Path
    config_path: Path


@dataclass(frozen=True)
class FinetuneInputCapture:
    """Complete capture including the exact command that was executed."""

    digest: str
    core: CoreInputCapture
    command: tuple[str, ...]
    payload: dict[str, Any]


TaskLoader = Callable[[Path], Sequence[Mapping[str, Any]]]
ParentVerifier = Callable[[Path, Mapping[str, Any]], None]
VLMValidator = Callable[[Path], None]
EnvironmentLoader = Callable[[Path], Mapping[str, Any]]
VerificationRuntimeLoader = Callable[[Path], Mapping[str, Any]]
ProjectionVerifier = Callable[[Path, Path], Mapping[str, Any]]
ChildSmokeValidator = Callable[[Path, Path], Mapping[str, Any]]
ParserPreflight = Callable[
    [Sequence[str], Path, Mapping[str, Any]],
    Mapping[str, Any],
]


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ArmOnlyFinetuneError(f"value is not strict JSON: {exc}") from exc
    return text.encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _require_no_symlink_components(path: Path, *, label: str) -> Path:
    absolute = _absolute(path)
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"{label} path component is missing: {current}") from exc
        if stat.S_ISLNK(mode):
            raise ArmOnlyFinetuneError(
                f"{label} must not contain a symbolic-link component: {current}"
            )
    return absolute


def _identity(path: Path, *, directory: bool | None = None) -> tuple[int, ...]:
    try:
        item = path.lstat()
    except FileNotFoundError as exc:
        raise ControlledInputChangedError(f"controlled input disappeared: {path}") from exc
    if stat.S_ISLNK(item.st_mode):
        raise ArmOnlyFinetuneError(f"controlled input is a symbolic link: {path}")
    if directory is True and not stat.S_ISDIR(item.st_mode):
        raise ArmOnlyFinetuneError(f"controlled input is not a directory: {path}")
    if directory is False and not stat.S_ISREG(item.st_mode):
        raise ArmOnlyFinetuneError(f"controlled input is not a regular file: {path}")
    return (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )


def _hash_regular_file(path: Path) -> tuple[int, str]:
    before = _identity(path, directory=False)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ControlledInputChangedError(f"cannot open controlled input {path}: {exc}") from exc
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        opened_identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if opened_identity != before or not stat.S_ISREG(opened.st_mode):
            raise ControlledInputChangedError(f"controlled input changed before hashing: {path}")
        while chunk := os.read(descriptor, _HASH_CHUNK_SIZE):
            digest.update(chunk)
        after = os.fstat(descriptor)
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if after_identity != before:
            raise ControlledInputChangedError(f"controlled input changed while hashing: {path}")
    finally:
        os.close(descriptor)
    if _identity(path, directory=False) != before:
        raise ControlledInputChangedError(f"controlled input was replaced while hashing: {path}")
    return before[3], digest.hexdigest()


def _descriptor_identity(descriptor: int) -> tuple[int, ...]:
    item = os.fstat(descriptor)
    return (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )


def _safetensors_structure(
    path: Path,
    *,
    label: str,
    require_finite: bool = False,
) -> list[dict[str, Any]]:
    """Parse a stable regular safetensors file without materializing its tensors."""

    path = _require_no_symlink_components(path, label=label)
    before = _identity(path, directory=False)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ArmOnlyFinetuneError(f"cannot open {label} {path}: {exc}") from exc
    try:
        if _descriptor_identity(descriptor) != before:
            raise ControlledInputChangedError(f"{label} changed before parsing: {path}")
        try:
            from safetensors import safe_open
            if require_finite:
                import torch
        except ImportError as exc:
            raise ArmOnlyFinetuneError(
                "safetensors and torch are required for checkpoint validation"
            ) from exc
        try:
            with safe_open(f"/proc/self/fd/{descriptor}", framework="pt", device="cpu") as tensors:
                keys = list(tensors.keys())
                if not keys or keys != sorted(keys) or len(keys) != len(set(keys)):
                    raise ArmOnlyFinetuneError(f"{label} must contain sorted unique tensor keys")
                structure = []
                for key in keys:
                    view = tensors.get_slice(key)
                    shape = list(view.get_shape())
                    if any(type(dimension) is not int or dimension < 0 for dimension in shape):
                        raise ArmOnlyFinetuneError(f"{label} tensor {key!r} has an invalid shape")
                    structure.append(
                        {"key": key, "dtype": str(view.get_dtype()), "shape": shape}
                    )
                    if require_finite:
                        tensor = tensors.get_tensor(key)
                        if not bool(torch.isfinite(tensor).all().item()):
                            raise ArmOnlyFinetuneError(
                                f"{label} tensor {key!r} contains a non-finite value"
                            )
                        del tensor
        except ArmOnlyFinetuneError:
            raise
        except Exception as exc:
            raise ArmOnlyFinetuneError(f"cannot parse {label} {path}: {exc}") from exc
        if _descriptor_identity(descriptor) != before:
            raise ControlledInputChangedError(f"{label} changed while parsing: {path}")
    finally:
        os.close(descriptor)
    if _identity(path, directory=False) != before:
        raise ControlledInputChangedError(f"{label} was replaced while parsing: {path}")
    return structure


def _tree_manifest(root: Path, *, label: str) -> tuple[Path, list[dict[str, Any]]]:
    root = _require_no_symlink_components(root, label=label)
    _identity(root, directory=True)
    files: list[Path] = []

    def visit(directory: Path) -> None:
        before = _identity(directory, directory=True)
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise ArmOnlyFinetuneError(f"cannot enumerate {label} {directory}: {exc}") from exc
        for entry in entries:
            path = Path(entry.path)
            try:
                mode = entry.stat(follow_symlinks=False).st_mode
            except OSError as exc:
                raise ControlledInputChangedError(f"cannot stat {label} entry {path}: {exc}") from exc
            if stat.S_ISLNK(mode):
                raise ArmOnlyFinetuneError(f"{label} contains a symbolic link: {path}")
            if stat.S_ISDIR(mode):
                visit(path)
            elif stat.S_ISREG(mode):
                files.append(path)
            else:
                raise ArmOnlyFinetuneError(f"{label} contains a non-regular entry: {path}")
        if _identity(directory, directory=True) != before:
            raise ControlledInputChangedError(f"{label} changed while being enumerated: {directory}")

    visit(root)
    manifest: list[dict[str, Any]] = []
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        size, digest = _hash_regular_file(path)
        manifest.append(
            {"path": path.relative_to(root).as_posix(), "size": size, "sha256": digest}
        )
    return root, manifest


def _read_regular_bytes(path: Path, *, max_bytes: int = _MAX_JSON_BYTES) -> bytes:
    size, _digest = _hash_regular_file(path)
    if size > max_bytes:
        raise ArmOnlyFinetuneError(f"JSON/YAML file is unexpectedly large: {path}")
    # _hash_regular_file already verifies identity and regular-file semantics;
    # verify them once more around the actual parse bytes.
    before = _identity(path, directory=False)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        data = bytearray()
        while chunk := os.read(descriptor, min(_HASH_CHUNK_SIZE, max_bytes + 1 - len(data))):
            data.extend(chunk)
            if len(data) > max_bytes:
                raise ArmOnlyFinetuneError(f"JSON/YAML file is unexpectedly large: {path}")
        after = os.fstat(descriptor)
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if after_identity != before:
            raise ControlledInputChangedError(f"file changed while being read: {path}")
    finally:
        os.close(descriptor)
    if _identity(path, directory=False) != before:
        raise ControlledInputChangedError(f"file was replaced while being read: {path}")
    return bytes(data)


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(_read_regular_bytes(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArmOnlyFinetuneError(f"{label} is not valid UTF-8 JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArmOnlyFinetuneError(f"{label} must contain a JSON object: {path}")
    # Reject NaN/Infinity accepted by Python's decoder.
    _canonical_json_bytes(value)
    return value


def _read_yaml_mapping(path: Path) -> Mapping[str, Any]:
    import yaml

    path = _require_no_symlink_components(path, label="training config")
    _identity(path, directory=False)
    try:
        value = yaml.safe_load(_read_regular_bytes(path).decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ArmOnlyFinetuneError(f"training config is not valid UTF-8 YAML: {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ArmOnlyFinetuneError("training config must contain a mapping")
    _canonical_json_bytes(value)
    return value


def _mapping(parent: Mapping[str, Any], key: str, *, label: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ArmOnlyFinetuneError(f"{label}.{key} must be a mapping")
    return value


def _exact(value: Any, expected: Any, *, label: str) -> None:
    """Recursively compare JSON-like values without Python's bool/int coercion."""

    if isinstance(expected, Mapping):
        if not isinstance(value, Mapping) or set(value) != set(expected):
            raise ArmOnlyFinetuneError(f"{label} must be exactly {expected!r}, got {value!r}")
        for key in expected:
            _exact(value[key], expected[key], label=f"{label}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(value, list) or len(value) != len(expected):
            raise ArmOnlyFinetuneError(f"{label} must be exactly {expected!r}, got {value!r}")
        for index, (actual_item, expected_item) in enumerate(zip(value, expected, strict=True)):
            _exact(actual_item, expected_item, label=f"{label}[{index}]")
        return
    if type(value) is not type(expected):
        raise ArmOnlyFinetuneError(f"{label} must be exactly {expected!r}, got {value!r}")
    if isinstance(expected, float) and (
        not math.isfinite(value) or not math.isfinite(expected)
    ):
        raise ArmOnlyFinetuneError(f"{label} must contain finite numbers")
    if value != expected:
        raise ArmOnlyFinetuneError(f"{label} must be exactly {expected!r}, got {value!r}")


def _derive_effective_processor_statistics(
    raw_statistics: Mapping[str, Any],
    features: Mapping[str, Any],
    *,
    use_imagenet_stats: bool,
) -> dict[str, Any]:
    """Mirror pinned LeRobot's runtime-only camera mean/std override."""

    if type(use_imagenet_stats) is not bool:
        raise ArmOnlyFinetuneError("use_imagenet_stats must be a strict boolean")
    if not isinstance(raw_statistics, Mapping) or not isinstance(features, Mapping):
        raise ArmOnlyFinetuneError("processor statistics require raw stats and features mappings")
    raw = copy.deepcopy(dict(raw_statistics))
    effective = copy.deepcopy(raw)
    _canonical_json_bytes(raw)

    camera_keys: list[str] = []
    depth_keys: list[str] = []
    for key, feature_value in features.items():
        if not isinstance(feature_value, Mapping) or feature_value.get("dtype") != "video":
            continue
        camera_keys.append(key)
        info = feature_value.get("info")
        if not isinstance(info, Mapping):
            raise ArmOnlyFinetuneError(f"camera feature {key} lacks an info mapping")
        is_depth_map = info.get("is_depth_map")
        if type(is_depth_map) is not bool:
            raise ArmOnlyFinetuneError(f"camera feature {key}.is_depth_map must be boolean")
        if is_depth_map:
            depth_keys.append(key)
    camera_keys.sort()
    depth_keys.sort()
    non_depth_camera_keys = sorted(set(camera_keys) - set(depth_keys))

    overrides: list[dict[str, Any]] = []
    if use_imagenet_stats:
        for camera_key in non_depth_camera_keys:
            camera_stats = effective.get(camera_key)
            if not isinstance(camera_stats, dict):
                raise ArmOnlyFinetuneError(
                    f"raw dataset statistics lack camera feature {camera_key}"
                )
            for statistic in ("mean", "std"):
                if statistic not in camera_stats:
                    raise ArmOnlyFinetuneError(
                        f"raw dataset statistics lack {camera_key}.{statistic}"
                    )
                camera_stats[statistic] = copy.deepcopy(
                    PINNED_IMAGENET_PROCESSOR_STATS[statistic]
                )
                overrides.append(
                    {
                        "feature": camera_key,
                        "statistic": statistic,
                        "source": "pinned_lerobot_imagenet_stats",
                    }
                )

    return {
        "schema_version": 1,
        "kind": "lerobot_effective_processor_statistics",
        "use_imagenet_stats": use_imagenet_stats,
        "camera_keys": camera_keys,
        "depth_keys": depth_keys,
        "non_depth_camera_keys": non_depth_camera_keys,
        "raw_statistics_sha256": _sha256_json(raw),
        "pinned_imagenet_stats": copy.deepcopy(PINNED_IMAGENET_PROCESSOR_STATS),
        "overrides": overrides,
        "effective_statistics": effective,
        "effective_statistics_sha256": _sha256_json(effective),
        "semantics": {
            "raw_dataset_metadata_is_not_rewritten": True,
            "override_only_non_depth_camera_mean_std": True,
            "all_saved_tensors_are_float32": True,
            "all_other_statistics_equal_raw_dataset_stats": True,
        },
    }


def validate_training_config(
    config_path: Path = CONFIG_PATH,
    *,
    dataset_root: Path = DATASET_ROOT,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    """Validate and hash the complete fixed human-readable config."""

    config_path = _require_no_symlink_components(config_path, label="training config")
    raw = _read_yaml_mapping(config_path)
    dataset = _mapping(raw, "dataset", label="training config")
    transforms = _mapping(dataset, "image_transforms", label="training config.dataset")
    wandb = _mapping(raw, "wandb", label="training config")
    job = _mapping(raw, "job", label="training config")

    checks = {
        "dataset.repo_id": (dataset.get("repo_id"), DATASET_REPO_ID),
        "dataset.root": (dataset.get("root"), str(_absolute(dataset_root))),
        "dataset.return_uint8": (dataset.get("return_uint8"), True),
        "dataset.image_transforms.enable": (transforms.get("enable"), False),
        "output_dir": (raw.get("output_dir"), str(_absolute(output_dir))),
        "job_name": (raw.get("job_name"), JOB_NAME),
        "seed": (raw.get("seed"), SEED),
        "batch_size": (raw.get("batch_size"), BATCH_SIZE),
        "num_workers": (raw.get("num_workers"), 4),
        "prefetch_factor": (raw.get("prefetch_factor"), 2),
        "persistent_workers": (raw.get("persistent_workers"), True),
        "steps": (raw.get("steps"), TRAINING_STEPS),
        "env_eval_freq": (raw.get("env_eval_freq"), 0),
        "eval_steps": (raw.get("eval_steps"), 0),
        "max_eval_samples": (raw.get("max_eval_samples"), 0),
        "log_freq": (raw.get("log_freq"), 20),
        "save_checkpoint": (raw.get("save_checkpoint"), True),
        "save_freq": (raw.get("save_freq"), SAVE_FREQ),
        "use_policy_training_preset": (raw.get("use_policy_training_preset"), True),
        "wandb.enable": (wandb.get("enable"), False),
        "job.target": (job.get("target"), "local"),
    }
    for label, (actual, expected) in checks.items():
        _exact(actual, expected, label=f"training config {label}")

    allowed_root = {
        "dataset",
        "output_dir",
        "job_name",
        "seed",
        "batch_size",
        "num_workers",
        "prefetch_factor",
        "persistent_workers",
        "steps",
        "env_eval_freq",
        "eval_steps",
        "max_eval_samples",
        "log_freq",
        "save_checkpoint",
        "save_freq",
        "use_policy_training_preset",
        "wandb",
        "job",
    }
    if set(raw) != allowed_root:
        raise ArmOnlyFinetuneError(
            "training config keys must be exactly the fixed allowlist; "
            f"unexpected={sorted(set(raw) - allowed_root)!r}, "
            f"missing={sorted(allowed_root - set(raw))!r}"
        )
    if set(dataset) != {"repo_id", "root", "return_uint8", "image_transforms"}:
        raise ArmOnlyFinetuneError("training config dataset keys drifted from the fixed allowlist")
    if set(transforms) != {"enable"} or set(wandb) != {"enable"} or set(job) != {"target"}:
        raise ArmOnlyFinetuneError("training config nested keys drifted from the fixed allowlist")

    size, digest = _hash_regular_file(config_path)
    return {"path": str(config_path), "size": size, "sha256": digest, "parsed": dict(raw)}


def _validate_dataset_feature(
    features: Mapping[str, Any], key: str, expected: Mapping[str, Any]
) -> None:
    actual = features.get(key)
    if not isinstance(actual, Mapping):
        raise ArmOnlyFinetuneError(f"dataset metadata feature {key!r} must be a mapping")
    for field, value in expected.items():
        _exact(actual.get(field), value, label=f"dataset feature {key}.{field}")


def _load_task_rows(dataset_root: Path) -> Sequence[Mapping[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise ArmOnlyFinetuneError("pyarrow is required for the real task metadata check") from exc
    task_path = dataset_root / "meta/tasks.parquet"
    try:
        return parquet.read_table(task_path, columns=["task_index", "task"]).to_pylist()
    except Exception as exc:
        raise ArmOnlyFinetuneError(f"cannot read task metadata {task_path}: {exc}") from exc


def _finite_numeric_shape(value: Any, *, label: str) -> tuple[int, ...]:
    if isinstance(value, list):
        if not value:
            raise ArmOnlyFinetuneError(f"{label} must not be empty")
        child_shapes = [_finite_numeric_shape(item, label=label) for item in value]
        if any(shape != child_shapes[0] for shape in child_shapes[1:]):
            raise ArmOnlyFinetuneError(f"{label} must be a rectangular numeric array")
        return (len(value), *child_shapes[0])
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArmOnlyFinetuneError(f"{label} must contain only finite numbers")
    if not math.isfinite(float(value)):
        raise ArmOnlyFinetuneError(f"{label} must contain only finite numbers")
    return ()


def _validate_arm_statistics(
    document: Mapping[str, Any],
    *,
    feature_names: Sequence[str],
) -> None:
    required_statistics = {*_VECTOR_STATISTICS, "count"}
    if set(document) != set(feature_names):
        raise ArmOnlyFinetuneError(
            "dataset statistics keys must exactly match dataset metadata features"
        )
    vector_shapes = {
        "action": (6,),
        "observation.state": (6,),
        "observation.images.scene": (3, 1, 1),
        "observation.images.wrist": (3, 1, 1),
    }
    for feature_name in feature_names:
        feature = document.get(feature_name)
        if not isinstance(feature, Mapping) or set(feature) != required_statistics:
            raise ArmOnlyFinetuneError(
                f"dataset stats.{feature_name} must contain exactly the fixed statistics"
            )
        expected_shape = vector_shapes.get(feature_name, (1,))
        for statistic in _VECTOR_STATISTICS:
            values = feature.get(statistic)
            if _finite_numeric_shape(
                values,
                label=f"dataset stats.{feature_name}.{statistic}",
            ) != expected_shape:
                raise ArmOnlyFinetuneError(
                    f"dataset stats.{feature_name}.{statistic} must have shape {expected_shape}"
                )
        count = feature.get("count")
        if (
            _finite_numeric_shape(count, label=f"dataset stats.{feature_name}.count") != (1,)
            or type(count[0]) is not int
            or count[0] <= 0
        ):
            raise ArmOnlyFinetuneError(
                f"dataset stats.{feature_name}.count must be one positive integer"
            )
        if feature_name in {"action", "observation.state"} and count != [EXPECTED_FRAMES]:
            raise ArmOnlyFinetuneError(
                f"dataset stats.{feature_name}.count must be [{EXPECTED_FRAMES}]"
            )


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ArmOnlyFinetuneError(f"{label} must be a lowercase SHA-256")
    return value


def _manifest_inventory(value: Any, *, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ArmOnlyFinetuneError(f"{label} must be a file list")
    normalized: list[dict[str, Any]] = []
    previous = ""
    for index, item in enumerate(value):
        if not isinstance(item, Mapping) or set(item) != {"path", "size", "sha256"}:
            raise ArmOnlyFinetuneError(f"{label}[{index}] must be an exact file record")
        path = item.get("path")
        size = item.get("size")
        digest = item.get("sha256")
        if (
            not isinstance(path, str)
            or not path
            or path.startswith("/")
            or ".." in Path(path).parts
            or path <= previous
        ):
            raise ArmOnlyFinetuneError(f"{label} paths must be safe, unique, and sorted")
        if type(size) is not int or size < 0:
            raise ArmOnlyFinetuneError(f"{label}[{index}].size must be a non-negative integer")
        _require_sha256(digest, label=f"{label}[{index}].sha256")
        normalized.append({"path": path, "size": size, "sha256": digest})
        previous = path
    return normalized


def _manifest_directories(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list):
        raise ArmOnlyFinetuneError(f"{label} must be a directory list")
    previous = ""
    for item in value:
        if (
            not isinstance(item, str)
            or not item
            or item.startswith("/")
            or ".." in Path(item).parts
            or item <= previous
        ):
            raise ArmOnlyFinetuneError(
                f"{label} paths must be safe, unique, and sorted"
            )
        previous = item
    return list(value)


def _dataset_directory_topology(dataset_root: Path) -> list[str]:
    directories: list[str] = []

    def visit(directory: Path) -> None:
        before = _identity(directory, directory=True)
        directories.append(directory.relative_to(dataset_root).as_posix())
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise ArmOnlyFinetuneError(
                f"cannot enumerate dataset directory topology {directory}: {exc}"
            ) from exc
        for entry in entries:
            mode = entry.stat(follow_symlinks=False).st_mode
            if stat.S_ISLNK(mode):
                raise ArmOnlyFinetuneError(
                    f"dataset directory topology contains a symbolic link: {entry.path}"
                )
            if stat.S_ISDIR(mode):
                visit(Path(entry.path))
            elif not stat.S_ISREG(mode):
                raise ArmOnlyFinetuneError(
                    f"dataset directory topology contains a non-regular entry: {entry.path}"
                )
        if _identity(directory, directory=True) != before:
            raise ControlledInputChangedError(
                f"dataset directory topology changed while enumerating: {directory}"
            )

    for name in ("meta", "data", "videos"):
        visit(dataset_root / name)
    directories.sort()
    return directories


def validate_arm_only_manifest_document(
    document: Mapping[str, Any],
    *,
    dataset_root: Path,
    destination_files: Sequence[Mapping[str, Any]],
    destination_directories: Sequence[str],
    controller_contract: Path,
) -> None:
    """Validate the independent arm-only materialization record."""

    expected_top_level = {
        "schema_version",
        "kind",
        "projection_version",
        "source",
        "destination",
        "transform",
        "controller_contract",
        "verification",
        "document_sha256",
    }
    if set(document) != expected_top_level:
        raise ArmOnlyFinetuneError("arm-only manifest top-level keys drifted")
    _exact(
        document.get("kind"),
        DATASET_MANIFEST_KIND,
        label="arm-only manifest kind",
    )
    _exact(
        document.get("schema_version"),
        DATASET_MANIFEST_SCHEMA_VERSION,
        label="arm-only manifest schema_version",
    )
    _exact(
        document.get("projection_version"),
        "exact_first_6_of_7_v1",
        label="arm-only manifest projection_version",
    )
    claimed_document_digest = _require_sha256(
        document.get("document_sha256"), label="arm-only manifest document_sha256"
    )
    unsigned = dict(document)
    unsigned.pop("document_sha256")
    _exact(
        claimed_document_digest,
        _sha256_json(unsigned),
        label="arm-only manifest canonical document_sha256",
    )

    source = _mapping(document, "source", label="arm-only manifest")
    destination = _mapping(document, "destination", label="arm-only manifest")
    common_keys = {
        "repo_id",
        "root",
        "episode_count",
        "frame_count",
        "fps",
        "task",
        "files_manifest_sha256",
        "files",
        "directories_sha256",
        "directories",
    }
    if set(source) != common_keys | {"upstream_training_input_digest"}:
        raise ArmOnlyFinetuneError("arm-only manifest source keys drifted")
    if set(destination) != common_keys:
        raise ArmOnlyFinetuneError("arm-only manifest destination keys drifted")
    source_checks = {
        "repo_id": SOURCE_DATASET_REPO_ID,
        "root": str(SOURCE_DATASET_ROOT),
        "episode_count": EXPECTED_EPISODES,
        "frame_count": EXPECTED_FRAMES,
        "fps": FPS,
        "task": EXPECTED_TASK,
        "upstream_training_input_digest": PARENT_INPUT_DIGEST,
        "files_manifest_sha256": SOURCE_FILES_MANIFEST_SHA256,
        "directories_sha256": SOURCE_DIRECTORIES_SHA256,
    }
    destination_checks = {
        "repo_id": DATASET_REPO_ID,
        "root": str(dataset_root),
        "episode_count": EXPECTED_EPISODES,
        "frame_count": EXPECTED_FRAMES,
        "fps": FPS,
        "task": EXPECTED_TASK,
    }
    for field, expected in source_checks.items():
        _exact(source.get(field), expected, label=f"arm-only manifest source.{field}")
    for field, expected in destination_checks.items():
        _exact(destination.get(field), expected, label=f"arm-only manifest destination.{field}")
    source_files = _manifest_inventory(source.get("files"), label="arm-only manifest source.files")
    if len(source_files) != 197:
        raise ArmOnlyFinetuneError("arm-only manifest source.files must contain 197 records")
    _exact(
        _sha256_json(source_files),
        source.get("files_manifest_sha256"),
        label="arm-only manifest source file-list digest",
    )
    source_directories = _manifest_directories(
        source.get("directories"), label="arm-only manifest source.directories"
    )
    _exact(
        _sha256_json(source_directories),
        source.get("directories_sha256"),
        label="arm-only manifest source directory-topology digest",
    )
    declared_destination_files = _manifest_inventory(
        destination.get("files"), label="arm-only manifest destination.files"
    )
    expected_destination_files = [dict(item) for item in destination_files]
    _exact(
        declared_destination_files,
        expected_destination_files,
        label="arm-only manifest destination file list",
    )
    if any(item["path"] == DATASET_MANIFEST_RELATIVE.as_posix() for item in declared_destination_files):
        raise ArmOnlyFinetuneError("arm-only manifest must exclude itself from destination.files")
    _exact(
        destination.get("files_manifest_sha256"),
        _sha256_json(declared_destination_files),
        label="arm-only manifest destination file-list digest",
    )
    declared_destination_directories = _manifest_directories(
        destination.get("directories"),
        label="arm-only manifest destination.directories",
    )
    _exact(
        declared_destination_directories,
        list(destination_directories),
        label="arm-only manifest destination directory topology",
    )
    _exact(
        declared_destination_directories,
        source_directories,
        label="arm-only manifest one-to-one source/destination directory topology",
    )
    _exact(
        destination.get("directories_sha256"),
        _sha256_json(declared_destination_directories),
        label="arm-only manifest destination directory-topology digest",
    )

    transform = _mapping(document, "transform", label="arm-only manifest")
    expected_transform = {
        "projected_features": {
            key: {
                "source_dtype": "float32",
                "source_shape": [7],
                "destination_dtype": "float32",
                "destination_shape": [6],
                "operation": "destination_value[i] = source_value[i] for i in 0..5",
            }
            for key in ("action", "observation.state")
        },
        "retained_feature_names": ARM_JOINT_NAMES,
        "removed_feature": {
            "name": "jaw_joint1.pos",
            "source_index": 6,
            "removed_from": ["action", "observation.state"],
            "runtime_authority": (
                "jaw must be controlled independently; this dataset emits no jaw action"
            ),
        },
        "rewritten_metadata": [
            "meta/info.json feature shapes/names",
            "meta/stats.json action/state vector statistics",
            "meta/episodes/**/*.parquet action/state vector statistics",
            "data/**/*.parquet Arrow and Hugging Face schemas",
        ],
        "unchanged": [
            "episode/frame/task/index/timestamp columns",
            "all non-action/state parquet columns",
            "all videos byte-for-byte",
            "all unrelated metadata byte-for-byte",
        ],
    }
    _exact(dict(transform), expected_transform, label="arm-only manifest transform")

    controller = _mapping(document, "controller_contract", label="arm-only manifest")
    contract_path = _require_no_symlink_components(controller_contract, label="controller contract")
    contract_size, contract_sha = _hash_regular_file(contract_path)
    contract_document = _read_json_object(contract_path, label="controller contract")
    limits = contract_document.get("effective_step_limits")
    if (
        not isinstance(limits, list)
        or len(limits) != 7
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
            for value in limits
        )
    ):
        raise ArmOnlyFinetuneError("controller contract effective_step_limits are invalid")
    expected_controller = {
        "path": str(contract_path),
        "size": contract_size,
        "sha256": contract_sha,
        "retained_arm_effective_step_limits_rad": [float(value) for value in limits[:6]],
        "removed_jaw_effective_step_limit_rad": float(limits[6]),
    }
    _exact(dict(controller), expected_controller, label="arm-only manifest controller_contract")
    expected_verification = {
        "exact_first_six_float32_bits": True,
        "one_to_one_file_paths": True,
        "copied_files_byte_identical": True,
        "destination_files_independent_regular_single_link": True,
        "source_unchanged_before_publish": True,
    }
    verification = _mapping(document, "verification", label="arm-only manifest")
    _exact(dict(verification), expected_verification, label="arm-only manifest verification")


def _verify_arm_only_projection(
    destination_root: Path, controller_contract: Path
) -> Mapping[str, Any]:
    try:
        materializer = _offline_script_import(
            "firefly_y6_arm_only_v1_materializer",
            Path("scripts/materialize_firefly_y6_arm_only_v1.py"),
        )

        return materializer.validate_arm_only_dataset(
            SOURCE_DATASET_ROOT,
            destination_root,
            controller_contract_path=controller_contract,
        )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise ArmOnlyFinetuneError(f"arm-only bit-exact projection validation failed: {exc}") from exc


def validate_dataset_metadata(
    dataset_root: Path = DATASET_ROOT,
    *,
    task_loader: TaskLoader = _load_task_rows,
    controller_contract: Path = CONTROLLER_CONTRACT,
    projection_verifier: ProjectionVerifier = _verify_arm_only_projection,
) -> dict[str, Any]:
    """Require exact v3 6-D arm features while preserving cameras and task."""

    dataset_root = _require_no_symlink_components(dataset_root, label="arm-only dataset")
    _identity(dataset_root, directory=True)
    info = _read_json_object(dataset_root / "meta/info.json", label="dataset metadata")
    checks = {
        "codebase_version": (info.get("codebase_version"), "v3.0"),
        "fps": (info.get("fps"), FPS),
        "total_episodes": (info.get("total_episodes"), EXPECTED_EPISODES),
        "total_frames": (info.get("total_frames"), EXPECTED_FRAMES),
        "total_tasks": (info.get("total_tasks"), 1),
        "robot_type": (info.get("robot_type"), "hex_zenoh_observer"),
        "splits": (info.get("splits"), {"train": "0:50"}),
    }
    for label, (actual, expected) in checks.items():
        _exact(actual, expected, label=f"dataset metadata {label}")

    features = info.get("features")
    if not isinstance(features, Mapping):
        raise ArmOnlyFinetuneError("dataset metadata features must be a mapping")
    expected_keys = {
        "action",
        "observation.state",
        "observation.images.scene",
        "observation.images.wrist",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    }
    if set(features) != expected_keys:
        raise ArmOnlyFinetuneError(
            "dataset feature keys must preserve only state/action, scene/wrist, and v3 indexes; "
            f"unexpected={sorted(set(features) - expected_keys)!r}, "
            f"missing={sorted(expected_keys - set(features))!r}"
        )
    for key in ("observation.state", "action"):
        _validate_dataset_feature(
            features,
            key,
            {"dtype": "float32", "shape": [6], "names": ARM_JOINT_NAMES},
        )
    for key in ("observation.images.scene", "observation.images.wrist"):
        _validate_dataset_feature(
            features,
            key,
            {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channels"],
            },
        )
        video_info = features[key].get("info")
        if not isinstance(video_info, Mapping):
            raise ArmOnlyFinetuneError(f"dataset feature {key}.info must be a mapping")
        for field, expected in {
            "is_depth_map": False,
            "video.height": 480,
            "video.width": 640,
            "video.fps": FPS,
            "video.channels": 3,
            "has_audio": False,
        }.items():
            _exact(video_info.get(field), expected, label=f"dataset feature {key}.info.{field}")
    for key, dtype in {
        "timestamp": "float32",
        "frame_index": "int64",
        "episode_index": "int64",
        "index": "int64",
        "task_index": "int64",
    }.items():
        _validate_dataset_feature(features, key, {"dtype": dtype, "shape": [1], "names": None})

    statistics = _read_json_object(
        dataset_root / "meta/stats.json",
        label="dataset statistics",
    )
    _validate_arm_statistics(statistics, feature_names=list(features))
    rows = list(task_loader(dataset_root))
    if rows != [{"task_index": 0, "task": EXPECTED_TASK}]:
        raise ArmOnlyFinetuneError(
            "arm-only dataset must preserve exactly the one full-cycle task row; "
            f"got {rows!r}"
        )

    files: list[dict[str, Any]] = []
    for name in ("meta", "data", "videos"):
        _root, subtree = _tree_manifest(dataset_root / name, label=f"dataset {name}")
        for entry in subtree:
            files.append({**entry, "path": f"{name}/{entry['path']}"})
    files.sort(key=lambda item: item["path"])
    if not any(entry["path"].startswith("data/") and entry["path"].endswith(".parquet") for entry in files):
        raise ArmOnlyFinetuneError("arm-only dataset contains no data parquet")
    camera_prefixes = {
        "videos/observation.images.scene/",
        "videos/observation.images.wrist/",
    }
    found_prefixes = {
        prefix for prefix in camera_prefixes if any(entry["path"].startswith(prefix) for entry in files)
    }
    if found_prefixes != camera_prefixes:
        raise ArmOnlyFinetuneError("arm-only dataset must preserve scene and wrist MP4 trees")

    arm_manifest_path = dataset_root / DATASET_MANIFEST_RELATIVE
    arm_manifest = _read_json_object(arm_manifest_path, label="arm-only dataset manifest")
    arm_manifest_size, arm_manifest_sha = _hash_regular_file(arm_manifest_path)
    destination_files = [
        entry for entry in files if entry["path"] != DATASET_MANIFEST_RELATIVE.as_posix()
    ]
    destination_directories = _dataset_directory_topology(dataset_root)
    validate_arm_only_manifest_document(
        arm_manifest,
        dataset_root=dataset_root,
        destination_files=destination_files,
        destination_directories=destination_directories,
        controller_contract=controller_contract,
    )
    projection_verification = dict(projection_verifier(dataset_root, controller_contract))
    _canonical_json_bytes(projection_verification)
    _exact(
        projection_verification.get("safe_for_arm_only_training"),
        True,
        label="arm-only materializer verification",
    )

    return {
        "repo_id": DATASET_REPO_ID,
        "root": str(dataset_root),
        "metadata": {
            "codebase_version": "v3.0",
            "fps": FPS,
            "total_episodes": EXPECTED_EPISODES,
            "total_frames": EXPECTED_FRAMES,
            "total_tasks": 1,
            "features": info["features"],
            "task": EXPECTED_TASK,
            "statistics": statistics,
        },
        "arm_only_manifest": {
            "path": DATASET_MANIFEST_RELATIVE.as_posix(),
            "size": arm_manifest_size,
            "sha256": arm_manifest_sha,
            "document": arm_manifest,
        },
        "projection_verification": projection_verification,
        "directories": destination_directories,
        "directories_sha256": _sha256_json(destination_directories),
        "files": files,
        "files_sha256": _sha256_json(files),
    }


def _validate_parent_attestation_document(document: Mapping[str, Any]) -> None:
    _exact(document.get("kind"), PARENT_ATTESTATION_KIND, label="parent attestation kind")
    _exact(
        document.get("schema_version"),
        PARENT_ATTESTATION_SCHEMA_VERSION,
        label="parent attestation schema_version",
    )
    inputs = document.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ArmOnlyFinetuneError("parent attestation inputs must be an object")
    digest = _sha256_json(inputs)
    _exact(document.get("input_digest"), digest, label="parent attestation input_digest")
    _exact(
        document.get("token"),
        f"{PARENT_ATTESTATION_TOKEN_PREFIX}{digest}",
        label="parent attestation token",
    )
    verification = document.get("verification")
    if not isinstance(verification, Mapping):
        raise ArmOnlyFinetuneError("parent attestation verification must be an object")
    _exact(verification.get("training_returncode"), 0, label="parent training returncode")
    _exact(
        verification.get("inputs_unchanged_after_training"),
        True,
        label="parent inputs_unchanged_after_training",
    )
    _exact(
        verification.get("postflight_input_digest"),
        digest,
        label="parent postflight_input_digest",
    )


def _offline_package_import(module_name: str):
    _validate_verification_dependency_pins()
    package_name = "lerobot_robot_hexbase"
    if package_name not in sys.modules:
        package_dir = REPOSITORY_ROOT / package_name
        package = types.ModuleType(package_name)
        package.__file__ = str(package_dir / "__init__.py")
        package.__package__ = package_name
        package.__path__ = [str(package_dir)]
        package.__spec__ = importlib.machinery.ModuleSpec(package_name, loader=None, is_package=True)
        sys.modules[package_name] = package
    return __import__(f"{package_name}.{module_name}", fromlist=[module_name])


def _validate_verification_dependency_pins() -> None:
    for relative, expected_digest in _VERIFICATION_DEPENDENCY_PINS.items():
        path = _require_no_symlink_components(
            REPOSITORY_ROOT / relative,
            label="pinned verification dependency",
        )
        _size, digest = _hash_regular_file(path)
        if digest != expected_digest:
            raise ArmOnlyFinetuneError(
                f"pinned verification dependency drifted: {relative.as_posix()}"
            )


def _offline_script_import(module_name: str, relative: Path):
    _validate_verification_dependency_pins()
    source = _require_no_symlink_components(
        REPOSITORY_ROOT / relative,
        label="pinned verification script",
    )
    loader = importlib.machinery.SourceFileLoader(module_name, str(source))
    spec = importlib.util.spec_from_loader(module_name, loader)
    if spec is None:
        raise ArmOnlyFinetuneError(f"cannot construct loader for {relative.as_posix()}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        loader.exec_module(module)
    except Exception:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise
    return module


def _verify_parent_training_attestation(
    parent_checkpoint: Path, document: Mapping[str, Any]
) -> None:
    """Re-run the unchanged schema-1 verifier against the original dataset."""

    module = _offline_package_import("firefly_training_attestation")
    inputs = document["inputs"]
    dataset = inputs.get("dataset")
    contract = inputs.get("controller_contract")
    expected_task = inputs.get("expected_task")
    if not isinstance(dataset, Mapping) or not isinstance(contract, Mapping):
        raise ArmOnlyFinetuneError("parent attestation lacks dataset/controller bindings")
    try:
        current = module.capture_training_inputs(
            Path(dataset["root"]),
            dataset["repo_id"],
            controller_contract=Path(contract["path"]),
            expected_task=expected_task,
        )
        module.verify_training_attestation(parent_checkpoint, current)
    except (KeyError, OSError, ValueError, RuntimeError) as exc:
        raise ArmOnlyFinetuneError(f"parent schema-1 training attestation failed: {exc}") from exc


def _validate_vlm_bundle(vlm_bundle: Path) -> None:
    module = _offline_package_import("smolvlm_bundle")
    try:
        canonical, _manifest = module.validate_materialized_smolvlm_bundle(vlm_bundle)
    except (OSError, ValueError) as exc:
        raise ArmOnlyFinetuneError(f"parent VLM bundle failed validation: {exc}") from exc
    if canonical != vlm_bundle:
        raise ArmOnlyFinetuneError(f"parent VLM bundle canonical path drifted: {canonical}")


def _policy_feature_shapes(document: Mapping[str, Any], *, label: str, dimension: int) -> None:
    input_features = document.get("input_features")
    output_features = document.get("output_features")
    if not isinstance(input_features, Mapping) or not isinstance(output_features, Mapping):
        raise ArmOnlyFinetuneError(f"{label} lacks policy input/output features")
    expected_inputs = {
        "observation.state": {"type": "STATE", "shape": [dimension]},
        "observation.images.scene": {"type": "VISUAL", "shape": [3, 480, 640]},
        "observation.images.wrist": {"type": "VISUAL", "shape": [3, 480, 640]},
    }
    expected_outputs = {"action": {"type": "ACTION", "shape": [dimension]}}
    if input_features != expected_inputs or output_features != expected_outputs:
        raise ArmOnlyFinetuneError(
            f"{label} features must be exactly state[{dimension}], scene+wrist, action[{dimension}]"
        )


def _validate_parent_configs(
    parent_checkpoint: Path,
    document: Mapping[str, Any],
    *,
    expected_vlm_bundle: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    policy_config = _read_json_object(parent_checkpoint / "config.json", label="parent policy config")
    train_config = _read_json_object(
        parent_checkpoint / "train_config.json", label="parent train_config"
    )
    _exact(policy_config.get("type"), "smolvla", label="parent policy type")
    _policy_feature_shapes(policy_config, label="parent policy config", dimension=7)
    for field, expected in {
        "chunk_size": CHUNK_SIZE,
        "n_action_steps": N_ACTION_STEPS,
        "freeze_vision_encoder": True,
        "train_expert_only": True,
        "train_state_proj": True,
        "load_vlm_weights": False,
        "vlm_model_name": str(expected_vlm_bundle),
    }.items():
        _exact(policy_config.get(field), expected, label=f"parent policy config {field}")
    dataset = _mapping(train_config, "dataset", label="parent train_config")
    train_policy = _mapping(train_config, "policy", label="parent train_config")
    wandb = _mapping(train_config, "wandb", label="parent train_config")
    _exact(train_config.get("steps"), 100_000, label="parent train_config steps")
    _exact(
        dataset.get("repo_id"),
        "local/firefly_y6_fullcycle_v4",
        label="parent train_config dataset.repo_id",
    )
    _exact(train_policy.get("type"), "smolvla", label="parent train_config policy.type")
    _policy_feature_shapes(train_policy, label="parent train_config policy", dimension=7)
    _exact(
        dict(train_policy),
        policy_config,
        label="parent train_config.policy exact saved policy config",
    )
    _exact(
        wandb.get("notes"),
        document.get("token"),
        label="parent train_config wandb.notes",
    )
    return policy_config, train_config


def validate_parent_checkpoint(
    parent_checkpoint: Path = PARENT_CHECKPOINT,
    *,
    expected_files: Mapping[str, tuple[int, str]] = PARENT_FILE_PINS,
    expected_input_digest: str = PARENT_INPUT_DIGEST,
    expected_vlm_bundle: Path = PARENT_VLM_BUNDLE,
    parent_verifier: ParentVerifier = _verify_parent_training_attestation,
    vlm_validator: VLMValidator = _validate_vlm_bundle,
) -> dict[str, Any]:
    """Validate the known 100k lineage and bind every parent file."""

    parent_checkpoint, files = _tree_manifest(parent_checkpoint, label="parent checkpoint")
    observed = {entry["path"]: (entry["size"], entry["sha256"]) for entry in files}
    if observed != dict(expected_files):
        missing = sorted(set(expected_files) - set(observed))
        unexpected = sorted(set(observed) - set(expected_files))
        mismatched = sorted(
            name for name in set(expected_files) & set(observed) if observed[name] != expected_files[name]
        )
        raise ArmOnlyFinetuneError(
            "parent checkpoint does not match the pinned 100k file manifest; "
            f"missing={missing!r}, unexpected={unexpected!r}, mismatched={mismatched!r}"
        )
    attestation = _read_json_object(
        parent_checkpoint / PARENT_ATTESTATION_FILENAME,
        label="parent training attestation",
    )
    _validate_parent_attestation_document(attestation)
    _exact(
        attestation.get("input_digest"),
        expected_input_digest,
        label="pinned parent training input digest",
    )
    parent_verifier(parent_checkpoint, attestation)

    expected_vlm_bundle = _require_no_symlink_components(
        expected_vlm_bundle, label="parent VLM bundle"
    )
    policy_config, train_config = _validate_parent_configs(
        parent_checkpoint,
        attestation,
        expected_vlm_bundle=expected_vlm_bundle,
    )
    vlm_validator(expected_vlm_bundle)
    _vlm_root, vlm_files = _tree_manifest(expected_vlm_bundle, label="parent VLM bundle")
    model_structure = _safetensors_structure(
        parent_checkpoint / "model.safetensors",
        label="parent model safetensors",
    )
    return {
        "path": str(parent_checkpoint),
        "checkpoint_step": 100_000,
        "files": files,
        "files_sha256": _sha256_json(files),
        "policy_config": policy_config,
        "train_config": train_config,
        "model_safetensors": {
            "structure": model_structure,
            "structure_sha256": _sha256_json(model_structure),
        },
        "training_attestation": {
            "filename": PARENT_ATTESTATION_FILENAME,
            "sha256": observed[PARENT_ATTESTATION_FILENAME][1],
            "input_digest": attestation["input_digest"],
            "token": attestation["token"],
            "verified_with_legacy_schema1_verifier": True,
        },
        "vlm_bundle": {
            "path": str(expected_vlm_bundle),
            "files": vlm_files,
            "files_sha256": _sha256_json(vlm_files),
        },
    }


def _capture_environment(training_python: Path) -> Mapping[str, Any]:
    training_python = _require_no_symlink_components(training_python, label="training Python")
    working_directory = _require_no_symlink_components(
        TRAINING_WORKING_DIRECTORY,
        label="training working directory",
    )
    _identity(working_directory, directory=True)
    size, digest = _hash_regular_file(training_python)
    if training_python == TRAINING_PYTHON and digest != TRAINING_PYTHON_SHA256:
        raise ArmOnlyFinetuneError("fixed training Python executable hash drifted")
    probe = """
import importlib
import json
import os
import platform
import sys
from importlib.metadata import version
from lerobot.utils.constants import IMAGENET_STATS

distributions = ["lerobot", "torch", "transformers", "safetensors", "datasets", "pyarrow", "PyYAML", "draccus", "numpy"]
modules = {
    "lerobot": "lerobot",
    "torch": "torch",
    "transformers": "transformers",
    "safetensors": "safetensors",
    "datasets": "datasets",
    "pyarrow": "pyarrow",
    "PyYAML": "yaml",
    "draccus": "draccus",
    "numpy": "numpy",
}
print(json.dumps({
    "python_version": sys.version,
    "platform": platform.platform(),
    "packages": {name: version(name) for name in distributions},
    "module_files": {
        name: os.path.realpath(importlib.import_module(module).__file__)
        for name, module in modules.items()
    },
    "imagenet_stats": IMAGENET_STATS,
    "sitecustomize_loaded": "sitecustomize" in sys.modules,
    "usercustomize_loaded": "usercustomize" in sys.modules,
    "sys_path": sys.path,
}, sort_keys=True))
"""
    environment = build_training_environment()
    try:
        result = subprocess.run(
            [str(training_python), "-I", "-s", "-c", probe],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            cwd=working_directory,
        )
    except OSError as exc:
        raise ArmOnlyFinetuneError(f"cannot probe fixed training environment: {exc}") from exc
    if result.returncode != 0:
        raise ArmOnlyFinetuneError(
            f"fixed training environment probe failed with status {result.returncode}: {result.stderr}"
        )
    try:
        details = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ArmOnlyFinetuneError("fixed training environment probe returned invalid JSON") from exc
    if not isinstance(details, Mapping):
        raise ArmOnlyFinetuneError("fixed training environment probe returned a non-object")
    packages = details.get("packages")
    if packages != EXPECTED_PACKAGE_VERSIONS:
        raise ArmOnlyFinetuneError(
            "fixed training package versions drifted from the verified environment; "
            f"got {packages!r}"
        )
    _exact(
        details.get("imagenet_stats"),
        PINNED_IMAGENET_PROCESSOR_STATS,
        label="pinned LeRobot ImageNet processor statistics",
    )
    if details.get("sitecustomize_loaded") is not False:
        raise ArmOnlyFinetuneError("isolated training Python loaded sitecustomize")
    if details.get("usercustomize_loaded") is not False:
        raise ArmOnlyFinetuneError("isolated training Python loaded usercustomize")
    expected_sys_path = [
        str(TRAINING_ENVIRONMENT_ROOT / "lib/python312.zip"),
        str(TRAINING_ENVIRONMENT_ROOT / "lib/python3.12"),
        str(TRAINING_ENVIRONMENT_ROOT / "lib/python3.12/lib-dynload"),
        str(LEROBOT_SITE_PACKAGES),
    ]
    if details.get("sys_path") != expected_sys_path:
        raise ArmOnlyFinetuneError(
            f"isolated training sys.path drifted: {details.get('sys_path')!r}"
        )
    module_files = details.get("module_files")
    if not isinstance(module_files, Mapping) or set(module_files) != set(
        EXPECTED_PACKAGE_VERSIONS
    ):
        raise ArmOnlyFinetuneError("isolated training module origins are incomplete")
    site_packages_resolved = LEROBOT_SITE_PACKAGES.resolve(strict=True)
    for name, value in module_files.items():
        if not isinstance(value, str):
            raise ArmOnlyFinetuneError(f"isolated module {name} has no regular origin")
        origin = Path(value)
        try:
            origin.relative_to(site_packages_resolved)
        except ValueError as exc:
            raise ArmOnlyFinetuneError(
                f"isolated module {name} resolved outside pinned site-packages: {origin}"
            ) from exc
        _require_no_symlink_components(origin, label=f"isolated module {name}")
        _identity(origin, directory=False)
    semantic_sources: list[dict[str, Any]] = []
    for relative, expected_digest in LEROBOT_SEMANTIC_SOURCE_PINS.items():
        source = _require_no_symlink_components(
            LEROBOT_SITE_PACKAGES / relative,
            label="LeRobot continuation source",
        )
        source_size, source_digest = _hash_regular_file(source)
        if source_digest != expected_digest:
            raise ArmOnlyFinetuneError(
                f"verified LeRobot continuation semantics drifted: {relative}"
            )
        semantic_sources.append(
            {
                "path": relative,
                "size": source_size,
                "sha256": source_digest,
            }
        )
    environment_records = [
        {
            "name": name,
            "value_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
        }
        for name, value in sorted(environment.items())
    ]
    return {
        "python_executable": str(training_python),
        "python_executable_size": size,
        "python_executable_sha256": digest,
        "python_flags": ["-I", "-s"],
        "working_directory": str(working_directory),
        "offline_overrides": dict(sorted(OFFLINE_ENVIRONMENT.items())),
        "removed_environment_variables": sorted(REMOVED_ENVIRONMENT_VARIABLES),
        "environment_variable_names": [item["name"] for item in environment_records],
        "environment_sha256": _sha256_json(environment_records),
        "lerobot_semantic_sources": semantic_sources,
        "lerobot_semantic_sources_sha256": _sha256_json(semantic_sources),
        "semantic_source_scope": (
            "pinned CLI parsing, dataset construction and runtime ImageNet statistics, "
            "dataset feature projection, SmolVLA continuation, processor load, and child "
            "smoke paths; not a whole-environment source manifest"
        ),
        **dict(details),
    }


def _capture_current_verification_runtime(training_python: Path) -> Mapping[str, Any]:
    """Fail closed unless this validator is the pinned isolated process."""

    import importlib
    import platform
    from importlib.metadata import version

    training_python = _require_no_symlink_components(
        training_python,
        label="verification runtime Python",
    )
    _exact(training_python, TRAINING_PYTHON, label="verification runtime Python path")
    snapshot = _bootstrap_runtime_snapshot()
    environment = dict(os.environ)
    violations = _bootstrap_runtime_violations(snapshot, environment)
    if violations:
        raise ArmOnlyFinetuneError(
            "current verification runtime is not the pinned isolated process: "
            + ", ".join(violations)
        )
    expected_orig_argv_prefix = [
        str(TRAINING_PYTHON),
        "-I",
        "-s",
        _BOOTSTRAP_SCRIPT,
    ]
    original_argv = list(getattr(sys, "orig_argv", []))
    if original_argv[:4] != expected_orig_argv_prefix:
        raise ArmOnlyFinetuneError("verification runtime original argv prefix drifted")
    invocation = original_argv[4:]
    allowed_invocations = [[], ["--preflight"], ["--verify-run"]]
    if invocation not in allowed_invocations:
        raise ArmOnlyFinetuneError(
            f"verification runtime invocation mode is not capturable: {invocation!r}"
        )
    expected_sys_path = [
        str(TRAINING_ENVIRONMENT_ROOT / "lib/python312.zip"),
        str(TRAINING_ENVIRONMENT_ROOT / "lib/python3.12"),
        str(TRAINING_ENVIRONMENT_ROOT / "lib/python3.12/lib-dynload"),
        str(LEROBOT_SITE_PACKAGES),
    ]
    _exact(sys.path, expected_sys_path, label="verification runtime sys.path")

    distributions = {
        "lerobot": "lerobot",
        "torch": "torch",
        "transformers": "transformers",
        "safetensors": "safetensors",
        "datasets": "datasets",
        "pyarrow": "pyarrow",
        "PyYAML": "yaml",
        "draccus": "draccus",
        "numpy": "numpy",
    }
    packages = {name: version(name) for name in distributions}
    _exact(packages, EXPECTED_PACKAGE_VERSIONS, label="verification runtime packages")
    site_packages = LEROBOT_SITE_PACKAGES.resolve(strict=True)
    module_files: dict[str, str] = {}
    for name, module_name in distributions.items():
        module = importlib.import_module(module_name)
        origin_value = getattr(module, "__file__", None)
        if not isinstance(origin_value, str):
            raise ArmOnlyFinetuneError(
                f"verification runtime module {name} has no regular origin"
            )
        origin = Path(os.path.realpath(origin_value))
        try:
            origin.relative_to(site_packages)
        except ValueError as exc:
            raise ArmOnlyFinetuneError(
                f"verification runtime module {name} is outside pinned site-packages: {origin}"
            ) from exc
        _require_no_symlink_components(origin, label=f"verification runtime module {name}")
        _identity(origin, directory=False)
        module_files[name] = str(origin)

    _validate_verification_dependency_pins()
    python_size, python_digest = _hash_regular_file(training_python)
    _exact(
        python_digest,
        TRAINING_PYTHON_SHA256,
        label="verification runtime Python executable digest",
    )
    environment_records = [
        {
            "name": name,
            "value_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
        }
        for name, value in sorted(environment.items())
    ]
    return {
        "python_executable": str(training_python),
        "python_executable_size": python_size,
        "python_executable_sha256": python_digest,
        "python_version": sys.version,
        "platform": platform.platform(),
        "flags": {
            "isolated": snapshot["isolated"],
            "no_user_site": snapshot["no_user_site"],
            "ignore_environment": snapshot["ignore_environment"],
            "safe_path": snapshot["safe_path"],
        },
        "working_directory": snapshot["cwd"],
        "script": snapshot["script"],
        "argv0": snapshot["argv0"],
        "original_argv_prefix": expected_orig_argv_prefix,
        "allowed_invocations": allowed_invocations,
        "current_invocation_validated": True,
        "sitecustomize_loaded": snapshot["sitecustomize_loaded"],
        "usercustomize_loaded": snapshot["usercustomize_loaded"],
        "sys_path": list(sys.path),
        "packages": packages,
        "module_files": module_files,
        "environment_variable_names": [item["name"] for item in environment_records],
        "environment_sha256": _sha256_json(environment_records),
        "repository_import_path_injected": False,
        "dependency_pins_sha256": _sha256_json(
            [
                {"path": path.as_posix(), "sha256": digest}
                for path, digest in _VERIFICATION_DEPENDENCY_PINS.items()
            ]
        ),
    }


def _run_cli_parser_preflight(
    command: Sequence[str],
    training_python: Path,
    expected: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Parse the complete training argv and derive 6-D features without training."""

    del expected  # The caller performs the independent exact comparison.
    if list(command[:5]) != [
        str(_absolute(training_python)),
        "-I",
        "-s",
        "-m",
        "lerobot.scripts.lerobot_train",
    ]:
        raise ArmOnlyFinetuneError("parser preflight command prefix drifted")
    output_dir = Path(_argument(command, "output_dir"))
    validation_output = output_dir.with_name(f".{output_dir.name}.parser-preflight-must-not-exist")
    if validation_output.exists() or validation_output.is_symlink():
        raise ArmOnlyFinetuneError(f"parser preflight sentinel path exists: {validation_output}")
    program = r'''
import json
import sys
from pathlib import Path

from lerobot.configs import FeatureType
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
import lerobot.policies  # Register the same policy choices imported by lerobot_train.
from lerobot.utils.feature_utils import dataset_to_policy_features

validation_output = Path(sys.argv[1])
training_argv = sys.argv[2:]
sys.argv = ["firefly-parser-preflight", *training_argv]

def inspect_config(config: TrainPipelineConfig):
    bound_output = config.output_dir
    config.output_dir = validation_output
    config.validate()
    config.output_dir = bound_output
    metadata = LeRobotDatasetMetadata(
        config.dataset.repo_id,
        root=config.dataset.root,
        revision=config.dataset.revision,
        force_cache_sync=False,
    )
    features = dataset_to_policy_features(metadata.features)
    config.policy.output_features = {
        key: feature for key, feature in features.items() if feature.type is FeatureType.ACTION
    }
    if not config.policy.input_features:
        config.policy.input_features = {
            key: feature for key, feature in features.items()
            if key not in config.policy.output_features
        }
    print("FIREFLY_PARSER_PREFLIGHT=" + json.dumps(config.to_dict(), sort_keys=True))

parser.wrap()(inspect_config)()
'''
    subprocess_command = [
        str(_absolute(training_python)),
        "-I",
        "-s",
        "-c",
        program,
        str(validation_output),
        *command[5:],
    ]
    try:
        result = subprocess.run(
            subprocess_command,
            check=False,
            capture_output=True,
            text=True,
            env=build_training_environment(),
            cwd=TRAINING_WORKING_DIRECTORY,
            timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ArmOnlyFinetuneError(f"isolated CLI parser preflight could not run: {exc}") from exc
    if result.returncode != 0:
        raise ArmOnlyFinetuneError(
            "isolated CLI parser preflight failed with status "
            f"{result.returncode}: {result.stderr[-4000:]}"
        )
    prefix = "FIREFLY_PARSER_PREFLIGHT="
    payloads = [line[len(prefix) :] for line in result.stdout.splitlines() if line.startswith(prefix)]
    if len(payloads) != 1:
        raise ArmOnlyFinetuneError("isolated CLI parser preflight returned no unique result")
    try:
        parsed = json.loads(payloads[0])
    except json.JSONDecodeError as exc:
        raise ArmOnlyFinetuneError("isolated CLI parser preflight returned invalid JSON") from exc
    if not isinstance(parsed, Mapping):
        raise ArmOnlyFinetuneError("isolated CLI parser preflight returned a non-object")
    return dict(parsed)


def capture_core_inputs(
    *,
    dataset_root: Path = DATASET_ROOT,
    parent_checkpoint: Path = PARENT_CHECKPOINT,
    controller_contract: Path = CONTROLLER_CONTRACT,
    config_path: Path = CONFIG_PATH,
    output_dir: Path = OUTPUT_DIR,
    training_python: Path = TRAINING_PYTHON,
    task_loader: TaskLoader = _load_task_rows,
    projection_verifier: ProjectionVerifier = _verify_arm_only_projection,
    parent_verifier: ParentVerifier = _verify_parent_training_attestation,
    vlm_validator: VLMValidator = _validate_vlm_bundle,
    verification_runtime_loader: VerificationRuntimeLoader = (
        _capture_current_verification_runtime
    ),
    environment_loader: EnvironmentLoader = _capture_environment,
    parser_preflight: ParserPreflight = _run_cli_parser_preflight,
    expected_parent_files: Mapping[str, tuple[int, str]] = PARENT_FILE_PINS,
    expected_parent_input_digest: str = PARENT_INPUT_DIGEST,
    expected_vlm_bundle: Path = PARENT_VLM_BUNDLE,
) -> CoreInputCapture:
    verification_runtime = dict(verification_runtime_loader(training_python))
    _canonical_json_bytes(verification_runtime)
    dataset = validate_dataset_metadata(
        dataset_root,
        task_loader=task_loader,
        controller_contract=controller_contract,
        projection_verifier=projection_verifier,
    )
    parent = validate_parent_checkpoint(
        parent_checkpoint,
        expected_files=expected_parent_files,
        expected_input_digest=expected_parent_input_digest,
        expected_vlm_bundle=expected_vlm_bundle,
        parent_verifier=parent_verifier,
        vlm_validator=vlm_validator,
    )
    config = validate_training_config(
        config_path,
        dataset_root=dataset_root,
        output_dir=output_dir,
    )
    raw_statistics = dataset["metadata"].get("statistics")
    features = dataset["metadata"].get("features")
    parent_dataset_config = _mapping(
        parent["train_config"],
        "dataset",
        label="captured parent train_config",
    )
    use_imagenet_stats = parent_dataset_config.get("use_imagenet_stats")
    processor_statistics_contract = _derive_effective_processor_statistics(
        raw_statistics,
        features,
        use_imagenet_stats=use_imagenet_stats,
    )
    controller_contract = _require_no_symlink_components(
        controller_contract, label="controller contract"
    )
    contract_size, contract_sha = _hash_regular_file(controller_contract)
    environment = dict(environment_loader(training_python))
    _canonical_json_bytes(environment)
    implementation_files: list[dict[str, Any]] = []
    for relative in _VERIFICATION_IMPLEMENTATION_FILES:
        path = _require_no_symlink_components(
            REPOSITORY_ROOT / relative,
            label="verification implementation",
        )
        size, digest = _hash_regular_file(path)
        implementation_files.append(
            {"path": relative.as_posix(), "size": size, "sha256": digest}
        )

    payload: dict[str, Any] = {
        "schema_version": CHILD_ATTESTATION_SCHEMA_VERSION,
        "kind": CHILD_ATTESTATION_KIND,
        "verification_runtime": verification_runtime,
        "dataset": dataset,
        "processor_statistics_contract": processor_statistics_contract,
        "parent_checkpoint": parent,
        "controller_contract": {
            "path": str(controller_contract),
            "size": contract_size,
            "sha256": contract_sha,
        },
        "training_config": config,
        "training_contract": {
            "policy_schema_role": POLICY_SCHEMA_ROLE,
            "authorized_runtime_role": AUTHORIZED_RUNTIME_ROLE,
            "observation_limitations": ARM_ONLY_OBSERVATION_LIMITATIONS,
            "output_dir": str(_absolute(output_dir)),
            "job_name": JOB_NAME,
            "steps": TRAINING_STEPS,
            "save_freq": SAVE_FREQ,
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "optimizer_lr": OPTIMIZER_LR,
            "scheduler_warmup_steps": SCHEDULER_WARMUP_STEPS,
            "scheduler_decay_steps": SCHEDULER_DECAY_STEPS,
            "scheduler_decay_lr": SCHEDULER_DECAY_LR,
            "n_obs_steps": 1,
            "chunk_size": CHUNK_SIZE,
            "n_action_steps": N_ACTION_STEPS,
            "use_amp": False,
            "freeze_vision_encoder": True,
            "train_expert_only": True,
            "train_state_proj": True,
            "image_augmentation": False,
            "use_imagenet_stats": use_imagenet_stats,
            "processor_statistics_semantics": (
                "raw dataset stats remain bound and unchanged; pinned LeRobot replaces only "
                "non-depth camera mean/std with pinned ImageNet float32 values at runtime"
            ),
            "resume": False,
            "library_offline_flags": True,
            "network_control_scope": "library_offline_flags_only",
        },
        "environment": environment,
        "verification_implementation": {
            "files": implementation_files,
            "files_sha256": _sha256_json(implementation_files),
        },
    }
    provisional_digest = _sha256_json(payload)
    provisional = CoreInputCapture(
        digest=provisional_digest,
        token=f"{CHILD_TOKEN_PREFIX}{provisional_digest}",
        payload=payload,
        dataset_root=_absolute(dataset_root),
        parent_checkpoint=_absolute(parent_checkpoint),
        controller_contract=controller_contract,
        config_path=_absolute(config_path),
    )
    preflight_command = build_training_command(
        provisional,
        output_dir=output_dir,
        training_python=training_python,
    )
    provisional_final = finalize_capture(provisional, preflight_command)
    expected_parsed = _expected_child_train_config(
        provisional_final,
        output_dir=output_dir,
    )
    parsed = dict(parser_preflight(preflight_command, training_python, expected_parsed))
    _exact(parsed, expected_parsed, label="isolated full-argv parser preflight")
    normalized_parsed = copy.deepcopy(parsed)
    normalized_parsed["wandb"]["notes"] = "<BOUND_CORE_TOKEN>"
    normalized_command = [
        "--wandb.notes=<BOUND_CORE_TOKEN>"
        if item.startswith("--wandb.notes=")
        else item
        for item in preflight_command
    ]
    payload["cli_parser_preflight"] = {
        "isolated_python_flags": ["-I", "-s"],
        "library_offline_flags": True,
        "full_argv_normalized": normalized_command,
        "full_argv_normalized_sha256": _sha256_json(normalized_command),
        "parsed_train_config_normalized": normalized_parsed,
        "parsed_train_config_normalized_sha256": _sha256_json(normalized_parsed),
        "dataset_features_derived_before_training": True,
        "state_dimension": 6,
        "action_dimension": 6,
    }
    digest = _sha256_json(payload)
    return CoreInputCapture(
        digest=digest,
        token=f"{CHILD_TOKEN_PREFIX}{digest}",
        payload=payload,
        dataset_root=_absolute(dataset_root),
        parent_checkpoint=_absolute(parent_checkpoint),
        controller_contract=controller_contract,
        config_path=_absolute(config_path),
    )


def build_training_command(
    capture: CoreInputCapture,
    *,
    output_dir: Path = OUTPUT_DIR,
    training_python: Path = TRAINING_PYTHON,
) -> list[str]:
    """Build the closed command; callers cannot append policy/network overrides."""

    output_dir = _absolute(output_dir)
    training_python = _absolute(training_python)
    vlm_bundle = capture.payload["parent_checkpoint"]["vlm_bundle"]["path"]
    return [
        str(training_python),
        "-I",
        "-s",
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--config_path={capture.config_path}",
        f"--dataset.repo_id={DATASET_REPO_ID}",
        f"--dataset.root={capture.dataset_root}",
        "--dataset.episodes=null",
        "--dataset.revision=null",
        "--dataset.streaming=false",
        "--dataset.eval_split=0.0",
        "--dataset.return_uint8=true",
        "--dataset.image_transforms.enable=false",
        f"--output_dir={output_dir}",
        f"--job_name={JOB_NAME}",
        "--resume=false",
        "--save_checkpoint=true",
        "--save_checkpoint_to_hub=false",
        "--wandb.enable=false",
        f"--wandb.notes={capture.token}",
        "--job.target=local",
        f"--policy.path={capture.parent_checkpoint}",
        "--policy.pretrained_revision=null",
        "--policy.input_features=null",
        "--policy.output_features=null",
        "--policy.load_vlm_weights=false",
        f"--policy.vlm_model_name={vlm_bundle}",
        "--policy.use_peft=false",
        "--policy.push_to_hub=false",
        "--policy.repo_id=null",
        "--peft=null",
        "--rename_map={}",
        "--policy.device=cuda",
        "--policy.use_amp=false",
        "--policy.n_obs_steps=1",
        f"--policy.chunk_size={CHUNK_SIZE}",
        f"--policy.n_action_steps={N_ACTION_STEPS}",
        f"--policy.optimizer_lr={OPTIMIZER_LR}",
        f"--policy.scheduler_warmup_steps={SCHEDULER_WARMUP_STEPS}",
        f"--policy.scheduler_decay_steps={SCHEDULER_DECAY_STEPS}",
        f"--policy.scheduler_decay_lr={SCHEDULER_DECAY_LR}",
        "--policy.freeze_vision_encoder=true",
        "--policy.train_expert_only=true",
        "--policy.train_state_proj=true",
        f"--seed={SEED}",
        f"--batch_size={BATCH_SIZE}",
        "--num_workers=4",
        "--prefetch_factor=2",
        "--persistent_workers=true",
        f"--steps={TRAINING_STEPS}",
        "--env_eval_freq=0",
        "--eval_steps=0",
        "--max_eval_samples=0",
        "--log_freq=20",
        f"--save_freq={SAVE_FREQ}",
        "--use_policy_training_preset=true",
    ]


def finalize_capture(capture: CoreInputCapture, command: Sequence[str]) -> FinetuneInputCapture:
    command_tuple = tuple(command)
    payload = {**capture.payload, "command": list(command_tuple)}
    return FinetuneInputCapture(
        digest=_sha256_json(payload),
        core=capture,
        command=command_tuple,
        payload=payload,
    )


def _require_new_output(path: Path) -> Path:
    absolute = _absolute(path)
    current = Path(absolute.anchor)
    missing = False
    for component in absolute.parts[1:]:
        current /= component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            missing = True
            continue
        if missing:
            raise ArmOnlyFinetuneError(f"output has an inconsistent missing ancestor: {current}")
        if stat.S_ISLNK(mode):
            raise ArmOnlyFinetuneError(f"output path contains a symbolic link: {current}")
    if absolute.exists():
        raise FileExistsError(f"attested fine-tuning requires a new output directory: {absolute}")
    return absolute


def build_training_environment() -> dict[str, str]:
    environment = dict(FIXED_BASE_ENVIRONMENT)
    for name in INHERITED_ENVIRONMENT_ALLOWLIST:
        value = os.environ.get(name)
        if value is not None:
            environment[name] = value
    environment.update(OFFLINE_ENVIRONMENT)
    return environment


def _argument(command: Sequence[str], name: str) -> str:
    prefix = f"--{name}="
    values = [item[len(prefix) :] for item in command if item.startswith(prefix)]
    if len(values) != 1:
        raise ArmOnlyFinetuneError(f"command must contain exactly one --{name}= value")
    return values[0]


def _expected_child_policy(capture: FinetuneInputCapture) -> dict[str, Any]:
    parent = capture.payload["parent_checkpoint"].get("policy_config")
    if not isinstance(parent, Mapping):
        raise ArmOnlyFinetuneError("captured parent policy config is missing")
    inputs, outputs = (
        {
            "observation.state": {"type": "STATE", "shape": [6]},
            "observation.images.scene": {"type": "VISUAL", "shape": [3, 480, 640]},
            "observation.images.wrist": {"type": "VISUAL", "shape": [3, 480, 640]},
        },
        {"action": {"type": "ACTION", "shape": [6]}},
    )
    expected = copy.deepcopy(dict(parent))
    expected.update(
        {
            "input_features": inputs,
            "output_features": outputs,
            "pretrained_path": str(capture.core.parent_checkpoint),
            "optimizer_lr": OPTIMIZER_LR,
            "scheduler_warmup_steps": SCHEDULER_WARMUP_STEPS,
            "scheduler_decay_steps": SCHEDULER_DECAY_STEPS,
            "scheduler_decay_lr": SCHEDULER_DECAY_LR,
        }
    )
    return expected


def _expected_child_train_config(
    capture: FinetuneInputCapture,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    parent = capture.payload["parent_checkpoint"].get("train_config")
    if not isinstance(parent, Mapping):
        raise ArmOnlyFinetuneError("captured parent train config is missing")
    expected = copy.deepcopy(dict(parent))
    dataset = expected.get("dataset")
    optimizer = expected.get("optimizer")
    scheduler = expected.get("scheduler")
    wandb = expected.get("wandb")
    job = expected.get("job")
    if not all(isinstance(item, dict) for item in (dataset, optimizer, scheduler, wandb, job)):
        raise ArmOnlyFinetuneError("captured parent train config lacks fixed nested configs")
    dataset.update(
        {
            "repo_id": DATASET_REPO_ID,
            "root": str(capture.core.dataset_root),
            "episodes": None,
            "revision": None,
            "return_uint8": True,
            "streaming": False,
            "eval_split": 0.0,
        }
    )
    transforms = dataset.get("image_transforms")
    if not isinstance(transforms, dict):
        raise ArmOnlyFinetuneError("captured parent image transform config is missing")
    transforms["enable"] = False
    expected.update(
        {
            "policy": _expected_child_policy(capture),
            "output_dir": str(_absolute(output_dir)),
            "job_name": JOB_NAME,
            "resume": False,
            "seed": SEED,
            "num_workers": 4,
            "batch_size": BATCH_SIZE,
            "prefetch_factor": 2,
            "persistent_workers": True,
            "steps": TRAINING_STEPS,
            "env_eval_freq": 0,
            "log_freq": 20,
            "eval_steps": 0,
            "max_eval_samples": 0,
            "save_checkpoint": True,
            "save_freq": SAVE_FREQ,
            "use_policy_training_preset": True,
            "save_checkpoint_to_hub": False,
            "peft": None,
            "rename_map": {},
            "checkpoint_path": None,
        }
    )
    optimizer.update(
        {
            "type": "adamw",
            "lr": OPTIMIZER_LR,
            "weight_decay": expected["policy"]["optimizer_weight_decay"],
            "grad_clip_norm": expected["policy"]["optimizer_grad_clip_norm"],
            "betas": expected["policy"]["optimizer_betas"],
            "eps": expected["policy"]["optimizer_eps"],
        }
    )
    scheduler.update(
        {
            "type": "cosine_decay_with_warmup",
            "num_warmup_steps": SCHEDULER_WARMUP_STEPS,
            "num_decay_steps": SCHEDULER_DECAY_STEPS,
            "peak_lr": OPTIMIZER_LR,
            "decay_lr": SCHEDULER_DECAY_LR,
        }
    )
    wandb.update({"enable": False, "notes": capture.core.token})
    job["target"] = "local"
    return expected


def _expected_processor_documents(
    policy: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    inputs = copy.deepcopy(policy["input_features"])
    outputs = copy.deepcopy(policy["output_features"])
    normalization_mapping = copy.deepcopy(policy["normalization_mapping"])
    all_features = {**inputs, **outputs}
    preprocessor = {
        "name": "policy_preprocessor",
        "steps": [
            {"registry_name": "rename_observations_processor", "config": {"rename_map": {}}},
            {"registry_name": "to_batch_processor", "config": {}},
            {"registry_name": "smolvla_new_line_processor", "config": {}},
            {
                "registry_name": "tokenizer_processor",
                "config": {
                    "max_length": policy["tokenizer_max_length"],
                    "task_key": "task",
                    "padding_side": "right",
                    "padding": "max_length",
                    "truncation": True,
                    "tokenizer_name": policy["vlm_model_name"],
                },
            },
            {
                "registry_name": "device_processor",
                "config": {"device": policy["device"], "float_dtype": None},
            },
            {
                "registry_name": "normalizer_processor",
                "config": {
                    "eps": 1e-8,
                    "features": all_features,
                    "norm_map": normalization_mapping,
                },
                "state_file": "policy_preprocessor_step_5_normalizer_processor.safetensors",
            },
        ],
    }
    postprocessor = {
        "name": "policy_postprocessor",
        "steps": [
            {
                "registry_name": "unnormalizer_processor",
                "config": {
                    "eps": 1e-8,
                    "features": outputs,
                    "norm_map": normalization_mapping,
                },
                "state_file": "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
            },
            {
                "registry_name": "device_processor",
                "config": {"device": "cpu", "float_dtype": None},
            },
        ],
    }
    return preprocessor, postprocessor


def _validate_statistics_safetensors(
    path: Path,
    *,
    label: str,
    statistics: Mapping[str, Any],
    preprocessor_normalizer: bool,
) -> None:
    path = _require_no_symlink_components(path, label=label)
    before = _identity(path, directory=False)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        if _descriptor_identity(descriptor) != before:
            raise ControlledInputChangedError(f"{label} changed before parsing: {path}")
        try:
            import numpy as np
            from safetensors import safe_open
        except ImportError as exc:
            raise ArmOnlyFinetuneError(
                "numpy and safetensors are required for processor state validation"
            ) from exc
        required_keys = {
            f"{feature}.{statistic}"
            for feature in statistics
            for statistic in (*_VECTOR_STATISTICS, "count")
        }
        try:
            with safe_open(f"/proc/self/fd/{descriptor}", framework="numpy") as tensors:
                available = set(tensors.keys())
                if available != required_keys:
                    raise ArmOnlyFinetuneError(
                        f"{label} tensor keys must exactly match bound effective "
                        "processor statistics; "
                        f"missing={sorted(required_keys - available)!r}, "
                        f"unexpected={sorted(available - required_keys)!r}"
                    )
                for feature in statistics:
                    bound = statistics.get(feature)
                    if not isinstance(bound, Mapping):
                        raise ArmOnlyFinetuneError(f"captured dataset statistics lack {feature}")
                    for statistic in (*_VECTOR_STATISTICS, "count"):
                        key = f"{feature}.{statistic}"
                        value = tensors.get_tensor(key)
                        expected_value = np.asarray(bound[statistic], dtype=np.float32)
                        expected_shape = expected_value.shape
                        if (
                            preprocessor_normalizer
                            and feature.startswith("observation.images.")
                            and statistic == "count"
                        ):
                            expected_shape = (1, 1, 1)
                        if value.dtype != np.dtype("float32") or value.shape != expected_shape:
                            raise ArmOnlyFinetuneError(
                                f"{label} tensor {key} must be float32{expected_shape}"
                            )
                        if not bool(np.isfinite(value).all()):
                            raise ArmOnlyFinetuneError(f"{label} tensor {key} must be finite")
                        if not np.array_equal(value.reshape(-1), expected_value.reshape(-1)):
                            raise ArmOnlyFinetuneError(
                                f"{label} tensor {key} differs from bound effective "
                                "processor statistics"
                            )
        except ArmOnlyFinetuneError:
            raise
        except Exception as exc:
            raise ArmOnlyFinetuneError(f"cannot parse {label} {path}: {exc}") from exc
        if _descriptor_identity(descriptor) != before:
            raise ControlledInputChangedError(f"{label} changed while parsing: {path}")
    finally:
        os.close(descriptor)
    if _identity(path, directory=False) != before:
        raise ControlledInputChangedError(f"{label} was replaced while parsing: {path}")


def _validate_processor_artifacts(
    pretrained_model: Path,
    capture: FinetuneInputCapture,
    policy: Mapping[str, Any],
) -> None:
    expected_preprocessor, expected_postprocessor = _expected_processor_documents(policy)
    observed_preprocessor = _read_json_object(
        pretrained_model / "policy_preprocessor.json",
        label="child policy preprocessor",
    )
    observed_postprocessor = _read_json_object(
        pretrained_model / "policy_postprocessor.json",
        label="child policy postprocessor",
    )
    _exact(
        observed_preprocessor,
        expected_preprocessor,
        label="child policy preprocessor exact 6-D pipeline",
    )
    _exact(
        observed_postprocessor,
        expected_postprocessor,
        label="child policy postprocessor exact 6-D pipeline",
    )
    dataset_metadata = capture.payload["dataset"].get("metadata")
    if not isinstance(dataset_metadata, Mapping):
        raise ArmOnlyFinetuneError("captured dataset metadata are missing")
    parent_train_config = capture.payload["parent_checkpoint"].get("train_config")
    if not isinstance(parent_train_config, Mapping):
        raise ArmOnlyFinetuneError("captured parent train_config is missing")
    parent_dataset_config = _mapping(
        parent_train_config,
        "dataset",
        label="captured parent train_config",
    )
    expected_contract = _derive_effective_processor_statistics(
        dataset_metadata.get("statistics"),
        dataset_metadata.get("features"),
        use_imagenet_stats=parent_dataset_config.get("use_imagenet_stats"),
    )
    _exact(
        capture.payload.get("processor_statistics_contract"),
        expected_contract,
        label="captured raw/effective processor statistics contract",
    )
    statistics = expected_contract["effective_statistics"]
    for filename, label, preprocessor_normalizer in (
        (
            "policy_preprocessor_step_5_normalizer_processor.safetensors",
            "child preprocessor normalizer state",
            True,
        ),
        (
            "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
            "child postprocessor unnormalizer state",
            False,
        ),
    ):
        _validate_statistics_safetensors(
            pretrained_model / filename,
            label=label,
            statistics=statistics,
            preprocessor_normalizer=preprocessor_normalizer,
        )


def _validate_smoke_result(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "status": "passed",
        "policy_type": "smolvla",
        "input_state_dimension": 6,
        "output_action_dimension": 6,
        "selected_action_shape": [1, 6],
        "postprocessed_action_shape": [1, 6],
        "strict_weights": True,
        "local_files_only": True,
        "processor_roundtrip": True,
        "finite_action": True,
    }
    _exact(dict(value), expected, label="offline child load smoke result")
    return expected


def _run_child_load_smoke(
    pretrained_model: Path,
    training_python: Path,
) -> Mapping[str, Any]:
    """Load one child with pinned LeRobot and run one deterministic 6-D action."""

    pretrained_model = _require_no_symlink_components(
        pretrained_model,
        label="child smoke pretrained model",
    )
    program = r'''
import json
import sys

import torch
from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

model_path, task = sys.argv[1:]
config = PreTrainedConfig.from_pretrained(model_path, local_files_only=True)
assert config.type == "smolvla"
assert config.input_features["observation.state"].shape == (6,)
assert config.output_features["action"].shape == (6,)
assert set(config.input_features) == {
    "observation.state", "observation.images.scene", "observation.images.wrist"
}
assert config.device == "cuda" and torch.cuda.is_available()
assert config.use_amp is False and config.load_vlm_weights is False
policy_class = get_policy_class("smolvla")
policy = policy_class.from_pretrained(
    model_path,
    config=config,
    local_files_only=True,
    strict=True,
)
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=config,
    pretrained_path=model_path,
    pretrained_revision=None,
)
torch.manual_seed(0)
batch = {
    "observation.state": torch.zeros(6, dtype=torch.float32),
    "observation.images.scene": torch.zeros(3, 480, 640, dtype=torch.float32),
    "observation.images.wrist": torch.zeros(3, 480, 640, dtype=torch.float32),
    "task": task,
}
with torch.inference_mode():
    processed = preprocessor(batch)
    assert tuple(processed["observation.state"].shape) == (1, 6)
    noise = torch.zeros((1, 50, 32), device=config.device, dtype=torch.float32)
    selected = policy.select_action(processed, noise=noise)
    postprocessed = postprocessor(selected)
assert tuple(selected.shape) == (1, 6)
assert tuple(postprocessed.shape) == (1, 6)
assert bool(torch.isfinite(selected).all().item())
assert bool(torch.isfinite(postprocessed).all().item())
print("FIREFLY_CHILD_SMOKE=" + json.dumps({
    "status": "passed",
    "policy_type": "smolvla",
    "input_state_dimension": 6,
    "output_action_dimension": 6,
    "selected_action_shape": list(selected.shape),
    "postprocessed_action_shape": list(postprocessed.shape),
    "strict_weights": True,
    "local_files_only": True,
    "processor_roundtrip": True,
    "finite_action": True,
}, sort_keys=True))
'''
    command = [
        str(_absolute(training_python)),
        "-I",
        "-s",
        "-c",
        program,
        str(pretrained_model),
        EXPECTED_TASK,
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=build_training_environment(),
            cwd=TRAINING_WORKING_DIRECTORY,
            timeout=600,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ArmOnlyFinetuneError(f"offline child load smoke could not run: {exc}") from exc
    if result.returncode != 0:
        raise ArmOnlyFinetuneError(
            "offline child load smoke failed with status "
            f"{result.returncode}: {result.stderr[-4000:]}"
        )
    prefix = "FIREFLY_CHILD_SMOKE="
    payloads = [line[len(prefix) :] for line in result.stdout.splitlines() if line.startswith(prefix)]
    if len(payloads) != 1:
        raise ArmOnlyFinetuneError("offline child load smoke returned no unique result")
    try:
        value = json.loads(payloads[0])
    except json.JSONDecodeError as exc:
        raise ArmOnlyFinetuneError("offline child load smoke returned invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise ArmOnlyFinetuneError("offline child load smoke returned a non-object")
    return _validate_smoke_result(value)


def _validate_child_checkpoint(
    pretrained_model: Path,
    capture: FinetuneInputCapture,
    *,
    output_dir: Path,
) -> list[dict[str, Any]]:
    policy_config = _read_json_object(pretrained_model / "config.json", label="child policy config")
    train_config = _read_json_object(pretrained_model / "train_config.json", label="child train_config")
    expected_policy = _expected_child_policy(capture)
    _exact(
        policy_config,
        expected_policy,
        label="child policy exact parent inheritance plus allowlisted changes",
    )
    _policy_feature_shapes(policy_config, label="child policy config", dimension=6)
    expected_train_config = _expected_child_train_config(capture, output_dir=output_dir)
    _exact(
        train_config,
        expected_train_config,
        label="child train_config exact parent inheritance plus allowlisted changes",
    )

    _root, files = _tree_manifest(pretrained_model, label="child pretrained model")
    if any(entry["path"] == CHILD_ATTESTATION_FILENAME for entry in files):
        raise ArmOnlyFinetuneError("child checkpoint was already attested")
    names = {entry["path"] for entry in files}
    if names != _REQUIRED_CHILD_FILES:
        raise ArmOnlyFinetuneError(
            "child checkpoint files must be exactly the fixed LeRobot artifact set; "
            f"missing={sorted(_REQUIRED_CHILD_FILES - names)!r}, "
            f"unexpected={sorted(names - _REQUIRED_CHILD_FILES)!r}"
        )
    _validate_processor_artifacts(pretrained_model, capture, expected_policy)
    model_structure = _safetensors_structure(
        pretrained_model / "model.safetensors",
        label="child model safetensors",
        require_finite=True,
    )
    _exact(
        model_structure,
        capture.payload["parent_checkpoint"]["model_safetensors"]["structure"],
        label="child model safetensors structure inherited from parent",
    )
    return files


def _discover_checkpoints(output_dir: Path) -> list[Path]:
    output_dir = _require_no_symlink_components(output_dir, label="training output")
    checkpoints = output_dir / "checkpoints"
    _identity(checkpoints, directory=True)
    models: list[Path] = []
    actual_directories: set[str] = set()
    seen_last = False
    with os.scandir(checkpoints) as iterator:
        entries = sorted(iterator, key=lambda entry: entry.name)
    for entry in entries:
        path = Path(entry.path)
        mode = entry.stat(follow_symlinks=False).st_mode
        if entry.name == "last":
            if seen_last:
                raise ArmOnlyFinetuneError("checkpoints contains more than one last entry")
            seen_last = True
            if not stat.S_ISLNK(mode):
                raise ArmOnlyFinetuneError("checkpoints/last must be LeRobot's symbolic link")
            if os.readlink(path) != "020000":
                raise ArmOnlyFinetuneError("checkpoints/last must target exactly 020000")
            continue
        if (
            stat.S_ISLNK(mode)
            or not stat.S_ISDIR(mode)
            or entry.name not in set(_EXPECTED_CHECKPOINT_DIRECTORIES.values())
        ):
            raise ArmOnlyFinetuneError(f"unexpected checkpoint entry: {path}")
        actual_directories.add(entry.name)
        model = path / "pretrained_model"
        _identity(model, directory=True)
        models.append(model)
    expected_directories = set(_EXPECTED_CHECKPOINT_DIRECTORIES.values())
    if actual_directories != expected_directories or not seen_last:
        raise ArmOnlyFinetuneError(
            "expected checkpoint directories 010000 and 020000 plus last exactly once; "
            f"got directories={sorted(actual_directories)!r}, seen_last={seen_last}"
        )
    return models


def _json_file_bytes(document: Mapping[str, Any]) -> bytes:
    return json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"


def _write_exclusive_json(path: Path, document: Mapping[str, Any]) -> None:
    _identity(path.parent, directory=True)
    payload = _json_file_bytes(document)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o444)
    except OSError as exc:
        raise ArmOnlyFinetuneError(f"cannot exclusively create attestation artifact {path}: {exc}") from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_noreplace(
    staged: Path,
    final: Path,
    *,
    fsync_parent: bool = True,
) -> None:
    rename_noreplace = 1
    at_fdcwd = -100
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as exc:
        raise ArmOnlyFinetuneError("renameat2 is required for atomic attestation publish") from exc
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        at_fdcwd,
        os.fsencode(staged),
        at_fdcwd,
        os.fsencode(final),
        rename_noreplace,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise ArmOnlyFinetuneError(
            "cannot atomically publish attestation artifact without replacement "
            f"{final}: [Errno {error_number}] {os.strerror(error_number)}"
        )
    if fsync_parent:
        _fsync_directory(final.parent)


def _training_step_record(checkpoint_dir: Path, *, expected_step: int) -> dict[str, Any]:
    path = checkpoint_dir / "training_state/training_step.json"
    document = _read_json_object(path, label="child training step")
    if set(document) != {"step", "num_processes", "batch_size"}:
        raise ArmOnlyFinetuneError("child training_step.json keys drifted")
    _exact(document.get("step"), expected_step, label="child training state step")
    _exact(document.get("batch_size"), BATCH_SIZE, label="child training state batch size")
    _exact(document.get("num_processes"), 1, label="child training state num_processes")
    size, digest = _hash_regular_file(path)
    return {
        "path": "training_state/training_step.json",
        "size": size,
        "sha256": digest,
        "document": document,
    }


def _attestation_document(
    capture: FinetuneInputCapture,
    *,
    checkpoint_files: Sequence[Mapping[str, Any]],
    checkpoint_step: int,
) -> dict[str, Any]:
    return {
        "schema_version": CHILD_ATTESTATION_SCHEMA_VERSION,
        "kind": CHILD_ATTESTATION_KIND,
        "policy_schema_role": POLICY_SCHEMA_ROLE,
        "authorized_runtime_role": AUTHORIZED_RUNTIME_ROLE,
        "limitations": ARM_ONLY_OBSERVATION_LIMITATIONS,
        "token": capture.core.token,
        "core_input_digest": capture.core.digest,
        "input_digest": capture.digest,
        "inputs": capture.payload,
        "run_commit": {
            "filename": RUN_COMMIT_FILENAME,
            "run_id": capture.digest,
        },
        "checkpoint": {
            "step": checkpoint_step,
            "files": list(checkpoint_files),
            "files_sha256": _sha256_json(list(checkpoint_files)),
        },
        "verification": {
            "training_returncode": 0,
            "inputs_unchanged_after_training": True,
            "postflight_input_digest": capture.digest,
        },
    }


def _validate_child_attestation_document(
    pretrained_model: Path,
    capture: FinetuneInputCapture,
    document: Mapping[str, Any],
    *,
    expected_files: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    required_top_level = {
        "schema_version",
        "kind",
        "policy_schema_role",
        "authorized_runtime_role",
        "limitations",
        "token",
        "core_input_digest",
        "input_digest",
        "inputs",
        "run_commit",
        "checkpoint",
        "verification",
    }
    if set(document) != required_top_level:
        raise ArmOnlyFinetuneError("child attestation top-level keys drifted")
    checks = {
        "schema_version": (document.get("schema_version"), CHILD_ATTESTATION_SCHEMA_VERSION),
        "kind": (document.get("kind"), CHILD_ATTESTATION_KIND),
        "policy_schema_role": (
            document.get("policy_schema_role"),
            POLICY_SCHEMA_ROLE,
        ),
        "authorized_runtime_role": (
            document.get("authorized_runtime_role"),
            AUTHORIZED_RUNTIME_ROLE,
        ),
        "limitations": (
            document.get("limitations"),
            ARM_ONLY_OBSERVATION_LIMITATIONS,
        ),
        "token": (document.get("token"), capture.core.token),
        "core_input_digest": (document.get("core_input_digest"), capture.core.digest),
        "input_digest": (document.get("input_digest"), capture.digest),
        "inputs": (document.get("inputs"), capture.payload),
    }
    for label, (actual, expected) in checks.items():
        _exact(actual, expected, label=f"child attestation {label}")
    run_commit = _mapping(document, "run_commit", label="child attestation")
    _exact(
        dict(run_commit),
        {"filename": RUN_COMMIT_FILENAME, "run_id": capture.digest},
        label="child attestation run commit reference",
    )
    verification = _mapping(document, "verification", label="child attestation")
    if set(verification) != {
        "training_returncode",
        "inputs_unchanged_after_training",
        "postflight_input_digest",
    }:
        raise ArmOnlyFinetuneError("child attestation verification keys drifted")
    _exact(verification.get("training_returncode"), 0, label="child training returncode")
    _exact(
        verification.get("inputs_unchanged_after_training"),
        True,
        label="child inputs unchanged",
    )
    _exact(
        verification.get("postflight_input_digest"),
        capture.digest,
        label="child postflight digest",
    )
    checkpoint = _mapping(document, "checkpoint", label="child attestation")
    if set(checkpoint) != {"step", "files", "files_sha256"}:
        raise ArmOnlyFinetuneError("child attestation checkpoint keys drifted")
    checkpoint_files = checkpoint.get("files")
    if not isinstance(checkpoint_files, list):
        raise ArmOnlyFinetuneError("child attestation checkpoint.files must be a list")
    step_name = pretrained_model.parent.name
    step_by_directory = {
        directory: step for step, directory in _EXPECTED_CHECKPOINT_DIRECTORIES.items()
    }
    if step_name not in step_by_directory:
        raise ArmOnlyFinetuneError("child attestation is outside a canonical checkpoint directory")
    step = step_by_directory[step_name]
    _exact(checkpoint.get("step"), step, label="child checkpoint step")
    output_dir = _absolute(Path(capture.payload["training_contract"]["output_dir"]))
    expected_model = output_dir / "checkpoints" / step_name / "pretrained_model"
    if _absolute(pretrained_model) != expected_model:
        raise ArmOnlyFinetuneError("child attestation checkpoint path drifted from bound output")
    _exact(
        checkpoint.get("files_sha256"),
        _sha256_json(checkpoint_files),
        label="child checkpoint manifest digest",
    )
    if expected_files is None:
        _root, current = _tree_manifest(pretrained_model, label="child pretrained model")
        expected_files = [
            entry for entry in current if entry["path"] != CHILD_ATTESTATION_FILENAME
        ]
    _exact(list(expected_files), checkpoint_files, label="child checkpoint files")
    return dict(document)


def _run_commit_document(
    capture: FinetuneInputCapture,
    *,
    children: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": CHILD_ATTESTATION_SCHEMA_VERSION,
        "kind": RUN_COMMIT_KIND,
        "policy_schema_role": POLICY_SCHEMA_ROLE,
        "authorized_runtime_role": AUTHORIZED_RUNTIME_ROLE,
        "limitations": ARM_ONLY_OBSERVATION_LIMITATIONS,
        "token": capture.core.token,
        "core_input_digest": capture.core.digest,
        "input_digest": capture.digest,
        "run_id": capture.digest,
        "last_target": "020000",
        "children": list(children),
        "children_sha256": _sha256_json(list(children)),
    }
    document["document_sha256"] = _sha256_json(document)
    return document


def _manifest_file_sha256(
    files: Sequence[Mapping[str, Any]],
    filename: str,
) -> str:
    matches = [entry.get("sha256") for entry in files if entry.get("path") == filename]
    if len(matches) != 1:
        raise ArmOnlyFinetuneError(f"checkpoint manifest must contain exactly one {filename}")
    return _require_sha256(matches[0], label=f"checkpoint {filename} digest")


def _validate_run_commit(
    output_dir: Path,
    capture: FinetuneInputCapture,
    *,
    marker_path: Path | None = None,
) -> dict[Path, dict[str, Any]]:
    output_dir = _require_no_symlink_components(output_dir, label="attested training output")
    if marker_path is None:
        marker_path = output_dir / RUN_COMMIT_FILENAME
    marker = _read_json_object(marker_path, label="run commit marker")
    required_marker_keys = {
        "schema_version",
        "kind",
        "policy_schema_role",
        "authorized_runtime_role",
        "limitations",
        "token",
        "core_input_digest",
        "input_digest",
        "run_id",
        "last_target",
        "children",
        "children_sha256",
        "document_sha256",
    }
    if set(marker) != required_marker_keys:
        raise ArmOnlyFinetuneError("run commit marker keys drifted")
    unsigned = {key: value for key, value in marker.items() if key != "document_sha256"}
    _exact(marker.get("document_sha256"), _sha256_json(unsigned), label="run commit digest")
    for key, expected in {
        "schema_version": CHILD_ATTESTATION_SCHEMA_VERSION,
        "kind": RUN_COMMIT_KIND,
        "policy_schema_role": POLICY_SCHEMA_ROLE,
        "authorized_runtime_role": AUTHORIZED_RUNTIME_ROLE,
        "limitations": ARM_ONLY_OBSERVATION_LIMITATIONS,
        "token": capture.core.token,
        "core_input_digest": capture.core.digest,
        "input_digest": capture.digest,
        "run_id": capture.digest,
        "last_target": "020000",
    }.items():
        _exact(marker.get(key), expected, label=f"run commit {key}")
    children = marker.get("children")
    if not isinstance(children, list) or len(children) != 2:
        raise ArmOnlyFinetuneError("run commit must bind exactly two child checkpoints")
    _exact(marker.get("children_sha256"), _sha256_json(children), label="run children digest")
    models = _discover_checkpoints(output_dir)
    models_by_step = {int(model.parent.name): model for model in models}
    documents: dict[Path, dict[str, Any]] = {}
    expected_records = []
    for step in (10_000, 20_000):
        directory = _EXPECTED_CHECKPOINT_DIRECTORIES[step]
        model = models_by_step[step]
        path = model / CHILD_ATTESTATION_FILENAME
        document = _read_json_object(path, label=f"child {step} attestation")
        document = _validate_child_attestation_document(model, capture, document)
        size, digest = _hash_regular_file(path)
        training_step = _training_step_record(model.parent, expected_step=step)
        record = children[len(expected_records)]
        if not isinstance(record, Mapping):
            raise ArmOnlyFinetuneError("run commit child record must be an object")
        smoke = record.get("offline_load_smoke")
        if not isinstance(smoke, Mapping):
            raise ArmOnlyFinetuneError("run commit child smoke result is missing")
        smoke = _validate_smoke_result(smoke)
        expected_record = {
            "step": step,
            "directory": directory,
            "attestation_path": (
                f"checkpoints/{directory}/pretrained_model/{CHILD_ATTESTATION_FILENAME}"
            ),
            "attestation_size": size,
            "attestation_sha256": digest,
            "checkpoint_files_sha256": document["checkpoint"]["files_sha256"],
            "model_sha256": _manifest_file_sha256(
                document["checkpoint"]["files"], "model.safetensors"
            ),
            "training_step": training_step,
            "offline_load_smoke": smoke,
        }
        _exact(dict(record), expected_record, label=f"run commit child {step}")
        expected_records.append(expected_record)
        documents[model] = document
    _exact(children, expected_records, label="run commit ordered children")
    if len({record["model_sha256"] for record in expected_records}) != 2:
        raise ArmOnlyFinetuneError("10000 and 20000 child model.safetensors must be distinct")
    return documents


def verify_child_attestation(
    pretrained_model: Path,
    capture: FinetuneInputCapture,
) -> dict[str, Any]:
    pretrained_model = _require_no_symlink_components(
        pretrained_model,
        label="child pretrained model",
    )
    output_dir = _absolute(Path(capture.payload["training_contract"]["output_dir"]))
    documents = _validate_run_commit(output_dir, capture)
    try:
        return documents[pretrained_model]
    except KeyError as exc:
        raise ArmOnlyFinetuneError(
            "requested child is not one of the jointly committed checkpoints"
        ) from exc


def _cleanup_attestation_publication(paths: Sequence[Path]) -> None:
    parents: set[Path] = set()
    for path in paths:
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISREG(mode):
            try:
                path.unlink()
                parents.add(path.parent)
            except OSError:
                pass
    for parent in parents:
        try:
            _fsync_directory(parent)
        except OSError:
            pass


def _publish_attestation_set(
    output_dir: Path,
    capture: FinetuneInputCapture,
    plans: Sequence[
        tuple[Path, Sequence[Mapping[str, Any]], Mapping[str, Any], Mapping[str, Any]]
    ],
) -> list[Path]:
    marker = output_dir / RUN_COMMIT_FILENAME
    marker_staged = output_dir / f".{RUN_COMMIT_FILENAME}.staged"
    final_paths = [model / CHILD_ATTESTATION_FILENAME for model, *_rest in plans]
    staged_paths = [model / _STAGED_ATTESTATION_FILENAME for model, *_rest in plans]
    cleanup_paths = [marker, marker_staged, *final_paths, *staged_paths]
    try:
        staged_records = []
        for (model, files, document, smoke), staged in zip(plans, staged_paths, strict=True):
            _write_exclusive_json(staged, document)
            _fsync_directory(model)
            observed = _read_json_object(staged, label="staged child attestation")
            _validate_child_attestation_document(
                model,
                capture,
                observed,
                expected_files=files,
            )
            size, digest = _hash_regular_file(staged)
            step = int(model.parent.name)
            staged_records.append((model, files, smoke, size, digest, step))

        for staged, final in zip(staged_paths, final_paths, strict=True):
            _publish_noreplace(staged, final)

        children = []
        for model, files, smoke, staged_size, staged_digest, step in staged_records:
            final = model / CHILD_ATTESTATION_FILENAME
            observed = _read_json_object(final, label="published child attestation")
            observed = _validate_child_attestation_document(
                model,
                capture,
                observed,
                expected_files=files,
            )
            size, digest = _hash_regular_file(final)
            _exact(size, staged_size, label="published child attestation size")
            _exact(digest, staged_digest, label="published child attestation digest")
            directory = _EXPECTED_CHECKPOINT_DIRECTORIES[step]
            children.append(
                {
                    "step": step,
                    "directory": directory,
                    "attestation_path": (
                        f"checkpoints/{directory}/pretrained_model/{CHILD_ATTESTATION_FILENAME}"
                    ),
                    "attestation_size": size,
                    "attestation_sha256": digest,
                    "checkpoint_files_sha256": observed["checkpoint"]["files_sha256"],
                    "model_sha256": _manifest_file_sha256(
                        observed["checkpoint"]["files"], "model.safetensors"
                    ),
                    "training_step": _training_step_record(model.parent, expected_step=step),
                    "offline_load_smoke": _validate_smoke_result(smoke),
                }
            )
        if len({child["model_sha256"] for child in children}) != 2:
            raise ArmOnlyFinetuneError("10000 and 20000 child model.safetensors must be distinct")
        _discover_checkpoints(output_dir)
        commit_document = _run_commit_document(capture, children=children)
        _write_exclusive_json(marker_staged, commit_document)
        _fsync_directory(output_dir)
        observed_marker = _read_json_object(marker_staged, label="staged run commit")
        _exact(observed_marker, commit_document, label="staged run commit document")
        _validate_run_commit(output_dir, capture, marker_path=marker_staged)
        _fsync_directory(output_dir)
        # This successful NOREPLACE rename is the sole commit point.  Do not add
        # validation, writes, or other fallible work after it.
        _publish_noreplace(marker_staged, marker, fsync_parent=False)
    except Exception:
        _cleanup_attestation_publication(cleanup_paths)
        raise
    return final_paths


Runner = Callable[..., subprocess.CompletedProcess[Any]]


def run_attested_finetune(
    *,
    dataset_root: Path = DATASET_ROOT,
    parent_checkpoint: Path = PARENT_CHECKPOINT,
    controller_contract: Path = CONTROLLER_CONTRACT,
    config_path: Path = CONFIG_PATH,
    output_dir: Path = OUTPUT_DIR,
    training_python: Path = TRAINING_PYTHON,
    runner: Runner = subprocess.run,
    task_loader: TaskLoader = _load_task_rows,
    projection_verifier: ProjectionVerifier = _verify_arm_only_projection,
    parent_verifier: ParentVerifier = _verify_parent_training_attestation,
    vlm_validator: VLMValidator = _validate_vlm_bundle,
    verification_runtime_loader: VerificationRuntimeLoader = (
        _capture_current_verification_runtime
    ),
    environment_loader: EnvironmentLoader = _capture_environment,
    parser_preflight: ParserPreflight = _run_cli_parser_preflight,
    child_smoke_validator: ChildSmokeValidator = _run_child_load_smoke,
    expected_parent_files: Mapping[str, tuple[int, str]] = PARENT_FILE_PINS,
    expected_parent_input_digest: str = PARENT_INPUT_DIGEST,
    expected_vlm_bundle: Path = PARENT_VLM_BUNDLE,
) -> list[Path]:
    """Run the fixed job, recapture inputs, and attest both expected checkpoints."""

    output_dir = _require_new_output(output_dir)
    capture_kwargs = {
        "dataset_root": dataset_root,
        "parent_checkpoint": parent_checkpoint,
        "controller_contract": controller_contract,
        "config_path": config_path,
        "output_dir": output_dir,
        "training_python": training_python,
        "task_loader": task_loader,
        "projection_verifier": projection_verifier,
        "parent_verifier": parent_verifier,
        "vlm_validator": vlm_validator,
        "verification_runtime_loader": verification_runtime_loader,
        "environment_loader": environment_loader,
        "parser_preflight": parser_preflight,
        "expected_parent_files": expected_parent_files,
        "expected_parent_input_digest": expected_parent_input_digest,
        "expected_vlm_bundle": expected_vlm_bundle,
    }
    before_core = capture_core_inputs(**capture_kwargs)
    command = build_training_command(
        before_core,
        output_dir=output_dir,
        training_python=training_python,
    )
    before = finalize_capture(before_core, command)
    environment = build_training_environment()
    result = runner(
        list(before.command),
        check=False,
        env=environment,
        cwd=str(TRAINING_WORKING_DIRECTORY),
    )
    if result.returncode != 0:
        raise ArmOnlyFinetuneError(
            f"LeRobot fine-tuning failed with status {result.returncode}; no child was attested"
        )

    after_core = capture_core_inputs(**capture_kwargs)
    after_command = build_training_command(
        after_core,
        output_dir=output_dir,
        training_python=training_python,
    )
    after = finalize_capture(after_core, after_command)
    if after.digest != before.digest or after.payload != before.payload:
        raise ControlledInputChangedError("controlled fine-tuning inputs changed during training")

    models = _discover_checkpoints(output_dir)
    plans: list[
        tuple[Path, Sequence[Mapping[str, Any]], Mapping[str, Any], Mapping[str, Any]]
    ] = []
    for model in models:
        files = _validate_child_checkpoint(model, before, output_dir=output_dir)
        step = int(model.parent.name)
        smoke = dict(child_smoke_validator(model, training_python))
        _validate_smoke_result(smoke)
        plans.append(
            (
                model,
                files,
                _attestation_document(before, checkpoint_files=files, checkpoint_step=step),
                smoke,
            )
        )
    written = _publish_attestation_set(output_dir, before, plans)
    return written


def verify_fixed_attested_run() -> dict[str, Any]:
    """Read-only verification of the one fixed jointly committed child run."""

    core = capture_core_inputs()
    command = build_training_command(core)
    capture = finalize_capture(core, command)
    documents = _validate_run_commit(OUTPUT_DIR, capture)
    marker_path = OUTPUT_DIR / RUN_COMMIT_FILENAME
    marker_size, marker_digest = _hash_regular_file(marker_path)
    children = []
    for model, document in sorted(
        documents.items(), key=lambda item: int(item[0].parent.name)
    ):
        attestation_path = model / CHILD_ATTESTATION_FILENAME
        attestation_size, attestation_digest = _hash_regular_file(attestation_path)
        children.append(
            {
                "step": document["checkpoint"]["step"],
                "checkpoint_directory": model.parent.name,
                "pretrained_model": str(model),
                "attestation_size": attestation_size,
                "attestation_sha256": attestation_digest,
                "checkpoint_files_sha256": document["checkpoint"]["files_sha256"],
                "model_sha256": _manifest_file_sha256(
                    document["checkpoint"]["files"], "model.safetensors"
                ),
            }
        )
    return {
        "schema_version": CHILD_ATTESTATION_SCHEMA_VERSION,
        "kind": "firefly_smolvla_arm_only_v1_finetune_verification_result",
        "status": "verified-read-only-no-training-started",
        "policy_schema_role": POLICY_SCHEMA_ROLE,
        "authorized_runtime_role": AUTHORIZED_RUNTIME_ROLE,
        "limitations": ARM_ONLY_OBSERVATION_LIMITATIONS,
        "output_dir": str(OUTPUT_DIR),
        "core_input_digest": core.digest,
        "input_digest": capture.digest,
        "run_commit": {
            "path": str(marker_path),
            "size": marker_size,
            "sha256": marker_digest,
        },
        "last_target": "020000",
        "children": children,
    }


def build_preview_command() -> list[str]:
    """Return a non-executable preview without reading any controlled inputs."""

    placeholder = CoreInputCapture(
        digest="<verified-core-input-sha256>",
        token=f"{CHILD_TOKEN_PREFIX}<verified-core-input-sha256>",
        payload={
            "parent_checkpoint": {"vlm_bundle": {"path": str(PARENT_VLM_BUNDLE)}}
        },
        dataset_root=DATASET_ROOT,
        parent_checkpoint=PARENT_CHECKPOINT,
        controller_contract=CONTROLLER_CONTRACT,
        config_path=CONFIG_PATH,
    )
    return build_training_command(placeholder)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the immutable offline command without inspecting inputs or starting training.",
    )
    mode.add_argument(
        "--preflight",
        action="store_true",
        help="Validate and hash all fixed inputs without starting training.",
    )
    mode.add_argument(
        "--verify-run",
        action="store_true",
        help="Read-only verify the fixed 10k/20k jointly committed output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    invocation = list(sys.argv[1:] if argv is None else argv)
    _bootstrap_or_reexec(invocation)
    args = parse_args(argv)
    if args.dry_run:
        runtime = _bootstrap_runtime_snapshot()
        print(
            json.dumps(
                {
                    "environment": OFFLINE_ENVIRONMENT,
                    "verification_runtime": {
                        "python_executable": runtime["executable"],
                        "python_executable_sha256": runtime["executable_sha256"],
                        "flags": {
                            "isolated": runtime["isolated"],
                            "no_user_site": runtime["no_user_site"],
                            "ignore_environment": runtime["ignore_environment"],
                            "safe_path": runtime["safe_path"],
                        },
                        "working_directory": runtime["cwd"],
                        "script": runtime["script"],
                        "sitecustomize_loaded": runtime["sitecustomize_loaded"],
                        "usercustomize_loaded": runtime["usercustomize_loaded"],
                        "environment_variable_names": sorted(os.environ),
                    },
                    "command": build_preview_command(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.preflight:
        try:
            core = capture_core_inputs()
            command = build_training_command(core)
            capture = finalize_capture(core, command)
        except (
            ArmOnlyFinetuneError,
            FileNotFoundError,
            FileExistsError,
            OSError,
            ValueError,
        ) as exc:
            print(f"arm-only v1 fine-tuning preflight refused: {exc}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "status": "preflight-passed-no-training-started",
                    "policy_schema_role": POLICY_SCHEMA_ROLE,
                    "authorized_runtime_role": AUTHORIZED_RUNTIME_ROLE,
                    "limitations": ARM_ONLY_OBSERVATION_LIMITATIONS,
                    "core_input_digest": core.digest,
                    "input_digest": capture.digest,
                    "command": list(capture.command),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.verify_run:
        try:
            result = verify_fixed_attested_run()
        except (
            ArmOnlyFinetuneError,
            FileNotFoundError,
            FileExistsError,
            OSError,
            ValueError,
        ) as exc:
            print(f"arm-only v1 fixed run verification refused: {exc}", file=sys.stderr)
            return 2
        sys.stdout.write(_canonical_json_bytes(result).decode("utf-8") + "\n")
        return 0
    try:
        written = run_attested_finetune()
    except (ArmOnlyFinetuneError, FileNotFoundError, FileExistsError, OSError, ValueError) as exc:
        print(f"arm-only v1 attested fine-tuning refused: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "attested",
                "policy_schema_role": POLICY_SCHEMA_ROLE,
                "authorized_runtime_role": AUTHORIZED_RUNTIME_ROLE,
                "limitations": ARM_ONLY_OBSERVATION_LIMITATIONS,
                "attestations": [str(path) for path in written],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
