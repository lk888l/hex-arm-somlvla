from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import run_firefly_y6
from lerobot_robot_hexbase.firefly_deployment import DeploymentSourceCapture
from lerobot_robot_hexbase.firefly_deployment import (
    GraspPhaseCorrectionConfig,
    GraspPhaseCorrector,
)
from lerobot_robot_hexbase.hex_zenoh_follower import _validated_visual_telemetry
from lerobot_robot_hexbase.firefly_visual_latch import (
    BlobEvidence,
    FrameStatus,
    LatchDecision,
    VisualEvidence,
    VisualLatchPhase,
    VisualPhaseLatchConfig,
)
from lerobot_robot_hexbase.hex_zenoh_follower import CameraFrameMetadata


_TASK = "Pick up the red block and place it in the blue tray."
_VLM_SNAPSHOT = "/tmp/fake-materialized-smolvlm"


def _raw_observation() -> dict:
    observation = {f"joint_{index}.pos": float(index - 1) for index in range(1, 7)}
    observation["jaw_joint1.pos"] = 6.0
    observation["scene"] = np.zeros((4, 5, 3), dtype=np.uint8)
    observation["wrist"] = np.ones((4, 5, 3), dtype=np.uint8)
    return observation


def _arm6_stub_infer():
    def infer(_observation):
        return tuple((0.0,) * 6 for _ in range(10))

    infer.policy_runtime_contract = run_firefly_y6.PolicyRuntimeContract(
        policy_type="smolvla",
        runtime_role="arm6_seeking",
        observation_state_dimension=6,
        policy_action_dimension=6,
    )
    return infer


def _install_fake_policy_stack(
    monkeypatch,
    policy_type: str,
    *,
    chunk_size: int = 50,
    predicted: torch.Tensor | None = None,
    feature_dim: int = 7,
    postprocessor_transform=None,
):
    from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
    from lerobot.policies import factory
    from lerobot.policies import utils as policy_utils

    input_features = {
        "observation.state": PolicyFeature(
            type=FeatureType.STATE, shape=(feature_dim,)
        ),
        "observation.images.scene": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 480, 640)
        ),
        "observation.images.wrist": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 480, 640)
        ),
    }
    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(feature_dim,)),
    }
    if predicted is None:
        predicted = torch.arange(
            50 * feature_dim, dtype=torch.float32
        ).reshape(1, 50, feature_dim)
    config = SimpleNamespace(
        type=policy_type,
        input_features=input_features,
        output_features=output_features,
        chunk_size=chunk_size,
        n_action_steps=chunk_size,
        device="original-device",
        use_amp=False,
    )
    records: dict[str, object] = {
        "policy_types": [],
        "postprocessor_inputs": [],
    }

    monkeypatch.setattr(
        PreTrainedConfig,
        "from_pretrained",
        classmethod(lambda cls, checkpoint, **kwargs: config),
    )

    class FakePolicy:
        @classmethod
        def from_pretrained(cls, checkpoint, **kwargs):
            records["policy_load"] = (checkpoint, kwargs)
            return cls()

        def predict_action_chunk(self, prepared):
            records["policy_input"] = prepared
            return predicted

    def get_policy_class(name):
        records["policy_types"].append(name)
        return FakePolicy

    def preprocessor(prepared):
        records["preprocessor_input"] = prepared
        return {"preprocessed": prepared}

    def postprocessor(action):
        records["postprocessor_inputs"].append(action.detach().clone())
        if postprocessor_transform is None:
            return action + 1000
        return postprocessor_transform(action)

    def make_pre_post_processors(**kwargs):
        records["processor_kwargs"] = kwargs
        return preprocessor, postprocessor

    def prepare_observation(observation, device, task=None, robot_type=None):
        records["prepared"] = {
            "observation": observation,
            "device": device,
            "task": task,
            "robot_type": robot_type,
        }
        return {"prepared": True}

    monkeypatch.setattr(factory, "get_policy_class", get_policy_class)
    monkeypatch.setattr(factory, "make_pre_post_processors", make_pre_post_processors)
    monkeypatch.setattr(policy_utils, "prepare_observation_for_inference", prepare_observation)
    monkeypatch.setattr(
        run_firefly_y6,
        "validate_materialized_smolvlm_bundle",
        lambda path: (Path(path), {"revision": "test"}),
    )
    return config, records


@pytest.mark.parametrize(
    ("policy_type", "expected_task"),
    [
        ("act", None),
        ("smolvla", _TASK),
    ],
)
def test_load_policy_detects_type_supplies_task_and_postprocesses_each_step(
    monkeypatch, policy_type, expected_task
):
    config, records = _install_fake_policy_stack(monkeypatch, policy_type)
    checkpoint = Path("/tmp/fake-checkpoint")

    vlm_snapshot = _VLM_SNAPSHOT if policy_type == "smolvla" else None
    infer = run_firefly_y6.load_policy(
        checkpoint, "cpu", 10, _TASK, vlm_snapshot
    )

    assert records["policy_types"] == [policy_type]
    assert config.device == "cpu"
    assert config.n_action_steps == 10
    loaded_checkpoint, load_kwargs = records["policy_load"]
    assert loaded_checkpoint == checkpoint
    assert load_kwargs == {
        "config": config,
        "local_files_only": True,
        "strict": True,
    }
    expected_overrides = {"device_processor": {"device": "cpu"}}
    if policy_type == "smolvla":
        expected_overrides["tokenizer_processor"] = {
            "tokenizer_name": _VLM_SNAPSHOT
        }
        assert config.vlm_model_name == _VLM_SNAPSHOT
        assert config.load_vlm_weights is False
    assert records["processor_kwargs"] == {
        "policy_cfg": config,
        "pretrained_path": str(checkpoint),
        "preprocessor_overrides": expected_overrides,
    }

    chunk = infer(_raw_observation())

    prepared = records["prepared"]
    assert prepared["device"] == torch.device("cpu")
    assert prepared["task"] == expected_task
    assert prepared["robot_type"] == "hex_zenoh_follower"
    model_observation = prepared["observation"]
    np.testing.assert_array_equal(
        model_observation["observation.state"], np.arange(7, dtype=np.float32)
    )
    assert set(model_observation) == {
        "observation.state",
        "observation.images.scene",
        "observation.images.wrist",
    }
    assert len(records["postprocessor_inputs"]) == 10
    assert all(value.shape == (1, 7) for value in records["postprocessor_inputs"])
    assert len(chunk) == 10
    assert chunk[0] == tuple(float(value) for value in range(1000, 1007))
    assert chunk[-1] == tuple(float(value) for value in range(1063, 1070))


def test_load_policy_uses_canonical_state_order(monkeypatch):
    _config, records = _install_fake_policy_stack(monkeypatch, "smolvla")
    infer = run_firefly_y6.load_policy(
        Path("/tmp/fake-checkpoint"), "cpu", 10, _TASK, _VLM_SNAPSHOT
    )
    raw = _raw_observation()
    raw = dict(reversed(list(raw.items())))

    infer(raw)

    np.testing.assert_array_equal(
        records["prepared"]["observation"]["observation.state"],
        np.arange(7, dtype=np.float32),
    )


@pytest.mark.parametrize(
    ("policy_type", "task", "steps", "chunk_size", "match"),
    [
        ("diffusion", _TASK, 10, 50, "unsupported deployment policy type"),
        ("smolvla", "   ", 10, 50, "requires a non-empty task"),
        ("act", _TASK, 9, 50, "fixed at 10 action steps"),
        ("act", _TASK, 10, 9, "cannot provide 10 deployment steps"),
    ],
)
def test_load_policy_rejects_unsupported_or_unsafe_configuration(
    monkeypatch, policy_type, task, steps, chunk_size, match
):
    _config, records = _install_fake_policy_stack(
        monkeypatch, policy_type, chunk_size=chunk_size
    )

    with pytest.raises(ValueError, match=match):
        run_firefly_y6.load_policy(Path("/tmp/fake-checkpoint"), "cpu", steps, task)

    assert records["policy_types"] == []


@pytest.mark.parametrize(
    ("feature_name", "shape"),
    [
        ("observation.state", (6,)),
        ("action", (6,)),
        ("observation.images.scene", (3, 224, 224)),
        ("observation.images.wrist", (480, 640, 3)),
    ],
)
def test_load_policy_rejects_checkpoint_feature_shape(monkeypatch, feature_name, shape):
    from lerobot.configs import FeatureType, PolicyFeature

    config, records = _install_fake_policy_stack(monkeypatch, "smolvla")
    features = (
        config.output_features if feature_name == "action" else config.input_features
    )
    feature_type = FeatureType.ACTION if feature_name == "action" else features[feature_name].type
    features[feature_name] = PolicyFeature(type=feature_type, shape=shape)

    with pytest.raises(ValueError, match="checkpoint feature"):
        run_firefly_y6.load_policy(Path("/tmp/fake-checkpoint"), "cpu", 10, _TASK)

    assert records["policy_types"] == []


def test_load_policy_rejects_extra_camera(monkeypatch):
    from lerobot.configs import FeatureType, PolicyFeature

    config, records = _install_fake_policy_stack(monkeypatch, "smolvla")
    config.input_features["observation.images.extra"] = PolicyFeature(
        type=FeatureType.VISUAL, shape=(3, 480, 640)
    )
    with pytest.raises(ValueError, match="expected exactly"):
        run_firefly_y6.load_policy(Path("/tmp/fake-checkpoint"), "cpu", 10, _TASK)
    assert records["policy_types"] == []


@pytest.mark.parametrize(
    ("predicted", "match"),
    [
        (torch.zeros((1, 9, 7)), "returned 9 steps"),
        (torch.zeros((1, 50, 6)), r"expected \[10, 7\]"),
        (
            torch.full((1, 50, 7), float("nan")),
            "returned NaN or infinite action values",
        ),
    ],
)
def test_load_policy_rejects_malformed_or_nonfinite_chunks(monkeypatch, predicted, match):
    _config, _records = _install_fake_policy_stack(
        monkeypatch, "smolvla", predicted=predicted
    )
    infer = run_firefly_y6.load_policy(
        Path("/tmp/fake-checkpoint"), "cpu", 10, _TASK, _VLM_SNAPSHOT
    )

    with pytest.raises(ValueError, match=match):
        infer(_raw_observation())


def test_load_policy_arm6_requires_explicit_role_and_smolvla(monkeypatch):
    config, records = _install_fake_policy_stack(
        monkeypatch,
        "smolvla",
        feature_dim=6,
    )
    checkpoint = Path("/tmp/fake-checkpoint")

    with pytest.raises(ValueError, match="checkpoint feature"):
        run_firefly_y6.load_policy(
            checkpoint,
            "cpu",
            10,
            _TASK,
            _VLM_SNAPSHOT,
        )
    assert records["policy_types"] == []

    infer = run_firefly_y6.load_policy(
        checkpoint,
        "cpu",
        10,
        _TASK,
        _VLM_SNAPSHOT,
        expected_runtime_role="arm6_seeking",
    )
    assert infer.policy_runtime_contract == run_firefly_y6.PolicyRuntimeContract(
        policy_type="smolvla",
        runtime_role="arm6_seeking",
        observation_state_dimension=6,
        policy_action_dimension=6,
    )
    assert config.input_features["observation.state"].shape == (6,)

    act_config, _ = _install_fake_policy_stack(
        monkeypatch,
        "act",
        feature_dim=6,
    )
    with pytest.raises(ValueError, match="accepts only a SmolVLA"):
        run_firefly_y6.load_policy(
            checkpoint,
            "cpu",
            10,
            _TASK,
            expected_runtime_role="arm6_seeking",
        )
    assert act_config.type == "act"


def test_load_policy_arm6_observation_never_reads_jaw(monkeypatch):
    _config, records = _install_fake_policy_stack(
        monkeypatch,
        "smolvla",
        feature_dim=6,
    )
    infer = run_firefly_y6.load_policy(
        Path("/tmp/fake-checkpoint"),
        "cpu",
        10,
        _TASK,
        _VLM_SNAPSHOT,
        expected_runtime_role="arm6_seeking",
    )

    class JawReadTrap(dict):
        def __getitem__(self, key):
            if key == "jaw_joint1.pos":
                raise AssertionError("arm6 model attempted to read jaw feedback")
            return super().__getitem__(key)

    chunk = infer(JawReadTrap(_raw_observation()))

    np.testing.assert_array_equal(
        records["prepared"]["observation"]["observation.state"],
        np.arange(6, dtype=np.float32),
    )
    assert len(chunk) == 10
    assert all(len(row) == 6 for row in chunk)


@pytest.mark.parametrize(
    ("predicted", "transform", "match"),
    [
        (
            torch.zeros((1, 50, 7), dtype=torch.float32),
            None,
            "normalized action dimension 7, expected 6",
        ),
        (
            torch.full((1, 50, 6), float("nan")),
            None,
            "must be a finite number",
        ),
        (
            torch.zeros((1, 50, 6), dtype=torch.bool),
            lambda action: action,
            "must be a finite number",
        ),
    ],
)
def test_load_policy_arm6_rejects_7d_nan_and_bool_postprocess(
    monkeypatch,
    predicted,
    transform,
    match,
):
    _config, _records = _install_fake_policy_stack(
        monkeypatch,
        "smolvla",
        predicted=predicted,
        feature_dim=6,
        postprocessor_transform=transform,
    )
    infer = run_firefly_y6.load_policy(
        Path("/tmp/fake-checkpoint"),
        "cpu",
        10,
        _TASK,
        _VLM_SNAPSHOT,
        expected_runtime_role="arm6_seeking",
    )

    with pytest.raises(ValueError, match=match):
        infer(_raw_observation())


def test_arm6_prepare_binds_capture_before_slew_and_blocks_jaw_writers(monkeypatch):
    infer = _arm6_stub_infer()
    reference = (0.0,) * 6 + (0.31,)
    runtime = run_firefly_y6.bind_arm6_seeking_runtime(infer, reference)
    raw6 = tuple((0.10,) * 6 for _ in range(10))
    captured = {}
    original_adapter = run_firefly_y6.adapt_action_chunk_slew_rate

    def inspect_adapter(raw_actions, *args, **kwargs):
        captured["candidate"] = raw_actions
        return original_adapter(raw_actions, *args, **kwargs)

    monkeypatch.setattr(
        run_firefly_y6,
        "adapt_action_chunk_slew_rate",
        inspect_adapter,
    )
    robot = SimpleNamespace(
        config=SimpleNamespace(
            arm_step_limit_rad=0.05,
            ee_step_limit_rad=0.01,
        )
    )
    actions, telemetry = run_firefly_y6.prepare_policy_action_chunk(
        SimpleNamespace(enable_action_slew_rate_adapter=True),
        robot,
        raw6,
        reference,
        arm6_runtime=runtime,
    )

    assert all(len(row) == 7 and row[6] == pytest.approx(0.31) for row in captured["candidate"])
    assert all(row[6] == pytest.approx(0.31) for row in actions)
    assert all(len(row) == 6 for row in telemetry["raw_actions"])
    assert telemetry["arm6_authority"] == {
        "policy_role": "arm6_seeking",
        "observation_state_dimension": 6,
        "raw_policy_action_dimension": 6,
        "candidate_action_dimension": 7,
        "jaw_candidate_source": "startup_accepted_reference",
        "capture_jaw_target_rad": pytest.approx(0.31),
    }

    malicious_corrector = SimpleNamespace(
        config=SimpleNamespace(enabled=True),
        apply=lambda *_args: pytest.fail("enabled jaw writer must not run"),
    )
    with pytest.raises(ValueError, match="forbids an enabled"):
        run_firefly_y6.prepare_policy_action_chunk(
            SimpleNamespace(enable_action_slew_rate_adapter=False),
            robot,
            raw6,
            reference,
            corrector=malicious_corrector,
            arm6_runtime=runtime,
        )

    with pytest.raises(AttributeError):
        runtime.capture_jaw_target_rad = 0.32

    with pytest.raises(ValueError, match="interval must be exactly"):
        run_firefly_y6.Arm6SeekingRuntime(
            contract=infer.policy_runtime_contract,
            startup_accepted_reference7=reference,
            capture_jaw_target_rad=0.31,
            capture_jaw_range_rad=(0.20, 0.40),
        )
    with pytest.raises(ValueError, match="must be immutable"):
        run_firefly_y6.Arm6SeekingRuntime(
            contract=infer.policy_runtime_contract,
            startup_accepted_reference7=list(reference),
            capture_jaw_target_rad=0.31,
            capture_jaw_range_rad=(0.285, 0.325),
        )


def test_run_act_arm6_rejects_capture_before_first_inference_submission(monkeypatch):
    workers = []
    inference_calls = []

    class Worker:
        def __init__(self, infer):
            self.infer = infer
            self.submissions = []
            self.closed = False
            workers.append(self)

        def submit(self, observation):
            self.submissions.append(observation)
            self.infer(observation)

        def close(self):
            self.closed = True

    class Robot:
        action_features = tuple(
            [f"joint_{index}.pos" for index in range(1, 7)]
            + ["jaw_joint1.pos"]
        )
        command_target = dict.fromkeys(action_features, 0.0)
        command_target["jaw_joint1.pos"] = 0.284

        def get_observation(self):
            return dict(self.command_target)

    class Recorder:
        def add_observation(self, _observation):
            pass

    infer = _arm6_stub_infer()

    def counted_infer(observation):
        inference_calls.append(observation)
        return infer(observation)

    counted_infer.policy_runtime_contract = infer.policy_runtime_contract
    monkeypatch.setattr(run_firefly_y6, "InferenceWorker", Worker)
    monkeypatch.setattr(run_firefly_y6, "build_visual_latch_runtime", lambda _config: None)

    with pytest.raises(ValueError, match="outside the arm6_seeking interval"):
        run_firefly_y6.run_act(
            SimpleNamespace(
                fps=30,
                n_action_steps=10,
                act_chunk_inference_mode="synchronous_hold",
                grasp_phase_correction=None,
            ),
            Robot(),
            counted_infer,
            Recorder(),
            [False],
            30.0,
        )

    assert inference_calls == []
    assert len(workers) == 1
    assert workers[0].submissions == []
    assert workers[0].closed is True


def test_run_act_never_reuses_old_chunk_when_next_replan_misses_deadline(monkeypatch):
    events: list[str] = []
    workers = []

    class FakeWorker:
        def __init__(self, infer):
            self.infer = infer
            self.closed = False
            workers.append(self)

        def submit(self, observation):
            events.append("submit")
            self.submitted = observation

        def get_nowait(self):
            return None

        def close(self):
            self.closed = True

    class FakeRobot:
        action_features = tuple(
            [f"joint_{index}.pos" for index in range(1, 7)] + ["jaw_joint1.pos"]
        )
        command_target = dict.fromkeys(action_features, 0.0)
        stopped_reason = None

        def __init__(self):
            self.sent = []
            self.preflight = []

        def get_observation(self):
            return dict.fromkeys(self.action_features, 0.0)

        def preflight_actions(self, actions):
            self.preflight.append(actions)

        def record_event(self, *args, **kwargs):
            return None

        def send_action_with_metadata(self, action, **kwargs):
            events.append("send")
            self.sent.append((action, kwargs))

        def stop(self, reason):
            self.stopped_reason = reason

    class FakeRecorder:
        def add_observation(self, observation):
            return None

    initial_chunk = SimpleNamespace(
        generation=1,
        actions=tuple((0.0,) * 7 for _ in range(10)),
        latency_s=0.01,
        error=None,
    )
    monkeypatch.setattr(run_firefly_y6, "InferenceWorker", FakeWorker)
    monkeypatch.setattr(
        run_firefly_y6,
        "wait_for_initial_chunk",
        lambda robot, worker, observation, recorder, stopping: initial_chunk,
    )
    monkeypatch.setattr(run_firefly_y6.time, "sleep", lambda duration: None)
    robot = FakeRobot()
    config = SimpleNamespace(fps=30, n_action_steps=10)

    with pytest.raises(
        run_firefly_y6.SafetyStopError,
        match="next policy chunk was not ready after 10 command steps",
    ):
        run_firefly_y6.run_act(
            config,
            robot,
            infer=lambda observation: None,
            recorder=FakeRecorder(),
            stopping=[False],
            duration_s=1.0,
        )

    assert events == ["submit", *(["send"] * 10)]
    assert len(robot.sent) == 10
    assert len(robot.preflight) == 1
    assert len(robot.preflight[0]) == 11
    assert workers[0].closed is True


def test_run_act_synchronous_hold_replans_from_post_chunk_observation(monkeypatch):
    feature_names = tuple(
        [f"joint_{index}.pos" for index in range(1, 7)] + ["jaw_joint1.pos"]
    )
    stopping = [False]
    wait_observations = []
    workers = []

    class FakeWorker:
        def __init__(self, _infer):
            self.direct_submissions = 0
            self.closed = False
            workers.append(self)

        def submit(self, _observation):
            self.direct_submissions += 1

        def close(self):
            self.closed = True

    class FakeRobot:
        action_features = feature_names
        stopped_reason = None

        def __init__(self):
            self.measured = dict.fromkeys(feature_names, 0.0)
            self.command_target = dict(self.measured)
            self.sent = []

        def get_observation(self):
            return dict(self.measured)

        def preflight_actions(self, _actions):
            pass

        def record_event(self, *_args, **_kwargs):
            pass

        def send_action_with_metadata(self, action, **_metadata):
            self.command_target = dict(action)
            self.measured = dict(action)
            self.sent.append(dict(action))

        def stop(self, reason):
            self.stopped_reason = reason

    class FakeRecorder:
        def add_observation(self, _observation):
            pass

    initial_chunk = SimpleNamespace(
        generation=1,
        actions=tuple((0.01 * index,) * 7 for index in range(1, 11)),
        latency_s=0.01,
        error=None,
    )

    def fake_wait(_robot, _worker, observation, _recorder, _stopping):
        wait_observations.append(dict(observation))
        if len(wait_observations) == 1:
            return initial_chunk
        stopping[0] = True
        return None

    monkeypatch.setattr(run_firefly_y6, "InferenceWorker", FakeWorker)
    monkeypatch.setattr(run_firefly_y6, "wait_for_initial_chunk", fake_wait)
    monkeypatch.setattr(run_firefly_y6.time, "sleep", lambda _duration: None)
    robot = FakeRobot()

    completed = run_firefly_y6.run_act(
        SimpleNamespace(
            fps=30,
            n_action_steps=10,
            act_chunk_inference_mode="synchronous_hold",
            enable_action_slew_rate_adapter=False,
        ),
        robot,
        infer=lambda _observation: None,
        recorder=FakeRecorder(),
        stopping=stopping,
        duration_s=3.0,
    )

    assert completed is False
    assert len(robot.sent) == 10
    assert wait_observations[0]["joint_1.pos"] == pytest.approx(0.0)
    assert wait_observations[1]["joint_1.pos"] == pytest.approx(0.10)
    assert workers[0].direct_submissions == 0
    assert workers[0].closed is True


def test_run_shadow_slew_limits_chunk_and_records_raw_and_adapted(monkeypatch):
    stopping = [False]
    feature_names = tuple(
        [f"joint_{index}.pos" for index in range(1, 7)] + ["jaw_joint1.pos"]
    )
    raw_chunk = tuple((1.0,) * 7 for _ in range(10))

    class FakeRobot:
        action_features = feature_names
        command_target = dict.fromkeys(feature_names, 0.0)
        config = SimpleNamespace(arm_step_limit_rad=0.05, ee_step_limit_rad=0.01)

        def __init__(self):
            self.events = []
            self.sent = []

        def get_observation(self):
            return dict.fromkeys(feature_names, 0.0)

        def record_event(self, event, **fields):
            self.events.append((event, fields))

        def send_action_with_metadata(self, action, **_metadata):
            self.sent.append(action)
            if len(self.sent) == 10:
                stopping[0] = True

    class FakeRecorder:
        def add_observation(self, _observation):
            pass

    monkeypatch.setattr(run_firefly_y6.time, "sleep", lambda _duration: None)
    robot = FakeRobot()
    run_firefly_y6.run_shadow(
        SimpleNamespace(
            fps=30,
            n_action_steps=10,
            enable_action_slew_rate_adapter=True,
        ),
        robot,
        infer=lambda _observation: raw_chunk,
        recorder=FakeRecorder(),
        stopping=stopping,
    )

    assert len(robot.sent) == 10
    for index, action in enumerate(robot.sent, 1):
        assert tuple(action[name] for name in feature_names[:6]) == pytest.approx(
            (0.05 * index,) * 6
        )
        assert action[feature_names[6]] == pytest.approx(0.01 * index)
    event, fields = robot.events[0]
    assert event == "inference_chunk"
    assert fields["raw_actions"] == raw_chunk
    assert fields["adapted_actions"][-1] == pytest.approx((0.5,) * 6 + (0.1,))
    assert fields["action_slew_rate"]["enabled"] is True
    assert fields["action_slew_rate"]["step_limited"] == (True,) * 10
    assert fields["action_slew_rate"]["limited_value_count"] == 70


def test_run_act_slew_limits_before_preflight_and_records_telemetry(monkeypatch):
    feature_names = tuple(
        [f"joint_{index}.pos" for index in range(1, 7)] + ["jaw_joint1.pos"]
    )
    raw_chunk = tuple((1.0,) * 7 for _ in range(10))

    class FakeWorker:
        def __init__(self, _infer):
            self.closed = False

        def submit(self, _observation):
            pass

        def get_nowait(self):
            return None

        def close(self):
            self.closed = True

    class FakeRobot:
        action_features = feature_names
        command_target = dict.fromkeys(feature_names, 0.0)
        stopped_reason = None
        config = SimpleNamespace(arm_step_limit_rad=0.05, ee_step_limit_rad=0.01)

        def __init__(self):
            self.events = []
            self.sent = []
            self.preflight = []

        def get_observation(self):
            return dict.fromkeys(feature_names, 0.0)

        def preflight_actions(self, actions):
            self.preflight.append(actions)

        def record_event(self, event, **fields):
            self.events.append((event, fields))

        def send_action_with_metadata(self, action, **_metadata):
            self.sent.append(action)

        def stop(self, reason):
            self.stopped_reason = reason

    class FakeRecorder:
        def add_observation(self, _observation):
            pass

    initial_chunk = SimpleNamespace(
        generation=1,
        actions=raw_chunk,
        latency_s=0.01,
        error=None,
    )
    monkeypatch.setattr(run_firefly_y6, "InferenceWorker", FakeWorker)
    monkeypatch.setattr(
        run_firefly_y6,
        "wait_for_initial_chunk",
        lambda *_args: initial_chunk,
    )
    monkeypatch.setattr(run_firefly_y6.time, "sleep", lambda _duration: None)
    robot = FakeRobot()

    with pytest.raises(run_firefly_y6.SafetyStopError, match="next policy chunk"):
        run_firefly_y6.run_act(
            SimpleNamespace(
                fps=30,
                n_action_steps=10,
                enable_action_slew_rate_adapter=True,
            ),
            robot,
            infer=lambda _observation: None,
            recorder=FakeRecorder(),
            stopping=[False],
            duration_s=3.0,
        )

    assert len(robot.preflight) == 1
    assert robot.preflight[0][0] == (0.0,) * 7
    assert robot.preflight[0][-1] == pytest.approx((0.5,) * 6 + (0.1,))
    assert len(robot.sent) == 10
    event, fields = robot.events[0]
    assert event == "inference_chunk"
    assert fields["raw_actions"] == raw_chunk
    assert fields["adapted_actions"][-1] == pytest.approx((0.5,) * 6 + (0.1,))
    assert fields["action_slew_rate"]["limited_step_count"] == 10


def test_run_act_interrupts_initial_wait_and_releases_before_worker_close(monkeypatch):
    events: list[str] = []
    stopping = [False]

    class FakeWorker:
        def __init__(self, _infer):
            pass

        def submit(self, _observation):
            events.append("submit")

        def get_nowait(self):
            return None

        def close(self):
            events.append("worker_close")

    class FakeRobot:
        action_features = ("joint_1.pos",)
        command_target = {"joint_1.pos": 0.0}
        config = SimpleNamespace(command_hz=30.0)

        def get_observation(self):
            return {"joint_1.pos": 0.0}

        def send_action(self, _action):
            events.append("send")
            stopping[0] = True

        def stop(self, reason):
            events.append(("stop", reason))

    class FakeRecorder:
        def add_observation(self, _observation):
            pass

    monkeypatch.setattr(run_firefly_y6, "InferenceWorker", FakeWorker)
    monkeypatch.setattr(run_firefly_y6.time, "sleep", lambda _duration: None)
    completed = run_firefly_y6.run_act(
        SimpleNamespace(fps=30, n_action_steps=10),
        FakeRobot(),
        infer=lambda _observation: None,
        recorder=FakeRecorder(),
        stopping=stopping,
        duration_s=3.0,
    )

    assert completed is False
    assert events == [
        "submit",
        "send",
        ("stop", "operator interrupt during policy execution"),
        "worker_close",
    ]


def _install_main_fakes(monkeypatch, tmp_path, events, captures):
    config = SimpleNamespace(
        act_duration_s=3.0,
        output_dir=str(tmp_path),
        record_video=False,
        fps=30,
        n_action_steps=10,
        inference_seed=None,
        policy_runtime_role="legacy_action7",
        grasp_phase_correction=GraspPhaseCorrectionConfig(),
        visual_phase_latch=VisualPhaseLatchConfig(),
        enable_action_slew_rate_adapter=False,
        checkpoint=str(tmp_path / "checkpoint"),
        dataset_repo_id="local/validated_v2",
        dataset_root=str(tmp_path / "dataset"),
        dataset_controller_contract=str(tmp_path / "contract.json"),
        smolvla_vlm_snapshot=None,
        task=_TASK,
        robot=SimpleNamespace(
            arm_timeout_behavior="ramp_stop",
            allow_arm_hold_timeout_without_ramp_stop=False,
            arm_step_limit_rad=0.035,
            ee_step_limit_rad=0.012,
            ee_closing_tracking_error_rad=None,
            resolved_arm_step_limits_rad=lambda: (0.035,) * 6,
            cameras={},
        ),
    )
    args = SimpleNamespace(
        config=tmp_path / "deploy.yaml",
        stage="shadow",
        act_duration_s=None,
        mark_reviewed=False,
        allow_arm_hold_timeout_without_ramp_stop=False,
        independent_estop_verified=False,
        hold_duration_s=1.0,
        operator_authorized_skip_replay_for_act_3s=False,
        operator_authorized_direct_act_30s=False,
        operator_authorized_direct_act_65s=False,
        operator_authorized_direct_act_90s=False,
        operator_authorized_act_start_reset=False,
        operator_authorized_act_ee_preopen=False,
        operator_authorized_tip_servo_execute=False,
        operator_authorized_tip_servo_open_only=False,
    )
    monkeypatch.setattr(run_firefly_y6, "parse_args", lambda: args)
    monkeypatch.setattr(run_firefly_y6, "load_config", lambda _path: config)

    def capture(_config):
        events.append("capture")
        return captures.pop(0)

    monkeypatch.setattr(run_firefly_y6, "capture_deployment_sources", capture)
    monkeypatch.setattr(
        run_firefly_y6,
        "require_acceptance_order",
        lambda *_args, **_kwargs: events.append("acceptance"),
    )
    monkeypatch.setattr(
        run_firefly_y6,
        "require_arm_hold_timeout_override_gate",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        run_firefly_y6, "require_hardware_safety_gate", lambda *_args: None
    )
    monkeypatch.setattr(
        run_firefly_y6,
        "robot_config_for_stage",
        lambda *_args: events.append("robot_config") or object(),
    )
    monkeypatch.setattr(
        run_firefly_y6,
        "load_policy",
        lambda *_args, **_kwargs: events.append("preload") or (lambda _observation: []),
    )
    monkeypatch.setattr(run_firefly_y6.signal, "signal", lambda *_args: None)
    return config


def _enable_tip_servo_execute_main_profile(config, *, allow_close=False):
    config.act_duration_s = 30.0
    config.visual_phase_latch = replace(
        config.visual_phase_latch,
        enabled=True,
        tip_servo=replace(
            config.visual_phase_latch.tip_servo,
            mode=run_firefly_y6.TipServoMode.EXECUTE,
            allow_close=allow_close,
        ),
    )


def _enable_tip_servo_open_only_main_profile(config):
    config.act_duration_s = 30.0
    config.visual_phase_latch = replace(
        config.visual_phase_latch,
        enabled=True,
        tip_servo=replace(
            config.visual_phase_latch.tip_servo,
            mode=run_firefly_y6.TipServoMode.EXECUTE,
            probe_profile=run_firefly_y6.TipServoProbeProfile.OPEN_ONLY,
            coarse_x_norm=(0.50, 0.545),
            coarse_y_norm=(0.42, 0.56),
            coarse_area_fraction=(0.017, 0.029),
            coarse_capture_jaw_rad=(0.285, 0.325),
            tracking_x_norm=(0.40, 0.56),
            tracking_y_norm=(0.40, 0.62),
            tracking_area_fraction=(0.014, 0.041),
            max_arm_speed_rad_s=0.07,
            max_takeover_reference_error_rad=0.035,
            arm_target_settle_rad=(0.002,) * 6,
            opening_timeout_ms=1500.0,
            descent_step_m=0.0,
            max_total_descent_m=0.0,
            allow_close=False,
        ),
    )


def test_parse_args_exposes_fresh_tip_servo_execute_authorization(monkeypatch, tmp_path):
    monkeypatch.setattr(
        run_firefly_y6.sys,
        "argv",
        [
            "run_firefly_y6.py",
            "--config",
            str(tmp_path / "deploy.yaml"),
            "--stage",
            "act",
            "--act-duration-s",
            "30",
            "--operator-authorized-tip-servo-execute",
        ],
    )

    args = run_firefly_y6.parse_args()

    assert args.operator_authorized_tip_servo_execute is True


def test_parse_args_exposes_fresh_tip_servo_open_only_authorization(monkeypatch, tmp_path):
    monkeypatch.setattr(
        run_firefly_y6.sys,
        "argv",
        [
            "run_firefly_y6.py",
            "--config",
            str(tmp_path / "deploy.yaml"),
            "--stage",
            "act",
            "--act-duration-s",
            "30",
            "--operator-authorized-tip-servo-open-only",
        ],
    )

    args = run_firefly_y6.parse_args()

    assert args.operator_authorized_tip_servo_open_only is True


def test_main_tip_servo_execute_act30_requires_fresh_flag_before_connect(
    monkeypatch, tmp_path
):
    events = []
    config = _install_main_fakes(monkeypatch, tmp_path, events, [])
    _enable_tip_servo_execute_main_profile(config)
    args = run_firefly_y6.parse_args()
    args.stage = "act"
    args.operator_authorized_tip_servo_execute = False
    args.operator_authorized_act_ee_preopen = False

    with pytest.raises(
        PermissionError,
        match="requires a fresh --operator-authorized-tip-servo-execute",
    ):
        run_firefly_y6.main()

    assert events == []


@pytest.mark.parametrize(
    ("stage", "tip_execute_authorized"),
    (("act", True), ("shadow", False)),
)
def test_main_tip_servo_execute_gate_allows_act30_with_fresh_flag_and_nonact_without_it(
    monkeypatch, tmp_path, stage, tip_execute_authorized
):
    events = []
    capture = DeploymentSourceCapture("a" * 64, "b" * 64)
    config = _install_main_fakes(
        monkeypatch, tmp_path, events, [capture, capture]
    )
    _enable_tip_servo_execute_main_profile(config)
    args = run_firefly_y6.parse_args()
    args.stage = stage
    args.operator_authorized_tip_servo_execute = tip_execute_authorized
    args.operator_authorized_act_ee_preopen = False
    provenance = {}

    monkeypatch.setattr(
        run_firefly_y6,
        "require_unchanged_deployment_sources",
        lambda *_args: events.append("unchanged"),
    )
    monkeypatch.setattr(
        run_firefly_y6,
        "invalidate_stage_reviews",
        lambda *_args: events.append("invalidate"),
    )

    class FakeRobot:
        def __init__(self, _config):
            events.append("robot")

        def record_event(self, event, **fields):
            events.append(event)
            if event == "deployment_provenance":
                provenance.update(fields)

        def connect(self):
            events.append("connect")

        def disconnect(self):
            events.append("disconnect")

    class FakeRecorder:
        def __init__(self, *_args):
            events.append("recorder")

        def close(self):
            events.append("recorder_close")

    monkeypatch.setattr(run_firefly_y6, "HexZenohFollower", FakeRobot)
    monkeypatch.setattr(run_firefly_y6, "StageVideoRecorder", FakeRecorder)
    monkeypatch.setattr(
        run_firefly_y6,
        "run_shadow",
        lambda *_args: events.append("run_shadow"),
    )
    monkeypatch.setattr(run_firefly_y6, "run_act", lambda *_args: True)

    assert run_firefly_y6.main() == 0
    assert "connect" in events
    assert provenance["operator_authorized_tip_servo_probe_act_30s"] is (
        tip_execute_authorized
    )


@pytest.mark.parametrize(
    "legacy_authorization",
    (
        "operator_authorized_skip_replay_for_act_3s",
        "operator_authorized_direct_act_30s",
        "operator_authorized_direct_act_65s",
        "operator_authorized_direct_act_90s",
        "operator_authorized_act_start_reset",
        "operator_authorized_act_ee_preopen",
    ),
)
def test_main_tip_servo_execute_flag_is_mutually_exclusive_with_legacy_authorizations(
    monkeypatch, tmp_path, legacy_authorization
):
    events = []
    config = _install_main_fakes(monkeypatch, tmp_path, events, [])
    _enable_tip_servo_execute_main_profile(config)
    args = run_firefly_y6.parse_args()
    args.stage = "act"
    args.operator_authorized_tip_servo_execute = True
    args.operator_authorized_act_ee_preopen = False
    setattr(args, legacy_authorization, True)

    with pytest.raises(PermissionError, match="mutually exclusive"):
        run_firefly_y6.main()

    assert events == []


def test_main_tip_servo_execute_profile_rejects_allow_close_before_connect(
    monkeypatch, tmp_path
):
    events = []
    config = _install_main_fakes(monkeypatch, tmp_path, events, [])
    _enable_tip_servo_execute_main_profile(config, allow_close=True)
    args = run_firefly_y6.parse_args()
    args.stage = "act"
    args.operator_authorized_tip_servo_execute = True
    args.operator_authorized_act_ee_preopen = False

    with pytest.raises(
        PermissionError,
        match="exact reviewed 25-mm, allow_close=false probe profile",
    ):
        run_firefly_y6.main()

    assert events == []


@pytest.mark.parametrize(
    ("stage", "act_duration_s", "mark_reviewed", "error_match"),
    (
        ("act", 90.0, False, "valid only for the exact execute profile"),
        ("act", 30.0, True, "valid only for the exact execute profile"),
        ("shadow", 30.0, False, "valid only for the ACT stage"),
    ),
)
def test_main_tip_servo_execute_flag_rejects_act90_review_and_nonact_use(
    monkeypatch, tmp_path, stage, act_duration_s, mark_reviewed, error_match
):
    events = []
    config = _install_main_fakes(monkeypatch, tmp_path, events, [])
    _enable_tip_servo_execute_main_profile(config)
    config.act_duration_s = act_duration_s
    args = run_firefly_y6.parse_args()
    args.stage = stage
    args.mark_reviewed = mark_reviewed
    args.operator_authorized_tip_servo_execute = True
    args.operator_authorized_act_ee_preopen = False

    with pytest.raises(PermissionError, match=error_match):
        run_firefly_y6.main()

    assert events == []


def test_main_tip_servo_open_only_act30_requires_fresh_flag_before_connect(
    monkeypatch, tmp_path
):
    events = []
    config = _install_main_fakes(monkeypatch, tmp_path, events, [])
    _enable_tip_servo_open_only_main_profile(config)
    args = run_firefly_y6.parse_args()
    args.stage = "act"

    with pytest.raises(
        PermissionError,
        match="requires a fresh --operator-authorized-tip-servo-open-only",
    ):
        run_firefly_y6.main()

    assert events == []


def test_main_tip_servo_open_only_profile_rejects_legacy_tip_flag_before_connect(
    monkeypatch, tmp_path
):
    events = []
    config = _install_main_fakes(monkeypatch, tmp_path, events, [])
    _enable_tip_servo_open_only_main_profile(config)
    args = run_firefly_y6.parse_args()
    args.stage = "act"
    args.operator_authorized_tip_servo_execute = True

    with pytest.raises(PermissionError, match="cannot authorize OPEN_ONLY"):
        run_firefly_y6.main()

    assert events == []


@pytest.mark.parametrize(
    ("stage", "act_duration_s", "mark_reviewed"),
    (
        ("shadow", 3.0, False),
        ("hold", 3.0, False),
        ("replay", 3.0, False),
        ("act", 10.0, False),
        ("act", 30.0, False),
        ("act", 65.0, False),
        ("act", 3.0, True),
    ),
)
def test_main_operator_replay_skip_rejects_every_out_of_scope_use(
    monkeypatch, tmp_path, stage, act_duration_s, mark_reviewed
):
    events = []
    config = _install_main_fakes(monkeypatch, tmp_path, events, [])
    config.act_duration_s = act_duration_s
    args = run_firefly_y6.parse_args()
    args.stage = stage
    args.mark_reviewed = mark_reviewed
    args.operator_authorized_skip_replay_for_act_3s = True

    with pytest.raises(PermissionError, match="valid only for a non-review"):
        run_firefly_y6.main()

    assert events == []


def test_main_operator_replay_skip_is_rechecked_and_recorded(monkeypatch, tmp_path):
    events = []
    capture = DeploymentSourceCapture("a" * 64, "b" * 64)
    config = _install_main_fakes(monkeypatch, tmp_path, events, [capture, capture])
    args = run_firefly_y6.parse_args()
    args.stage = "act"
    args.operator_authorized_skip_replay_for_act_3s = True
    acceptance_calls = []
    provenance = {}

    monkeypatch.setattr(
        run_firefly_y6,
        "require_acceptance_order",
        lambda *positional, **keywords: acceptance_calls.append((positional, keywords)),
    )
    monkeypatch.setattr(
        run_firefly_y6,
        "require_unchanged_deployment_sources",
        lambda *_args: events.append("unchanged"),
    )
    monkeypatch.setattr(
        run_firefly_y6,
        "invalidate_stage_reviews",
        lambda *_args: events.append("invalidate"),
    )
    monkeypatch.setattr(
        run_firefly_y6,
        "load_episode_actions",
        lambda *_args: pytest.fail("act_3s replay override must not load replay data"),
    )

    class FakeRobot:
        def __init__(self, _config):
            events.append("robot")

        def record_event(self, event, **fields):
            events.append(event)
            if event == "deployment_provenance":
                provenance.update(fields)

        def connect(self):
            events.append("connect")

        def disconnect(self):
            events.append("disconnect")

    class FakeRecorder:
        def __init__(self, *_args):
            events.append("recorder")

        def close(self):
            events.append("recorder_close")

    monkeypatch.setattr(run_firefly_y6, "HexZenohFollower", FakeRobot)
    monkeypatch.setattr(run_firefly_y6, "StageVideoRecorder", FakeRecorder)
    monkeypatch.setattr(run_firefly_y6, "run_act", lambda *_args: True)

    assert config.act_duration_s == 3.0
    assert run_firefly_y6.main() == 0
    assert len(acceptance_calls) == 2
    assert all(
        keywords == {
            "operator_authorized_skip_replay_for_act_3s": True,
            "operator_authorized_direct_act_30s": False,
            "operator_authorized_tip_servo_probe_act_30s": False,
            "operator_authorized_tip_servo_open_only_act_30s": False,
            "operator_authorized_direct_act_65s": False,
            "operator_authorized_direct_act_90s": False,
        }
        for _positional, keywords in acceptance_calls
    )
    assert provenance["operator_authorized_skip_replay_for_act_3s"] is True


@pytest.mark.parametrize(
    ("stage", "act_duration_s", "mark_reviewed"),
    (
        ("shadow", 30.0, False),
        ("hold", 30.0, False),
        ("replay", 30.0, False),
        ("act", 3.0, False),
        ("act", 10.0, False),
        ("act", 65.0, False),
        ("act", 30.0, True),
    ),
)
def test_main_operator_direct_act_30s_rejects_every_out_of_scope_use(
    monkeypatch, tmp_path, stage, act_duration_s, mark_reviewed
):
    events = []
    config = _install_main_fakes(monkeypatch, tmp_path, events, [])
    config.act_duration_s = act_duration_s
    args = run_firefly_y6.parse_args()
    args.stage = stage
    args.mark_reviewed = mark_reviewed
    args.operator_authorized_direct_act_30s = True

    with pytest.raises(PermissionError, match="valid only for a non-review"):
        run_firefly_y6.main()

    assert events == []


def test_main_operator_direct_act_30s_is_rechecked_and_recorded(monkeypatch, tmp_path):
    events = []
    capture = DeploymentSourceCapture("a" * 64, "b" * 64)
    config = _install_main_fakes(monkeypatch, tmp_path, events, [capture, capture])
    config.act_duration_s = 30.0
    args = run_firefly_y6.parse_args()
    args.stage = "act"
    args.operator_authorized_direct_act_30s = True
    acceptance_calls = []
    provenance = {}

    monkeypatch.setattr(
        run_firefly_y6,
        "require_acceptance_order",
        lambda *positional, **keywords: acceptance_calls.append((positional, keywords)),
    )
    monkeypatch.setattr(
        run_firefly_y6,
        "require_unchanged_deployment_sources",
        lambda *_args: events.append("unchanged"),
    )
    monkeypatch.setattr(
        run_firefly_y6,
        "invalidate_stage_reviews",
        lambda *_args: events.append("invalidate"),
    )
    monkeypatch.setattr(
        run_firefly_y6,
        "load_episode_actions",
        lambda *_args: pytest.fail("direct act_30s must not load replay data"),
    )

    class FakeRobot:
        def __init__(self, _config):
            events.append("robot")

        def record_event(self, event, **fields):
            events.append(event)
            if event == "deployment_provenance":
                provenance.update(fields)

        def connect(self):
            events.append("connect")

        def disconnect(self):
            events.append("disconnect")

    class FakeRecorder:
        def __init__(self, *_args):
            events.append("recorder")

        def close(self):
            events.append("recorder_close")

    monkeypatch.setattr(run_firefly_y6, "HexZenohFollower", FakeRobot)
    monkeypatch.setattr(run_firefly_y6, "StageVideoRecorder", FakeRecorder)
    monkeypatch.setattr(run_firefly_y6, "run_act", lambda *_args: True)

    assert run_firefly_y6.main() == 0
    assert len(acceptance_calls) == 2
    assert all(
        keywords
        == {
            "operator_authorized_skip_replay_for_act_3s": False,
            "operator_authorized_direct_act_30s": True,
            "operator_authorized_tip_servo_probe_act_30s": False,
            "operator_authorized_tip_servo_open_only_act_30s": False,
            "operator_authorized_direct_act_65s": False,
            "operator_authorized_direct_act_90s": False,
        }
        for _positional, keywords in acceptance_calls
    )
    assert provenance["operator_authorized_direct_act_30s"] is True


@pytest.mark.parametrize(
    ("stage", "act_duration_s", "mark_reviewed"),
    (
        ("shadow", 65.0, False),
        ("hold", 65.0, False),
        ("replay", 65.0, False),
        ("act", 3.0, False),
        ("act", 10.0, False),
        ("act", 30.0, False),
        ("act", 65.0, True),
    ),
)
def test_main_operator_direct_act_65s_rejects_every_out_of_scope_use(
    monkeypatch, tmp_path, stage, act_duration_s, mark_reviewed
):
    events = []
    config = _install_main_fakes(monkeypatch, tmp_path, events, [])
    config.act_duration_s = act_duration_s
    args = run_firefly_y6.parse_args()
    args.stage = stage
    args.mark_reviewed = mark_reviewed
    args.operator_authorized_direct_act_65s = True

    with pytest.raises(PermissionError, match="valid only for a non-review"):
        run_firefly_y6.main()

    assert events == []


def test_main_operator_direct_act_65s_is_rechecked_and_recorded(monkeypatch, tmp_path):
    events = []
    capture = DeploymentSourceCapture("a" * 64, "b" * 64)
    config = _install_main_fakes(monkeypatch, tmp_path, events, [capture, capture])
    config.act_duration_s = 65.0
    args = run_firefly_y6.parse_args()
    args.stage = "act"
    args.operator_authorized_direct_act_65s = True
    acceptance_calls = []
    provenance = {}
    act_durations = []

    monkeypatch.setattr(
        run_firefly_y6,
        "require_acceptance_order",
        lambda *positional, **keywords: acceptance_calls.append((positional, keywords)),
    )
    monkeypatch.setattr(
        run_firefly_y6,
        "require_unchanged_deployment_sources",
        lambda *_args: events.append("unchanged"),
    )
    monkeypatch.setattr(
        run_firefly_y6,
        "invalidate_stage_reviews",
        lambda *_args: events.append("invalidate"),
    )
    monkeypatch.setattr(
        run_firefly_y6,
        "load_episode_actions",
        lambda *_args: pytest.fail("direct act_65s must not load replay data"),
    )

    class FakeRobot:
        def __init__(self, _config):
            events.append("robot")

        def record_event(self, event, **fields):
            events.append(event)
            if event == "deployment_provenance":
                provenance.update(fields)

        def connect(self):
            events.append("connect")

        def disconnect(self):
            events.append("disconnect")

    class FakeRecorder:
        def __init__(self, *_args):
            events.append("recorder")

        def close(self):
            events.append("recorder_close")

    def fake_run_act(_config, _robot, _infer, _recorder, _stopping, duration_s):
        act_durations.append(duration_s)
        return True

    monkeypatch.setattr(run_firefly_y6, "HexZenohFollower", FakeRobot)
    monkeypatch.setattr(run_firefly_y6, "StageVideoRecorder", FakeRecorder)
    monkeypatch.setattr(run_firefly_y6, "run_act", fake_run_act)

    assert run_firefly_y6.main() == 0
    assert act_durations == [65.0]
    assert len(acceptance_calls) == 2
    assert all(
        keywords
        == {
            "operator_authorized_skip_replay_for_act_3s": False,
            "operator_authorized_direct_act_30s": False,
            "operator_authorized_tip_servo_probe_act_30s": False,
            "operator_authorized_tip_servo_open_only_act_30s": False,
            "operator_authorized_direct_act_65s": True,
            "operator_authorized_direct_act_90s": False,
        }
        for _positional, keywords in acceptance_calls
    )
    assert provenance["operator_authorized_direct_act_65s"] is True


def test_main_source_change_fails_before_mutation_or_hardware(monkeypatch, tmp_path):
    events = []
    before = DeploymentSourceCapture("a" * 64, "b" * 64)
    after = DeploymentSourceCapture("a" * 64, "c" * 64)
    _install_main_fakes(monkeypatch, tmp_path, events, [before, after])
    telemetry = tmp_path / "shadow.jsonl"
    telemetry.write_text("preserve existing telemetry\n")
    monkeypatch.setattr(
        run_firefly_y6,
        "require_unchanged_deployment_sources",
        lambda *_args: (_ for _ in ()).throw(PermissionError("source changed")),
    )
    monkeypatch.setattr(
        run_firefly_y6,
        "invalidate_stage_reviews",
        lambda *_args: events.append("invalidate"),
    )
    monkeypatch.setattr(
        run_firefly_y6,
        "HexZenohFollower",
        lambda *_args: events.append("robot") or object(),
    )
    monkeypatch.setattr(
        run_firefly_y6,
        "StageVideoRecorder",
        lambda *_args: events.append("recorder") or object(),
    )

    assert run_firefly_y6.main() == 2

    assert events == [
        "capture",
        "acceptance",
        "robot_config",
        "preload",
        "capture",
    ]
    assert telemetry.read_text() == "preserve existing telemetry\n"


def test_main_preloads_and_rechecks_before_provenance_and_connect(monkeypatch, tmp_path):
    events = []
    capture = DeploymentSourceCapture("a" * 64, "b" * 64)
    _install_main_fakes(monkeypatch, tmp_path, events, [capture, capture])
    monkeypatch.setattr(
        run_firefly_y6,
        "require_unchanged_deployment_sources",
        lambda *_args: events.append("unchanged"),
    )
    monkeypatch.setattr(
        run_firefly_y6,
        "invalidate_stage_reviews",
        lambda *_args: events.append("invalidate"),
    )

    class FakeRobot:
        def __init__(self, _config):
            events.append("robot")

        def record_event(self, event, **_kwargs):
            events.append(event)

        def connect(self):
            events.append("connect")

        def disconnect(self):
            events.append("disconnect")

    class FakeRecorder:
        def __init__(self, *_args):
            events.append("recorder")

        def close(self):
            events.append("recorder_close")

    monkeypatch.setattr(run_firefly_y6, "HexZenohFollower", FakeRobot)
    monkeypatch.setattr(run_firefly_y6, "StageVideoRecorder", FakeRecorder)
    monkeypatch.setattr(
        run_firefly_y6,
        "run_shadow",
        lambda *_args: events.append("run_shadow"),
    )

    assert run_firefly_y6.main() == 0
    assert events == [
        "capture",
        "acceptance",
        "robot_config",
        "preload",
        "capture",
        "unchanged",
        "acceptance",
        "invalidate",
        "robot",
        "recorder",
        "deployment_provenance",
        "connect",
        "run_shadow",
        "stage_complete",
        "disconnect",
        "recorder_close",
    ]


def test_main_arm6_role_is_explicitly_gated_loaded_and_recorded(monkeypatch, tmp_path):
    events = []
    capture = DeploymentSourceCapture("a" * 64, "b" * 64)
    config = _install_main_fakes(
        monkeypatch,
        tmp_path,
        events,
        [capture, capture],
    )
    config.policy_runtime_role = "arm6_seeking"
    _enable_tip_servo_open_only_main_profile(config)
    args = run_firefly_y6.parse_args()
    args.stage = "shadow"
    gate_calls = []
    load_calls = []
    provenance = {}
    monkeypatch.setattr(
        run_firefly_y6,
        "require_arm6_policy_stage_gate",
        lambda *positional, **keywords: gate_calls.append((positional, keywords)),
    )
    monkeypatch.setattr(
        run_firefly_y6,
        "load_policy",
        lambda *positional, **keywords: (
            load_calls.append((positional, keywords))
            or (lambda _observation: [])
        ),
    )
    monkeypatch.setattr(
        run_firefly_y6,
        "require_unchanged_deployment_sources",
        lambda *_args: None,
    )
    monkeypatch.setattr(run_firefly_y6, "invalidate_stage_reviews", lambda *_args: None)

    class FakeRobot:
        def __init__(self, _config):
            pass

        def record_event(self, event, **fields):
            if event == "deployment_provenance":
                provenance.update(fields)

        def connect(self):
            pass

        def disconnect(self):
            pass

    class FakeRecorder:
        def __init__(self, *_args):
            pass

        def close(self):
            pass

    monkeypatch.setattr(run_firefly_y6, "HexZenohFollower", FakeRobot)
    monkeypatch.setattr(run_firefly_y6, "StageVideoRecorder", FakeRecorder)
    monkeypatch.setattr(run_firefly_y6, "run_shadow", lambda *_args: None)

    assert run_firefly_y6.main() == 0
    assert len(gate_calls) == 1
    assert gate_calls[0][0][1:] == ("shadow", "shadow")
    assert load_calls[0][1]["expected_runtime_role"] == "arm6_seeking"
    assert provenance["policy_runtime_role"] == "arm6_seeking"


def test_main_signal_during_second_capture_never_mutates_or_connects(
    monkeypatch, tmp_path
):
    events = []
    capture_value = DeploymentSourceCapture("a" * 64, "b" * 64)
    _install_main_fakes(monkeypatch, tmp_path, events, [capture_value, capture_value])
    handlers = {}

    def install_handler(signum, handler):
        handlers[signum] = handler

    capture_count = 0

    def capture(_config):
        nonlocal capture_count
        capture_count += 1
        events.append("capture")
        if capture_count == 2:
            handlers[run_firefly_y6.signal.SIGTERM](run_firefly_y6.signal.SIGTERM, None)
        return capture_value

    monkeypatch.setattr(run_firefly_y6.signal, "signal", install_handler)
    monkeypatch.setattr(run_firefly_y6, "capture_deployment_sources", capture)
    monkeypatch.setattr(
        run_firefly_y6,
        "require_unchanged_deployment_sources",
        lambda *_args: events.append("unchanged"),
    )
    monkeypatch.setattr(
        run_firefly_y6,
        "invalidate_stage_reviews",
        lambda *_args: events.append("invalidate"),
    )
    monkeypatch.setattr(
        run_firefly_y6,
        "HexZenohFollower",
        lambda *_args: events.append("robot") or object(),
    )
    monkeypatch.setattr(
        run_firefly_y6,
        "StageVideoRecorder",
        lambda *_args: events.append("recorder") or object(),
    )

    assert run_firefly_y6.main() == 130
    assert events == [
        "capture",
        "acceptance",
        "robot_config",
        "preload",
        "capture",
    ]


@pytest.mark.parametrize("failing_cleanup", ["disconnect", "recorder_close"])
def test_main_cleanup_always_attempts_disconnect_and_recorder_close(
    monkeypatch, tmp_path, failing_cleanup
):
    events = []
    capture = DeploymentSourceCapture("a" * 64, "b" * 64)
    _install_main_fakes(monkeypatch, tmp_path, events, [capture, capture])
    monkeypatch.setattr(
        run_firefly_y6,
        "require_unchanged_deployment_sources",
        lambda *_args: events.append("unchanged"),
    )
    monkeypatch.setattr(
        run_firefly_y6,
        "invalidate_stage_reviews",
        lambda *_args: events.append("invalidate"),
    )

    class FakeRobot:
        def __init__(self, _config):
            events.append("robot")

        def record_event(self, event, **_kwargs):
            events.append(event)

        def connect(self):
            events.append("connect")

        def disconnect(self):
            events.append("disconnect")
            if failing_cleanup == "disconnect":
                raise RuntimeError("disconnect failed")

    class FakeRecorder:
        def __init__(self, *_args):
            events.append("recorder")

        def close(self):
            events.append("recorder_close")
            if failing_cleanup == "recorder_close":
                raise RuntimeError("recorder close failed")

    monkeypatch.setattr(run_firefly_y6, "HexZenohFollower", FakeRobot)
    monkeypatch.setattr(run_firefly_y6, "StageVideoRecorder", FakeRecorder)
    monkeypatch.setattr(
        run_firefly_y6,
        "run_shadow",
        lambda *_args: events.append("run_shadow"),
    )

    expected_error = (
        "disconnect failed" if failing_cleanup == "disconnect" else "recorder close failed"
    )
    with pytest.raises(RuntimeError, match=expected_error):
        run_firefly_y6.main()

    assert events[-3:] == ["stage_complete", "disconnect", "recorder_close"]


def test_grasp_phase_corrector_latches_then_releases_terminally():
    center = [0.395, -0.148, 2.324, 0.623, 0.299, 1.031]
    corrector = GraspPhaseCorrector(
        GraspPhaseCorrectionConfig(
            enabled=True,
            pickup_center_rad=center,
            pickup_tolerance_rad=[0.10, 0.10, 0.10, 0.12, 0.10, 0.15],
        )
    )

    def chunk(ee):
        return tuple((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ee) for _ in range(10))

    assisted, telemetry = corrector.apply(chunk(0.300), [0.0] * 6 + [0.400])
    assert assisted == chunk(0.300)
    assert telemetry["event"] == "premature_close_observed"
    assert telemetry["phase_after"] == "monitoring"

    pickup_reference = [*center, 0.400]
    for _ in range(5):
        assisted, telemetry = corrector.apply(chunk(0.400), pickup_reference)
        assert assisted == chunk(0.400)
        assert telemetry["phase_after"] == "monitoring"
    assisted, telemetry = corrector.apply(chunk(0.400), pickup_reference)
    assert telemetry["event"] == "grasp_latched"
    assert telemetry["phase_after"] == "latched"
    assert all(action[6] == pytest.approx(0.320) for action in assisted)

    release_reference = [0.800, -0.200, 2.000, 0.400, 0.0, 0.0, 0.320]
    for _ in range(2):
        assisted, telemetry = corrector.apply(chunk(0.370), release_reference)
        assert telemetry["phase_after"] == "latched"
        assert all(action[6] == pytest.approx(0.320) for action in assisted)
    assisted, telemetry = corrector.apply(chunk(0.370), release_reference)
    assert telemetry["event"] == "release_unlatched"
    assert telemetry["phase_after"] == "released"
    assert assisted == chunk(0.370)

    terminal, telemetry = corrector.apply(chunk(0.250), [0.0] * 7)
    assert telemetry["phase_before"] == telemetry["phase_after"] == "released"
    assert terminal == chunk(0.250)


def test_grasp_phase_corrector_does_not_fire_without_early_close_evidence():
    center = [0.395, -0.148, 2.324, 0.623, 0.299, 1.031]
    corrector = GraspPhaseCorrector(
        GraspPhaseCorrectionConfig(
            enabled=True,
            pickup_center_rad=center,
            pickup_tolerance_rad=[0.10, 0.10, 0.10, 0.12, 0.10, 0.15],
        )
    )
    pickup_reference = [*center, 0.400]
    natural_close = tuple((*center, 0.300) for _ in range(10))
    natural_open = tuple((*center, 0.400) for _ in range(10))

    assisted, telemetry = corrector.apply(natural_close, pickup_reference)
    assert assisted == natural_close
    assert telemetry["premature_close_seen"] is False
    for _ in range(8):
        assisted, telemetry = corrector.apply(natural_open, pickup_reference)
        assert assisted == natural_open
        assert telemetry["transformed"] is False
    assert telemetry["phase_after"] == "monitoring"
    assert telemetry["pickup_confirm_streak"] == 0


def test_disabled_grasp_corrector_can_still_enforce_an_act_start_pose_gate():
    config = GraspPhaseCorrectionConfig(
        enabled=False,
        act_start_center_rad=[0.0] * 7,
        act_start_tolerance_rad=[0.1] * 7,
    )
    assessment = run_firefly_y6.assess_act_start_pose(config, [0.05] * 7)
    assert assessment["enabled"] is True
    assert assessment["passed"] is True

    raw = tuple(tuple(float(index) for index in range(7)) for _ in range(10))
    assisted, telemetry = GraspPhaseCorrector(config).apply(raw, [0.0] * 7)
    assert assisted == raw
    assert telemetry["enabled"] is False
    assert telemetry["transformed"] is False


def test_prepare_corrects_before_slew_and_phase_latch_yaml_is_exact():
    class StubCorrector:
        def apply(self, raw_actions, reference):
            assert raw_actions == tuple((0.0,) * 6 + (0.400,) for _ in range(10))
            assert reference == (0.0,) * 6 + (0.300,)
            assisted = tuple((0.0,) * 6 + (0.200,) for _ in range(10))
            return assisted, {"enabled": True, "event": "grasp_latched"}

    robot = SimpleNamespace(
        config=SimpleNamespace(arm_step_limit_rad=0.05, ee_step_limit_rad=0.01)
    )
    raw = tuple((0.0,) * 6 + (0.400,) for _ in range(10))
    actions, telemetry = run_firefly_y6.prepare_policy_action_chunk(
        SimpleNamespace(enable_action_slew_rate_adapter=True),
        robot,
        raw,
        (0.0,) * 6 + (0.300,),
        corrector=StubCorrector(),
    )
    assert telemetry["raw_actions"] == raw
    assert telemetry["assisted_actions"][0][6] == pytest.approx(0.200)
    assert telemetry["grasp_phase_correction"]["event"] == "grasp_latched"
    assert actions[0][6] == pytest.approx(0.290)

    path = (
        Path(__file__).parents[1]
        / "configs"
        / (
            "deploy_firefly_y6_hold_validated_v3_trimmed_fast90_per_joint_"
            "gravity_ff_gr80_payload_kp60_kd2p5_eekp150_phase_latch_v1.local.yaml"
        )
    )
    config = run_firefly_y6.load_config(path)
    expected = GraspPhaseCorrectionConfig(
        enabled=True,
        pickup_center_rad=[0.395, -0.148, 2.324, 0.623, 0.299, 1.031],
        pickup_tolerance_rad=[0.10, 0.10, 0.10, 0.12, 0.10, 0.15],
    )
    assert config.grasp_phase_correction.provenance() == expected.provenance()
    assert "phase_latch_v1" in config.robot.client_name
    assert "phase_latch_v1" in config.output_dir
    execute = run_firefly_y6.robot_config_for_stage(
        config, "act", Path("act_90s.jsonl")
    )
    assert execute.mode == "execute"
    with pytest.raises(PermissionError, match="approved only for ACT"):
        run_firefly_y6.robot_config_for_stage(
            config, "shadow", Path("shadow.jsonl")
        )

    v2_path = path.with_name(path.name.replace("phase_latch_v1", "phase_latch_v2"))
    v2 = run_firefly_y6.load_config(v2_path)
    expected_v2 = GraspPhaseCorrectionConfig(
        enabled=True,
        pickup_center_rad=[0.395, -0.148, 2.324, 0.623, 0.299, 1.031],
        pickup_tolerance_rad=[0.10, 0.10, 0.10, 0.12, 0.10, 0.15],
        pickup_open_threshold_rad=0.340,
    )
    assert v2.grasp_phase_correction.provenance() == expected_v2.provenance()
    assert "phase_latch_v2" in v2.robot.client_name
    assert "phase_latch_v2" in v2.output_dir
    assert run_firefly_y6.robot_config_for_stage(
        v2, "act", Path("act_90s_v2.jsonl")
    ).mode == "execute"


def _v3_phase_config() -> GraspPhaseCorrectionConfig:
    return GraspPhaseCorrectionConfig(
        enabled=True,
        pickup_center_rad=[0.395, -0.148, 2.324, 0.623, 0.299, 1.031],
        pickup_tolerance_rad=[0.10, 0.10, 0.10, 0.12, 0.10, 0.15],
        pickup_open_threshold_rad=0.340,
        pregrasp_open_target_rad=0.380,
        pregrasp_open_ready_rad=0.370,
        raw_open_progress_deadline_chunks=45,
        pickup_progress_deadline_chunks=180,
        release_progress_deadline_chunks=90,
        act_start_center_rad=[
            -0.0077796346,
            -1.4772257805,
            3.0181598663,
            0.3821436167,
            -0.0341406241,
            -0.0972255841,
            0.3245764077,
        ],
        act_start_tolerance_rad=[0.10, 0.10, 0.08, 0.15, 0.08, 0.18, 0.08],
    )


def _constant_ee_chunk(value: float):
    return tuple((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, value) for _ in range(10))


def test_grasp_phase_v3_opens_early_close_then_latches_from_commanded_readiness():
    config = _v3_phase_config()
    corrector = GraspPhaseCorrector(config)

    assisted, telemetry = corrector.apply(_constant_ee_chunk(0.300), [0.0] * 7)
    assert telemetry["event"] == "premature_close_observed"
    assert telemetry["pregrasp_open_active"] is True
    assert all(action[6] == pytest.approx(0.380) for action in assisted)

    pickup_not_ready = [*config.pickup_center_rad, 0.369]
    assisted, telemetry = corrector.apply(
        _constant_ee_chunk(0.400), pickup_not_ready
    )
    assert telemetry["commanded_open_ready"] is False
    assert telemetry["pickup_confirm_streak"] == 0
    assert telemetry["phase_after"] == "monitoring"

    pickup_ready = [*config.pickup_center_rad, 0.370]
    for expected_streak in range(1, 6):
        assisted, telemetry = corrector.apply(
            _constant_ee_chunk(0.300), pickup_ready
        )
        assert telemetry["pickup_confirm_streak"] == expected_streak
        assert telemetry["phase_after"] == "monitoring"
        assert all(action[6] == pytest.approx(0.380) for action in assisted)

    assisted, telemetry = corrector.apply(
        _constant_ee_chunk(0.400), pickup_ready
    )
    assert telemetry["event"] == "grasp_latched"
    assert telemetry["phase_after"] == "latched"
    assert all(action[6] == pytest.approx(0.320) for action in assisted)

    assisted, telemetry = corrector.apply(
        _constant_ee_chunk(0.300), pickup_ready
    )
    assert telemetry["phase_after"] == "latched"
    assert all(action[6] == pytest.approx(0.320) for action in assisted)

    release_reference = [0.800, -0.200, 2.000, 0.400, 0.0, 0.0, 0.320]
    for _ in range(2):
        assisted, telemetry = corrector.apply(
            _constant_ee_chunk(0.370), release_reference
        )
        assert telemetry["phase_after"] == "latched"
        assert all(action[6] == pytest.approx(0.320) for action in assisted)
    assisted, telemetry = corrector.apply(
        _constant_ee_chunk(0.370), release_reference
    )
    assert telemetry["event"] == "release_unlatched"
    assert telemetry["phase_after"] == "released"
    assert assisted == _constant_ee_chunk(0.370)


def test_grasp_phase_v3_natural_pickup_is_terminal_passthrough():
    config = _v3_phase_config()
    corrector = GraspPhaseCorrector(config)

    opening = _constant_ee_chunk(0.400)
    assisted, telemetry = corrector.apply(opening, [0.0] * 7)
    assert telemetry["raw_open_seen"] is True
    assert telemetry["phase_after"] == "monitoring"
    assert assisted == opening

    pickup_reference = [*config.pickup_center_rad, 0.354]
    natural_close = _constant_ee_chunk(0.300)
    assisted, telemetry = corrector.apply(natural_close, pickup_reference)
    assert telemetry["event"] == "natural_pickup_observed"
    assert telemetry["phase_after"] == "natural_passthrough"
    assert telemetry["premature_close_seen"] is False
    assert assisted == natural_close

    terminal, telemetry = corrector.apply(_constant_ee_chunk(0.250), [0.0] * 7)
    assert telemetry["phase_before"] == telemetry["phase_after"] == "natural_passthrough"
    assert telemetry["transformed"] is False
    assert terminal == _constant_ee_chunk(0.250)


def test_grasp_phase_v3_times_out_without_full_chunk_raw_open_progress():
    config = _v3_phase_config()
    corrector = GraspPhaseCorrector(config)
    closed_pickup_reference = [*config.pickup_center_rad, 0.320]
    one_point_spike = tuple(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.340 if index == 9 else 0.320)
        for index in range(10)
    )

    assisted, telemetry = corrector.apply(one_point_spike, closed_pickup_reference)
    assert telemetry["raw_open_seen"] is False
    assert telemetry["phase_after"] == "monitoring"
    assert telemetry["progress_timeout"] is False
    assert assisted == one_point_spike

    for _ in range(43):
        assisted, telemetry = corrector.apply(
            _constant_ee_chunk(0.320), closed_pickup_reference
        )
        assert telemetry["phase_after"] == "monitoring"
        assert telemetry["progress_timeout"] is False
        assert telemetry["transformed"] is False
    assisted, telemetry = corrector.apply(
        _constant_ee_chunk(0.320), closed_pickup_reference
    )
    assert telemetry["event"] == "raw_open_progress_timeout"
    assert telemetry["progress_timeout"] is True
    assert telemetry["progress_timeout_kind"] == "raw_open"
    assert telemetry["chunk_count"] == 45
    assert telemetry["raw_open_seen"] is False
    assert assisted == _constant_ee_chunk(0.320)


def test_grasp_phase_v3_pickup_and_release_progress_deadlines():
    pickup_config = _v3_phase_config()
    pickup_config.pickup_progress_deadline_chunks = 3
    pickup_corrector = GraspPhaseCorrector(pickup_config)
    pickup_corrector.apply(_constant_ee_chunk(0.400), [0.0] * 7)
    _, telemetry = pickup_corrector.apply(_constant_ee_chunk(0.320), [0.0] * 7)
    assert telemetry["progress_timeout"] is False
    _, telemetry = pickup_corrector.apply(_constant_ee_chunk(0.320), [0.0] * 7)
    assert telemetry["event"] == "pickup_progress_timeout"
    assert telemetry["progress_timeout_kind"] == "pickup"

    release_config = _v3_phase_config()
    release_config.release_progress_deadline_chunks = 2
    release_corrector = GraspPhaseCorrector(release_config)
    release_corrector.apply(_constant_ee_chunk(0.300), [0.0] * 7)
    pickup_ready = [*release_config.pickup_center_rad, 0.380]
    for _ in range(6):
        _, telemetry = release_corrector.apply(
            _constant_ee_chunk(0.300), pickup_ready
        )
    assert telemetry["phase_after"] == "latched"
    _, telemetry = release_corrector.apply(
        _constant_ee_chunk(0.300), pickup_ready
    )
    assert telemetry["progress_timeout"] is False
    assisted, telemetry = release_corrector.apply(
        _constant_ee_chunk(0.300), pickup_ready
    )
    assert telemetry["event"] == "release_progress_timeout"
    assert telemetry["progress_timeout_kind"] == "release"
    assert all(action[6] == pytest.approx(0.320) for action in assisted)



def test_v3_start_gate_accepts_old_good_starts_and_rejects_failed_v2_start():
    config = _v3_phase_config()
    old_fast_start = [0.0349, -1.3895, 3.0110, 0.3983, 0.0280, 0.0524, 0.3699]
    old_v1_start = [0.0630, -1.4331, 3.0559, 0.2654, 0.0024, 0.0513, 0.3913]
    failed_v2_start = [0.0253, -1.3016, 2.8595, 0.4046, 0.0016, -0.0107, 0.3103]

    assert run_firefly_y6.assess_act_start_pose(config, old_fast_start)["passed"]
    assert run_firefly_y6.assess_act_start_pose(config, old_v1_start)["passed"]
    failed = run_firefly_y6.assess_act_start_pose(config, failed_v2_start)
    assert failed["passed"] is False
    assert failed["failed_indices"] == [1, 2]


def test_act_start_reset_trajectory_matches_current_88_frame_envelope():
    start = (
        -0.00611533,
        -1.33870196,
        2.84305787,
        0.46175706,
        0.04463899,
        0.09400427,
        0.30719605,
    )
    target = tuple(_v3_phase_config().act_start_center_rad)
    step_limits = (0.003, 0.003, 0.002, 0.003, 0.003, 0.003, 0.003)

    trajectory = run_firefly_y6.build_act_start_reset_trajectory(
        start, target, step_limits
    )

    assert len(trajectory) == 88
    assert trajectory[-1] == target
    previous = start
    for values in trajectory:
        assert all(
            abs(value - prior) <= limit + 1e-12
            for value, prior, limit in zip(
                values, previous, step_limits, strict=True
            )
        )
        previous = values
    with pytest.raises(ValueError, match="hard 87-frame envelope"):
        run_firefly_y6.build_act_start_reset_trajectory(
            start, target, step_limits, max_frames=87
        )


def test_act_start_reset_trajectory_covers_failed_only_j2_j4_restore():
    start = (
        -0.02254869,
        -1.20249045,
        2.97224998,
        0.03561712,
        0.02479311,
        -0.05294144,
        0.31427121,
    )
    target = (
        start[0],
        _v3_phase_config().act_start_center_rad[1],
        start[2],
        _v3_phase_config().act_start_center_rad[3],
        *start[4:],
    )
    step_limits = (0.003, 0.003, 0.002, 0.003, 0.003, 0.003, 0.003)

    trajectory = run_firefly_y6.build_act_start_reset_trajectory(
        start, target, step_limits
    )

    assert len(trajectory) == 116
    assert trajectory[-1] == target
    assert abs(target[1] - start[1]) < 0.30
    assert abs(target[3] - start[3]) < 0.36


def test_run_act_start_reset_preflights_and_requires_three_measured_passes(
    monkeypatch,
):
    phase_config = _v3_phase_config()
    target = tuple(phase_config.act_start_center_rad)
    start = list(target)
    start[1] += 0.11
    start[4] += 0.02
    names = tuple(
        [f"joint_{index}.pos" for index in range(1, 7)]
        + ["jaw_joint1.pos"]
    )
    events = []

    class FakeRobot:
        action_features = names
        stopped_reason = None
        config = SimpleNamespace(
            resolved_arm_step_limits_rad=lambda: (0.003, 0.003, 0.002, 0.003, 0.003, 0.003),
            ee_step_limit_rad=0.003,
            arm_tracking_error_rad=0.050,
            ee_tracking_error_rad=0.020,
        )

        def __init__(self):
            self.measured = tuple(start)
            self.command_target = dict(zip(names, self.measured, strict=True))

        def preflight_actions(self, actions):
            events.append(("preflight", tuple(actions)))

        def record_event(self, event, **fields):
            events.append((event, fields))

        def get_observation(self):
            return dict(zip(names, self.measured, strict=True))

        def send_action(self, action):
            self.measured = tuple(float(action[name]) for name in names)
            self.command_target = dict(action)
            events.append(("send", self.measured))

        def stop(self, reason):
            self.stopped_reason = reason
            events.append(("stop", reason))

    class FakeRecorder:
        def add_observation(self, _observation):
            events.append(("record",))

    monkeypatch.setattr(run_firefly_y6.time, "sleep", lambda _seconds: None)
    robot = FakeRobot()
    run_firefly_y6.run_act_start_reset(
        SimpleNamespace(fps=30, grasp_phase_correction=phase_config),
        robot,
        FakeRecorder(),
        [False],
    )

    preflight = events[0]
    assert preflight[0] == "preflight"
    assert preflight[1][0] == tuple(start)
    expected_target = list(start)
    expected_target[1] = target[1]
    assert preflight[1][-1] == tuple(expected_target)
    assert len(preflight[1]) == 38
    assert events[1][0] == "act_start_reset_begin"
    assert events[1][1]["corrected_indices"] == (1,)
    assert 4 in events[1][1]["preserved_indices"]
    assert sum(event[0] == "send" for event in events) == 40
    complete = next(event for event in events if event[0] == "act_start_reset_complete")
    assert complete[1]["movement_frames"] == 37
    assert complete[1]["movement_control_frames"] == 37
    assert complete[1]["movement_hold_frames"] == 0
    assert complete[1]["settle_frames"] == 3
    assert complete[1]["consecutive_gate_pass_frames"] == 3
    assert not any(event[0] == "stop" for event in events)


def test_run_act_start_reset_refuses_positive_j3_before_preflight_or_send():
    phase_config = _v3_phase_config()
    center = tuple(phase_config.act_start_center_rad)
    start = list(center)
    start[2] -= 0.10
    names = tuple(
        [f"joint_{index}.pos" for index in range(1, 7)]
        + ["jaw_joint1.pos"]
    )
    events = []

    class FakeRobot:
        action_features = names
        stopped_reason = None
        config = SimpleNamespace(
            resolved_arm_step_limits_rad=lambda: (0.003, 0.003, 0.002, 0.003, 0.003, 0.003),
            ee_step_limit_rad=0.003,
            arm_tracking_error_rad=0.050,
            ee_tracking_error_rad=0.020,
        )
        command_target = dict(zip(names, start, strict=True))

        def preflight_actions(self, _actions):
            pytest.fail("positive J3 reset must be refused before preflight")

        def send_action(self, _action):
            pytest.fail("positive J3 reset must not send")

        def stop(self, reason):
            self.stopped_reason = reason
            events.append(reason)

    with pytest.raises(
        run_firefly_y6.SafetyStopError,
        match="unvalidated positive joint_3 motion",
    ):
        run_firefly_y6.run_act_start_reset(
            SimpleNamespace(fps=30, grasp_phase_correction=phase_config),
            FakeRobot(),
            SimpleNamespace(),
            [False],
        )
    assert len(events) == 1


def test_run_act_start_reset_direct_candidate_budget_crosses_j2_j4_deadband(
    monkeypatch,
):
    phase_config = _v3_phase_config()
    center = tuple(phase_config.act_start_center_rad)
    start = (
        -0.02254869,
        -1.20249045,
        2.97224998,
        0.03561712,
        0.02479311,
        -0.05294144,
        0.31427121,
    )
    names = tuple(
        [f"joint_{index}.pos" for index in range(1, 7)]
        + ["jaw_joint1.pos"]
    )
    events = []
    sent = []
    leads = []

    class FakeRobot:
        action_features = names
        stopped_reason = None
        config = SimpleNamespace(
            resolved_arm_step_limits_rad=lambda: (0.003, 0.003, 0.002, 0.003, 0.003, 0.003),
            ee_step_limit_rad=0.003,
            arm_tracking_error_rad=0.050,
            ee_tracking_error_rad=0.020,
        )

        def __init__(self):
            self.measured = list(start)
            self.command_target = dict(zip(names, start, strict=True))

        def preflight_actions(self, actions):
            events.append(("preflight", tuple(actions)))

        def record_event(self, event, **fields):
            events.append((event, fields))

        def get_observation(self):
            return dict(zip(names, self.measured, strict=True))

        def send_action(self, action):
            desired = tuple(float(action[name]) for name in names)
            sent.append(desired)
            leads.append(
                tuple(
                    abs(desired[index] - self.measured[index])
                    for index in range(len(names))
                )
            )
            for index, deadband, max_step in (
                (1, 0.019, 0.003),
                (3, 0.0205, 0.0005),
            ):
                error = desired[index] - self.measured[index]
                if abs(error) >= deadband:
                    self.measured[index] += (
                        1.0 if error > 0.0 else -1.0
                    ) * min(abs(error), max_step)
            self.command_target = dict(action)

        def stop(self, reason):
            self.stopped_reason = reason
            events.append(("stop", reason))

    monkeypatch.setattr(run_firefly_y6.time, "sleep", lambda _seconds: None)
    robot = FakeRobot()
    run_firefly_y6.run_act_start_reset(
        SimpleNamespace(fps=30, grasp_phase_correction=phase_config),
        robot,
        SimpleNamespace(add_observation=lambda _observation: None),
        [False],
    )

    begin = next(event for event in events if event[0] == "act_start_reset_begin")
    assert begin[1]["corrected_indices"] == (1, 3)
    assert begin[1]["candidate_lead_budget_rad"] == (
        run_firefly_y6._ACT_START_RESET_CANDIDATE_LEAD_BUDGET_RAD
    )
    complete = next(event for event in events if event[0] == "act_start_reset_complete")
    assert complete[1]["movement_frames"] == 116
    assert 600 < complete[1]["movement_control_frames"] < 900
    assert not any(event[0] == "stop" for event in events)
    assert sent
    assert all(values[2] == pytest.approx(start[2]) for values in sent)
    assert max(values[1] for values in leads) <= 0.025 + 1e-12
    assert max(values[3] for values in leads) <= 0.025 + 1e-12
    expected_target = list(start)
    expected_target[1] = center[1]
    expected_target[3] = center[3]
    assert sent[-1] == pytest.approx(expected_target)


def test_run_act_start_reset_fast_follower_crosses_j4_deadband_with_bounded_lead(
    monkeypatch,
):
    phase_config = _v3_phase_config()
    center = tuple(phase_config.act_start_center_rad)
    start = list(center)
    start[2] += 0.1325
    start[3] -= 0.1678
    start[4] += 0.1341
    start = tuple(start)
    names = tuple(
        [f"joint_{index}.pos" for index in range(1, 7)]
        + ["jaw_joint1.pos"]
    )
    events = []
    sent = []
    leads = []

    class FakeRobot:
        action_features = names
        stopped_reason = None
        config = SimpleNamespace(
            resolved_arm_step_limits_rad=lambda: (0.0435,) * 6,
            ee_step_limit_rad=0.0165,
            arm_tracking_error_rad=0.050,
            ee_tracking_error_rad=0.020,
        )

        def __init__(self):
            self.measured = start
            self.command_target = dict(zip(names, start, strict=True))

        def preflight_actions(self, _actions):
            pass

        def record_event(self, event, **fields):
            events.append((event, fields))

        def get_observation(self):
            return dict(zip(names, self.measured, strict=True))

        def send_action(self, action):
            desired = tuple(float(action[name]) for name in names)
            leads.append(
                tuple(
                    abs(desired[index] - self.measured[index])
                    for index in range(len(names))
                )
            )
            measured = list(desired)
            joint_4_error = desired[3] - self.measured[3]
            measured[3] = self.measured[3]
            if abs(joint_4_error) >= 0.0205:
                measured[3] += (
                    1.0 if joint_4_error > 0.0 else -1.0
                ) * min(abs(joint_4_error), 0.0005)
            self.measured = tuple(measured)
            self.command_target = dict(action)
            sent.append(desired)

        def stop(self, reason):
            self.stopped_reason = reason

    monkeypatch.setattr(run_firefly_y6.time, "sleep", lambda _seconds: None)
    robot = FakeRobot()
    run_firefly_y6.run_act_start_reset(
        SimpleNamespace(fps=30, grasp_phase_correction=phase_config),
        robot,
        SimpleNamespace(add_observation=lambda _observation: None),
        [False],
    )

    begin = next(fields for event, fields in events if event == "act_start_reset_begin")
    complete = next(
        fields for event, fields in events if event == "act_start_reset_complete"
    )
    assert begin["corrected_indices"] == (2, 3, 4)
    assert begin["follower_step_limits_rad"] == (0.0435,) * 6 + (0.0165,)
    assert begin["waypoint_step_cap_rad"] == (
        run_firefly_y6._ACT_START_RESET_WAYPOINT_STEP_CAP_RAD
    )
    assert begin["step_limits_rad"] == pytest.approx(
        run_firefly_y6._ACT_START_RESET_WAYPOINT_STEP_CAP_RAD
    )
    assert begin["frames"] == 67
    assert sent[0] != pytest.approx(start)
    assert sent[0] == pytest.approx(
        run_firefly_y6.build_act_start_reset_trajectory(
            start,
            tuple(begin["target"]),
            tuple(begin["step_limits_rad"]),
        )[0]
    )
    assert all(
        lead[index]
        <= run_firefly_y6._ACT_START_RESET_CANDIDATE_LEAD_BUDGET_RAD[index]
        + 1e-12
        for lead in leads
        for index in begin["corrected_indices"]
    )
    assert complete["movement_frames"] == 67
    assert 67 < complete["movement_control_frames"] < 900
    assert not robot.stopped_reason


def test_run_act_start_reset_feedback_stall_stops_with_small_command_lead(
    monkeypatch,
):
    phase_config = _v3_phase_config()
    center = tuple(phase_config.act_start_center_rad)
    start = list(center)
    start[1] += 0.11
    names = tuple(
        [f"joint_{index}.pos" for index in range(1, 7)]
        + ["jaw_joint1.pos"]
    )
    sent = []
    stopped = []

    class FakeRobot:
        action_features = names
        stopped_reason = None
        config = SimpleNamespace(
            resolved_arm_step_limits_rad=lambda: (0.003, 0.003, 0.002, 0.003, 0.003, 0.003),
            ee_step_limit_rad=0.003,
            arm_tracking_error_rad=0.050,
            ee_tracking_error_rad=0.020,
        )
        command_target = dict(zip(names, start, strict=True))

        def preflight_actions(self, _actions):
            pass

        def record_event(self, _event, **_fields):
            pass

        def get_observation(self):
            return dict(zip(names, start, strict=True))

        def send_action(self, action):
            sent.append(tuple(float(action[name]) for name in names))

        def stop(self, reason):
            self.stopped_reason = reason
            stopped.append(reason)

    monkeypatch.setattr(run_firefly_y6.time, "sleep", lambda _seconds: None)
    with pytest.raises(
        run_firefly_y6.SafetyStopError,
        match="no feedback-paced progress",
    ):
        run_firefly_y6.run_act_start_reset(
            SimpleNamespace(fps=30, grasp_phase_correction=phase_config),
            FakeRobot(),
            SimpleNamespace(add_observation=lambda _observation: None),
            [False],
        )

    assert len(stopped) == 1
    assert sent
    assert (
        max(abs(values[1] - start[1]) for values in sent)
        <= run_firefly_y6._ACT_START_RESET_CANDIDATE_LEAD_BUDGET_RAD[1]
        + 1e-12
    )


def test_run_act_refuses_bad_v3_start_before_initial_inference(monkeypatch):
    events = []

    class FakeWorker:
        def __init__(self, _infer):
            events.append("worker_created")

        def close(self):
            events.append("worker_closed")

    class FakeRobot:
        action_features = tuple(
            [f"joint_{index}.pos" for index in range(1, 7)] + ["jaw_joint1.pos"]
        )
        stopped_reason = None

        def get_observation(self):
            values = [0.0253, -1.3016, 2.8595, 0.4046, 0.0016, -0.0107, 0.3103]
            return dict(zip(self.action_features, values, strict=True))

        def record_event(self, event, **fields):
            events.append((event, fields))

        def stop(self, reason):
            self.stopped_reason = reason
            events.append(("stop", reason))

    class FakeRecorder:
        def add_observation(self, _observation):
            events.append("record")

    monkeypatch.setattr(run_firefly_y6, "InferenceWorker", FakeWorker)
    monkeypatch.setattr(
        run_firefly_y6,
        "wait_for_initial_chunk",
        lambda *_args: pytest.fail("initial inference must not start"),
    )
    robot = FakeRobot()
    with pytest.raises(run_firefly_y6.SafetyStopError, match="start pose"):
        run_firefly_y6.run_act(
            SimpleNamespace(
                fps=30,
                n_action_steps=10,
                grasp_phase_correction=_v3_phase_config(),
            ),
            robot,
            infer=lambda _observation: None,
            recorder=FakeRecorder(),
            stopping=[False],
            duration_s=90.0,
        )

    assert events[0:2] == ["worker_created", "record"]
    assert events[2][0] == "act_start_pose_preflight"
    assert events[2][1]["passed"] is False
    assert events[-1] == "worker_closed"


_V5_CONFIG_NAME = (
    "deploy_firefly_y6_hold_validated_v3_trimmed_fast90_per_joint_"
    "gravity_ff_gr80_payload_kp60_kd2p5_eekp150_"
    "phase_latch_v5_recovery_preopen.local.yaml"
)
_V5_LATEST_START = (
    0.012776307761669159, -1.4754520654678345, 2.885098457336426,
    0.31943240761756897, 0.03146570548415184, 0.025195710361003876,
    0.3109680414199829,
)


def _v5_phase_config():
    config = _v3_phase_config()
    config.act_start_tolerance_rad = [0.10, 0.10, 0.15, 0.15, 0.08, 0.18, 0.08]
    config.act_start_ee_preopen_enabled = True
    config.__post_init__()
    return config


def _v5_config_path():
    return Path(__file__).parents[1] / "configs" / _V5_CONFIG_NAME


def test_phase_latch_v5_yaml_allowlist_and_fingerprints_are_exact():
    from lerobot_robot_hexbase.firefly_deployment import compute_deployment_fingerprint
    config = run_firefly_y6.load_config(_v5_config_path())
    legacy_checkpoint = Path(config.checkpoint)
    if not legacy_checkpoint.exists():
        pytest.skip(
            "external legacy v5 checkpoint artifact is unavailable: "
            f"{legacy_checkpoint}"
        )
    assert config.grasp_phase_correction.provenance() == _v5_phase_config().provenance()
    assert config.inference_seed == 1000
    assert "phase_latch_v5_recovery_preopen" in config.robot.client_name
    assert "phase_latch_v5_recovery_preopen" in config.output_dir
    assert compute_deployment_fingerprint(config) == "4e3d35bbfda2163e7d123137c78666cfcbc7dd71f6ad5fb24a1da7eecfc9e858"
    v2 = run_firefly_y6.load_config(_v5_config_path().with_name(_V5_CONFIG_NAME.replace("v5_recovery_preopen", "v2")))
    v3 = run_firefly_y6.load_config(_v5_config_path().with_name(_V5_CONFIG_NAME.replace("v5_recovery_preopen", "v3")))
    assert compute_deployment_fingerprint(v2) == "a6b1b88e2b2e32f1fda4a57e11daa78b88f057701dbc2eb9e51306dc5bb5d8bf"
    assert compute_deployment_fingerprint(v3) == "f25df42b2bd706adf8c435acd57b578f6342c1bdf4f9e9fadc522fe8859a4de3"
    assert run_firefly_y6.robot_config_for_stage(config, "act", Path("act_90s_v5.jsonl")).mode == "execute"
    assert run_firefly_y6.assess_act_start_pose(config.grasp_phase_correction, _V5_LATEST_START)["passed"]
    assert run_firefly_y6.assess_act_start_pose(config.grasp_phase_correction, (*_V5_LATEST_START[:6], 0.380))["passed"]
    failed = run_firefly_y6.assess_act_start_pose(config.grasp_phase_correction, [0.0253, -1.3016, 2.8595, 0.4046, 0.0016, -0.0107, 0.3103])
    assert failed["failed_indices"] == [1, 2]


@pytest.mark.parametrize(("field", "bad_value"), (
    ("act_start_tolerance_rad", [0.10, 0.10, 0.149, 0.15, 0.08, 0.18, 0.08]),
    ("act_start_ee_preopen_enabled", False),
    ("pregrasp_open_target_rad", 0.381),
    ("pregrasp_open_ready_rad", 0.371),
    ("pickup_open_threshold_rad", 0.341),
))
def test_phase_latch_v5_rejects_every_inexact_profile_field(field, bad_value):
    config = run_firefly_y6.load_config(_v5_config_path())
    setattr(config.grasp_phase_correction, field, bad_value)
    with pytest.raises(PermissionError, match="exact fingerprint-bound"):
        run_firefly_y6.robot_config_for_stage(config, "act", Path("act_90s_v5_inexact.jsonl"))


class _V5PreopenRobot:
    action_features = tuple([f"joint_{index}.pos" for index in range(1, 7)] + ["jaw_joint1.pos"])
    config = SimpleNamespace(
        resolved_arm_step_limits_rad=lambda: (0.003, 0.003, 0.002, 0.003, 0.003, 0.003),
        ee_step_limit_rad=0.003,
    )
    def __init__(self, *, command_start=_V5_LATEST_START, measured_start=_V5_LATEST_START,
                 follow=True, settle_ee=None, stopping=None, interrupt_after_sends=None):
        self.command_target = dict(zip(self.action_features, command_start, strict=True))
        self.measured = tuple(measured_start); self.follow = follow; self.settle_ee = settle_ee
        self.stopping = stopping; self.interrupt_after_sends = interrupt_after_sends
        self.preflights = []; self.sent = []; self.events = []; self.stops = []; self.stopped_reason = None
    def get_observation(self):
        measured = self.measured
        if self.settle_ee is not None and len(self.sent) >= 24:
            measured = (*measured[:6], self.settle_ee)
        return dict(zip(self.action_features, measured, strict=True))
    def preflight_actions(self, actions): self.preflights.append(tuple(actions))
    def record_event(self, event, **fields): self.events.append((event, fields))
    def send_action(self, action):
        desired = tuple(float(action[name]) for name in self.action_features)
        self.sent.append(desired); self.command_target = dict(action)
        if self.follow: self.measured = desired
        if self.interrupt_after_sends is not None and len(self.sent) == self.interrupt_after_sends:
            self.stopping[0] = True
    def stop(self, reason): self.stopped_reason = reason; self.stops.append(reason)


def _run_v5_preopen(monkeypatch, robot, stopping=None):
    monkeypatch.setattr(run_firefly_y6.time, "sleep", lambda _seconds: None)
    run_firefly_y6.run_act_ee_preopen(
        SimpleNamespace(fps=30, grasp_phase_correction=_v5_phase_config()), robot,
        SimpleNamespace(add_observation=lambda _observation: None),
        [False] if stopping is None else stopping,
    )


def test_run_act_ee_preopen_is_exact_24_frame_arm_hold(monkeypatch):
    robot = _V5PreopenRobot(); _run_v5_preopen(monkeypatch, robot)
    assert len(robot.preflights) == 1 and len(robot.preflights[0]) == 25
    assert robot.preflights[0][0] == _V5_LATEST_START
    movement = robot.sent[:24]
    assert len(movement) == 24
    assert all(values[:6] == _V5_LATEST_START[:6] for values in robot.sent)
    assert all(left[6] < right[6] for left, right in zip((_V5_LATEST_START, *movement[:-1]), movement, strict=True))
    assert max(right[6] - left[6] for left, right in zip((_V5_LATEST_START, *movement[:-1]), movement, strict=True)) <= 0.003 + 1e-12
    assert movement[-1][6] == 0.380
    assert len(robot.sent) == 27
    assert all(values == (*_V5_LATEST_START[:6], 0.380) for values in robot.sent[23:])
    assert robot.stops == []
    begin = next(fields for event, fields in robot.events if event == "act_start_ee_preopen_begin")
    complete = next(fields for event, fields in robot.events if event == "act_start_ee_preopen_complete")
    assert begin["movement_frames"] == complete["movement_frames"] == 24
    assert begin["opening_travel_rad"] == pytest.approx(0.0690319585800171)
    assert complete["settle_frames"] == complete["consecutive_pass_frames"] == 3


@pytest.mark.parametrize(("kind", "message"), (("travel", "opening travel"), ("command_nan", "finite"), ("measured_nan", "finite")))
def test_run_act_ee_preopen_nonfinite_and_travel_fail_closed(monkeypatch, kind, message):
    command = list(_V5_LATEST_START); measured = list(_V5_LATEST_START)
    if kind == "travel": command[6] = measured[6] = 0.299
    elif kind == "command_nan": command[6] = float("nan")
    else: measured[6] = float("nan")
    robot = _V5PreopenRobot(command_start=command, measured_start=measured)
    with pytest.raises(run_firefly_y6.SafetyStopError, match=message): _run_v5_preopen(monkeypatch, robot)
    assert len(robot.stops) == 1 and robot.sent == []


@pytest.mark.parametrize("settle_ee", (0.369, 0.391))
def test_run_act_ee_preopen_low_or_overshoot_feedback_never_settles(monkeypatch, settle_ee):
    robot = _V5PreopenRobot(follow=False, settle_ee=settle_ee)
    with pytest.raises(run_firefly_y6.SafetyStopError, match="did not settle"): _run_v5_preopen(monkeypatch, robot)
    assert len(robot.stops) == 1
    assert len(robot.sent) == 24 + run_firefly_y6._ACT_START_EE_PREOPEN_SETTLE_FRAMES
    assert not any(event == "act_start_ee_preopen_complete" for event, _ in robot.events)


@pytest.mark.parametrize("phase", ("movement", "settle"))
def test_run_act_ee_preopen_interrupts_fail_closed(monkeypatch, phase):
    stopping = [phase == "movement"]
    robot = _V5PreopenRobot(stopping=stopping, interrupt_after_sends=24 if phase == "settle" else None)
    with pytest.raises(run_firefly_y6.SafetyStopError, match="operator interrupt"): _run_v5_preopen(monkeypatch, robot, stopping)
    assert len(robot.stops) == 1 and len(robot.sent) == (0 if phase == "movement" else 24)
    assert not any(event == "act_start_ee_preopen_complete" for event, _ in robot.events)


def test_load_policy_reseeds_each_inference_generation(monkeypatch):
    _config, _records = _install_fake_policy_stack(monkeypatch, "smolvla")
    python_seeds = []
    numpy_seeds = []
    torch_seeds = []
    monkeypatch.setattr(run_firefly_y6.random, "seed", python_seeds.append)
    monkeypatch.setattr(run_firefly_y6.np.random, "seed", numpy_seeds.append)
    monkeypatch.setattr(torch, "manual_seed", torch_seeds.append)

    infer = run_firefly_y6.load_policy(
        Path("/tmp/fake-checkpoint"),
        "cpu",
        10,
        _TASK,
        _VLM_SNAPSHOT,
        inference_seed=1000,
    )
    infer(_raw_observation())
    infer(_raw_observation())

    assert python_seeds == [1000, 1001]
    assert numpy_seeds == [1000, 1001]
    assert torch_seeds == [1000, 1001]


def test_phase_latch_v3_yaml_and_seed_pair_are_exact():
    path = (
        Path(__file__).parents[1]
        / "configs"
        / (
            "deploy_firefly_y6_hold_validated_v3_trimmed_fast90_per_joint_"
            "gravity_ff_gr80_payload_kp60_kd2p5_eekp150_phase_latch_v3.local.yaml"
        )
    )
    config = run_firefly_y6.load_config(path)
    assert config.inference_seed == 1000
    assert config.grasp_phase_correction.provenance() == _v3_phase_config().provenance()
    assert "phase_latch_v3" in config.robot.client_name
    assert "phase_latch_v3" in config.output_dir
    assert run_firefly_y6.robot_config_for_stage(
        config, "act", Path("act_90s_v3.jsonl")
    ).mode == "execute"

    config.inference_seed = 1001
    with pytest.raises(PermissionError, match="v1, v2, v3, or v5"):
        run_firefly_y6.robot_config_for_stage(
            config, "act", Path("act_90s_v3_bad_seed.jsonl")
        )


def test_run_act_progress_timeout_records_then_stops_before_preflight_or_send(
    monkeypatch,
):
    events = []

    class FakeWorker:
        def __init__(self, _infer):
            pass

        def close(self):
            events.append("worker_closed")

    class StubCorrector:
        def __init__(self, _config):
            pass

        def apply(self, raw_actions, _reference):
            return raw_actions, {
                "enabled": True,
                "progress_timeout": True,
                "progress_timeout_kind": "pickup",
                "raw_open_progress_deadline_chunks": 45,
                "pickup_progress_deadline_chunks": 180,
                "release_progress_deadline_chunks": 90,
            }

    class FakeRobot:
        action_features = tuple(
            [f"joint_{index}.pos" for index in range(1, 7)] + ["jaw_joint1.pos"]
        )
        command_target = dict.fromkeys(action_features, 0.0)
        stopped_reason = None

        def get_observation(self):
            return dict.fromkeys(self.action_features, 0.0)

        def record_event(self, event, **fields):
            events.append((event, fields))

        def preflight_actions(self, _actions):
            pytest.fail("timeout chunk must not reach preflight")

        def send_action_with_metadata(self, _action, **_metadata):
            pytest.fail("timeout chunk must not be sent")

        def stop(self, reason):
            self.stopped_reason = reason
            events.append(("stop", reason))

    class FakeRecorder:
        def add_observation(self, _observation):
            pass

    initial_chunk = SimpleNamespace(
        generation=180,
        actions=_constant_ee_chunk(0.320),
        latency_s=0.01,
        error=None,
    )
    monkeypatch.setattr(run_firefly_y6, "InferenceWorker", FakeWorker)
    monkeypatch.setattr(run_firefly_y6, "GraspPhaseCorrector", StubCorrector)
    monkeypatch.setattr(
        run_firefly_y6,
        "wait_for_initial_chunk",
        lambda *_args: initial_chunk,
    )
    with pytest.raises(
        run_firefly_y6.SafetyStopError,
        match="demonstrated pickup pose",
    ):
        run_firefly_y6.run_act(
            SimpleNamespace(
                fps=30,
                n_action_steps=10,
                enable_action_slew_rate_adapter=False,
                grasp_phase_correction=SimpleNamespace(act_start_center_rad=[]),
            ),
            FakeRobot(),
            infer=lambda _observation: None,
            recorder=FakeRecorder(),
            stopping=[False],
            duration_s=90.0,
        )

    assert events[0][0] == "inference_chunk"
    assert events[0][1]["grasp_phase_correction"]["progress_timeout_kind"] == "pickup"
    assert events[1][0] == "stop"
    assert events[-1] == "worker_closed"


_VISUAL_ACTION_FEATURES = tuple(
    [f"joint_{index}.pos" for index in range(1, 7)] + ["jaw_joint1.pos"]
)
_VISUAL_HOME = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.314)
_VISUAL_HOME_TOLERANCE = (0.02,) * 7


def _visual_config(**overrides) -> VisualPhaseLatchConfig:
    return replace(VisualPhaseLatchConfig(enabled=True), **overrides)


def _visual_runtime(config: VisualPhaseLatchConfig | None = None):
    return run_firefly_y6.VisualLatchRuntime(
        config or _visual_config(),
        home_center_rad=_VISUAL_HOME,
        home_tolerance_rad=_VISUAL_HOME_TOLERANCE,
    )


def _red_grasp_image() -> np.ndarray:
    """Raw, already-rotated D405 image; rotating it again misses the close ROI."""

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[211:307, 234:330] = (255, 0, 0)
    return image


def _visual_observation(
    *,
    image: np.ndarray | None = None,
    arm: tuple[float, ...] = (0.0,) * 6,
    ee: float = 0.41,
) -> dict[str, object]:
    observation = {
        name: float(value)
        for name, value in zip(
            _VISUAL_ACTION_FEATURES,
            (*arm, ee),
            strict=True,
        )
    }
    observation["wrist"] = (
        np.zeros((480, 640, 3), dtype=np.uint8) if image is None else image
    )
    return observation


class _VisualRuntimeRobot:
    action_features = _VISUAL_ACTION_FEATURES

    def __init__(self, *, sequence: int = 1, captured_at_s: float = 9.99):
        self.sequence = sequence
        self.captured_at_s = captured_at_s
        self.command_target = dict.fromkeys(self.action_features, 0.0)

    def get_camera_frame_metadata(self):
        return {
            "wrist": CameraFrameMetadata(
                camera_name="wrist",
                sequence=self.sequence,
                captured_at_s=self.captured_at_s,
                observed_at_s=self.captured_at_s,
                age_s=0.0,
                timestamp_source="host_perf_counter",
                is_new_frame=True,
            )
        }

    def get_tcp_pose(self, _measured_arm_q):
        return (0.0, 0.0, 0.10), np.eye(3)


def _tip_red_image(
    side_px: int = 74, *, center_px: tuple[int, int] = (294, 235)
) -> np.ndarray:
    """One coarse/tracking red target centred short of the close-ready ROI."""

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    center_x, center_y = center_px
    x0 = center_x - side_px // 2
    y0 = center_y - side_px // 2
    image[y0 : y0 + side_px, x0 : x0 + side_px] = (255, 0, 0)
    return image


def _tip_test_rotation(arm_q) -> np.ndarray:
    q4, q5, q6 = (float(value) for value in arm_q[3:6])
    c4, s4 = np.cos(q4), np.sin(q4)
    c5, s5 = np.cos(q5), np.sin(q5)
    c6, s6 = np.cos(q6), np.sin(q6)
    rx = np.array(((1.0, 0.0, 0.0), (0.0, c4, -s4), (0.0, s4, c4)))
    ry = np.array(((c5, 0.0, s5), (0.0, 1.0, 0.0), (-s5, 0.0, c5)))
    rz = np.array(((c6, -s6, 0.0), (s6, c6, 0.0), (0.0, 0.0, 1.0)))
    return rz @ ry @ rx


class _TipRuntimeRobot(_VisualRuntimeRobot):
    """Software-only follower with a full-rank FK for DLS integration tests."""

    def __init__(self):
        super().__init__(sequence=0, captured_at_s=9.99)
        self.command_target = dict(
            zip(_VISUAL_ACTION_FEATURES, (0.0,) * 6 + (0.41,), strict=True)
        )
        arm_limits = SimpleNamespace(
            pos_min=(-2.0,) * 6,
            pos_max=(2.0,) * 6,
            vel_max=(3.0,) * 6,
        )
        self.controller_limits = (arm_limits, SimpleNamespace())
        self.config = SimpleNamespace(
            command_hz=30.0,
            resolved_arm_step_limits_rad=lambda: (0.0435,) * 6,
        )
        self.preflights = []
        self.sent = []
        self.events = []
        self.stopped_reason = None

    def get_tcp_pose(self, arm_q):
        q = tuple(float(value) for value in arm_q)
        # Make J2 the world-Z degree of freedom while retaining a full-rank
        # six-axis numerical Jacobian around the test pose.
        return (q[0], q[2], 0.10 + q[1]), _tip_test_rotation(q)

    def preflight_actions(self, actions):
        self.preflights.append(actions)
        self.events.append(("preflight", actions))

    def send_action_with_metadata(self, action, **metadata):
        _validated_visual_telemetry(metadata["visual_telemetry"])
        self.command_target = {
            name: float(action[name]) for name in self.action_features
        }
        self.sent.append((dict(self.command_target), metadata))
        self.events.append(("send", dict(self.command_target), metadata))

    def record_event(self, event, **fields):
        self.events.append((event, fields))

    def stop(self, reason):
        self.stopped_reason = reason
        self.events.append(("stop", reason))


def _tip_runtime() -> run_firefly_y6.VisualLatchRuntime:
    visual = _visual_config()
    visual = replace(
        visual,
        tip_servo=replace(
            visual.tip_servo,
            mode=run_firefly_y6.TipServoMode.EXECUTE,
        ),
    )
    return _visual_runtime(visual)


def _tip_open_only_runtime() -> run_firefly_y6.VisualLatchRuntime:
    holder = SimpleNamespace(
        act_duration_s=30.0,
        visual_phase_latch=_visual_config(),
    )
    _enable_tip_servo_open_only_main_profile(holder)
    return _visual_runtime(holder.visual_phase_latch)


def _tip_reference(robot: _TipRuntimeRobot) -> tuple[float, ...]:
    return tuple(float(robot.command_target[name]) for name in robot.action_features)


def _send_tip_frame(
    robot: _TipRuntimeRobot,
    runtime: run_firefly_y6.VisualLatchRuntime,
    clock: dict[str, int],
    *,
    image: np.ndarray,
    action_source=run_firefly_y6.VisualActionSource.FRESH_POLICY,
    measured_arm: tuple[float, ...] | None = None,
    reference: tuple[float, ...] | None = None,
    policy_action: tuple[float, ...] | None = None,
    measured_ee: float = 0.41,
    fresh: bool = True,
    elapsed_ns: int = 33_000_000,
    arm6_runtime: run_firefly_y6.Arm6SeekingRuntime | None = None,
):
    clock["now_ns"] += elapsed_ns
    if fresh:
        robot.sequence += 1
        robot.captured_at_s = (clock["now_ns"] - 10_000_000) / 1_000_000_000.0
    if reference is None:
        reference = _tip_reference(robot)
    if measured_arm is None:
        measured_arm = tuple(reference[:6])
    if policy_action is None:
        policy_action = tuple(reference)
    observation = _visual_observation(
        image=image,
        arm=tuple(float(value) for value in measured_arm),
        ee=measured_ee,
    )
    kwargs = {"action_source": action_source}
    if arm6_runtime is not None:
        kwargs["arm6_runtime"] = arm6_runtime
    return run_firefly_y6.send_visual_control_cycle(
        robot,
        runtime,
        observation,
        policy_action,
        reference,
        **kwargs,
    )


def _drive_tip_runtime_to_descending(
    robot: _TipRuntimeRobot,
    runtime: run_firefly_y6.VisualLatchRuntime,
    clock: dict[str, int],
):
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    coarse = _tip_red_image(74)
    _send_tip_frame(robot, runtime, clock, image=blank)
    takeover = None
    for _ in range(runtime.tip_config.takeover_confirm_frames):
        takeover = _send_tip_frame(robot, runtime, clock, image=coarse)
    assert takeover is not None
    assert takeover.discard_policy_chunk is True
    assert runtime.tip_state.phase is run_firefly_y6.TipServoPhase.OPENING

    opening = []
    for _ in range(runtime.tip_config.open_confirm_frames):
        opening.append(
            _send_tip_frame(
                robot,
                runtime,
                clock,
                image=coarse,
                action_source=run_firefly_y6.VisualActionSource.TIP_SERVO,
            )
        )
    assert runtime.tip_state.phase is run_firefly_y6.TipServoPhase.DESCENDING
    return takeover, opening


def test_open_only_seeking_allows_policy_arm_but_isolates_both_jaw_writers(
    monkeypatch,
):
    clock = {"now_ns": 10_000_000_000}
    monkeypatch.setattr(
        run_firefly_y6.time,
        "perf_counter_ns",
        lambda: clock["now_ns"],
    )
    robot = _TipRuntimeRobot()
    runtime = _tip_open_only_runtime()
    accepted = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.31)
    policy = (0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05)
    robot.command_target = dict(
        zip(_VISUAL_ACTION_FEATURES, accepted, strict=True)
    )
    # Force the legacy/main latch to request its 0.41-rad pregrasp target.
    # OPEN_ONLY must ignore that jaw writer just as it ignores policy jaw.
    runtime.latch.pregrasp_open_seen = True

    cycle = _send_tip_frame(
        robot,
        runtime,
        clock,
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        measured_arm=accepted[:6],
        reference=accepted,
        policy_action=policy,
        measured_ee=accepted[6],
    )

    assert runtime.tip_state.phase is run_firefly_y6.TipServoPhase.SEEKING
    assert runtime.tip_state.capture_jaw_target_rad == pytest.approx(accepted[6])
    assert cycle.action[0] == pytest.approx(policy[0])
    assert cycle.action[6] == pytest.approx(accepted[6])
    assert cycle.jaw_owner == "tip_capture_hold"
    tip = cycle.telemetry["tip_servo"]
    assert tip["jaw_owner"] == "tip_capture_hold"
    assert tip["capture_jaw_target_rad"] == pytest.approx(accepted[6])
    assert tip["raw_policy_jaw_target_rad"] == pytest.approx(policy[6])
    assert tip["main_latch_jaw_target_rad_ignored"] == pytest.approx(0.41)
    assert tip["planner_jaw_target_rad"] == pytest.approx(accepted[6])
    assert tip["published_jaw_target_rad"] == pytest.approx(accepted[6])
    assert tip["previous_accepted_jaw_target_rad"] == pytest.approx(accepted[6])
    assert tip["no_close_guard_passed"] is True
    assert tip["seeking_timeout_ms"] == pytest.approx(20_000.0)


def test_arm6_open_only_has_no_policy_jaw_and_compact_source_telemetry(monkeypatch):
    clock = {"now_ns": 10_000_000_000}
    monkeypatch.setattr(
        run_firefly_y6.time,
        "perf_counter_ns",
        lambda: clock["now_ns"],
    )
    robot = _TipRuntimeRobot()
    runtime = _tip_open_only_runtime()
    accepted = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.31)
    robot.command_target = dict(
        zip(_VISUAL_ACTION_FEATURES, accepted, strict=True)
    )
    arm6_runtime = run_firefly_y6.bind_arm6_seeking_runtime(
        _arm6_stub_infer(), accepted
    )
    # Force the main latch to request 0.41 rad. It remains telemetry-only;
    # neither it nor the six-axis policy has authority over the capture jaw.
    runtime.latch.pregrasp_open_seen = True

    cycle = _send_tip_frame(
        robot,
        runtime,
        clock,
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        measured_arm=accepted[:6],
        reference=accepted,
        policy_action=(0.02, 0.0, 0.0, 0.0, 0.0, 0.0, accepted[6]),
        measured_ee=accepted[6],
        arm6_runtime=arm6_runtime,
    )

    assert cycle.action[0] == pytest.approx(0.02)
    assert cycle.action[6] == pytest.approx(accepted[6])
    assert cycle.jaw_owner == "tip_capture_hold"
    tip = cycle.telemetry["tip_servo"]
    assert "raw_policy_jaw_target_rad" not in tip
    assert tip["policy_role"] == "arm6_seeking"
    assert tip["jaw_candidate_source"] == "startup_accepted_reference"
    assert tip["main_latch_jaw_target_rad_ignored"] == pytest.approx(0.41)
    assert tip["capture_jaw_target_rad"] == pytest.approx(accepted[6])
    assert tip["seeking_timeout_ms"] == pytest.approx(20_000.0)
    assert _validated_visual_telemetry(cycle.telemetry) == cycle.telemetry

    def primitive_nodes(value):
        if isinstance(value, dict):
            return 1 + sum(primitive_nodes(child) for child in value.values())
        if isinstance(value, list):
            return 1 + sum(primitive_nodes(child) for child in value)
        return 1

    assert primitive_nodes(cycle.telemetry) <= 128

    with pytest.raises(ValueError, match="differs from the immutable startup"):
        _send_tip_frame(
            robot,
            runtime,
            clock,
            image=np.zeros((480, 640, 3), dtype=np.uint8),
            reference=accepted,
            policy_action=(0.0,) * 6 + (0.30,),
            measured_ee=accepted[6],
            arm6_runtime=arm6_runtime,
        )


def test_arm6_takeover_discards_chunk_resets_smoother_and_removes_policy_jaw(
    monkeypatch,
):
    clock = {"now_ns": 10_000_000_000}
    monkeypatch.setattr(
        run_firefly_y6.time,
        "perf_counter_ns",
        lambda: clock["now_ns"],
    )
    robot = _TipRuntimeRobot()
    runtime = _tip_open_only_runtime()
    accepted = (0.0,) * 6 + (0.31,)
    robot.command_target = dict(
        zip(_VISUAL_ACTION_FEATURES, accepted, strict=True)
    )
    arm6_runtime = run_firefly_y6.bind_arm6_seeking_runtime(
        _arm6_stub_infer(), accepted
    )
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    coarse = _tip_red_image(74, center_px=(334, 235))
    candidate = accepted

    _send_tip_frame(
        robot,
        runtime,
        clock,
        image=blank,
        reference=accepted,
        policy_action=candidate,
        measured_ee=accepted[6],
        arm6_runtime=arm6_runtime,
    )
    takeover = None
    for _ in range(runtime.tip_config.takeover_confirm_frames + 2):
        takeover = _send_tip_frame(
            robot,
            runtime,
            clock,
            image=coarse,
            reference=accepted,
            policy_action=candidate,
            measured_ee=accepted[6],
            arm6_runtime=arm6_runtime,
        )
        if takeover.discard_policy_chunk:
            break

    assert takeover is not None
    assert takeover.discard_policy_chunk is True
    assert takeover.smoother_reset_required is True
    assert takeover.replan_reason == "tip_servo_takeover"
    assert takeover.action == pytest.approx(accepted)
    tip = takeover.telemetry["tip_servo"]
    assert tip["discard_policy_chunk"] is True
    assert tip["smoother_reset_required"] is True
    assert tip["policy_role"] == "arm6_seeking"
    assert "raw_policy_jaw_target_rad" not in tip
    assert _validated_visual_telemetry(takeover.telemetry) == takeover.telemetry

    exclusive_reference = _tip_reference(robot)
    opening = _send_tip_frame(
        robot,
        runtime,
        clock,
        image=coarse,
        action_source=run_firefly_y6.VisualActionSource.TIP_SERVO,
        reference=exclusive_reference,
        policy_action=exclusive_reference,
        measured_ee=exclusive_reference[6],
        arm6_runtime=arm6_runtime,
    )
    opening_tip = opening.telemetry["tip_servo"]
    assert opening_tip["jaw_candidate_source"] == "tip_servo_exclusive_reference"
    assert "raw_policy_jaw_target_rad" not in opening_tip
    assert _validated_visual_telemetry(opening.telemetry) == opening.telemetry


def test_open_only_seeking_timeout_publishes_exact_accepted_hold_then_stops(
    monkeypatch,
):
    clock = {"now_ns": 10_000_000_000}
    monkeypatch.setattr(
        run_firefly_y6.time,
        "perf_counter_ns",
        lambda: clock["now_ns"],
    )
    robot = _TipRuntimeRobot()
    runtime = _tip_open_only_runtime()
    accepted = (0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.31)
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    robot.command_target = dict(
        zip(_VISUAL_ACTION_FEATURES, accepted, strict=True)
    )

    _send_tip_frame(
        robot,
        runtime,
        clock,
        image=blank,
        measured_arm=accepted[:6],
        reference=accepted,
        policy_action=(*accepted[:6], 0.0),
        measured_ee=accepted[6],
    )
    robot.events.clear()

    with pytest.raises(
        run_firefly_y6.SafetyStopError,
        match="external:seeking_timeout",
    ):
        _send_tip_frame(
            robot,
            runtime,
            clock,
            image=blank,
            measured_arm=accepted[:6],
            reference=accepted,
            policy_action=(*accepted[:6], 0.0),
            measured_ee=0.0,
            elapsed_ns=20_000_000_000,
        )

    assert runtime.tip_state.phase is run_firefly_y6.TipServoPhase.FAULT_HOLD
    assert runtime.tip_state.fault_reason == "seeking_timeout"
    assert [event[0] for event in robot.events[:3]] == [
        "preflight",
        "send",
        "stop",
    ]
    sent_action = tuple(
        robot.events[1][1][name] for name in robot.action_features
    )
    assert sent_action == pytest.approx(accepted)
    tip = robot.events[1][2]["visual_telemetry"]["tip_servo"]
    assert tip["jaw_owner"] == "tip_fault_hold"
    assert tip["capture_jaw_target_rad"] == pytest.approx(accepted[6])
    assert tip["published_jaw_target_rad"] == pytest.approx(accepted[6])
    assert tip["previous_accepted_jaw_target_rad"] == pytest.approx(accepted[6])
    assert tip["no_close_guard_passed"] is True
    assert tip["seeking_elapsed_ms"] == pytest.approx(20_000.0)
    assert tip["seeking_timeout_ms"] == pytest.approx(20_000.0)


@pytest.mark.parametrize("invalid_capture", [0.284, 0.326])
def test_open_only_rejects_invalid_accepted_capture_with_one_exact_hold(
    monkeypatch,
    invalid_capture,
):
    clock = {"now_ns": 10_000_000_000}
    monkeypatch.setattr(
        run_firefly_y6.time,
        "perf_counter_ns",
        lambda: clock["now_ns"],
    )
    robot = _TipRuntimeRobot()
    runtime = _tip_open_only_runtime()
    accepted = (0.01, 0.0, 0.0, 0.0, 0.0, 0.0, invalid_capture)
    robot.command_target = dict(
        zip(_VISUAL_ACTION_FEATURES, accepted, strict=True)
    )

    with pytest.raises(run_firefly_y6.SafetyStopError):
        _send_tip_frame(
            robot,
            runtime,
            clock,
            image=np.zeros((480, 640, 3), dtype=np.uint8),
            measured_arm=accepted[:6],
            reference=accepted,
            policy_action=(*accepted[:6], 0.0),
            measured_ee=invalid_capture,
        )

    assert runtime.tip_state.phase is run_firefly_y6.TipServoPhase.FAULT_HOLD
    assert [event[0] for event in robot.events[:3]] == [
        "preflight",
        "send",
        "stop",
    ]
    sent_action = tuple(
        robot.events[1][1][name] for name in robot.action_features
    )
    assert sent_action == pytest.approx(accepted)
    tip = robot.events[1][2]["visual_telemetry"]["tip_servo"]
    assert tip["capture_jaw_target_rad"] is None
    assert tip["published_jaw_target_rad"] == pytest.approx(invalid_capture)
    assert tip["no_close_guard_passed"] is False


def test_open_only_bound_capture_rejects_low_accepted_reference_without_new_close(
    monkeypatch,
):
    clock = {"now_ns": 10_000_000_000}
    monkeypatch.setattr(
        run_firefly_y6.time,
        "perf_counter_ns",
        lambda: clock["now_ns"],
    )
    robot = _TipRuntimeRobot()
    runtime = _tip_open_only_runtime()
    capture = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.31)
    robot.command_target = dict(
        zip(_VISUAL_ACTION_FEATURES, capture, strict=True)
    )
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    _send_tip_frame(
        robot,
        runtime,
        clock,
        image=blank,
        measured_arm=capture[:6],
        reference=capture,
        policy_action=(*capture[:6], 0.0),
        measured_ee=capture[6],
    )
    robot.events.clear()
    already_accepted_low = (*capture[:6], 0.20)

    with pytest.raises(run_firefly_y6.SafetyStopError):
        _send_tip_frame(
            robot,
            runtime,
            clock,
            image=blank,
            measured_arm=capture[:6],
            reference=already_accepted_low,
            policy_action=(*capture[:6], 0.0),
            measured_ee=capture[6],
        )

    assert [event[0] for event in robot.events[:3]] == [
        "preflight",
        "send",
        "stop",
    ]
    sent_action = tuple(
        robot.events[1][1][name] for name in robot.action_features
    )
    # The guard does not synthesize another closing target. It republishes the
    # controller's already-accepted reference exactly once, then disconnects.
    assert sent_action == pytest.approx(already_accepted_low)
    tip = robot.events[1][2]["visual_telemetry"]["tip_servo"]
    assert tip["capture_jaw_target_rad"] == pytest.approx(capture[6])
    assert tip["previous_accepted_jaw_target_rad"] == pytest.approx(0.20)
    assert tip["published_jaw_target_rad"] == pytest.approx(0.20)
    assert tip["no_close_guard_passed"] is False


def test_open_only_final_publish_assertion_rejects_downstream_jaw_tampering(
    monkeypatch,
):
    clock = {"now_ns": 10_000_000_000}
    monkeypatch.setattr(
        run_firefly_y6.time,
        "perf_counter_ns",
        lambda: clock["now_ns"],
    )
    robot = _TipRuntimeRobot()
    runtime = _tip_open_only_runtime()
    accepted = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.31)
    robot.command_target = dict(
        zip(_VISUAL_ACTION_FEATURES, accepted, strict=True)
    )
    original_apply = runtime.j23_target_smoother.apply

    def tamper_jaw(desired_action, command_reference, *, forced_reset_reason=None):
        smoothed, telemetry = original_apply(
            desired_action,
            command_reference,
            forced_reset_reason=forced_reset_reason,
        )
        return (*smoothed[:6], smoothed[6] - 0.001), telemetry

    monkeypatch.setattr(runtime.j23_target_smoother, "apply", tamper_jaw)

    with pytest.raises(
        run_firefly_y6.SafetyStopError,
        match="preflight-safe command: TipServoError",
    ):
        _send_tip_frame(
            robot,
            runtime,
            clock,
            image=np.zeros((480, 640, 3), dtype=np.uint8),
            measured_arm=accepted[:6],
            reference=accepted,
            policy_action=accepted,
            measured_ee=accepted[6],
        )

    assert robot.sent == []
    assert [event[0] for event in robot.events] == ["stop"]


def test_open_only_final_publish_assertion_rejects_post_takeover_arm_tampering(
    monkeypatch,
):
    clock = {"now_ns": 10_000_000_000}
    monkeypatch.setattr(
        run_firefly_y6.time,
        "perf_counter_ns",
        lambda: clock["now_ns"],
    )
    robot = _TipRuntimeRobot()
    runtime = _tip_open_only_runtime()
    accepted = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.31)
    robot.command_target = dict(
        zip(_VISUAL_ACTION_FEATURES, accepted, strict=True)
    )
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    coarse = _tip_red_image(74, center_px=(334, 235))

    _send_tip_frame(
        robot,
        runtime,
        clock,
        image=blank,
        measured_arm=accepted[:6],
        reference=accepted,
        policy_action=accepted,
        measured_ee=accepted[6],
    )
    for _ in range(runtime.tip_config.takeover_confirm_frames):
        _send_tip_frame(
            robot,
            runtime,
            clock,
            image=coarse,
            measured_arm=accepted[:6],
            reference=accepted,
            policy_action=accepted,
            measured_ee=accepted[6],
        )
    assert runtime.tip_state.phase is run_firefly_y6.TipServoPhase.OPENING

    original_apply = runtime.j23_target_smoother.apply

    def tamper_arm(desired_action, command_reference, *, forced_reset_reason=None):
        smoothed, telemetry = original_apply(
            desired_action,
            command_reference,
            forced_reset_reason=forced_reset_reason,
        )
        tampered = list(smoothed)
        tampered[1] += 0.001
        return tuple(tampered), telemetry

    monkeypatch.setattr(runtime.j23_target_smoother, "apply", tamper_arm)
    robot.sent.clear()
    robot.events.clear()

    with pytest.raises(
        run_firefly_y6.SafetyStopError,
        match="preflight-safe command: TipServoError",
    ):
        _send_tip_frame(
            robot,
            runtime,
            clock,
            image=coarse,
            action_source=run_firefly_y6.VisualActionSource.TIP_SERVO,
            measured_arm=accepted[:6],
            reference=accepted,
            policy_action=(*accepted[:6], 0.0),
            measured_ee=accepted[6],
        )

    assert robot.sent == []
    assert [event[0] for event in robot.events] == ["stop"]


def test_open_only_tip_accumulates_hold_frames_then_opens_without_arm_motion(
    monkeypatch,
):
    clock = {"now_ns": 10_000_000_000}
    monkeypatch.setattr(
        run_firefly_y6.time,
        "perf_counter_ns",
        lambda: clock["now_ns"],
    )
    robot = _TipRuntimeRobot()
    runtime = _tip_open_only_runtime()
    accepted = (0.034, 0.0, 0.0, 0.0, 0.0, 0.0, 0.31)
    measured_at_capture = (0.0,) * 6
    robot.command_target = dict(
        zip(_VISUAL_ACTION_FEATURES, accepted, strict=True)
    )
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    coarse = _tip_red_image(74, center_px=(334, 235))

    _send_tip_frame(
        robot,
        runtime,
        clock,
        image=blank,
        measured_arm=measured_at_capture,
        reference=accepted,
        policy_action=accepted,
        measured_ee=0.31,
    )
    sources = (
        (run_firefly_y6.VisualActionSource.FRESH_POLICY,) * 2
        + (run_firefly_y6.VisualActionSource.INFERENCE_HOLD,) * 5
        + (run_firefly_y6.VisualActionSource.FRESH_POLICY,)
    )
    cycles = [
        _send_tip_frame(
            robot,
            runtime,
            clock,
            image=coarse,
            action_source=source,
            measured_arm=measured_at_capture,
            reference=accepted,
            policy_action=accepted,
            measured_ee=0.31,
        )
        for source in sources
    ]

    assert [cycle.telemetry["tip_servo"]["event"] for cycle in cycles] == (
        ["coarse_candidate"] * 5
        + ["coarse_ready_waiting_authorization"] * 2
        + ["coarse_takeover"]
    )
    assert all(cycle.discard_policy_chunk is False for cycle in cycles[:-1])
    takeover = cycles[-1]
    assert takeover.discard_policy_chunk is True
    assert takeover.action[:6] == pytest.approx(accepted[:6])
    assert takeover.action[6] == pytest.approx(0.31)
    assert runtime.tip_state.takeover_q_rad == pytest.approx(measured_at_capture)
    assert runtime.tip_state.last_arm_target_q_rad == pytest.approx(accepted[:6])
    assert runtime.tip_state.phase is run_firefly_y6.TipServoPhase.OPENING

    active_cycles = []
    measured_ee = 0.31
    for _ in range(20):
        cycle = _send_tip_frame(
            robot,
            runtime,
            clock,
            image=coarse,
            action_source=run_firefly_y6.VisualActionSource.TIP_SERVO,
            measured_arm=accepted[:6],
            measured_ee=measured_ee,
        )
        active_cycles.append(cycle)
        measured_ee = cycle.action[6]
        if runtime.tip_probe_complete:
            break

    assert runtime.tip_probe_complete is True
    assert runtime.tip_state.phase is run_firefly_y6.TipServoPhase.PROBE_COMPLETE
    assert active_cycles[-1].telemetry["tip_servo"]["event"] == (
        "open_only_probe_complete"
    )
    assert all(
        cycle.action[:6] == pytest.approx(accepted[:6])
        for cycle in active_cycles
    )
    assert takeover.arm_owner == "hold"
    assert all(cycle.arm_owner == "hold" for cycle in active_cycles)
    assert all(
        cycle.telemetry["tip_servo"]["arm_owner"] == "hold"
        for cycle in active_cycles
    )
    ee_targets = [takeover.action[6]] + [
        cycle.action[6] for cycle in active_cycles
    ]
    assert all(
        0.0 <= next_value - previous <= runtime.config.ee_step_limit_rad + 1e-12
        for previous, next_value in zip(
            ee_targets,
            ee_targets[1:],
        )
    )
    assert ee_targets[-1] == pytest.approx(
        runtime.tip_config.pregrasp_open_target_rad
    )
    for index, cycle in enumerate(active_cycles):
        tip = cycle.telemetry["tip_servo"]
        assert tip["profile"] == "open_only"
        assert tip["commanded_descent_m"] == pytest.approx(0.0)
        assert tip["probe_close_allowed"] is False
        assert "ik_condition_number" not in tip
        assert "ik_position_residual_m" not in tip
        assert "ik_orientation_residual_rad" not in tip
        assert "ik_reason" not in tip
        assert tip["jaw_owner"] == (
            "tip_capture_hold" if index == 0 else "tip_open"
        )
        assert tip["capture_jaw_target_rad"] == pytest.approx(accepted[6])
        assert tip["published_jaw_target_rad"] == pytest.approx(cycle.action[6])
        previous = takeover if index == 0 else active_cycles[index - 1]
        assert tip["previous_accepted_jaw_target_rad"] == pytest.approx(
            previous.action[6]
        )
        assert tip["no_close_guard_passed"] is True
        assert tip["seeking_timeout_ms"] == pytest.approx(20_000.0)


def test_open_only_tip_opening_timeout_publishes_hold_before_stop(monkeypatch):
    clock = {"now_ns": 10_000_000_000}
    monkeypatch.setattr(
        run_firefly_y6.time,
        "perf_counter_ns",
        lambda: clock["now_ns"],
    )
    robot = _TipRuntimeRobot()
    runtime = _tip_open_only_runtime()
    accepted = (0.034, 0.0, 0.0, 0.0, 0.0, 0.0, 0.31)
    measured_at_capture = (0.0,) * 6
    robot.command_target = dict(
        zip(_VISUAL_ACTION_FEATURES, accepted, strict=True)
    )
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    coarse = _tip_red_image(74, center_px=(334, 235))

    _send_tip_frame(
        robot,
        runtime,
        clock,
        image=blank,
        measured_arm=measured_at_capture,
        reference=accepted,
        policy_action=accepted,
        measured_ee=0.31,
    )
    for _ in range(runtime.tip_config.takeover_confirm_frames):
        _send_tip_frame(
            robot,
            runtime,
            clock,
            image=coarse,
            measured_arm=measured_at_capture,
            reference=accepted,
            policy_action=accepted,
            measured_ee=0.31,
        )
    assert runtime.tip_state.phase is run_firefly_y6.TipServoPhase.OPENING

    robot.events.clear()
    with pytest.raises(
        run_firefly_y6.SafetyStopError,
        match="external:opening_timeout",
    ):
        _send_tip_frame(
            robot,
            runtime,
            clock,
            image=coarse,
            action_source=run_firefly_y6.VisualActionSource.TIP_SERVO,
            measured_arm=measured_at_capture,
            reference=accepted,
            policy_action=accepted,
            measured_ee=0.31,
            elapsed_ns=int(runtime.tip_config.opening_timeout_ms * 1_000_000),
        )

    assert runtime.tip_state.phase is run_firefly_y6.TipServoPhase.FAULT_HOLD
    assert runtime.tip_state.fault_reason == "opening_timeout"
    assert [event[0] for event in robot.events[:3]] == [
        "preflight",
        "send",
        "stop",
    ]
    sent_action = robot.events[1][1]
    assert tuple(
        sent_action[name] for name in robot.action_features
    ) == pytest.approx(accepted)
    assert robot.stopped_reason is not None
    assert "external:opening_timeout" in robot.stopped_reason


def test_execute_tip_takeover_holds_accepted_reference_then_rebinds_settled_fk(
    monkeypatch,
):
    clock = {"now_ns": 10_000_000_000}
    monkeypatch.setattr(
        run_firefly_y6.time,
        "perf_counter_ns",
        lambda: clock["now_ns"],
    )
    robot = _TipRuntimeRobot()
    runtime = _tip_runtime()
    accepted = (0.0, 0.015, -0.015, 0.0, 0.0, 0.0, 0.41)
    robot.command_target = dict(
        zip(_VISUAL_ACTION_FEATURES, accepted, strict=True)
    )
    measured_at_capture = (0.0,) * 6
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    coarse = _tip_red_image(74)

    _send_tip_frame(
        robot,
        runtime,
        clock,
        image=blank,
        measured_arm=measured_at_capture,
        reference=accepted,
        policy_action=accepted,
    )
    cycles = []
    for frame_index in range(runtime.tip_config.takeover_confirm_frames):
        policy = (
            accepted
            if frame_index < runtime.tip_config.takeover_confirm_frames - 1
            else (*accepted[:6], 0.10)
        )
        cycles.append(
            _send_tip_frame(
                robot,
                runtime,
                clock,
                image=coarse,
                measured_arm=measured_at_capture,
                reference=accepted,
                policy_action=policy,
            )
        )

    assert all(
        cycle.telemetry["pickup_policy_close_intent"]["active"] is False
        for cycle in cycles
    )
    assert all(cycle.discard_policy_chunk is False for cycle in cycles[:-1])
    takeover = cycles[-1]
    tip = takeover.telemetry["tip_servo"]
    assert tip["event"] == "coarse_takeover"
    assert tip["takeover_reference_settled"] is False
    assert tip["takeover_geometry_rebound"] is False
    assert takeover.action[:6] == pytest.approx(accepted[:6])
    assert takeover.action[6] == pytest.approx(
        runtime.tip_config.pregrasp_open_target_rad
    )
    assert takeover.discard_policy_chunk is True
    assert takeover.smoother_reset_required is True
    assert takeover.replan_reason == "tip_servo_takeover"
    assert takeover.arm_owner == "hold"
    assert runtime.replan_epoch == 1
    assert runtime.tip_state.phase is run_firefly_y6.TipServoPhase.OPENING
    capture_geometry = runtime.tip_state.takeover_position_m
    assert capture_geometry == pytest.approx((0.0, 0.0, 0.10))

    # OPENING owns the exact accepted target but cannot descend while the
    # measured arm still trails it, or on the first high-speed catch-up frame.
    unsettled = _send_tip_frame(
        robot,
        runtime,
        clock,
        image=coarse,
        action_source=run_firefly_y6.VisualActionSource.TIP_SERVO,
        measured_arm=measured_at_capture,
        reference=accepted,
        policy_action=(*accepted[:6], 0.10),
    )
    catch_up = _send_tip_frame(
        robot,
        runtime,
        clock,
        image=coarse,
        action_source=run_firefly_y6.VisualActionSource.TIP_SERVO,
        measured_arm=accepted[:6],
        reference=accepted,
        policy_action=(*accepted[:6], 0.10),
    )
    for cycle in (unsettled, catch_up):
        assert cycle.action[:6] == pytest.approx(accepted[:6])
        assert cycle.action[6] == pytest.approx(0.41)
        assert cycle.telemetry["tip_servo"]["takeover_geometry_rebound"] is False
        assert cycle.telemetry["pickup_policy_close_intent"] == {
            "sampled": False,
            "action_source": run_firefly_y6.VisualActionSource.TIP_SERVO.value,
            "observed": False,
            "active": False,
            "remaining_ms": 0.0,
        }
    assert runtime.tip_state.phase is run_firefly_y6.TipServoPhase.OPENING
    assert runtime.tip_state.commanded_descent_m == pytest.approx(0.0)
    assert runtime.tip_state.takeover_position_m == pytest.approx(capture_geometry)

    settled_cycles = []
    for _ in range(runtime.tip_config.open_confirm_frames):
        settled_cycles.append(
            _send_tip_frame(
                robot,
                runtime,
                clock,
                image=coarse,
                action_source=run_firefly_y6.VisualActionSource.TIP_SERVO,
                measured_arm=accepted[:6],
                reference=accepted,
                policy_action=(*accepted[:6], 0.10),
            )
        )
    assert all(
        cycle.telemetry["tip_servo"]["takeover_reference_settled"] is True
        and cycle.telemetry["tip_servo"]["takeover_geometry_rebound"] is True
        for cycle in settled_cycles
    )
    expected_settled_position, _ = robot.get_tcp_pose(accepted[:6])
    assert runtime.tip_state.takeover_position_m == pytest.approx(
        expected_settled_position
    )
    assert runtime.tip_state.takeover_position_m != pytest.approx(capture_geometry)
    assert runtime.tip_state.phase is run_firefly_y6.TipServoPhase.DESCENDING
    assert runtime.tip_state.commanded_descent_m == pytest.approx(0.0)
    assert runtime.replan_epoch == 1


def test_tip_servo_descent_moves_j23_without_sampling_policy_intent(monkeypatch):
    clock = {"now_ns": 10_000_000_000}
    monkeypatch.setattr(
        run_firefly_y6.time,
        "perf_counter_ns",
        lambda: clock["now_ns"],
    )
    robot = _TipRuntimeRobot()
    runtime = _tip_runtime()
    _drive_tip_runtime_to_descending(robot, runtime, clock)
    reference = _tip_reference(robot)

    cycle = _send_tip_frame(
        robot,
        runtime,
        clock,
        image=_tip_red_image(74),
        action_source=run_firefly_y6.VisualActionSource.TIP_SERVO,
        measured_arm=reference[:6],
        reference=reference,
        policy_action=(*reference[:6], 0.10),
    )

    tip_decision = cycle.telemetry["tip_servo"]
    assert tip_decision["event"] == "descent_step"
    assert tip_decision["intent"] == run_firefly_y6.TipServoIntent.DESCEND.value
    assert cycle.arm_owner == "visual_tip_servo"
    assert cycle.action[1] < reference[1]
    assert cycle.action[2] == pytest.approx(reference[2])
    assert cycle.telemetry["pickup_policy_close_intent"] == {
        "sampled": False,
        "action_source": run_firefly_y6.VisualActionSource.TIP_SERVO.value,
        "observed": False,
        "active": False,
        "remaining_ms": 0.0,
    }
    assert runtime.tip_state.commanded_descent_m == pytest.approx(
        runtime.tip_config.descent_step_m
    )
    assert runtime.replan_epoch == 1


@pytest.mark.parametrize("hold_kind", ("duplicate", "area_drop"))
def test_tip_servo_visual_hold_first_frame_is_exact_accepted_arm_reference(
    monkeypatch,
    hold_kind,
):
    clock = {"now_ns": 10_000_000_000}
    monkeypatch.setattr(
        run_firefly_y6.time,
        "perf_counter_ns",
        lambda: clock["now_ns"],
    )
    robot = _TipRuntimeRobot()
    runtime = _tip_runtime()
    _drive_tip_runtime_to_descending(robot, runtime, clock)
    reference = _tip_reference(robot)
    descent = _send_tip_frame(
        robot,
        runtime,
        clock,
        image=_tip_red_image(74),
        action_source=run_firefly_y6.VisualActionSource.TIP_SERVO,
        measured_arm=reference[:6],
        reference=reference,
    )
    assert descent.telemetry["tip_servo"]["event"] == "descent_step"

    if hold_kind == "duplicate":
        reference = _tip_reference(robot)
        held = _send_tip_frame(
            robot,
            runtime,
            clock,
            image=_tip_red_image(74),
            action_source=run_firefly_y6.VisualActionSource.TIP_SERVO,
            measured_arm=reference[:6],
            reference=reference,
            policy_action=(*reference[:6], 0.10),
            fresh=False,
        )
        expected_event = "visual_frame_hold"
    else:
        reference = _tip_reference(robot)
        peak = _send_tip_frame(
            robot,
            runtime,
            clock,
            image=_tip_red_image(77),
            action_source=run_firefly_y6.VisualActionSource.TIP_SERVO,
            measured_arm=reference[:6],
            reference=reference,
        )
        assert peak.telemetry["tip_servo"]["event"] == "descent_step"
        reference = _tip_reference(robot)
        held = _send_tip_frame(
            robot,
            runtime,
            clock,
            image=_tip_red_image(70),
            action_source=run_firefly_y6.VisualActionSource.TIP_SERVO,
            measured_arm=reference[:6],
            reference=reference,
            policy_action=(*reference[:6], 0.10),
        )
        expected_event = "red_area_drawdown_hold"

    tip_decision = held.telemetry["tip_servo"]
    assert tip_decision["event"] == expected_event
    assert held.action[:6] == pytest.approx(reference[:6])
    assert held.action[6] == pytest.approx(0.41)
    assert held.telemetry["pickup_policy_close_intent"]["sampled"] is False
    assert held.telemetry["pickup_policy_close_intent"]["active"] is False


def test_tip_planner_fault_publishes_latch_hold_then_stops(monkeypatch):
    clock = {"now_ns": 10_000_000_000}
    monkeypatch.setattr(
        run_firefly_y6.time,
        "perf_counter_ns",
        lambda: clock["now_ns"],
    )
    robot = _TipRuntimeRobot()
    runtime = _tip_runtime()
    _drive_tip_runtime_to_descending(robot, runtime, clock)
    reference = _tip_reference(robot)
    _send_tip_frame(
        robot,
        runtime,
        clock,
        image=_tip_red_image(74),
        action_source=run_firefly_y6.VisualActionSource.TIP_SERVO,
        measured_arm=reference[:6],
        reference=reference,
    )
    reference = _tip_reference(robot)
    _send_tip_frame(
        robot,
        runtime,
        clock,
        image=_tip_red_image(77),
        action_source=run_firefly_y6.VisualActionSource.TIP_SERVO,
        measured_arm=reference[:6],
        reference=reference,
    )
    for _ in range(runtime.tip_config.area_drop_fault_frames - 1):
        reference = _tip_reference(robot)
        held = _send_tip_frame(
            robot,
            runtime,
            clock,
            image=_tip_red_image(70),
            action_source=run_firefly_y6.VisualActionSource.TIP_SERVO,
            measured_arm=reference[:6],
            reference=reference,
        )
        assert held.action[:6] == pytest.approx(reference[:6])

    reference = _tip_reference(robot)
    robot.events.clear()
    with pytest.raises(
        run_firefly_y6.SafetyStopError,
        match="red_area_significant_drop",
    ):
        _send_tip_frame(
            robot,
            runtime,
            clock,
            image=_tip_red_image(70),
            action_source=run_firefly_y6.VisualActionSource.TIP_SERVO,
            measured_arm=reference[:6],
            reference=reference,
        )

    event_names = [event[0] for event in robot.events]
    assert event_names[:3] == ["preflight", "send", "stop"]
    assert "visual_phase_transition" in event_names[3:]
    assert "tip_servo_transition" in event_names[3:]
    sent_action = robot.events[1][1]
    assert tuple(sent_action[name] for name in robot.action_features) == pytest.approx(
        reference
    )
    assert runtime.tip_state.phase is run_firefly_y6.TipServoPhase.FAULT_HOLD
    assert robot.stopped_reason is not None


def _latch_decision(
    phase_before: VisualLatchPhase,
    phase_after: VisualLatchPhase,
    *,
    event: str | None = None,
    ee_target_rad: float | None = None,
    arm_hold: bool = False,
    fault_reason: str | None = None,
    policy_close_intent: bool = False,
) -> LatchDecision:
    return LatchDecision(
        phase_before=phase_before,
        phase_after=phase_after,
        event=event,
        ee_target_rad=ee_target_rad,
        arm_hold=arm_hold,
        pregrasp_open_seen=True,
        policy_close_intent=policy_close_intent,
        grasp_latched=phase_after
        not in {VisualLatchPhase.SEEKING, VisualLatchPhase.CLOSE_PENDING},
        release_started=phase_after
        in {
            VisualLatchPhase.RELEASING,
            VisualLatchPhase.RELEASED,
            VisualLatchPhase.HOME_CLOSING,
            VisualLatchPhase.DONE,
        },
        fault_reason=fault_reason,
        close_streak=0,
        grasp_streak=0,
        lift_streak=0,
        transport_streak=0,
        box_streak=0,
        settle_streak=0,
        release_streak=0,
        home_ready_streak=0,
        home_settle_streak=0,
        contact_gap_rad=0.0,
        held_contact_ok=None,
        arm_speed_rad_s=0.0,
        jaw_speed_rad_s=0.0,
        tcp_delta_z_m=0.0,
        visual_usable=True,
    )


def test_j23_target_smoother_bounds_step_delta_and_preserves_other_axes():
    smoother = run_firefly_y6.J23TargetSmoother()
    reference = (0.0,) * 7
    desired = (0.7, 0.1, -0.1, 0.4, 0.5, 0.6, 0.42)

    first, first_telemetry = smoother.apply(desired, reference)
    second, _ = smoother.apply(desired, first)
    third, third_telemetry = smoother.apply(desired, second)

    assert first == pytest.approx((0.7, 0.005, -0.005, 0.4, 0.5, 0.6, 0.42))
    assert second[1:3] == pytest.approx((0.015, -0.015))
    assert third[1:3] == pytest.approx((0.027, -0.030))
    assert third[0] == desired[0]
    assert third[3:] == desired[3:]
    assert first_telemetry["step"] == pytest.approx([0.005, -0.005])
    assert third_telemetry["step"] == pytest.approx([0.012, -0.015])
    assert third_telemetry["step_limits_rad"] == pytest.approx([0.012, 0.015])
    assert third_telemetry["delta_step_limit_rad"] == pytest.approx(0.005)
    assert third[1] <= desired[1]
    assert third[2] >= desired[2]
    assert _validated_visual_telemetry(
        {"j23_target_smoothing": third_telemetry}
    ) is not None


def test_j23_target_smoother_brakes_at_reference_before_reversing():
    smoother = run_firefly_y6.J23TargetSmoother()
    desired_positive = (0.0, 0.2, 0.2, 0.0, 0.0, 0.0, 0.4)
    reference = (0.0,) * 7
    for _ in range(3):
        reference, _ = smoother.apply(desired_positive, reference)

    desired_negative = list(reference)
    desired_negative[1] -= 0.1
    desired_negative[2] -= 0.1
    brake, telemetry = smoother.apply(desired_negative, reference)

    assert brake[1:3] == pytest.approx(reference[1:3])
    assert telemetry["reset_reason"] == ["target_reversal", "target_reversal"]
    assert telemetry["urgent_deceleration_exception"] == [True, True]
    assert telemetry["urgent_deceleration_exception_count"] == 2

    reversed_action, reversed_telemetry = smoother.apply(desired_negative, brake)
    assert reversed_action[1] - brake[1] == pytest.approx(-0.005)
    assert reversed_action[2] - brake[2] == pytest.approx(-0.005)
    assert reversed_telemetry["previous_step"] == pytest.approx([0.0, 0.0])


def test_j23_target_smoother_clamps_short_remaining_distance_without_overshoot():
    smoother = run_firefly_y6.J23TargetSmoother()
    desired_far = (0.0, 0.2, 0.2, 0.0, 0.0, 0.0, 0.4)
    reference = (0.0,) * 7
    for _ in range(3):
        reference, _ = smoother.apply(desired_far, reference)

    desired_near = list(reference)
    desired_near[1] += 0.001
    desired_near[2] += 0.001
    reached, telemetry = smoother.apply(desired_near, reference)

    assert reached[1:3] == pytest.approx(desired_near[1:3])
    assert telemetry["reset_reason"] == ["target_reached", "target_reached"]
    assert telemetry["urgent_deceleration_exception"] == [True, True]


@pytest.mark.parametrize("reset_reason", ["nonfresh_hold", "visual_arm_hold"])
def test_j23_target_smoother_hold_is_exact_and_resets(reset_reason):
    smoother = run_firefly_y6.J23TargetSmoother()
    desired = (0.1, 0.2, -0.2, 0.3, 0.4, 0.5, 0.42)
    reference = (0.0,) * 7
    moving, _ = smoother.apply(desired, reference)
    moving, _ = smoother.apply(desired, moving)
    hold_desired = (0.6, 1.0, -1.0, 0.7, 0.8, 0.9, 0.32)

    held, telemetry = smoother.apply(
        hold_desired,
        moving,
        forced_reset_reason=reset_reason,
    )

    assert held[1:3] == pytest.approx(moving[1:3])
    assert held[0] == hold_desired[0]
    assert held[3:] == hold_desired[3:]
    assert telemetry["reset_reason"] == [reset_reason, reset_reason]
    resumed, resumed_telemetry = smoother.apply(hold_desired, held)
    assert abs(resumed[1] - held[1]) == pytest.approx(0.005)
    assert abs(resumed[2] - held[2]) == pytest.approx(0.005)
    assert resumed_telemetry["previous_step"] == pytest.approx([0.0, 0.0])


def test_send_visual_cycle_smooths_after_composition_and_emits_plain_telemetry(
    monkeypatch,
):
    class Robot(_VisualRuntimeRobot):
        def __init__(self):
            super().__init__()
            self.preflight = None
            self.sent = None

        def preflight_actions(self, actions):
            self.preflight = actions

        def send_action_with_metadata(self, action, **metadata):
            self.sent = (action, metadata)
            _validated_visual_telemetry(metadata["visual_telemetry"])

        def record_event(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(
        run_firefly_y6.time,
        "perf_counter_ns",
        lambda: 10_000_000_000,
    )
    robot = Robot()
    runtime = _visual_runtime()
    reference = (0.0,) * 6 + (0.41,)
    policy = (0.1, 0.2, -0.2, 0.3, 0.4, 0.5, 0.41)

    cycle = run_firefly_y6.send_visual_control_cycle(
        robot,
        runtime,
        _visual_observation(),
        policy,
        reference,
    )

    assert cycle.action == pytest.approx(
        (0.1, 0.005, -0.005, 0.3, 0.4, 0.5, 0.41)
    )
    assert robot.preflight == [reference, cycle.action]
    sent_action, metadata = robot.sent
    assert tuple(sent_action[name] for name in _VISUAL_ACTION_FEATURES) == pytest.approx(
        cycle.action
    )
    smoothing = metadata["visual_telemetry"]["j23_target_smoothing"]
    assert smoothing["raw"] == pytest.approx([0.2, -0.2])
    assert smoothing["smoothed"] == pytest.approx([0.005, -0.005])
    assert smoothing["reason"] == ["delta_step_limited", "delta_step_limited"]


def test_send_visual_nonfresh_hold_is_exact_and_next_fresh_cycle_restarts_ramp(
    monkeypatch,
):
    class Robot(_VisualRuntimeRobot):
        def preflight_actions(self, _actions):
            pass

        def send_action_with_metadata(self, action, **metadata):
            _validated_visual_telemetry(metadata["visual_telemetry"])

        def record_event(self, *_args, **_kwargs):
            pass

    times = iter((10_000_000_000, 10_033_000_000, 10_066_000_000))
    monkeypatch.setattr(
        run_firefly_y6.time,
        "perf_counter_ns",
        lambda: next(times),
    )
    robot = Robot()
    runtime = _visual_runtime()
    reference = (0.0,) * 6 + (0.41,)
    desired = (0.0, 0.2, -0.2, 0.0, 0.0, 0.0, 0.41)
    moving = run_firefly_y6.send_visual_control_cycle(
        robot,
        runtime,
        _visual_observation(),
        desired,
        reference,
    )

    robot.sequence = 2
    robot.captured_at_s = 10.023
    held = run_firefly_y6.send_visual_control_cycle(
        robot,
        runtime,
        _visual_observation(),
        moving.action,
        moving.action,
        action_source=run_firefly_y6.VisualActionSource.INFERENCE_HOLD,
    )

    assert held.action == moving.action
    hold_telemetry = held.telemetry["j23_target_smoothing"]
    assert hold_telemetry["reset_reason"] == [
        "nonfresh_hold",
        "nonfresh_hold",
    ]

    robot.sequence = 3
    robot.captured_at_s = 10.056
    resumed_desired = list(held.action)
    resumed_desired[1] += 0.2
    resumed_desired[2] -= 0.2
    resumed = run_firefly_y6.send_visual_control_cycle(
        robot,
        runtime,
        _visual_observation(),
        resumed_desired,
        held.action,
    )

    assert resumed.action[1] - held.action[1] == pytest.approx(0.005)
    assert resumed.action[2] - held.action[2] == pytest.approx(-0.005)
    assert resumed.telemetry["j23_target_smoothing"][
        "previous_step"
    ] == pytest.approx([0.0, 0.0])


@pytest.mark.parametrize(
    (
        "action_source",
        "expected_sampled",
        "expected_observed",
        "expected_j23",
        "expected_reset_reason",
    ),
    [
        (
            run_firefly_y6.VisualActionSource.FRESH_POLICY,
            True,
            True,
            (0.005, -0.005),
            (None, None),
        ),
        (
            run_firefly_y6.VisualActionSource.INFERENCE_HOLD,
            False,
            False,
            (0.0, 0.0),
            ("nonfresh_hold", "nonfresh_hold"),
        ),
        (
            run_firefly_y6.VisualActionSource.TIP_SERVO,
            False,
            False,
            (0.005, -0.005),
            (None, None),
        ),
    ],
)
def test_visual_action_source_separates_intent_sampling_from_j23_hold_reset(
    monkeypatch,
    action_source,
    expected_sampled,
    expected_observed,
    expected_j23,
    expected_reset_reason,
):
    class Robot(_VisualRuntimeRobot):
        def preflight_actions(self, _actions):
            pass

        def send_action_with_metadata(self, _action, **metadata):
            _validated_visual_telemetry(metadata["visual_telemetry"])

        def record_event(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(
        run_firefly_y6.time,
        "perf_counter_ns",
        lambda: 10_000_000_000,
    )
    robot = Robot(captured_at_s=9.99)
    runtime = _visual_runtime()
    reference = (0.0,) * 6 + (0.41,)
    close_desired = (0.0, 0.2, -0.2, 0.0, 0.0, 0.0, 0.32)

    cycle = run_firefly_y6.send_visual_control_cycle(
        robot,
        runtime,
        _visual_observation(ee=0.41),
        close_desired,
        reference,
        action_source=action_source,
    )

    intent = cycle.telemetry["pickup_policy_close_intent"]
    assert intent["sampled"] is expected_sampled
    assert intent["action_source"] == action_source.value
    assert intent["observed"] is expected_observed
    assert intent["active"] is expected_observed
    assert cycle.action[1:3] == pytest.approx(expected_j23)
    assert cycle.telemetry["j23_target_smoothing"]["reset_reason"] == list(
        expected_reset_reason
    )


def test_visual_runtime_uses_fresh_d405_metadata_without_second_rotation(monkeypatch):
    now_ns = 10_000_000_000
    monkeypatch.setattr(run_firefly_y6.time, "perf_counter_ns", lambda: now_ns)
    image = _red_grasp_image()
    robot = _VisualRuntimeRobot(captured_at_s=9.99)
    runtime = _visual_runtime()
    original_detect = runtime.detector.detect
    detector_inputs = []

    def detect(rgb, **metadata):
        detector_inputs.append((rgb, metadata))
        return original_detect(rgb, **metadata)

    runtime.detector.detect = detect
    reference = (0.0,) * 6 + (0.41,)
    cycle = runtime.compose_cycle(
        robot,
        _visual_observation(image=image),
        (0.1,) * 6 + (0.20,),
        reference,
    )

    assert detector_inputs[0][0] is image
    assert detector_inputs[0][1] == {
        "frame_id": 1,
        "captured_at_ns": 9_990_000_000,
        "now_ns": now_ns,
    }
    assert cycle.telemetry["frame"]["status"] == FrameStatus.FRESH_UNIQUE.value
    assert cycle.telemetry["frame"]["age_ms"] == pytest.approx(10.0)
    assert cycle.telemetry["red"]["accepted"] is True

    # This makes the test discriminating: another 180-degree rotation would
    # move this blob outside the calibrated close ROI.
    rotated = np.rot90(image, 2).copy()
    rotated_evidence = original_detect(
        rotated,
        frame_id=2,
        captured_at_ns=9_995_000_000,
        now_ns=now_ns,
    )
    assert rotated_evidence.red.accepted is False


def test_visual_runtime_only_overrides_ee_and_holds_arm_when_requested(monkeypatch):
    monkeypatch.setattr(
        run_firefly_y6.time,
        "perf_counter_ns",
        lambda: 10_000_000_000,
    )
    robot = _VisualRuntimeRobot()
    runtime = _visual_runtime()
    evidence = VisualEvidence(
        frame_id=1,
        captured_at_ns=9_990_000_000,
        evaluated_at_ns=10_000_000_000,
        frame_age_ms=10.0,
        frame_status=FrameStatus.FRESH_UNIQUE,
        red=BlobEvidence(accepted=True, held_accepted=True),
        blue=BlobEvidence(),
    )
    runtime.detector = SimpleNamespace(detect=lambda *_args, **_kwargs: evidence)
    decisions = iter(
        (
            _latch_decision(
                VisualLatchPhase.SEEKING,
                VisualLatchPhase.SEEKING,
                ee_target_rad=0.41,
            ),
            _latch_decision(
                VisualLatchPhase.CLOSING,
                VisualLatchPhase.CLOSING,
                ee_target_rad=0.32,
                arm_hold=True,
            ),
        )
    )
    runtime.latch = SimpleNamespace(
        phase=VisualLatchPhase.SEEKING,
        step=lambda *_args, **_kwargs: next(decisions),
    )
    observation = _visual_observation()
    policy = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.10)
    reference = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.35)

    passthrough = runtime.compose_cycle(robot, observation, policy, reference)
    held = runtime.compose_cycle(robot, observation, policy, reference)

    assert passthrough.action[:6] == policy[:6]
    assert passthrough.action[6] == pytest.approx(0.3665)
    assert held.action[:6] == reference[:6]
    assert held.action[6] == pytest.approx(0.3335)
    assert passthrough.requires_replan is False
    assert held.requires_replan is True
    assert held.replan_reason == "arm_hold_entered"
    assert held.telemetry["arm_hold"] is True


def test_visual_runtime_duplicate_d405_frame_does_not_advance_close_streak(
    monkeypatch,
):
    times = iter((10_000_000_000, 10_033_000_000, 10_066_000_000))
    monkeypatch.setattr(run_firefly_y6.time, "perf_counter_ns", lambda: next(times))
    image = _red_grasp_image()
    robot = _VisualRuntimeRobot(sequence=1, captured_at_s=9.99)
    runtime = _visual_runtime(replace(_visual_config(), close_confirm_frames=3))
    observation = _visual_observation(image=image)
    action = (0.0,) * 6 + (0.32,)
    reference = (0.0,) * 6 + (0.41,)

    first = runtime.compose_cycle(robot, observation, action, reference)
    robot.sequence = 2
    robot.captured_at_s = 10.023
    second = runtime.compose_cycle(robot, observation, action, reference)
    duplicate = runtime.compose_cycle(robot, observation, action, reference)

    assert first.decision.close_streak == 0
    assert second.decision.phase_after is VisualLatchPhase.CLOSE_PENDING
    assert second.decision.close_streak == 1
    assert duplicate.decision.phase_after is VisualLatchPhase.CLOSE_PENDING
    assert duplicate.decision.close_streak == 1
    assert duplicate.telemetry["frame"]["status"] == FrameStatus.DUPLICATE.value
    assert duplicate.telemetry["visual_usable"] is False


def test_visual_runtime_bounds_cached_pickup_policy_close_intent(monkeypatch):
    times = iter(
        (
            10_000_000_000,
            10_033_000_000,
            10_099_000_000,
            10_134_000_000,
        )
    )
    monkeypatch.setattr(run_firefly_y6.time, "perf_counter_ns", lambda: next(times))
    config = _visual_config(
        close_confirm_frames=9,
        pickup_policy_close_intent_hold_ms=100.0,
    )
    runtime = _visual_runtime(config)
    robot = _VisualRuntimeRobot(sequence=1, captured_at_s=9.99)
    observation = _visual_observation(image=_red_grasp_image())
    reference = (0.0,) * 6 + (0.41,)

    runtime.compose_cycle(robot, observation, reference, reference)

    robot.sequence = 2
    robot.captured_at_s = 10.023
    seeded = runtime.compose_cycle(
        robot,
        observation,
        (0.0,) * 6 + (0.32,),
        reference,
    )
    robot.sequence = 3
    robot.captured_at_s = 10.089
    held = runtime.compose_cycle(
        robot,
        observation,
        reference,
        reference,
        action_source=run_firefly_y6.VisualActionSource.INFERENCE_HOLD,
    )
    robot.sequence = 4
    robot.captured_at_s = 10.124
    expired = runtime.compose_cycle(
        robot,
        observation,
        (0.0,) * 6 + (0.32,),
        reference,
        action_source=run_firefly_y6.VisualActionSource.INFERENCE_HOLD,
    )

    assert seeded.decision.policy_close_intent is True
    assert seeded.telemetry["pickup_policy_close_intent"] == {
        "sampled": True,
        "action_source": run_firefly_y6.VisualActionSource.FRESH_POLICY.value,
        "observed": True,
        "active": True,
        "remaining_ms": pytest.approx(100.0),
    }
    assert held.decision.policy_close_intent is True
    assert held.telemetry["pickup_policy_close_intent"] == {
        "sampled": False,
        "action_source": run_firefly_y6.VisualActionSource.INFERENCE_HOLD.value,
        "observed": False,
        "active": True,
        "remaining_ms": pytest.approx(34.0),
    }
    assert expired.decision.policy_close_intent is False
    assert expired.telemetry["pickup_policy_close_intent"] == {
        "sampled": False,
        "action_source": run_firefly_y6.VisualActionSource.INFERENCE_HOLD.value,
        "observed": False,
        "active": False,
        "remaining_ms": pytest.approx(0.0),
    }


def test_visual_runtime_starts_pickup_intent_cache_only_after_pregrasp_open(
    monkeypatch,
):
    times = iter(
        (
            10_000_000_000,
            10_033_000_000,
            10_066_000_000,
            10_099_000_000,
            10_167_000_000,
        )
    )
    monkeypatch.setattr(run_firefly_y6.time, "perf_counter_ns", lambda: next(times))
    config = _visual_config(
        close_confirm_frames=9,
        pickup_policy_close_intent_hold_ms=100.0,
    )
    runtime = _visual_runtime(config)
    robot = _VisualRuntimeRobot(sequence=1, captured_at_s=9.99)
    closed_reference = (0.0,) * 6 + (0.314,)
    open_reference = (0.0,) * 6 + (0.41,)
    close_action = (0.0,) * 6 + (0.32,)

    initial_closed = runtime.compose_cycle(
        robot,
        _visual_observation(ee=0.314),
        close_action,
        closed_reference,
    )
    robot.sequence = 2
    robot.captured_at_s = 10.023
    natural_open = runtime.compose_cycle(
        robot,
        _visual_observation(ee=config.pregrasp_open_ready_rad),
        open_reference,
        open_reference,
    )
    robot.sequence = 3
    robot.captured_at_s = 10.056
    post_open_close = runtime.compose_cycle(
        robot,
        _visual_observation(ee=0.41),
        close_action,
        open_reference,
    )
    robot.sequence = 4
    robot.captured_at_s = 10.089
    bounded_hold = runtime.compose_cycle(
        robot,
        _visual_observation(ee=0.41),
        close_action,
        open_reference,
        action_source=run_firefly_y6.VisualActionSource.INFERENCE_HOLD,
    )
    robot.sequence = 5
    robot.captured_at_s = 10.157
    expired_hold = runtime.compose_cycle(
        robot,
        _visual_observation(ee=0.41),
        close_action,
        open_reference,
        action_source=run_firefly_y6.VisualActionSource.INFERENCE_HOLD,
    )

    assert initial_closed.telemetry["pickup_policy_close_intent"] == {
        "sampled": True,
        "action_source": run_firefly_y6.VisualActionSource.FRESH_POLICY.value,
        "observed": False,
        "active": False,
        "remaining_ms": pytest.approx(0.0),
    }
    assert natural_open.decision.pregrasp_open_seen is True
    assert natural_open.telemetry["pickup_policy_close_intent"] == {
        "sampled": True,
        "action_source": run_firefly_y6.VisualActionSource.FRESH_POLICY.value,
        "observed": False,
        "active": False,
        "remaining_ms": pytest.approx(0.0),
    }
    assert post_open_close.telemetry["pickup_policy_close_intent"] == {
        "sampled": True,
        "action_source": run_firefly_y6.VisualActionSource.FRESH_POLICY.value,
        "observed": True,
        "active": True,
        "remaining_ms": pytest.approx(100.0),
    }
    assert bounded_hold.telemetry["pickup_policy_close_intent"] == {
        "sampled": False,
        "action_source": run_firefly_y6.VisualActionSource.INFERENCE_HOLD.value,
        "observed": False,
        "active": True,
        "remaining_ms": pytest.approx(67.0),
    }
    assert expired_hold.telemetry["pickup_policy_close_intent"] == {
        "sampled": False,
        "action_source": run_firefly_y6.VisualActionSource.INFERENCE_HOLD.value,
        "observed": False,
        "active": False,
        "remaining_ms": pytest.approx(0.0),
    }


@pytest.mark.parametrize("ready_source", ["measured", "reference"])
def test_visual_runtime_accepts_fresh_close_on_open_ready_boundary(
    monkeypatch,
    ready_source,
):
    monkeypatch.setattr(
        run_firefly_y6.time,
        "perf_counter_ns",
        lambda: 10_000_000_000,
    )
    config = _visual_config(close_confirm_frames=9)
    runtime = _visual_runtime(config)
    robot = _VisualRuntimeRobot(captured_at_s=9.99)
    measured_ee = (
        config.pregrasp_open_ready_rad if ready_source == "measured" else 0.314
    )
    reference_ee = (
        config.pregrasp_open_ready_rad if ready_source == "reference" else 0.314
    )
    cycle = runtime.compose_cycle(
        robot,
        _visual_observation(ee=measured_ee),
        (0.0,) * 6 + (0.32,),
        (0.0,) * 6 + (reference_ee,),
    )

    assert cycle.decision.pregrasp_open_seen is True
    assert cycle.telemetry["pickup_policy_close_intent"] == {
        "sampled": True,
        "action_source": run_firefly_y6.VisualActionSource.FRESH_POLICY.value,
        "observed": True,
        "active": True,
        "remaining_ms": pytest.approx(config.pickup_policy_close_intent_hold_ms),
    }


def test_visual_wait_accumulates_close_frames_during_bounded_policy_hold(
    monkeypatch,
):
    perf_times = iter(
        (
            10_000_000_000,
            10_033_000_000,
            10_066_000_000,
            10_099_000_000,
        )
    )
    monkeypatch.setattr(
        run_firefly_y6.time,
        "perf_counter_ns",
        lambda: next(perf_times),
    )
    monkeypatch.setattr(run_firefly_y6.time, "sleep", lambda _duration: None)
    config = _visual_config(
        close_confirm_frames=3,
        pickup_policy_close_intent_hold_ms=200.0,
    )
    runtime = _visual_runtime(config)
    telemetry = []

    class Robot(_VisualRuntimeRobot):
        config = SimpleNamespace(command_hz=30.0)

        def __init__(self):
            super().__init__(sequence=1, captured_at_s=9.99)
            self.command_target = dict(
                zip(_VISUAL_ACTION_FEATURES, (0.0,) * 6 + (0.41,), strict=True)
            )

        def get_observation(self):
            self.sequence += 1
            self.captured_at_s += 0.033
            return _visual_observation(image=_red_grasp_image())

        def preflight_actions(self, _actions):
            pass

        def send_action_with_metadata(self, _action, **metadata):
            telemetry.append(metadata["visual_telemetry"])

        def record_event(self, _event, **_fields):
            pass

    stale = SimpleNamespace(error=None, generation=1)
    fresh = SimpleNamespace(error=None, generation=2)
    results = iter((None, None, stale, fresh))

    class Worker:
        def __init__(self):
            self.submissions = 0

        def submit(self, _observation):
            self.submissions += 1

        def get_nowait(self):
            return next(results)

    class Recorder:
        def add_observation(self, _observation):
            pass

    robot = Robot()
    observation = _visual_observation(image=_red_grasp_image())
    reference = (0.0,) * 6 + (0.41,)
    runtime.compose_cycle(robot, observation, reference, reference)
    robot.sequence = 2
    robot.captured_at_s = 10.023
    seeded = runtime.compose_cycle(
        robot,
        observation,
        (0.0,) * 6 + (0.32,),
        reference,
    )
    assert seeded.decision.close_streak == 1

    worker = Worker()
    result = run_firefly_y6.wait_for_initial_chunk(
        robot,
        worker,
        observation,
        Recorder(),
        [False],
        runtime,
    )

    assert result is fresh
    assert worker.submissions == 2
    assert runtime.phase is VisualLatchPhase.CLOSING
    assert runtime.replan_epoch == 1
    assert [entry["pickup_policy_close_intent"] for entry in telemetry] == [
        {
            "sampled": False,
            "action_source": run_firefly_y6.VisualActionSource.INFERENCE_HOLD.value,
            "observed": False,
            "active": True,
            "remaining_ms": pytest.approx(167.0),
        },
        {
            "sampled": False,
            "action_source": run_firefly_y6.VisualActionSource.INFERENCE_HOLD.value,
            "observed": False,
            "active": True,
            "remaining_ms": pytest.approx(134.0),
        },
    ]


def test_visual_wait_uses_accepted_command_target_not_contact_displaced_measurement(
    monkeypatch,
):
    stopping = [False]
    captured = {}

    class Worker:
        def submit(self, _observation):
            pass

        def get_nowait(self):
            return None

    class Robot:
        action_features = _VISUAL_ACTION_FEATURES
        command_target = dict(
            zip(_VISUAL_ACTION_FEATURES, (0.1,) * 6 + (0.32,), strict=True)
        )
        config = SimpleNamespace(command_hz=30.0)

        def get_observation(self):
            return _visual_observation(arm=(0.2,) * 6, ee=0.35)

    class Recorder:
        def add_observation(self, _observation):
            pass

    runtime = SimpleNamespace(replan_epoch=0)

    def send_cycle(
        _robot,
        _runtime,
        _observation,
        policy_action,
        reference,
        *,
        action_source,
    ):
        captured["policy_action"] = policy_action
        captured["reference"] = reference
        captured["action_source"] = action_source
        stopping[0] = True

    monkeypatch.setattr(run_firefly_y6, "send_visual_control_cycle", send_cycle)
    monkeypatch.setattr(run_firefly_y6.time, "sleep", lambda _duration: None)
    result = run_firefly_y6.wait_for_initial_chunk(
        Robot(),
        Worker(),
        _visual_observation(ee=0.35),
        Recorder(),
        stopping,
        runtime,
    )

    assert result is None
    assert captured["policy_action"] == pytest.approx((0.1,) * 6 + (0.32,))
    assert captured["reference"] == pytest.approx((0.1,) * 6 + (0.32,))
    assert captured["action_source"] is run_firefly_y6.VisualActionSource.INFERENCE_HOLD


def test_visual_wait_discards_inference_from_prior_transition_epoch(monkeypatch):
    stale = SimpleNamespace(generation=1, error=None)
    fresh = SimpleNamespace(generation=2, error=None)
    results = iter((None, stale, fresh))
    submissions = []

    class Worker:
        def submit(self, observation):
            submissions.append(observation["sample"])

        def get_nowait(self):
            return next(results)

    class Robot:
        action_features = _VISUAL_ACTION_FEATURES
        command_target = dict.fromkeys(_VISUAL_ACTION_FEATURES, 0.0)
        config = SimpleNamespace(command_hz=30.0)

        def get_observation(self):
            observation = _visual_observation()
            observation["sample"] = "post_transition"
            return observation

    class Recorder:
        def add_observation(self, _observation):
            pass

    runtime = SimpleNamespace(replan_epoch=4)

    def transition_cycle(*_args, **_kwargs):
        runtime.replan_epoch = 5

    monkeypatch.setattr(
        run_firefly_y6,
        "send_visual_control_cycle",
        transition_cycle,
    )
    monkeypatch.setattr(run_firefly_y6.time, "sleep", lambda _duration: None)
    initial = _visual_observation()
    initial["sample"] = "initial"

    result = run_firefly_y6.wait_for_initial_chunk(
        Robot(), Worker(), initial, Recorder(), [False], runtime
    )

    assert result is fresh
    assert submissions == ["initial", "post_transition"]


def test_visual_fault_publishes_reference_hold_before_stopping(monkeypatch):
    events = []

    class Robot(_VisualRuntimeRobot):
        stopped_reason = None

        def get_camera_frame_metadata(self):
            return {}

        def preflight_actions(self, actions):
            events.append(("preflight", actions))

        def send_action_with_metadata(self, action, **metadata):
            events.append(("send", action, metadata))

        def record_event(self, event, **fields):
            events.append((event, fields))

        def stop(self, reason):
            self.stopped_reason = reason
            events.append(("stop", reason))

    monkeypatch.setattr(
        run_firefly_y6.time,
        "perf_counter_ns",
        lambda: 10_000_000_000,
    )
    robot = Robot()
    runtime = _visual_runtime()
    reference = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.35)

    with pytest.raises(run_firefly_y6.SafetyStopError, match="fail-closed HOLD"):
        run_firefly_y6.send_visual_control_cycle(
            robot,
            runtime,
            _visual_observation(),
            (1.0,) * 6 + (0.10,),
            reference,
        )

    assert [event[0] for event in events] == [
        "preflight",
        "send",
        "stop",
        "visual_phase_transition",
    ]
    sent = events[1][1]
    assert tuple(sent[name] for name in _VISUAL_ACTION_FEATURES) == reference
    assert events[1][2]["visual_telemetry"]["arm_hold"] is True


def test_visual_fault_releases_sessions_even_when_transition_telemetry_fails(
    monkeypatch,
):
    events = []

    class Robot(_VisualRuntimeRobot):
        stopped_reason = None

        def get_camera_frame_metadata(self):
            return {}

        def preflight_actions(self, _actions):
            events.append("preflight")

        def send_action_with_metadata(self, _action, **_metadata):
            events.append("send")

        def record_event(self, _event, **_fields):
            events.append("transition_log")
            raise OSError("telemetry disk full")

        def stop(self, reason):
            self.stopped_reason = reason
            events.append("stop")

    monkeypatch.setattr(
        run_firefly_y6.time,
        "perf_counter_ns",
        lambda: 10_000_000_000,
    )
    robot = Robot()
    runtime = _visual_runtime()
    reference = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.35)

    with pytest.raises(run_firefly_y6.SafetyStopError, match="fail-closed HOLD"):
        run_firefly_y6.send_visual_control_cycle(
            robot,
            runtime,
            _visual_observation(),
            (1.0,) * 6 + (0.10,),
            reference,
        )

    assert events == ["preflight", "send", "stop", "transition_log"]
    assert robot.stopped_reason is not None


def test_released_clamps_ee_open_until_home_then_reaches_done(monkeypatch):
    times = iter(
        (
            10_000_000_000,
            10_033_000_000,
            10_066_000_000,
            10_099_000_000,
            10_132_000_000,
        )
    )
    monkeypatch.setattr(run_firefly_y6.time, "perf_counter_ns", lambda: next(times))
    config = _visual_config(
        home_ready_confirm_cycles=1,
        home_settle_confirm_cycles=1,
    )
    runtime = _visual_runtime(config)
    runtime.latch.phase = VisualLatchPhase.RELEASED
    runtime.latch.release_started = True
    runtime.latch.grasp_latched = True
    robot = _VisualRuntimeRobot()
    policy_close = (0.5,) * 6 + (0.10,)

    released = runtime.compose_cycle(
        robot,
        _visual_observation(ee=0.42),
        policy_close,
        (0.0,) * 6 + (0.42,),
    )
    assert released.decision.phase_after is VisualLatchPhase.RELEASED
    assert released.action[:6] == policy_close[:6]
    assert released.action[6] == pytest.approx(config.release_target_rad)

    robot.sequence = 2
    robot.captured_at_s = 10.023
    home_started = runtime.compose_cycle(
        robot,
        _visual_observation(ee=0.42),
        policy_close,
        (0.0,) * 6 + (0.42,),
    )
    assert home_started.decision.phase_after is VisualLatchPhase.HOME_CLOSING

    robot.sequence = 3
    robot.captured_at_s = 10.056
    runtime.compose_cycle(
        robot,
        _visual_observation(ee=config.home_target_rad),
        policy_close,
        (0.0,) * 6 + (config.home_target_rad,),
    )
    robot.sequence = 4
    robot.captured_at_s = 10.089
    done = runtime.compose_cycle(
        robot,
        _visual_observation(ee=config.home_target_rad),
        policy_close,
        (0.0,) * 6 + (config.home_target_rad,),
    )

    assert done.decision.phase_after is VisualLatchPhase.DONE
    assert done.action[:6] == (0.0,) * 6
    assert done.action[6] == pytest.approx(config.home_target_rad)
    assert done.requires_replan is True
    json.dumps(done.telemetry, allow_nan=False)
    assert _validated_visual_telemetry(done.telemetry) == done.telemetry


def test_run_act_visual_transition_discards_remaining_chunk_and_replans(
    monkeypatch,
):
    runtime = SimpleNamespace(
        phase=VisualLatchPhase.SEEKING,
        replan_epoch=0,
        tip_exclusive_active=False,
    )
    waits = []
    sent_policy_actions = []
    stopping = [False]

    class Worker:
        def __init__(self, _infer):
            self.closed = False

        def close(self):
            self.closed = True

    class Robot:
        action_features = _VISUAL_ACTION_FEATURES
        command_target = dict.fromkeys(_VISUAL_ACTION_FEATURES, 0.0)
        stopped_reason = None

        def get_observation(self):
            return _visual_observation()

        def preflight_actions(self, _actions):
            pass

        def record_event(self, *_args, **_kwargs):
            pass

        def stop(self, reason):
            self.stopped_reason = reason

    class Recorder:
        def add_observation(self, _observation):
            pass

    chunk = SimpleNamespace(
        generation=1,
        actions=tuple((float(index),) * 7 for index in range(10)),
        latency_s=0.01,
        error=None,
    )

    def wait(_robot, _worker, _observation, _recorder, _stopping, visual=None):
        waits.append(visual)
        if len(waits) == 1:
            return chunk
        runtime.phase = VisualLatchPhase.DONE
        return chunk

    def send_cycle(
        _robot,
        _runtime,
        _observation,
        policy_action,
        _reference,
        **_kwargs,
    ):
        sent_policy_actions.append(policy_action)
        runtime.replan_epoch += 1
        decision = _latch_decision(
            VisualLatchPhase.CLOSE_PENDING,
            VisualLatchPhase.CLOSING,
            event="grasp_latched",
            ee_target_rad=0.32,
            arm_hold=True,
        )
        return run_firefly_y6.VisualControlCycle(
            action=(0.0,) * 7,
            decision=decision,
            telemetry={},
            requires_replan=True,
        )

    monkeypatch.setattr(run_firefly_y6, "InferenceWorker", Worker)
    monkeypatch.setattr(run_firefly_y6, "build_visual_latch_runtime", lambda _config: runtime)
    monkeypatch.setattr(run_firefly_y6, "wait_for_initial_chunk", wait)
    monkeypatch.setattr(run_firefly_y6, "send_visual_control_cycle", send_cycle)
    monkeypatch.setattr(run_firefly_y6.time, "sleep", lambda _duration: None)

    completed = run_firefly_y6.run_act(
        SimpleNamespace(
            fps=30,
            n_action_steps=10,
            act_chunk_inference_mode="synchronous_hold",
            enable_action_slew_rate_adapter=False,
            grasp_phase_correction=None,
        ),
        Robot(),
        infer=lambda _observation: None,
        recorder=Recorder(),
        stopping=stopping,
        duration_s=3.0,
    )

    assert completed is True
    assert sent_policy_actions == [chunk.actions[0]]
    assert waits == [runtime, runtime]


def test_run_act_tip_takeover_enters_exclusive_without_rewaiting_or_reinferring(
    monkeypatch,
):
    runtime = SimpleNamespace(
        phase=VisualLatchPhase.SEEKING,
        replan_epoch=0,
        tip_exclusive_active=False,
        tip_probe_complete=False,
    )
    waits = []
    sent_policy_actions = []
    exclusive_calls = []
    workers = []

    class Worker:
        def __init__(self, _infer):
            self.submissions = []
            self.closed = False
            workers.append(self)

        def submit(self, observation):
            self.submissions.append(observation)

        def close(self):
            self.closed = True

    class Robot:
        action_features = _VISUAL_ACTION_FEATURES
        command_target = dict.fromkeys(_VISUAL_ACTION_FEATURES, 0.0)
        stopped_reason = None

        def get_observation(self):
            return _visual_observation()

        def preflight_actions(self, _actions):
            pass

        def record_event(self, *_args, **_kwargs):
            pass

        def stop(self, reason):
            self.stopped_reason = reason

    class Recorder:
        def add_observation(self, _observation):
            pass

    chunk = SimpleNamespace(
        generation=1,
        actions=tuple((float(index),) * 7 for index in range(10)),
        latency_s=0.01,
        error=None,
    )

    def wait(_robot, worker, observation, _recorder, _stopping, visual=None):
        waits.append(visual)
        worker.submit(observation)
        return chunk

    def send_cycle(
        _robot,
        _runtime,
        _observation,
        policy_action,
        _reference,
        **_kwargs,
    ):
        sent_policy_actions.append(policy_action)
        runtime.replan_epoch += 1
        runtime.tip_exclusive_active = True
        return run_firefly_y6.VisualControlCycle(
            action=(0.0,) * 7,
            decision=_latch_decision(
                VisualLatchPhase.SEEKING,
                VisualLatchPhase.SEEKING,
            ),
            telemetry={},
            requires_replan=False,
            replan_reason="tip_servo_takeover",
            discard_policy_chunk=True,
            smoother_reset_required=True,
            arm_owner="hold",
        )

    def run_exclusive(
        _robot,
        _runtime,
        _recorder,
        _stopping,
        *,
        deadline_monotonic_s,
    ):
        exclusive_calls.append(deadline_monotonic_s)
        runtime.tip_probe_complete = True
        runtime.tip_exclusive_active = False
        return True

    monkeypatch.setattr(run_firefly_y6, "InferenceWorker", Worker)
    monkeypatch.setattr(
        run_firefly_y6,
        "build_visual_latch_runtime",
        lambda _config: runtime,
    )
    monkeypatch.setattr(run_firefly_y6, "wait_for_initial_chunk", wait)
    monkeypatch.setattr(run_firefly_y6, "send_visual_control_cycle", send_cycle)
    monkeypatch.setattr(
        run_firefly_y6,
        "run_tip_servo_probe_exclusive",
        run_exclusive,
    )
    monkeypatch.setattr(run_firefly_y6.time, "sleep", lambda _duration: None)

    completed = run_firefly_y6.run_act(
        SimpleNamespace(
            fps=30,
            n_action_steps=10,
            act_chunk_inference_mode="synchronous_hold",
            enable_action_slew_rate_adapter=False,
            grasp_phase_correction=None,
        ),
        Robot(),
        infer=lambda _observation: None,
        recorder=Recorder(),
        stopping=[False],
        duration_s=3.0,
    )

    assert completed is True
    assert runtime.tip_probe_complete is True
    assert waits == [runtime]
    assert len(workers) == 1
    assert len(workers[0].submissions) == 1
    assert workers[0].closed is True
    assert sent_policy_actions == [chunk.actions[0]]
    assert len(exclusive_calls) == 1


def test_run_act_arm6_takeover_submits_no_further_inference(monkeypatch):
    runtime = SimpleNamespace(
        phase=VisualLatchPhase.SEEKING,
        replan_epoch=0,
        tip_exclusive_active=False,
        tip_probe_complete=False,
    )
    waits = []
    workers = []
    exclusive_bindings = []

    class Worker:
        def __init__(self, _infer):
            self.submissions = []
            self.closed = False
            workers.append(self)

        def submit(self, observation):
            self.submissions.append(observation)

        def close(self):
            self.closed = True

    class Robot:
        action_features = _VISUAL_ACTION_FEATURES
        command_target = dict(
            zip(_VISUAL_ACTION_FEATURES, (0.0,) * 6 + (0.31,), strict=True)
        )
        stopped_reason = None

        def get_observation(self):
            return _visual_observation(ee=0.31)

        def preflight_actions(self, _actions):
            pass

        def record_event(self, *_args, **_kwargs):
            pass

        def stop(self, reason):
            self.stopped_reason = reason

    class Recorder:
        def add_observation(self, _observation):
            pass

    chunk = SimpleNamespace(
        generation=1,
        actions=tuple((0.01,) * 6 for _ in range(10)),
        latency_s=0.01,
        error=None,
    )

    def wait(
        _robot,
        worker,
        observation,
        _recorder,
        _stopping,
        visual,
        arm6_runtime,
    ):
        waits.append((visual, arm6_runtime))
        worker.submit(observation)
        return chunk

    def send_cycle(
        _robot,
        _runtime,
        _observation,
        policy_action,
        _reference,
        **kwargs,
    ):
        assert len(policy_action) == 7
        assert policy_action[6] == pytest.approx(0.31)
        assert kwargs["arm6_runtime"] is waits[0][1]
        runtime.replan_epoch += 1
        runtime.tip_exclusive_active = True
        return run_firefly_y6.VisualControlCycle(
            action=(0.0,) * 6 + (0.31,),
            decision=_latch_decision(
                VisualLatchPhase.SEEKING,
                VisualLatchPhase.SEEKING,
            ),
            telemetry={},
            requires_replan=False,
            replan_reason="tip_servo_takeover",
            discard_policy_chunk=True,
            smoother_reset_required=True,
            arm_owner="hold",
        )

    def run_exclusive(
        _robot,
        _runtime,
        _recorder,
        _stopping,
        *,
        deadline_monotonic_s,
        arm6_runtime,
    ):
        assert deadline_monotonic_s > 0
        exclusive_bindings.append(arm6_runtime)
        runtime.tip_probe_complete = True
        runtime.tip_exclusive_active = False
        return True

    monkeypatch.setattr(run_firefly_y6, "InferenceWorker", Worker)
    monkeypatch.setattr(run_firefly_y6, "build_visual_latch_runtime", lambda _config: runtime)
    monkeypatch.setattr(run_firefly_y6, "wait_for_initial_chunk", wait)
    monkeypatch.setattr(run_firefly_y6, "send_visual_control_cycle", send_cycle)
    monkeypatch.setattr(run_firefly_y6, "run_tip_servo_probe_exclusive", run_exclusive)
    monkeypatch.setattr(run_firefly_y6.time, "sleep", lambda _duration: None)

    completed = run_firefly_y6.run_act(
        SimpleNamespace(
            fps=30,
            n_action_steps=10,
            act_chunk_inference_mode="synchronous_hold",
            enable_action_slew_rate_adapter=False,
            grasp_phase_correction=None,
        ),
        Robot(),
        _arm6_stub_infer(),
        Recorder(),
        [False],
        30.0,
    )

    assert completed is True
    assert len(waits) == 1
    assert len(workers) == 1
    assert len(workers[0].submissions) == 1
    assert workers[0].closed is True
    assert exclusive_bindings == [waits[0][1]]


def test_run_act_tip_execute_deadline_without_takeover_stops_and_cannot_complete(
    monkeypatch,
):
    runtime = SimpleNamespace(
        phase=VisualLatchPhase.SEEKING,
        tip_mode=run_firefly_y6.TipServoMode.EXECUTE,
        tip_state=SimpleNamespace(phase=run_firefly_y6.TipServoPhase.SEEKING),
        tip_exclusive_active=False,
    )
    workers = []

    class Worker:
        def __init__(self, _infer):
            self.closed = False
            workers.append(self)

        def close(self):
            self.closed = True

    class Robot:
        action_features = _VISUAL_ACTION_FEATURES
        stopped_reason = None

        def __init__(self):
            self.events = []

        def get_observation(self):
            return _visual_observation()

        def record_event(self, event, **_fields):
            self.events.append(event)

        def stop(self, reason):
            self.stopped_reason = reason
            self.events.append("stop")

    class Recorder:
        def add_observation(self, _observation):
            pass

    chunk = SimpleNamespace(
        generation=1,
        actions=tuple((0.0,) * 7 for _ in range(10)),
        latency_s=0.01,
        error=None,
    )
    monotonic_values = iter((100.0, 130.0))

    monkeypatch.setattr(run_firefly_y6, "InferenceWorker", Worker)
    monkeypatch.setattr(
        run_firefly_y6,
        "build_visual_latch_runtime",
        lambda _config: runtime,
    )
    monkeypatch.setattr(
        run_firefly_y6,
        "wait_for_initial_chunk",
        lambda _robot, _worker, _observation, _recorder, _stopping, _visual: chunk,
    )
    monkeypatch.setattr(
        run_firefly_y6,
        "send_visual_control_cycle",
        lambda *_args, **_kwargs: pytest.fail(
            "deadline-before-takeover must not enter a visual control cycle"
        ),
    )
    monkeypatch.setattr(
        run_firefly_y6.time,
        "monotonic",
        lambda: next(monotonic_values),
    )
    robot = Robot()

    with pytest.raises(
        run_firefly_y6.SafetyStopError,
        match="ended before the reviewed probe progress gate",
    ):
        run_firefly_y6.run_act(
            SimpleNamespace(
                fps=30,
                n_action_steps=10,
                act_chunk_inference_mode="synchronous_hold",
                enable_action_slew_rate_adapter=False,
                grasp_phase_correction=None,
            ),
            robot,
            infer=lambda _observation: None,
            recorder=Recorder(),
            stopping=[False],
            duration_s=30.0,
        )

    assert robot.events == ["stop"]
    assert "stage_complete" not in robot.events
    assert robot.stopped_reason is not None
    assert "terminal tip phase=seeking" in robot.stopped_reason
    assert len(workers) == 1
    assert workers[0].closed is True


def test_disabled_visual_wait_preserves_legacy_send_path(monkeypatch):
    stopping = [False]
    events = []

    class Worker:
        def submit(self, _observation):
            pass

        def get_nowait(self):
            return None

    class Robot:
        action_features = _VISUAL_ACTION_FEATURES
        command_target = dict.fromkeys(_VISUAL_ACTION_FEATURES, 0.0)
        config = SimpleNamespace(command_hz=30.0)

        def get_observation(self):
            return _visual_observation()

        def send_action(self, action):
            events.append(action)
            stopping[0] = True

    class Recorder:
        def add_observation(self, _observation):
            pass

    monkeypatch.setattr(
        run_firefly_y6,
        "send_visual_control_cycle",
        lambda *_args, **_kwargs: pytest.fail("disabled path entered visual control"),
    )
    monkeypatch.setattr(run_firefly_y6.time, "sleep", lambda _duration: None)

    result = run_firefly_y6.wait_for_initial_chunk(
        Robot(), Worker(), _visual_observation(), Recorder(), stopping
    )

    assert result is None
    assert len(events) == 1
    assert tuple(events[0].values()) == (0.0,) * 7
    assert run_firefly_y6.build_visual_latch_runtime(SimpleNamespace()) is None
