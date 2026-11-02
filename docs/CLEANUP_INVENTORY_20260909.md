# 旧文件与清理候选完整路径（2026-09-09）

下面记录的是实际检查到的本地文件。除了清单末尾的可再生缓存，本次均未删除。删除影响先看[诊断报告](AUDIT_AND_OPTIMIZATION_20260909.md)。容量按普通文件分配空间统计，不跟随软链接；父子项不能重复相加。

## 大目录与训练状态

| 路径 | 容量 |
| --- | ---: |
| [datasets/local/firefly_y6_fullcycle_v4](../datasets/local/firefly_y6_fullcycle_v4) | 1.457 GiB |
| [datasets/local/firefly_y6_fullcycle_v4_arm_only_v1](../datasets/local/firefly_y6_fullcycle_v4_arm_only_v1) | 1.456 GiB |
| [outputs/train/firefly_y6_smolvla_fullcycle_v4_100k_attested_run1](../outputs/train/firefly_y6_smolvla_fullcycle_v4_100k_attested_run1) | 1.500 GiB |
| [outputs/train/firefly_y6_smolvla_fullcycle_v4_arm_only_v1_finetune_20k_attested_run2](../outputs/train/firefly_y6_smolvla_fullcycle_v4_arm_only_v1_finetune_20k_attested_run2) | 3.000 GiB |
| [imports/nas_outputs_20260812](../imports/nas_outputs_20260812) | 0.845 GiB |
| [lerobot-robot-hexbase/outputs](../lerobot-robot-hexbase/outputs) | 0.527 GiB |
| [pip_cache](../pip_cache) | 1.503 GiB |
| [assets](../assets) | 0.849 GiB |
| [hf_cache](../hf_cache) | 2.757 GiB |
| [miniforge3/envs/lerobot-io](../miniforge3/envs/lerobot-io) | 8.967 GiB |
| [miniforge3/envs/lerobot-train](../miniforge3/envs/lerobot-train) | 8.533 GiB |
| [outputs/train/firefly_y6_smolvla_fullcycle_v4_100k_attested_run1/checkpoints/100000/training_state](../outputs/train/firefly_y6_smolvla_fullcycle_v4_100k_attested_run1/checkpoints/100000/training_state) | 0.384 GiB |
| [outputs/train/firefly_y6_smolvla_fullcycle_v4_arm_only_v1_finetune_20k_attested_run2/checkpoints/020000/training_state](../outputs/train/firefly_y6_smolvla_fullcycle_v4_arm_only_v1_finetune_20k_attested_run2/checkpoints/020000/training_state) | 0.384 GiB |
| [outputs/train/firefly_y6_smolvla_fullcycle_v4_arm_only_v1_finetune_20k_attested_run2/checkpoints/010000/training_state](../outputs/train/firefly_y6_smolvla_fullcycle_v4_arm_only_v1_finetune_20k_attested_run2/checkpoints/010000/training_state) | 0.384 GiB |

`pip_cache` 表中容量为清理前统计，目前其下载缓存已删除。`training_state` 是模型目录子项，不额外增加总量。

**当前 arm6 校验会核对两份 checkpoint、父模型、两套数据、run commit 和 training_step.json，不能单删 010000 或整个 training_state。** 只有决定放弃该旧部署链路后，才考虑按整套资产归档/删除。

## 历史 local 配置

这些是旧 smoke/validated_v2/validated_v3 路线的配置，很多引用的数据或模型已不在当前数据目录。建议从日常入口中移出，保留历史记录；不要误删当前 fullcycle_v4 和 arm6 配置。

- [lerobot-robot-hexbase/configs/deploy_firefly_y6.local.yaml](../lerobot-robot-hexbase/configs/deploy_firefly_y6.local.yaml)
- [lerobot-robot-hexbase/configs/deploy_firefly_y6_hold_validated_v2.local.yaml](../lerobot-robot-hexbase/configs/deploy_firefly_y6_hold_validated_v2.local.yaml)
- [lerobot-robot-hexbase/configs/deploy_firefly_y6_hold_validated_v3_trimmed.local.yaml](../lerobot-robot-hexbase/configs/deploy_firefly_y6_hold_validated_v3_trimmed.local.yaml)
- [lerobot-robot-hexbase/configs/deploy_firefly_y6_hold_validated_v3_trimmed_fast90_per_joint_gravity_ff_gr80_payload_j3j4_kp72_kd2p75_eekp150_phase_latch_v6_recovery_preopen.local.yaml](../lerobot-robot-hexbase/configs/deploy_firefly_y6_hold_validated_v3_trimmed_fast90_per_joint_gravity_ff_gr80_payload_j3j4_kp72_kd2p75_eekp150_phase_latch_v6_recovery_preopen.local.yaml)
- [lerobot-robot-hexbase/configs/deploy_firefly_y6_hold_validated_v3_trimmed_fast90_per_joint_gravity_ff_gr80_payload_kp60_kd2p5_eekp150.local.yaml](../lerobot-robot-hexbase/configs/deploy_firefly_y6_hold_validated_v3_trimmed_fast90_per_joint_gravity_ff_gr80_payload_kp60_kd2p5_eekp150.local.yaml)
- [lerobot-robot-hexbase/configs/deploy_firefly_y6_hold_validated_v3_trimmed_fast90_per_joint_gravity_ff_gr80_payload_kp60_kd2p5_eekp150_phase_latch_v1.local.yaml](../lerobot-robot-hexbase/configs/deploy_firefly_y6_hold_validated_v3_trimmed_fast90_per_joint_gravity_ff_gr80_payload_kp60_kd2p5_eekp150_phase_latch_v1.local.yaml)
- [lerobot-robot-hexbase/configs/deploy_firefly_y6_hold_validated_v3_trimmed_fast90_per_joint_gravity_ff_gr80_payload_kp60_kd2p5_eekp150_phase_latch_v2.local.yaml](../lerobot-robot-hexbase/configs/deploy_firefly_y6_hold_validated_v3_trimmed_fast90_per_joint_gravity_ff_gr80_payload_kp60_kd2p5_eekp150_phase_latch_v2.local.yaml)
- [lerobot-robot-hexbase/configs/deploy_firefly_y6_hold_validated_v3_trimmed_fast90_per_joint_gravity_ff_gr80_payload_kp60_kd2p5_eekp150_phase_latch_v3.local.yaml](../lerobot-robot-hexbase/configs/deploy_firefly_y6_hold_validated_v3_trimmed_fast90_per_joint_gravity_ff_gr80_payload_kp60_kd2p5_eekp150_phase_latch_v3.local.yaml)
- [lerobot-robot-hexbase/configs/deploy_firefly_y6_hold_validated_v3_trimmed_fast90_per_joint_gravity_ff_gr80_payload_kp60_kd2p5_eekp150_phase_latch_v4_approx_start.local.yaml](../lerobot-robot-hexbase/configs/deploy_firefly_y6_hold_validated_v3_trimmed_fast90_per_joint_gravity_ff_gr80_payload_kp60_kd2p5_eekp150_phase_latch_v4_approx_start.local.yaml)
- [lerobot-robot-hexbase/configs/deploy_firefly_y6_hold_validated_v3_trimmed_fast90_per_joint_gravity_ff_gr80_payload_kp60_kd2p5_eekp150_phase_latch_v5_recovery_preopen.local.yaml](../lerobot-robot-hexbase/configs/deploy_firefly_y6_hold_validated_v3_trimmed_fast90_per_joint_gravity_ff_gr80_payload_kp60_kd2p5_eekp150_phase_latch_v5_recovery_preopen.local.yaml)
- [lerobot-robot-hexbase/configs/deploy_firefly_y6_hold_validated_v3_trimmed_slew002.local.yaml](../lerobot-robot-hexbase/configs/deploy_firefly_y6_hold_validated_v3_trimmed_slew002.local.yaml)
- [lerobot-robot-hexbase/configs/deploy_firefly_y6_hold_validated_v3_trimmed_slew002_gravity_ff.local.yaml](../lerobot-robot-hexbase/configs/deploy_firefly_y6_hold_validated_v3_trimmed_slew002_gravity_ff.local.yaml)
- [lerobot-robot-hexbase/configs/deploy_firefly_y6_hold_validated_v3_trimmed_slew002_gravity_ff_gr80_payload.local.yaml](../lerobot-robot-hexbase/configs/deploy_firefly_y6_hold_validated_v3_trimmed_slew002_gravity_ff_gr80_payload.local.yaml)
- [lerobot-robot-hexbase/configs/deploy_firefly_y6_hold_validated_v3_trimmed_slew002_gravity_ff_gr80_payload_kp60_kd2p5.local.yaml](../lerobot-robot-hexbase/configs/deploy_firefly_y6_hold_validated_v3_trimmed_slew002_gravity_ff_gr80_payload_kp60_kd2p5.local.yaml)
- [lerobot-robot-hexbase/configs/record_firefly_y6.local.yaml](../lerobot-robot-hexbase/configs/record_firefly_y6.local.yaml)
- [lerobot-robot-hexbase/configs/record_firefly_y6_validated_v2.local.yaml](../lerobot-robot-hexbase/configs/record_firefly_y6_validated_v2.local.yaml)

## 历史训练与数据整理脚本

以下可作为旧流程归档候选，并非已证明可以独立删除；部分被测试或其他脚本导入。新版本流程确认替代后，再一起处理对应测试和配置。

- [lerobot-robot-hexbase/scripts/record_firefly_y6_validated_v2.py](../lerobot-robot-hexbase/scripts/record_firefly_y6_validated_v2.py)
- [lerobot-robot-hexbase/scripts/train_firefly_smolvla_30k.py](../lerobot-robot-hexbase/scripts/train_firefly_smolvla_30k.py)
- [lerobot-robot-hexbase/scripts/train_firefly_smolvla_v3_trimmed_30k.py](../lerobot-robot-hexbase/scripts/train_firefly_smolvla_v3_trimmed_30k.py)
- [lerobot-robot-hexbase/scripts/materialize_firefly_y6_trimmed.py](../lerobot-robot-hexbase/scripts/materialize_firefly_y6_trimmed.py)
- [lerobot-robot-hexbase/scripts/train_firefly_smolvla_arm_only_v1_finetune_20k.py](../lerobot-robot-hexbase/scripts/train_firefly_smolvla_arm_only_v1_finetune_20k.py)

## 与当前源码不同的备份

这 7 份 `.orig` 均已逐个比对，与对应当前文件不相同；不是相同内容的重复文件。本次保留，选择归档/删除前确认不需要回看旧实现。

- [lerobot-robot-hexbase/configs/deploy_firefly_y6_hold_fullcycle_v4_100k.local.yaml.orig](../lerobot-robot-hexbase/configs/deploy_firefly_y6_hold_fullcycle_v4_100k.local.yaml.orig)（4.3 KiB）
- [lerobot-robot-hexbase/lerobot_robot_hexbase/firefly_deployment.py.orig](../lerobot-robot-hexbase/lerobot_robot_hexbase/firefly_deployment.py.orig)（150.8 KiB）
- [lerobot-robot-hexbase/lerobot_robot_hexbase/firefly_visual_latch.py.orig](../lerobot-robot-hexbase/lerobot_robot_hexbase/firefly_visual_latch.py.orig)（51.8 KiB）
- [lerobot-robot-hexbase/scripts/audit_firefly_y6_visual_latch.py.orig](../lerobot-robot-hexbase/scripts/audit_firefly_y6_visual_latch.py.orig)（122.9 KiB）
- [lerobot-robot-hexbase/scripts/run_firefly_y6.py.orig](../lerobot-robot-hexbase/scripts/run_firefly_y6.py.orig)（113.1 KiB）
- [lerobot-robot-hexbase/tests/test_hex_zenoh_follower.py.orig](../lerobot-robot-hexbase/tests/test_hex_zenoh_follower.py.orig)（115.4 KiB）
- [lerobot-robot-hexbase/tests/test_run_firefly_y6.py.orig](../lerobot-robot-hexbase/tests/test_run_firefly_y6.py.orig)（149.5 KiB）

## 旧部署输出子目录

这些旧 validated_v2/v3 等试验输出约占 217 MiB。包含历史日志、视频及旧验收文件，可按不再复现的试验整组清理。当前 fullcycle_v4 试验和 `firefly_diagnostics` 未列入这组，后者包含现有部署依赖，不能随日志一并删除。

| 路径 | 容量 |
| --- | ---: |
| [lerobot-robot-hexbase/outputs/firefly_deployment](../lerobot-robot-hexbase/outputs/firefly_deployment) | 6.27 MiB |
| [lerobot-robot-hexbase/outputs/firefly_deployment_hold_fast90_arm003_j3_002_ee003_j3j4_kp72_kd2p75_eekp150_close060_phase_latch_v6_recovery_preopen_gravity_ff8_gr80_payload_validated_v3_trimmed_attested_run1](../lerobot-robot-hexbase/outputs/firefly_deployment_hold_fast90_arm003_j3_002_ee003_j3j4_kp72_kd2p75_eekp150_close060_phase_latch_v6_recovery_preopen_gravity_ff8_gr80_payload_validated_v3_trimmed_attested_run1) | 18.58 MiB |
| [lerobot-robot-hexbase/outputs/firefly_deployment_hold_fast90_arm003_j3_002_ee003_kp60_kd2p5_eekp150_close060_gravity_ff8_gr80_payload_validated_v3_trimmed_attested_run1](../lerobot-robot-hexbase/outputs/firefly_deployment_hold_fast90_arm003_j3_002_ee003_kp60_kd2p5_eekp150_close060_gravity_ff8_gr80_payload_validated_v3_trimmed_attested_run1) | 35.79 MiB |
| [lerobot-robot-hexbase/outputs/firefly_deployment_hold_fast90_arm003_j3_002_ee003_kp60_kd2p5_eekp150_close060_phase_latch_v1_gravity_ff8_gr80_payload_validated_v3_trimmed_attested_run1](../lerobot-robot-hexbase/outputs/firefly_deployment_hold_fast90_arm003_j3_002_ee003_kp60_kd2p5_eekp150_close060_phase_latch_v1_gravity_ff8_gr80_payload_validated_v3_trimmed_attested_run1) | 36.29 MiB |
| [lerobot-robot-hexbase/outputs/firefly_deployment_hold_fast90_arm003_j3_002_ee003_kp60_kd2p5_eekp150_close060_phase_latch_v2_gravity_ff8_gr80_payload_validated_v3_trimmed_attested_run1](../lerobot-robot-hexbase/outputs/firefly_deployment_hold_fast90_arm003_j3_002_ee003_kp60_kd2p5_eekp150_close060_phase_latch_v2_gravity_ff8_gr80_payload_validated_v3_trimmed_attested_run1) | 30.74 MiB |
| [lerobot-robot-hexbase/outputs/firefly_deployment_hold_fast90_arm003_j3_002_ee003_kp60_kd2p5_eekp150_close060_phase_latch_v3_gravity_ff8_gr80_payload_validated_v3_trimmed_attested_run1](../lerobot-robot-hexbase/outputs/firefly_deployment_hold_fast90_arm003_j3_002_ee003_kp60_kd2p5_eekp150_close060_phase_latch_v3_gravity_ff8_gr80_payload_validated_v3_trimmed_attested_run1) | 3.38 MiB |
| [lerobot-robot-hexbase/outputs/firefly_deployment_hold_fast90_arm003_j3_002_ee003_kp60_kd2p5_eekp150_close060_phase_latch_v5_recovery_preopen_gravity_ff8_gr80_payload_validated_v3_trimmed_attested_run1](../lerobot-robot-hexbase/outputs/firefly_deployment_hold_fast90_arm003_j3_002_ee003_kp60_kd2p5_eekp150_close060_phase_latch_v5_recovery_preopen_gravity_ff8_gr80_payload_validated_v3_trimmed_attested_run1) | 15.96 MiB |
| [lerobot-robot-hexbase/outputs/firefly_deployment_hold_slew002_ee002_kp40_gravity_ff6_validated_v3_trimmed_attested_run1](../lerobot-robot-hexbase/outputs/firefly_deployment_hold_slew002_ee002_kp40_gravity_ff6_validated_v3_trimmed_attested_run1) | 4.45 MiB |
| [lerobot-robot-hexbase/outputs/firefly_deployment_hold_slew002_ee002_kp40_gravity_ff8_gr80_payload_validated_v3_trimmed_attested_run1](../lerobot-robot-hexbase/outputs/firefly_deployment_hold_slew002_ee002_kp40_gravity_ff8_gr80_payload_validated_v3_trimmed_attested_run1) | 4.97 MiB |
| [lerobot-robot-hexbase/outputs/firefly_deployment_hold_slew002_ee002_kp40_validated_v3_trimmed_attested_run1](../lerobot-robot-hexbase/outputs/firefly_deployment_hold_slew002_ee002_kp40_validated_v3_trimmed_attested_run1) | 3.96 MiB |
| [lerobot-robot-hexbase/outputs/firefly_deployment_hold_slew002_ee002_kp60_kd2p5_gravity_ff8_gr80_payload_validated_v3_trimmed_attested_run1](../lerobot-robot-hexbase/outputs/firefly_deployment_hold_slew002_ee002_kp60_kd2p5_gravity_ff8_gr80_payload_validated_v3_trimmed_attested_run1) | 16.65 MiB |
| [lerobot-robot-hexbase/outputs/firefly_deployment_hold_slew003_kp40_validated_v2_attested_run1](../lerobot-robot-hexbase/outputs/firefly_deployment_hold_slew003_kp40_validated_v2_attested_run1) | 10.19 MiB |
| [lerobot-robot-hexbase/outputs/firefly_deployment_hold_slew003_validated_v2_attested_run1](../lerobot-robot-hexbase/outputs/firefly_deployment_hold_slew003_validated_v2_attested_run1) | 2.71 MiB |
| [lerobot-robot-hexbase/outputs/firefly_deployment_hold_slew005_ee002_kp40_validated_v2_attested_run1](../lerobot-robot-hexbase/outputs/firefly_deployment_hold_slew005_ee002_kp40_validated_v2_attested_run1) | 17.98 MiB |
| [lerobot-robot-hexbase/outputs/firefly_deployment_hold_slew005_ee002_kp40_validated_v3_trimmed_attested_run1](../lerobot-robot-hexbase/outputs/firefly_deployment_hold_slew005_ee002_kp40_validated_v3_trimmed_attested_run1) | 4.03 MiB |
| [lerobot-robot-hexbase/outputs/firefly_deployment_hold_slew_validated_v2_attested_run1](../lerobot-robot-hexbase/outputs/firefly_deployment_hold_slew_validated_v2_attested_run1) | 2.66 MiB |
| [lerobot-robot-hexbase/outputs/firefly_deployment_hold_validated_v2_attested_run1](../lerobot-robot-hexbase/outputs/firefly_deployment_hold_validated_v2_attested_run1) | 2.50 MiB |

## 旧 NAS 模型导入

优先考虑清理 `imports/nas_outputs_20260812/outputs/train/smolvla_firefly_expert_b8_30k`；整个导入目录约 0.845 GiB。保留 [AUDIT.md](../imports/nas_outputs_20260812/AUDIT.md) 可记录它为何被弃用。它不是当前 fullcycle_v4 的 100k 模型。

## 实际已清理

| 缓存路径 | 文件数 | 释放分配空间 |
| --- | ---: | ---: |
| `pip_cache/http-v2` | 614 | 1538.52 MiB |
| `pip_cache/wheels` | 4 | 0.10 MiB |
| `lerobot-robot-hexbase/.pytest_cache` | 5 | 0.14 MiB |
| `tools/__pycache__` | 3 | 0.02 MiB |
| `lerobot-robot-hexbase/scripts/__pycache__` | 20 | 0.75 MiB |
| `lerobot-robot-hexbase/tests/__pycache__` | 52 | 3.64 MiB |
| `lerobot-robot-hexbase/lerobot_robot_hexbase/__pycache__` | 21 | 0.88 MiB |
| `lerobot-robot-hexbase/lerobot_robot_hexbase/proto/__pycache__` | 7 | 0.03 MiB |

合计 726 个文件，1.508 GiB。未删除源码、录制、训练数据、模型、标定或部署证据。

`/home/kk/.cache/huggingface/lerobot/local`、工作区 `datasets/_archive` 和 `datasets/.trash` 目前为空。未扫描 NAS 或其他磁盘。
