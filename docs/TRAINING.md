# Firefly Y6 数据检查与训练

> **历史基线，禁止用于新的真机 checkpoint。** 本文现有命令指向旧数据集
> `firefly_y6_manual_20260811_172346`。该集有 14,002/41,991 个 action 在实时
> controller 位置范围外，并有 1,454 个相邻 30 Hz step 违规。旧 ACT checkpoint
> 的 shadow action 也未通过安全门。请先按 [录制文档](RECORDING.md) 完成新的
> `firefly_y6_manual_validated_v2` 全量录制与审计，然后重算 stats、从头训练；在
> 本文更新为 v2 命令前，不要把以下历史命令产物用于 replay/ACT。


以下第 1–7 节只保留旧实验的可复现记录，全部路径和命令都指向不合格的 legacy
数据，不是当前训练操作手册。LeRobot `0.6.0` 和 RTX 5070 Ti Laptop 12 GB 的环境
信息仍可作参考；真正的新训练必须先把所有数据路径、repo_id 和 output_dir 换成经
审计的 v2 产物。录制流程见 [RECORDING.md](RECORDING.md)。

## 1. 训练前固定数据路径

当前已核验的 50 条数据是：

```bash
export FIREFLY_DATASET_ROOT=/home/kk/.cache/huggingface/lerobot/local/firefly_y6_manual_20260811_172346
```

该目录包含 50 个 episode、41,991 帧、两路 `640×480 @ 30 FPS` RGB，以及 7 维
state/action。不要只复制 MP4；`meta/`、`data/`、`videos/` 缺一不可。

数据目前位于缓存目录，建议在训练前额外备份到工作区或另一块磁盘：

```bash
rsync -a --info=progress2 \
  /home/kk/.cache/huggingface/lerobot/local/firefly_y6_manual_20260811_172346/ \
  /home/kk/kk_data/lerobot/datasets/firefly_y6_manual_20260811_172346/
```

备份完成后，如果改用新位置，应重新设置：

```bash
export FIREFLY_DATASET_ROOT=/home/kk/kk_data/lerobot/datasets/firefly_y6_manual_20260811_172346
```

## 2. 激活训练环境并检查 GPU

每个新终端都要执行：

```bash
source /home/kk/kk_data/lerobot/activate-train.sh
cd /home/kk/kk_data/lerobot

python tools/diagnose.py
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")'
```

训练环境和采集环境分开：训练使用 `lerobot-train`，采集/相机/插件使用
`lerobot-io`。纯训练不需要启动 supervisor、launcher、`arm_teleop` 或相机。

## 3. 再做一次只读数据检查

```bash
source /home/kk/kk_data/lerobot/activate-train.sh
python -c 'import os; from lerobot.datasets.lerobot_dataset import LeRobotDataset; p=os.environ["FIREFLY_DATASET_ROOT"]; d=LeRobotDataset("local/firefly_y6_manual", root=p, download_videos=False); print("episodes=", d.num_episodes, "frames=", len(d)); print("first/last episode=", int(d[0]["episode_index"]), int(d[len(d)-1]["episode_index"])); print("state=", tuple(d[0]["observation.state"].shape), "action=", tuple(d[0]["action"].shape)); print("scene=", tuple(d[0]["observation.images.scene"].shape), "wrist=", tuple(d[0]["observation.images.wrist"].shape))'
```

期望：`episodes=50`、`frames=41991`、episode `0..49`、state/action `(7,)`、
两路图像 `(3, 480, 640)`。

## 4. 历史基线：ACT（仅供复现）

这批数据是单一视觉操作任务且只有 50 条示范，ACT 比 VLA 更适合作为第一条基线。
ACT 不读取 `single_task` 文本，因此当前数据内旧的任务描述不会影响 ACT。

为每次实验选择一个尚不存在的输出目录：

```bash
export FIREFLY_TRAIN_OUTPUT=/home/kk/kk_data/lerobot/outputs/train/firefly_y6_act_run1
```

历史复现命令（禁止将产物用于真机 replay/ACT）：

```bash
source /home/kk/kk_data/lerobot/activate-train.sh
cd /home/kk/kk_data/lerobot

lerobot-train \
  --dataset.repo_id=local/firefly_y6_manual \
  --dataset.root="$FIREFLY_DATASET_ROOT" \
  --dataset.return_uint8=true \
  --policy.type=act \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.push_to_hub=false \
  --batch_size=4 \
  --num_workers=4 \
  --steps=100000 \
  --save_freq=20000 \
  --wandb.enable=false \
  --output_dir="$FIREFLY_TRAIN_OUTPUT"
```

第一次运行可能下载 ResNet18 ImageNet 预训练权重。输出目录已存在时 LeRobot 会拒绝
覆盖；新实验请改 `run1` 为新的名称，不要删除旧 checkpoint 后复用目录。

另开终端观察：

```bash
watch -n 1 nvidia-smi
```

若 CUDA OOM，先把 `--batch_size=4` 改为 `2`，仍不足再改为 `1`。降低
`num_workers` 主要减少主机内存/解码进程，不直接解决显存不足。

## 5. checkpoint、最终模型和续训

训练会生成：

```text
$FIREFLY_TRAIN_OUTPUT/
└── checkpoints/
    ├── 020000/
    ├── 040000/
    ├── ...
    └── last -> 最后一个 step
```

用于加载策略的目录是：

```text
$FIREFLY_TRAIN_OUTPUT/checkpoints/last/pretrained_model/
```

其中包含 `model.safetensors`、`config.json`、`train_config.json` 和预/后处理器
状态。若训练中断，从最后 checkpoint 续训：

```bash
lerobot-train \
  --config_path="$FIREFLY_TRAIN_OUTPUT/checkpoints/last/pretrained_model/train_config.json" \
  --resume=true
```

续训必须保留整个 step 目录中的 `pretrained_model/` 和 `training_state/`；只复制
`model.safetensors` 不能恢复 optimizer、scheduler、随机数和精确训练步数。

训练 loss 只能用于发现发散或比较趋势，不能替代真实成功率。模型部署前应固定物体
初始分布和成功标准，并对多个 checkpoint 做相同次数的离线/真机评估。

## 6. SmolVLA 微调（旧流程已禁用）

不要复制、修改任务文字或继续训练旧的 50 条 `firefly_y6_manual`：其中 14,002 / 41,991
帧 action 超出 controller 原始位置范围，且 1,454 帧相邻 action 超出当前 30 Hz 步长
合同。修改 task 不能修复这些 action 标签。

也不要使用旧的 `rename_map` 方案。`smolvla_base` 的配置是 6 维 state/action 和三路
camera，而 Firefly 数据是 7 维 state/action 和 `scene/wrist` 两路相机；仅重命名图像
会留下错误的 checkpoint schema。

必须重新录制并审计 `local/firefly_y6_manual_validated_v2`，然后使用
`lerobot-robot-hexbase/configs/train_smolvla_firefly_y6.example.yaml`。训练 CLI 必须同时
传入 `--policy.path=lerobot/smolvla_base`、`--policy.input_features=null` 和
`--policy.output_features=null`，让 LeRobot 从新数据推断完整 schema。单步 smoke、正式
训练命令和数据集审计流程见 `lerobot-robot-hexbase/README.md`。

## 7. 换电脑训练

必须复制完整数据集目录：

```bash
rsync -a --info=progress2 \
  "$FIREFLY_DATASET_ROOT/" \
  用户名@训练电脑:/data/firefly_y6_manual_20260811_172346/
```

新电脑安装相同 LeRobot `0.6.0` 训练环境后，把命令改成：

```bash
--dataset.root=/data/firefly_y6_manual_20260811_172346
```

可使用当前工作区的以下文件重建环境：

```text
configs/lerobot-train.conda.yml
configs/lerobot-train.requirements.txt
```

不要直接复制整个 `miniforge3` 环境目录到另一台机器。SmolVLA 若不想重新下载模型，
还可额外复制 `/home/kk/kk_data/lerobot/hf_cache`；ACT 的 ResNet 权重也可在新机首次
训练时重新下载。

若要在新电脑继续中断的训练，还要复制整个训练 run 目录，包括 checkpoint 的
`pretrained_model/` 和 `training_state/`。纯训练不需要
`lerobot-robot-hexbase`；只有采集或真机部署才需要该插件和硬件链路。

## 8. 当前真机执行状态

`hex_zenoh_observer` 仍是只读录制插件；真正的分阶段执行器已经由
`hex_zenoh_follower` 和 `scripts/run_firefly_y6.py` 实现，并包含双 session、
liveliness、实时 description、范围/步长/跟踪误差和 watchdog 检查。

这不代表当前模型可以运动。旧 dataset/checkpoint 已知不合格，现有 controller
v0.3.7 又只广告 HOLD/FAULT、没有 RAMP_STOP，因此适配器会阻止旧模型
`mark-reviewed`、replay 和 ACT。后续只有在新 v2 数据完成全量审计、重算 stats、
从头训练，并让新 checkpoint 的 shadow 达到零拒绝后，才可继续分阶段验收；模型
运动还要求经断流验证的 RAMP_STOP controller。完整门槛见插件
[README](../lerobot-robot-hexbase/README.md)。
