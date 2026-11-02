# Firefly Y6 数据集录制

开始前必须先完成 [启动与验证](STARTUP.md)。本文按当前 LeRobot `0.6.0`、
`hex_zenoh_observer`、`hex_zenoh_command_tap`、D435 + D405 双 RGB 相机编写。

## 1. 数据对应关系

| 数据集字段 | 来源 | 形状/说明 |
| --- | --- | --- |
| `observation.state` | follower `arm1/ee1` 实测 q | `(7,)`，rad |
| `action` | 状态一致性校验后的 `arm_teleop` requested target q | `(7,)`，rad |
| `observation.images.scene` | D435 RGB | `640×480 @ 30 FPS` |
| `observation.images.wrist` | D405 RGB | `640×480 @ 30 FPS`，软件旋转 180° |

7 个关节的固定顺序是 `joint_1..joint_6, jaw_joint1`。不能把 follower 实测位置
同时写成 action；当前 command tap 已经将实测状态和目标命令分开。

command tap 会只读查询实时 controller description，并校验单点轨迹、有限值、
`RUNNING`/session holder、timeout 能力、100 ms freshness、20 ms 配对偏差、内侧位置
裕量和 30 Hz 逐步上限。协议没有逐命令 ACK 或 applied/clamped target 回显，因此 action
仍是 validated requested target，不应写成 controller 已应用目标。任何校验失败都会终止
本次录制；它不会停止独立运行的 `arm_teleop`，操作者仍须手动安全接管。

## 2. 准备本机配置

首次配置时：

```bash
source /home/kk/kk_data/lerobot/activate-io.sh
cd /home/kk/kk_data/lerobot/lerobot-robot-hexbase
test -f configs/record_firefly_y6.local.yaml
```

本机 local 文件已填好真实相机序列号和安全 smoke 参数，不要用 example 覆盖它。
仅在新电脑且 local 不存在时才复制 example，然后手工填写相机序列号。

`record_firefly_y6.local.yaml` 已被 Git 忽略，可保存本机相机序列号。至少核对以下
字段：

```yaml
robot:
  cameras:
    scene:
      serial_number_or_name: "030522071867"  # D435
    wrist:
      serial_number_or_name: "323622270296"  # D405
      rotation: 180

teleop:
  max_command_age_s: 0.100
  max_state_age_s: 0.100
  max_pair_skew_s: 0.020
  record_hz: 30.0
  arm_step_limit_rad: 0.035
  ee_step_limit_rad: 0.012
  arm_position_margin_rad: 0.020
  ee_position_margin_rad: 0.005

dataset:
  repo_id: local/firefly_y6_manual_validated_v2_smoke
  root: /home/kk/kk_data/lerobot/datasets/local/firefly_y6_manual_validated_v2_smoke
  single_task: Pick up the red block and place it in the blue tray.
  fps: 30
  episode_time_s: 10
  reset_time_s: 5
  num_episodes: 1
  push_to_hub: false

display_data: false
play_sounds: false
```

`rotation: 180` 是每帧读取后的软件变换，不会写入相机，也不会在下一次录制时
累积成 360°。

`single_task` 必须写真实任务。ACT 不读取语言描述，但 SmolVLA 等 VLA 会使用它；
不要继续使用旧的 `Move the follower arm through a short test trajectory.` 来描述红色
方块放入蓝色托盘的数据。

本机 local 文件永久作为 1×10 秒 smoke 配置使用。正式录制通过 CLI 显式覆盖数据集
参数，不要把 smoke local 原地改成正式配置。

## 3. 先录一条 smoke test

保持 supervisor、launcher 和 `arm_teleop` 运行，在第四个终端执行：

```bash
source /home/kk/kk_data/lerobot/activate-io.sh
cd /home/kk/kk_data/lerobot/lerobot-robot-hexbase

lerobot-record --config_path=configs/record_firefly_y6.local.yaml
```

应看到两台 RealSense connected、键盘监听已启用和
`Recording episode 0`。10 秒结束后应编码两路视频并正常断开。第一条通过后再录
50 条，避免长时间录制后才发现相机方向、任务描述或关节顺序错误。

## 4. 正式手动分段录制

```bash
source /home/kk/kk_data/lerobot/activate-io.sh
cd /home/kk/kk_data/lerobot/lerobot-robot-hexbase
echo "$HF_LEROBOT_HOME"

lerobot-record \
  --config_path=configs/record_firefly_y6.local.yaml \
  --dataset.repo_id=local/firefly_y6_manual_validated_v2 \
  --dataset.root=/home/kk/kk_data/lerobot/datasets/local/firefly_y6_manual_validated_v2 \
  --dataset.single_task='Pick up the red block and place it in the blue tray.' \
  --dataset.num_episodes=50 \
  --dataset.episode_time_s=600 \
  --dataset.reset_time_s=600
```

先 `source activate-io.sh` 很重要，用于加载正确的 Python、Zenoh 和 RealSense 环境。
本机 smoke YAML 和上面的正式命令都显式传入 `dataset.root`，因此数据位置不再依赖
启动 shell 的 `HF_LEROBOT_HOME`。每次新录制的 root 必须尚不存在；不要覆盖或 resume
旧数据集。

每条 episode 的操作顺序：

1. 看到 `Recording episode N` 后完成一次完整任务。
2. 动作完成后，在运行 `lerobot-record` 的终端轻按一次 `n`。
3. 看到 `Reset the environment` 后复位机械臂、红色方块和蓝色托盘。
4. 复位完成后轻按一次 `n`。
5. 等待下一条 `Recording episode N+1` 出现，再开始动作。

快捷键：

| 按键 | 作用 |
| --- | --- |
| `n` 或右方向键 | 提前结束当前录制/reset 阶段 |
| `r` 或左方向键 | 丢弃当前 episode 并重录 |
| `q` 或 `Esc` | 停止整个录制 |

优先使用字母键并且每次只按一下。方向键可能产生按键重复；如果第二次事件落在新
episode 读取第一帧之前，会出现
`You must add one or several frames ... before calling add_episode`。按键后先等待日志
切换阶段，不要连续敲 `n` 或长按方向键。

计划录满 50 条时，最好让 `num_episodes: 50` 自动结束。提前按 `q` 时，已经完成的
episode 通常会被 finalize，但仍必须按第 6 节检查，不能凭目录存在就认为完整。

## 5. 数据保存位置

当前路径固定为：

```text
smoke: /home/kk/kk_data/lerobot/datasets/local/firefly_y6_manual_validated_v2_smoke
正式:  /home/kk/kk_data/lerobot/datasets/local/firefly_y6_manual_validated_v2
```

显式指定 `dataset.root` 后不会自动给目录加时间戳。开始前确认目标不存在；如需另起
一次正式录制，同时更换 repo_id 和 root（例如都加 `_run02`），不要让新 repo_id 写入
旧 root，也不要对旧的 50 条数据使用 `--resume=true`。

## 6. 录完立即检查

先设置刚录好的实际目录：

```bash
export FIREFLY_DATASET_ROOT=/home/kk/kk_data/lerobot/datasets/local/firefly_y6_manual_validated_v2
```

查看元信息：

```bash
source /home/kk/kk_data/lerobot/activate-io.sh
python -c 'import json, os; p=os.environ["FIREFLY_DATASET_ROOT"] + "/meta/info.json"; d=json.load(open(p)); print("episodes=", d["total_episodes"]); print("frames=", d["total_frames"]); print("fps=", d["fps"]); print("features=", list(d["features"]))'
```

用 LeRobot 加载首尾样本：

```bash
python -c 'import os; from lerobot.datasets.lerobot_dataset import LeRobotDataset; p=os.environ["FIREFLY_DATASET_ROOT"]; d=LeRobotDataset("local/firefly_y6_manual_validated_v2", root=p, download_videos=False); print("episodes=", d.num_episodes, "frames=", len(d)); print("first=", d[0]["episode_index"].item(), "last=", d[len(d)-1]["episode_index"].item()); print("scene=", tuple(d[0]["observation.images.scene"].shape), "wrist=", tuple(d[0]["observation.images.wrist"].shape))'
```

完整数据集必须同时包含：

```text
meta/    # info、任务、episode 索引、统计量
data/    # observation/action 的 Parquet
videos/  # scene/wrist 的 MP4
```

不要只根据 MP4 文件数量判断 episode 数。LeRobot 会把很多 episode 自动打包进较大
的 Parquet/视频分片；`file-000.mp4`、`file-001.mp4` 是分片编号，不是只有 0、1
两条轨迹。真实数量以 `meta/info.json` 和
`meta/episodes/chunk-*/file-*.parquet` 为准。

最后运行只读安全审计；contract 必须来自本轮启动后完整通过的 30 Hz 探针：

```bash
python /home/kk/kk_data/lerobot/lerobot-robot-hexbase/scripts/audit_firefly_dataset.py \
  --root="$FIREFLY_DATASET_ROOT" \
  --repo-id=local/firefly_y6_manual_validated_v2 \
  --controller-contract=/home/kk/kk_data/lerobot/lerobot-robot-hexbase/outputs/firefly_diagnostics/controller_contract_live.json \
  --expected-task='Pick up the red block and place it in the blue tray.'
```

只有退出码 0 和 `safe_for_training: true` 才能训练。退出码 1 表示范围、步长、schema、
task 或视频等检查失败；2 表示输入/metadata 错误。不要手工修改审计失败的数据来放行。

## 7. 旧 50 条数据仅作归档与诊断

旧数据位于：

```text
/home/kk/.cache/huggingface/lerobot/local/firefly_y6_manual_20260811_172346
```

它在文件结构上完整：

- 50 个 episode，索引 0–49，共 41,991 帧。
- 总时长约 23 分 20 秒，单条 13.9–52.0 秒。
- state/action 均为 7 维，无 NaN、Inf 或空 episode。
- scene/wrist 各 41,991 帧、`640×480 @ 30 FPS`。
- D405 的 180° 旋转已写入视频。
- 一个 Parquet 包含全部帧；每路相机各有两个 MP4 分片，这是正常分片结果。

但 action 标签并不满足实时 controller 合同：joint_2 低于下限 12,185 帧，joint_3
高于上限 11,745 帧，二者并集 14,002/41,991（33.35%，50/50 episodes 受影响）；
同 episode 相邻 action 另有 1,454 个 30 Hz step 违规。对应旧 ACT shadow 的 894 个
候选 action 全部被位置或首步安全门拒绝。

因此这批旧数据和由它训练的 checkpoint 只能保留作离线诊断/对照，不能作为新的
真机 replay/ACT 放行依据，也不要通过扩 controller 限位或运行时裁剪来掩盖。正式
真机模型必须使用完整的新 v2 数据集、重算 stats、从头训练，并通过离线与 shadow
零拒绝验收。旧版 [训练文档](TRAINING.md) 记录的是历史基线命令，在更新为 v2 前
不得用于生成新的真机 checkpoint。

## 8. 录制故障速查

### `Failed to find Rerun Viewer executable`

将 `display_data` 保持为 `false`。这只关闭实时 GUI，不影响关节和两路视频写入。

### `No device connected` / `Failed to open RealSenseCamera`

```bash
source /home/kk/kk_data/lerobot/activate-io.sh
python -m pip show pyrealsense2
rs-enumerate-devices | rg 'Name|Serial Number'
lerobot-find-cameras realsense \
  --output-dir=/home/kk/kk_data/lerobot/outputs/captured_images
```

本机需使用 `pyrealsense2 2.58.3.10794`。重新插拔不会改变 librealsense 硬件
序列号；不要用 `lsusb` 的另一组编号替换 YAML。

### `latest frame is too old` 或内核 `USB disconnect/-71`

这是 USB 断流。更换数据线和物理 USB 控制器，避免 D435/D405 共用 Hub；失败的
当前 episode 不应作为正常数据训练。

### stale state / stale command

分别检查 launcher/controller 与 `arm_teleop`。不要简单放宽最大消息年龄。

### 空 episode 的 `ValueError`

通常是切换阶段时重复按键。已经保存的前一条不一定丢失，但必须读取
`meta/info.json`、Parquet 和视频确认。重新开始正式录制前，先把失败目录改名隔离
或明确确认后再删除，不要把半成品和完整数据混在同一训练路径。
