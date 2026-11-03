# Firefly Y6 采集链路启动与关闭

本文对应当前本机链路：双 Firefly Y6 + GR80、`hex-hw-supervisor`、
launcher/Zenoh、`arm_teleop`、D435 场景相机和倒装的 D405 腕部相机。
录制步骤见 [RECORDING.md](RECORDING.md)，训练步骤见
[TRAINING.md](TRAINING.md)。

## 0. 安全前提

`pc-always-clear.yaml` 持续发布 `EstopState.engaged=false`，它不是实际急停保护。
上电前必须确认：

- 有独立、可触及并已验证的断电或急停手段。
- leader/follower 周围无人、无障碍物，线缆不会被关节卷入。
- 两条 CAN 总线、机械臂、夹爪和相机供电稳定。
- 操作者知道 `arm0/ee0` 是 leader，`arm1/ee1` 是 follower。

本文只启动已经存在的控制链路。不要另外运行 `lerobot-setup-motors`、
`lerobot-calibrate`，也不要让其他程序同时占用 CAN 或 controller session。

## 1. 每次上机前检查

新开一个普通终端，不要使用 `sudo`：

```bash
source /home/kk/kk_data/lerobot/activate-io.sh
cd /home/kk/kk_data/lerobot/lerobot-robot-hexbase

python -c "import lerobot, lerobot_robot_hexbase; print(lerobot.__version__)"
python -m pip show pyrealsense2 | sed -n '1,8p'
ip -br link show | rg '^(can0|can1)\b'
rs-enumerate-devices | rg 'Name|Serial Number'
lsusb -t
```

期望结果：

- LeRobot 为 `0.6.0`。
- `pyrealsense2` 为已经实机验证的 `2.58.3.10794`。
- D435 的 LeRobot/librealsense 序列号为 `030522071867`。
- D405 的 LeRobot/librealsense 序列号为 `323622270296`。
- 两台 RealSense 都工作在 `5000M` 或 `10000M`；尽量分接不同 USB 3
  控制器，不要共用无源 Hub。

`lsusb` 显示的 USB 描述符序列号不一定等于 librealsense 序列号。配置相机时以
`rs-enumerate-devices` 或 `lerobot-find-cameras realsense` 的结果为准。

首次安装或重建采集环境，先按 [安装与环境恢复](INSTALL.md) 安装固定依赖和
RealSense USB 权限规则。依赖已完整恢复后，如需重新注册当前插件源码：

```bash
source /home/kk/kk_data/lerobot/activate-io.sh
cd /home/kk/kk_data/lerobot/lerobot-robot-hexbase
python -m pip install --no-deps -e '.[realsense,dev]'
python -m pip check
```

## 2. 按顺序启动四个终端

### 终端 1：supervisor

```bash
cd /home/kk/kk_data/git_clone/hex-hw-supervisor
./target/release/hex-hw-supervisor ./profiles/pc-always-clear.yaml
```

保持该进程在前台运行。若它退出，先停止后续进程并排查，不要绕过 supervisor。

### 终端 2：launcher、Zenoh 和双臂 controller

```bash
cd /home/kk/kk_data/hex-arm/remote-control
./launcher launch_real_dual_arm_teleop.yaml
```

另开终端检查 Zenoh TCP endpoint：

```bash
ss -ltnp | rg ':7447'
```

当前 launcher 配置监听 `tcp/0.0.0.0:7447`，本机采集插件连接
`tcp/127.0.0.1:7447`。

### 终端 3：leader-follower 遥操作

确认机械臂处于安全初始姿态后运行：

```bash
cd /home/kk/kk_data/hex-arm/remote-control
EE_KP=150 \
EE_SQUEEZE=0.04 \
EE_DFR_GAIN=0.2 \
EE_DFR_TAU_LIMIT=1.5 \
EE_DFR_LEADER_KD=0.05 \
ARM_KP=40 \
ARM_KD=2 \
MODE=dfr \
DFR_GAIN=0.1 \
DFR_TAU_LIMIT=3 \
./arm_teleop \
  hexmeow/50e41fe1cf464c2581a44f1339e6eb4e/arm0 \
  hexmeow/50e41fe1cf464c2581a44f1339e6eb4e/arm1 \
  hexmeow/50e41fe1cf464c2581a44f1339e6eb4e/ee0 \
  hexmeow/50e41fe1cf464c2581a44f1339e6eb4e/ee1
```

旧命令同时写过 `MODE=follow` 和 `MODE=dfr`；Shell 只采用最后一个值。这里保留
实际生效的 `MODE=dfr`，避免误解。

### 终端 4：验证 LeRobot 只读采集

先只读 follower 实测位置：

```bash
source /home/kk/kk_data/lerobot/activate-io.sh
cd /home/kk/kk_data/lerobot/lerobot-robot-hexbase
python scripts/probe_hex_capture.py --state-only --duration-s 10 --hz 10
```

再检查实测位置和遥操作目标命令配对：

```bash
python scripts/probe_hex_capture.py --duration-s 10 --hz 30 --max-age-s 0.1 \
  --contract-output outputs/firefly_diagnostics/controller_contract_live.json
```

期望连续打印 7 维 `state` 和 `target`，顺序都是：

```text
joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, jaw_joint1
```

`state` 是 follower 实测位置，`target` 是经过实时 description、内侧裕量和逐步检查的
`arm_teleop` requested target；二者应同方向变化，运动中允许有跟踪误差。完整探针必须
保持 30 Hz，否则一个采样周期会跨过多个控制 tick。探针默认只运行 10 秒，正常返回不是
崩溃；持续观察可把 `--duration-s` 改为 `3600`，用 `Ctrl+C` 结束。

先把 follower 移到有余量的中心工作区并保持静止；当前 Firefly 至少应让 requested
`joint_2 > -1.55`、`joint_3 < 3.12`，实际 smoke 建议留到约 `joint_2 > -1.53`、
`joint_3 < 3.05`。只有完整 30 Hz 探针成功时才会原子写出 contract；报错时不得开始
录制，也不要放宽 margin、freshness 或 step 上限绕过。

采集插件只做 Zenoh description query 与 state/command/status subscribe，不获取控制
session、不开 publisher，也不向机械臂发送命令。

## 3. 进入录制

上述四步全部通过后，再按照 [录制文档](RECORDING.md) 启动
`lerobot-record`。录制时保持终端 1、2、3 的进程持续运行。

## 4. 正常关闭顺序

1. 让 `lerobot-record` 完成最后一条并正常退出；需要提前结束时按 `q` 一次。
2. 在 `arm_teleop` 终端按 `Ctrl+C`。
3. 在 launcher 终端按 `Ctrl+C`，确认子进程退出。
4. 最后停止 `hex-hw-supervisor`。
5. 按硬件规定断开机械臂和相机电源。

不要先杀 supervisor、launcher 或 Zenoh，再让录制器继续写数据；这样会产生 stale
state/command 或不完整 episode。

## 5. 启动故障速查

### `ModuleNotFoundError: lerobot_robot_hexbase`

使用了系统 `python3` 或 `sudo python3`。重新 `source activate-io.sh`，然后使用
当前环境的 `python`；探针和录制均不需要 sudo。

### 连接 7447 超时

```bash
ss -ltnp | rg ':7447'
pgrep -af 'launcher|zenoh|arm_teleop|hex-hw-supervisor'
```

检查 launcher 是否仍在运行，以及 controller ID、`arm1/ee1` 前缀是否与本机 YAML
一致。

### stale follower state / command

- stale state：launcher/controller 的 joint state 已停止更新。
- stale command：`arm_teleop` 未持续发布 `arm1/ee1` 目标。

不要通过增大 `max_state_age_s` 或 `max_command_age_s` 掩盖进程、CAN 或 Zenoh 故障。

### 相机存在但 D435 无法启动

先核对 `python -m pip show pyrealsense2`。本机曾因 `2.56.5` 无法启动 D435，升级到
`2.58.3.10794` 后双相机端到端录制通过。不要把 `lsusb` 的描述符编号写入录制
YAML。

### `uvcvideo ... -71`、`USB disconnect` 或 `latest frame is too old`

这是相机/USB 链路断流，不是录制快捷键故障。更换可靠的 USB 3 数据线，将 D435
和 D405 分接不同控制器，随后重新运行 `rs-enumerate-devices` 和 `lsusb -t`。不要
继续复用旧帧，否则会污染训练数据。
