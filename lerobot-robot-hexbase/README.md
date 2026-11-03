# lerobot_robot_hexbase

这个仓库现在提供四个 LeRobot 0.6 插件：

- hexbase：原有的全向底盘控制插件。
- hex_zenoh_observer：只读 follower 机械臂、夹爪实际位置和相机。
- hex_zenoh_command_tap：只读并校验 arm_teleop 发给 follower 的 requested target，合格后才作为数据集 action。
- hex_zenoh_follower：带 shadow/execute 隔离、双 session 和 fail-closed 检查的真机执行插件。

包名带 lerobot_robot_ 前缀，安装后会被 LeRobot 的第三方插件发现机制自动导入，不需要修改 LeRobot 源码。

## 为什么从 Zenoh 读取

hex-gui 的 Arm (Zenoh) 模块表明，CAN-FD/CANopen 电机数据已经由 hex-hw-supervisor/controller 转换成 robot_api protobuf 并发布到 Zenoh。LeRobot 采集端复用这个边界，不再直接打开 CAN：

| 数据集内容 | Zenoh key | protobuf | 单位 |
| --- | --- | --- | --- |
| observation.state 前 6 维 | hexmeow/<controller>/arm1/arm/joint_state | JointState.q | rad |
| observation.state 第 7 维 | hexmeow/<controller>/ee1/ee/joint_state | JointState.q | rad |
| action 前 6 维 | hexmeow/<controller>/arm1/arm/command | 单点 JointTrajectory.q | rad |
| action 第 7 维 | hexmeow/<controller>/ee1/ee/command | 单点 JointTrajectory.q | rad |
| observation.images.scene | D435 RGB | RealSense | uint8 RGB |
| observation.images.wrist | D405 RGB | RealSense | uint8 RGB |

arm0/ee0 是 leader，arm1/ee1 是 follower。训练时 observation 应当是 follower 实测位置，action 应当是同一时刻的 follower 目标命令；不能把实测位置同时当作 action。

HexZenohObserver.send_action() 只校验并原样返回 action，不获取控制 session，也不向机械臂发命令。因此录制时仍由现有 arm_teleop 独占控制，LeRobot 不会和它争夺 CAN 或 controller 控制权。

录制 tap 每次连接都会只读查询 arm1/ee1 的实时 `ArmDescription` / `EeDescription`，并同时订阅 requested command、实测 joint_state 和 `RobotStatus`。它只接受 30 Hz 单点轨迹：q 必须完整有限，非空的 dq/kp/kd/tau_ff 必须维数正确且有限；command 与 state 必须小于 100 ms，arm/EE 以及同设备 command↔state 的接收偏差不超过 20 ms；status 必须为 `RUNNING` 且 holder 等于该 command 的 session_id，timeout 必须由 controller 广告支持。第一个通过其余全部检查的目标只建立 requested-command 和 arm/EE session 连续性基线，不把 action 与实测 state 间的正常跟踪误差当作 command step；第二个目标起相对上一条已接受 action 做逐轴步长检查，录制中任一 session ID 改变都会 fail-closed。有效步长取配置上限和 `vel_max / 30` 的较小值；为吸收非硬实时 30 Hz 采样的轻微调度抖动，本轮批准的录制上限为 arm 0.0435 rad、EE 0.0165 rad。arm 相比原始 0.035 增加约 24.3%（相比上一版 0.0385 增加约 13.0%），EE 相比原始 0.015 增加 10%。本机实时速度合同对应 arm 0.200、EE 0.100 rad/帧，因此录制门仍分别只使用其 21.75% 和 16.5%。

为避免训练标签再次顶住 controller 边界，录制范围会在实时范围内侧保留裕量：arm 每轴 0.020 rad，EE 0.005 rad。例如当前 Firefly description 下，joint_2 requested target 必须不低于约 -1.55 rad，joint_3 不高于约 3.12 rad。任何违规都会直接终止本次 `lerobot-record`，不会静默裁剪，也不会保存当前未完成 episode。

协议没有 controller 最终 applied/clamped target 或逐命令 ACK；这里记录的是经过上述合同验证的 requested target。tap 报错只会停止 LeRobot 录制，不会停止独立运行的 `arm_teleop` 或机械臂，操作者仍须接管并安全停止遥操作。

## 安装

在用于采集的环境中执行：

    conda activate /home/kk/kk_data/lerobot/miniforge3/envs/lerobot-io
    cd /home/kk/kk_data/lerobot/lerobot-robot-hexbase
    python -m pip install -e '.[realsense,dev]'

生成好的 arm、ee、controller 等 protobuf Python 文件已随插件保存。只有 hex-robot-proto 发生变化时才需要重新生成：

    ROBOT_PROTO_DIR=/home/kk/kk_data/git_clone/hex-robot-proto/proto bash scripts/gen_proto.sh

## 第一步：只验证机械臂位置

先启动 hex-hw-supervisor 和 launcher，但可以暂时不启动 arm_teleop：

    cd ~/k/g/hex-hw-supervisor
    ./target/release/hex-hw-supervisor ./profiles/pc-always-clear.yaml

    cd /home/kk/kk_data/hex-arm/remote-control
    ./launcher launch_real_dual_arm_teleop.yaml

然后从插件目录运行只读状态探针。不要使用 sudo，也不要调用系统 python3；插件和依赖安装在 lerobot-io 环境：

    /home/kk/kk_data/lerobot/miniforge3/envs/lerobot-io/bin/python \
      scripts/probe_hex_capture.py --state-only

如果已经激活 lerobot-io，也可以使用 python scripts/probe_hex_capture.py --state-only。

应连续打印 7 个 state 值，顺序为 joint_1 到 joint_6、jaw_joint1。移动 follower 或执行控制后，数值应变化。若这里超时，优先检查 controller ID、arm1/ee1 前缀、7447 端口和 launcher 是否正常。

## 第二步：验证 observation/action 配对

启动现有遥操作：

    cd /home/kk/kk_data/hex-arm/remote-control
    MODE=follow LEADER_MODE=gravity_comp ALIGN=absolute \
      ARM_KP=40 ARM_KD=2 EE_KP=150 EE_SQUEEZE=0.04 \
      DFR_GAIN=0 EE_DFR_GAIN=0 \
      ./arm_teleop \
      hexmeow/50e41fe1cf464c2581a44f1339e6eb4e/arm0 \
      hexmeow/50e41fe1cf464c2581a44f1339e6eb4e/arm1 \
      hexmeow/50e41fe1cf464c2581a44f1339e6eb4e/ee0 \
      hexmeow/50e41fe1cf464c2581a44f1339e6eb4e/ee1

这里必须只写一个 `MODE`。`MODE=follow` 配合 `LEADER_MODE=gravity_comp` 是无力反馈的
零重力遥操作；`DFR_GAIN=0` 和 `EE_DFR_GAIN=0` 显式关闭臂与夹爪的反射力矩。不要在
同一命令末尾再追加 `MODE=dfr`：Shell 会采用最后一个同名变量，使 leader 进入有阻力的
DFR 力反馈模式。launcher YAML 中两条臂的 `IDLE_MODE` 也应保持 `gravity_comp`。

再运行：

    /home/kk/kk_data/lerobot/miniforge3/envs/lerobot-io/bin/python \
      scripts/probe_hex_capture.py --hz 30 --max-age-s 0.1 \
      --contract-output outputs/firefly_diagnostics/controller_contract_validated_v2_arm0435_ee0165.json

完整探针必须保持 30 Hz，否则一次采样会跨过多个控制 tick，无法正确执行逐步检查。输出中的 state 是 follower 实测 q，target 是经过实时范围、裕量和步长检查的 arm_teleop requested target。两者应同方向变化并且数值接近，但在运动中允许存在正常跟踪误差。

该固定路径是 validated_v2 正式链路唯一使用的 controller contract；探针默认 arm/EE
step 为 0.0435/0.0165 rad。完整 30 Hz 探针成功后才原子写入该文件。controller、profile
或录制门参数变化时必须在正式录制前重新生成，不能沿用旧 contract。

## 第三步：接入 D435 和 D405

查看两台相机的序列号：

    rs-enumerate-devices

本机的 `configs/record_firefly_y6.local.yaml` 是独立的 1×10 秒 smoke；正式录制必须使用
`configs/record_firefly_y6_validated_v2.local.yaml`。两者都已填好本机相机序列号，不要用
example 覆盖。只有换到新电脑且 local 文件不存在时才复制 example、填写两个
`CHANGE_ME`，并保留 wrist 的 `rotation: 180`。

配置以 640x480、30 FPS 录制两路 RGB，并关闭 depth，以减少第一轮联调变量和 USB/编码负载。RGB 数据集稳定后，可把相应相机的 use_depth 改为 true；LeRobot 会增加 observation.images.<camera>_depth。

D435 和 D405 最好分别接到独立的 USB 3 控制器。若双路相机无法同时达到 30 FPS，先单独启用一台相机排查，再检查线材、USB 拓扑和分辨率。

## 第四步：录制 LeRobot 数据集

保持 supervisor、launcher 和 arm_teleop 三个进程运行。先让 follower 在中心工作区静止，并让 arm_teleop 稳定发布 1–2 秒；再运行本机 smoke 配置：

    /home/kk/kk_data/lerobot/miniforge3/envs/lerobot-io/bin/lerobot-record \
      --config_path=configs/record_firefly_y6.local.yaml

本机 local 配置已固定为全新的 `local/firefly_y6_manual_validated_v2_smoke`、1 个 episode、10 秒、30 FPS，不会续写旧的 `firefly_y6_manual`。它设置 `display_data: false`，所以不启动 Rerun 窗口，但两路视频和关节数据仍会写入。`dataset.root` 显式固定为 `/home/kk/kk_data/lerobot/datasets/local/firefly_y6_manual_validated_v2_smoke`，与训练环境的 `HF_LEROBOT_HOME` 一致，不依赖启动 shell。

只有 smoke 退出码为 0，并确认两路视频方向、7 维顺序、夹爪方向和 action/state 对齐后，
才通过正式录制 wrapper 启动或续录；不要直接运行 formal YAML，也不要临时修改 smoke
配置来“升级”为正式集：

    /home/kk/kk_data/lerobot/miniforge3/envs/lerobot-io/bin/python \
      scripts/record_firefly_y6_validated_v2.py

正式配置固定为 `local/firefly_y6_manual_validated_v2`、绝对 root
`/home/kk/kk_data/lerobot/datasets/local/firefly_y6_manual_validated_v2`、任务
`Pick up the red block and place it in the blue box.`、30 FPS、arm/EE step 0.0435/0.0165 rad、20 个
episode；单条只设 600 秒保护上限，演示完成后由操作者按 `n`/右方向键结束，并与
`outputs/firefly_diagnostics/controller_contract_validated_v2_arm0435_ee0165.json` 配对。wrapper
把本轮目标锁死为总计 20 条：root 不存在时以 `resume=false` 新建并录 20 条；root 已存在时先用
固定合同和任务文字完整审计已有数据，只有审计通过才以 `resume=true` 续录 `20 - 已有条数`。
达到 20 条时直接退出，不再连接相机或硬件。formal YAML 中静态的 `resume`/`num_episodes`
只是首次创建基准，wrapper 每次都会安全覆盖；不要绕过 wrapper 手工续录，因为 LeRobot 的
`dataset.num_episodes` 表示“本进程新增条数”而不是“最终总条数”。不要复制或拼接旧数据/smoke。
任务文字会进入每帧的语言条件、数据集 metadata、训练证明和部署指纹，四处必须逐字一致；不要在同一数据集中把
`box` 改成 `tray` 或换成另一句中文。若以后要改任务文字或重新录另一批数据，应使用全新的
repo/root 从 episode 0 开始。

录制得到的主要特征为：

- observation.state：shape (7,)，名称 joint_1.pos 到 joint_6.pos、jaw_joint1.pos。
- action：shape (7,)，名称和顺序与 observation.state 一致，内容为 validated requested target。
- observation.images.scene：D435 RGB。
- observation.images.wrist：D405 RGB。

若 validator 报错，`lerobot-record` 会以非零状态退出；违规帧和当前尚未提交的 episode 不会保存，之前已经提交的 episode 与 dataset root 会保留。这是故意的 fail-closed 行为，所以随后直接用 `resume=false` 的 formal YAML 会因 root 已存在而得到 `FileExistsError`。处理触发原因后重新运行同一条 `scripts/record_firefly_y6_validated_v2.py` 命令即可：它会先审计保留下来的数据，再自动计算剩余条数。若已有数据本身不完整或审计失败，wrapper 会拒绝续录且不会删除、修改数据，也不会连接硬件，此时应先人工审查/归档，而不是强行 resume。不要临时从命令行继续追高 freshness、margin 或 step；步长变化必须像本次一样同时版本化在线 tap、固定 contract 和离线 auditor。episode 内连续目标若超过 arm 0.0435 或 EE 0.0165 rad，仍属于 30 Hz action 步长违规；先把 follower 放在中心工作区并减慢对应关节或夹爪。只有超过 1 秒的非采样空窗才允许重新建立基线，不能用空窗逻辑绕过连续动作违规。

按 `n` 或右方向键会立刻结束当前 episode；`r`/左方向键丢弃并重录当前条，`q`/Esc
结束整次录制。本机正式配置把单条保护上限设为 600 秒，正常演示完成后应主动按
`n`，不必等满 600 秒。非流式 AV1 编码期间 tap 不会被调用；同一 arm/EE session 的
采样空窗超过 1 秒时，下一条合格命令只重新建立 step reference，而不把几秒累计位移
误判成单个 30 Hz step。session 变化、freshness、holder/status、限位和 episode 内连续
0.0435/0.0165 步长门仍然 fail-closed。

录制结束后必须用同一次探针导出的实时 contract 做只读离线审计；退出码 0 才能训练，
1 表示数据不合格，2 表示输入路径或 metadata 错误。smoke 示例：

```bash
PY=/home/kk/kk_data/lerobot/miniforge3/envs/lerobot-io/bin/python

$PY scripts/audit_firefly_dataset.py \
  --root /home/kk/kk_data/lerobot/datasets/local/firefly_y6_manual_validated_v2_smoke \
  --repo-id local/firefly_y6_manual_validated_v2_smoke \
  --controller-contract outputs/firefly_diagnostics/controller_contract_validated_v2_arm0435_ee0165.json \
  --expected-task 'Pick up the red block and place it in the blue box.'
```

正式数据不能靠手工替换 smoke 命令；必须按固定 repo/root/task/0.0435/0.0165 合同执行：

```bash
$PY scripts/audit_firefly_dataset.py \
  --root /home/kk/kk_data/lerobot/datasets/local/firefly_y6_manual_validated_v2 \
  --repo-id local/firefly_y6_manual_validated_v2 \
  --controller-contract outputs/firefly_diagnostics/controller_contract_validated_v2_arm0435_ee0165.json \
  --expected-task 'Pick up the red block and place it in the blue box.'
```

审计会检查 v3 metadata/parquet、7 维顺序、30 Hz 时间戳、NaN/Inf、raw+margin 限位、
episode 内相邻 action 的硬性步长、action/state 关系、任务映射和两路 MP4；缺少上述
固定 0.0435/0.0165 contract 时会 fail-closed。每个
episode 首帧的 `action-state` 差值是跟踪误差诊断：报告会给出均值、最大值及相对
30 Hz step reference 的超出计数，但它不属于相邻 action 步长，也不会单独令审计失败。

## SmolVLA 固定资产与 attested 训练

正式训练只接受全新的 `local/firefly_y6_manual_validated_v2`。单条 smoke、旧
`firefly_y6_manual`、任何旧 local 配置和旧 ACT/SmolVLA checkpoint 都是隔离输入，
不能用于 smoke training、正式训练或部署。截至 2026-08-13，正式 dataset root 和
正式 checkpoint 都尚不存在，因此当前还不能执行下面的训练命令。

本机先从两个精确 revision 的本地 Hugging Face snapshot 一次性物化 SmolVLM 资产和
SmolVLA base。禁止用 repo ID、`main`、Hub 下载结果或旧 checkpoint 代替。物化脚本会
核验 pinned revision/文件并拒绝覆盖已有输出；本机下列两个输出已经物化完成，应直接
复用。只有输出不存在时才按顺序执行物化命令：

```bash
REPO=/home/kk/kk_data/lerobot/lerobot-robot-hexbase
IO_PY=/home/kk/kk_data/lerobot/miniforge3/envs/lerobot-io/bin/python
SMOLVLM_SOURCE=/home/kk/kk_data/lerobot/hf_cache/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467
SMOLVLA_SOURCE=/home/kk/kk_data/lerobot/hf_cache/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205
SMOLVLM_BUNDLE=/home/kk/kk_data/lerobot/assets/firefly/smolvlm2-500m-video-instruct-7b375e1b73b11138ff12fe22c8f2822d8fe03467
SMOLVLA_BASE=/home/kk/kk_data/lerobot/assets/firefly/smolvla_base-c83c3163b8ca9b7e67c509fffd9121e66cb96205

"$IO_PY" "$REPO/scripts/materialize_firefly_smolvlm.py" \
  --source-snapshot "$SMOLVLM_SOURCE" \
  --output "$SMOLVLM_BUNDLE"

"$IO_PY" "$REPO/scripts/materialize_firefly_smolvla_base.py" \
  --source-snapshot "$SMOLVLA_SOURCE" \
  --smolvlm-bundle "$SMOLVLM_BUNDLE" \
  --output "$SMOLVLA_BASE"
```

不要删除现有 bundle 来重复物化。SmolVLA base 的 `config.json` 已把
`vlm_model_name` 固定到上述本地 SmolVLM bundle；训练时也只能通过
`--smolvla-base` 选择上述本地 SmolVLA base。

训练模板是
[`configs/train_smolvla_firefly_y6.example.yaml`](configs/train_smolvla_firefly_y6.example.yaml)。
其 `dataset.root` 是绝对路径
`/home/kk/kk_data/lerobot/datasets/local/firefly_y6_manual_validated_v2`，不能改成相对路径
或 `null`。base 原始 schema 是 6 维和三路通用相机，而本机是 7 维与 `scene/wrist`；
wrapper 会强制 `policy.input_features=null`、`policy.output_features=null`，使 LeRobot 从
正式 metadata 重建 schema，并禁止 `rename_map`。用户不得在 `--` 后重复传入这些字段。

训练必须从 attestation wrapper 进入。它会在训练前/后各执行一次 full dataset audit，
覆盖 `meta/`、`data/`、`videos/` 和固定 0.0435/0.0165 controller contract 的内容 manifest；将
manifest token 注入 `wandb.notes`；强制固定本地 base/VLM、本地 job、关闭 Hub/WandB
push 并保存 checkpoint；最后给每个 `pretrained_model` 写入
`firefly_dataset_attestation.json`。训练期间任一受控输入改变、训练失败、没有 checkpoint
或 attestation 复核失败都会 fail-closed；残留的未 attested checkpoint 禁止使用。

正式数据录制和固定合同审计通过后，先用全新输出目录做一次 attested 单步 smoke：

```bash
source /home/kk/kk_data/lerobot/activate-train.sh
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

REPO=/home/kk/kk_data/lerobot/lerobot-robot-hexbase
TRAIN_PY=/home/kk/kk_data/lerobot/miniforge3/envs/lerobot-train/bin/python
DATASET=/home/kk/kk_data/lerobot/datasets/local/firefly_y6_manual_validated_v2
CONTRACT="$REPO/outputs/firefly_diagnostics/controller_contract_validated_v2_arm0435_ee0165.json"
LOCAL_BASE=/home/kk/kk_data/lerobot/assets/firefly/smolvla_base-c83c3163b8ca9b7e67c509fffd9121e66cb96205
SMOKE_OUTPUT=/home/kk/kk_data/lerobot/outputs/train/firefly_y6_smolvla_validated_v2_attested_smoke_run1
cd "$REPO"

"$TRAIN_PY" "$REPO/scripts/train_firefly_smolvla_attested.py" \
  --dataset-root "$DATASET" \
  --repo-id local/firefly_y6_manual_validated_v2 \
  --controller-contract "$CONTRACT" \
  --expected-task 'Pick up the red block and place it in the blue box.' \
  --output-dir "$SMOKE_OUTPUT" \
  --smolvla-base "$LOCAL_BASE" \
  -- \
  --config_path="$REPO/configs/train_smolvla_firefly_y6.example.yaml" \
  --policy.device=cuda \
  --policy.use_amp=false \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.train_state_proj=true \
  --steps=1 \
  --log_freq=1 \
  --num_workers=0 \
  --persistent_workers=false
```

只有退出码 0、loss/grad norm 有限、日志 schema 为 7 维 `state/action` 和
`scene/wrist`，且单步 checkpoint 内存在并通过复核的
`firefly_dataset_attestation.json`，才用另一个全新目录正式训练：

本机也提供不依赖 shell 变量、不会被长路径换行破坏的一键 30k 启动器：

```bash
/home/kk/kk_data/lerobot/miniforge3/envs/lerobot-train/bin/python \
  scripts/train_firefly_smolvla_30k.py
```

```bash
TRAIN_OUTPUT=/home/kk/kk_data/lerobot/outputs/train/firefly_y6_smolvla_validated_v2_attested_run1

"$TRAIN_PY" "$REPO/scripts/train_firefly_smolvla_attested.py" \
  --dataset-root "$DATASET" \
  --repo-id local/firefly_y6_manual_validated_v2 \
  --controller-contract "$CONTRACT" \
  --expected-task 'Pick up the red block and place it in the blue box.' \
  --output-dir "$TRAIN_OUTPUT" \
  --smolvla-base "$LOCAL_BASE" \
  -- \
  --config_path="$REPO/configs/train_smolvla_firefly_y6.example.yaml" \
  --policy.device=cuda \
  --policy.use_amp=false \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.train_state_proj=true
```

`--smolvla-base` 由 wrapper 验证后强制注入 `policy.path`、`input_features=null`、
`output_features=null`、`load_vlm_weights=false`、`use_peft=false` 和本地
`vlm_model_name`。
`--output-dir`、所有 `--dataset.*`、`--save_checkpoint*`、`--wandb.*`、
`--policy.push_to_hub`、`--policy.pretrained_path`、`--peft`、`--job.target`、`--rename_map` 和
`--resume` 也都由 wrapper 控制，不要在 `--` 后重复传入。RTX 12 GB 先保持模板的
batch 1、30,000 steps、每 5,000 steps 保存，并冻结视觉编码器、只训练 action expert。
纯训练不需要 supervisor、launcher、`arm_teleop`、Zenoh 或相机。

## 安全和常见故障

- 采集插件绝不应调用 acquire_session、set_mode 或 Zenoh put；自动测试覆盖了这一只读约束。
- stale follower state 表示 joint_state 停更；不要通过增大 max_state_age_s 掩盖 controller/launcher 故障。
- stale follower command 表示 arm_teleop 没有持续发布 arm1 或 ee1 命令。
- expected 6 values 表示模型 DOF 或前缀选错，需要同步修改 arm_joint_names 和真实 protobuf 长度。
- camera not found 通常是配置中的序列号不对，或 pyrealsense2 没安装在 lerobot-io 这个 Python 环境。
- 第一批数据先不要启用 depth 和 streaming encoding；基础链路稳定后再逐项开启。

## Firefly Y6 真机部署

`hex_zenoh_follower` 默认是 `shadow`，此模式只订阅 arm1、ee1 和相机、查询 controller description，不申请 session、不切 ACTIVE、不发布命令。`execute` 由分阶段脚本临时选择，不能通过示例 YAML 意外启用。

执行插件从实时 `ArmDescription` / `EeDescription` 验证关节顺序、位置范围、速度范围、默认增益和 timeout 能力。它不会用 URDF 数值替代 controller 自描述，也不会裁剪不安全的 action。30 Hz 命令使用单点 `JointTrajectory`，并显式发送 `dq=0`、`tau_ff=0` 和 `t_from_start_ns=33,333,333`。默认 RAMP_STOP profile 使用 arm `kp=40`、`kd=2`；本机 controller 不广告 RAMP_STOP，必须改用独立的、持久且指纹绑定的 HOLD profile：arm `kp=20`、EE `kp=75`，arm/EE 单步门分别收紧为 0.005/0.002 rad，跟踪误差门收紧为 0.05/0.02 rad。

execute 取控有以下原子性与退出保证：

- arm1、ee1 分别申请独占 session；EE 失败会立即释放 arm。
- 每个 session 绑定唯一 Zenoh liveliness token，并等待 `/status` 同时满足 `RUNNING` 和 holder 等于自己的 session ID。
- 任一已有 holder（包括 `arm_teleop`）、holder 丢失、liveliness 消失、RPC/发布失败、过期状态或安全检查失败都会停止新命令并释放双 session。
- state 年龄上限 100 ms，command send gap 上限 250 ms。RAMP_STOP profile 的 arm/EE 单步上限为 0.035/0.012 rad；本机 HOLD profile 收紧为 0.005/0.002 rad，并都进一步受 description `vel_max / 30` 限制。
- RAMP_STOP profile 的跟踪误差门为 arm 0.15 rad、EE 0.05 rad；本机 HOLD profile 收紧为 0.05/0.02 rad，连续三帧超限即停止。action 缺键、NaN/Inf、description 越界和过大单步同样 fail-closed。

`configs/deploy_firefly_y6.local.yaml`、旧 `firefly_y6_manual` 数据集以及所有旧
ACT/SmolVLA checkpoint 现全部隔离。不得把它们传给 `scripts/run_firefly_y6.py` 的任何
阶段，包括 shadow；也不得对它们 `--mark-reviewed`、复制 acceptance 或借用旧输出。
shadow 虽不发布机械臂命令，仍会加载模型/数据并参与验收，因此不是旧输入的例外。

截至 2026-08-13，正式 `firefly_y6_manual_validated_v2` 和 attested SmolVLA checkpoint
尚不存在，所以当前没有可运行或可审核的正式部署配置，也不能执行本节后面的 shadow、
hold、replay 或策略命令。待本轮 20 条正式数据完成固定 0.0435/0.0165 合同审计、并且 attested 训练
成功后，再从 HOLD 模板新建配置：

    cp configs/deploy_firefly_y6_hold.example.yaml \
      configs/deploy_firefly_y6_hold_validated_v2.local.yaml

新配置必须填写真实相机序列号；`checkpoint` 必须指向正式训练输出中的明确编号
`pretrained_model` 目录（例如 `checkpoints/030000/pretrained_model`），禁止可变的
`checkpoints/last`；该目录必须含有效 `firefly_dataset_attestation.json`。
`smolvla_vlm_snapshot` 必须是
`/home/kk/kk_data/lerobot/assets/firefly/smolvlm2-500m-video-instruct-7b375e1b73b11138ff12fe22c8f2822d8fe03467`；
`dataset_root` 必须是
`/home/kk/kk_data/lerobot/datasets/local/firefly_y6_manual_validated_v2`；
`dataset_controller_contract` 必须是
`outputs/firefly_diagnostics/controller_contract_validated_v2_arm0435_ee0165.json`；并使用全新的
`output_dir/acceptance.json`。首次推理固定每 chunk 10 步，约 0.33 秒重规划。

执行脚本必须使用同时包含 CUDA PyTorch、Zenoh 和 RealSense 的环境；本机已验证的是 `lerobot-io`（`lerobot-train` 当前缺 Zenoh/RealSense）：

    PY=/home/kk/kk_data/lerobot/miniforge3/envs/lerobot-io/bin/python
    CFG=configs/deploy_firefly_y6_hold_validated_v2.local.yaml

验收顺序被配置中 `output_dir/acceptance.json` 强制锁定。审核门要求该阶段
零 `safety_rejection`/`safety_stop`、至少一个成功 cycle，并把批准记录绑定到 JSONL
和本轮视频的 SHA-256。每轮还会计算部署指纹，绑定 checkpoint 目录内全部文件内容、
任务文本、数据集全部 metadata/parquet、固定 0.0435/0.0165 controller contract 和
robot/control 配置。计算指纹前会重新执行完整数据集审计，并核对 checkpoint 的
`train_config.json` 具有精确的绝对 dataset repo/root、SmolVLA 类型、空 `rename_map`、
固定本地 SmolVLM 路径；同时验证 `firefly_dataset_attestation.json` 与当前
meta/data/videos/contract manifest 完全一致。任一路径祖先或受控目录内出现符号链接
都会 fail-closed。模型或 replay 数据完成离线加载后，脚本会
在删除旧 telemetry 或连接硬件前再次计算内容指纹和 inode/mtime 稳定性摘要；加载期间
有替换就直接拒绝。换模型、数据、contract 或配置后不能继承旧审核，必须从 shadow
重新开始。旧格式 marker、被改写的日志和旧视频都不能解锁下一阶段。
每一级正常完成后仍须人工检查日志和两路 MP4，再用 `--mark-reviewed` 批准；这个
命令不连接硬件。下列流程只适用于上述新正式配置；隔离的旧输入不得进入第一步 shadow。

1. Shadow。保持 supervisor/launcher 运行；`arm_teleop` 可以继续运行。它持续记录 observation、10 步预测 action、推理延迟和安全检查，按 Ctrl+C 正常结束：

       $PY scripts/run_firefly_y6.py --config "$CFG" --stage shadow
       # 检查配置 output_dir 下的 shadow.jsonl 和 shadow_*.mp4
       # 仅正式 attested v2 checkpoint 且 shadow 全部零 safety_rejection 时：
       $PY scripts/run_firefly_y6.py --config "$CFG" --stage shadow --mark-reviewed

2. 当前位置保持也必须等新的 v2 checkpoint 完成“零拒绝 shadow → 人工审核”后才会
被 ledger 解锁。解锁后先用遥操作把 follower
放在安全姿态，再停止 `arm_teleop`，确认实体总开关和下落保护就绪。HOLD profile
的首次取控只允许最长 1 秒，并且每次真机 execute 都要重新输入两个临时确认：

       $PY scripts/run_firefly_y6.py --config "$CFG" --stage hold \
         --hold-duration-s 1 \
         --allow-arm-hold-timeout-without-ramp-stop \
         --independent-estop-verified
       $PY scripts/run_firefly_y6.py --config "$CFG" --stage hold --mark-reviewed

HOLD 在命令流、主机或进程中断时可能继续以最后目标和增益上力，直到 controller
回收 session；软件 watchdog 对整机掉线无效，它不等价于 RAMP_STOP。HOLD 双字段和
低增益/低步长值存于 YAML 并进入部署指纹；CLI flag 只作本次人工风险确认，不会在
计算指纹后更改 robot 配置。

3. Episode 0 的 0.5 倍速回放。只有 HOLD 的 1 秒视频和 telemetry 人工审核通过后才会
解锁。先用遥操作把 follower 放到 episode 0 首帧附近，再停止 `arm_teleop`。脚本
会在取控前加载数据，取控后用实时 description 预检完整插值轨迹；arm 首帧误差超过
0.05 rad 或 EE 超过 0.02 会拒绝启动：

       $PY scripts/run_firefly_y6.py --config "$CFG" --stage replay \
         --allow-arm-hold-timeout-without-ramp-stop \
         --independent-estop-verified
       $PY scripts/run_firefly_y6.py --config "$CFG" --stage replay --mark-reviewed

4. SmolVLA 策略闭环（CLI 为兼容仍使用 `--stage act`）必须同时使用审计通过的完整正式
v2 dataset、由 attested wrapper 从固定本地 base 训练的新 checkpoint 和全新
output/acceptance 目录。旧 local、旧数据、旧 ACT/SmolVLA、smoke 数据以及未 attested
checkpoint 均不得运行下列命令。条件满足
后，后台线程做推理，前台严格 30 Hz 发送；下一 chunk 未在前 10 步完成前准备好时，
旧 action 不会复用，插件会停止发布并释放 session。逐级运行、查看、批准 3、10、
30 秒，每次执行都重新确认 HOLD 风险和独立断电手段：

       $PY scripts/run_firefly_y6.py --config "$CFG" --stage act \
         --act-duration-s 3 \
         --allow-arm-hold-timeout-without-ramp-stop \
         --independent-estop-verified
       $PY scripts/run_firefly_y6.py --config "$CFG" --stage act \
         --act-duration-s 3 --mark-reviewed

       $PY scripts/run_firefly_y6.py --config "$CFG" --stage act \
         --act-duration-s 10 \
         --allow-arm-hold-timeout-without-ramp-stop \
         --independent-estop-verified
       $PY scripts/run_firefly_y6.py --config "$CFG" --stage act \
         --act-duration-s 10 --mark-reviewed

       $PY scripts/run_firefly_y6.py --config "$CFG" --stage act \
         --act-duration-s 30 \
         --allow-arm-hold-timeout-without-ramp-stop \
         --independent-estop-verified
       $PY scripts/run_firefly_y6.py --config "$CFG" --stage act \
         --act-duration-s 30 --mark-reviewed

`--independent-estop-verified` 和
`--allow-arm-hold-timeout-without-ramp-stop` 都是本机 HOLD profile 每次 execute
必须显式给出的临时人工确认，不写入配置。前者只适用于已经单独验证过、独立于本
软件进程的物理断电或急停手段；`pc-always-clear.yaml` 只模拟“急停未触发”，不满足
该条件。没有这样的硬件手段时只能完成 shadow，不能运行 hold、replay 或策略。

### 只读限位反事实诊断

只能分析已存在的 JSONL；不得为了生成新 telemetry 而重新加载或运行隔离的旧 local/旧
模型。如果需要判断既有 shadow rejection 是否仅由 j2/j3 的几毫弧度边界差异造成，
可以对已有 JSONL 做纯离线重放。该工具不导入硬件模块、不连接 Zenoh，也不会修改
follower 或 controller：

    $PY scripts/analyze_shadow_envelope.py \
      --telemetry outputs/firefly_deployment/shadow.jsonl \
      --controller-contract outputs/firefly_diagnostics/controller_contract_20260812.json \
      --joint-2-min=-1.5900000524520874 \
      --joint-3-max=3.160000104904175 \
      --output outputs/firefly_diagnostics/act_shadow_expand_20mrad.json

诊断扩展最多 0.020 rad，并只允许 j2 下限、j3 上限向外扩。输出始终带有：

- `hypothetical_only=true`
- `eligible_for_execution=false`
- `eligible_for_acceptance=false`

它同时模拟 follower 的 fail-closed 语义：未通过的目标不会成为下一步 reference。因此即使假想位置范围通过，首步相对实测姿态过大仍会继续拒绝。诊断结果不得用于解锁 hold、replay、ACT，也不会改变 execute 始终使用实时 controller description 的硬门。
