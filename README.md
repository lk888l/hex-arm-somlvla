# LeRobot + SmolVLA 本机环境

2026-09-09 工程复查：[红块抓取诊断、优化顺序与旧数据清理建议](docs/AUDIT_AND_OPTIMIZATION_20260909.md)。
当前已有 fullcycle_v4 数据及 100k/arm6 20k 模型；最新两份部署配置为
`open_only` 张爪验证，张爪完成后结束，尚未接通完整抓取流程。
历史文档中的状态请按其日期阅读，当前文件情况见上述报告。

安装日期：2026-07-29

## 已固定版本

- LeRobot 0.6.0
- Python 3.12
- PyTorch 2.10.0 + CUDA 12.8
- torchvision 0.25.0
- TorchCodec 0.10.0
- FFmpeg 7.1.1

TorchCodec 0.10 与 PyTorch 2.10 配套。不要单独升级 TorchCodec；0.11
需要 PyTorch 2.11。

## 操作文档

按顺序使用以下文档：

1. [启动与关闭 Firefly Y6 采集链路](docs/STARTUP.md)
2. [录制、分段、保存位置与故障排查](docs/RECORDING.md)
3. [检查数据、训练 ACT/SmolVLA 与换电脑迁移](docs/TRAINING.md)

采集插件的协议映射与实现说明见
[lerobot-robot-hexbase/README.md](lerobot-robot-hexbase/README.md)。

## Git 管理与本地归档（2026-11-02）

根目录是整套工程的版本入口，保存环境定义、启动脚本、工具、文档、基础资产的
配置与来源凭证。`lerobot-robot-hexbase/` 作为正式 Git submodule 保存插件的
独立历史，父仓库通过 `.gitmodules` 和 gitlink 固定它的提交。

插件 Git 元数据统一存放在 `.git/modules/lerobot-robot-hexbase/`；子目录内的
`.git` 是指向该位置的文本文件，属于正常子模块布局。日常分别查看两层状态：

```bash
git status
git submodule status
git -C lerobot-robot-hexbase status
```

本次本地归档使用分支 `archive/git-layout-20261102`。插件的原 `master` 和远端
配置保留；发布时先推送插件归档提交，再推送父仓库，并先检查远端已有历史。
本地整理没有验证远端是否已包含这些提交。

插件中的 fullcycle-v4 录制配置、follower-only launcher、100k/arm6 两份部署
配置，以及它们依赖的 controller contract、schema-5/schema-6 审计 JSON 已按
原路径纳入版本管理。这些配置保存的是当前机器实值，迁移时需要核对设备和路径。
其他 `*.local.yaml`、`.orig` 和历史诊断文件保存在本地备份中。

数据集、训练输出、模型大权重、Miniforge 环境和下载缓存由 `.gitignore` 排除，
需要单独备份。基础 bundle 的文本配置、tokenizer、来源 lock 和两份 640 字节
归一化参数保留在 Git；`datasets/calibration/` 中的标定文件也允许纳入版本管理。
本次整理没有补全数据/模型备份，也不表示已完成新机复现验收。

整理前的根仓库元数据和小文件快照保存在
`.local-archive/git-layout-20261102-120215/`，其中 `README.txt` 说明备份范围，
`worktree-before.sha256.json` 记录校验和。该目录被 Git 忽略，包含旧临时 pack，
需独立保管。

## 激活环境

采集、RealSense、Zenoh 插件和真机链路检查：

```bash
source /home/kk/kk_data/lerobot/activate-io.sh
```

训练：

```bash
source /home/kk/kk_data/lerobot/activate-train.sh
```

激活只对当前终端有效，不会修改 `.bashrc`。每个新终端都要重新 `source`。
直接调用环境内二进制的绝对路径不会自动加载 `env.sh`，因此录制前仍应先激活
`lerobot-io`。当前 Firefly smoke/正式配置还显式固定了 `dataset.root`，避免数据位置
随 shell 环境漂移；旧配置若未写 root，仍可能落入 `/home/kk/.cache/huggingface/lerobot`。

## 验证

```bash
source /home/kk/kk_data/lerobot/activate-train.sh
python /home/kk/kk_data/lerobot/tools/diagnose.py
python /home/kk/kk_data/lerobot/tools/smoke_smolvla.py
```

## 数据与隐私

- 激活环境后的默认数据集目录：`/home/kk/kk_data/lerobot/datasets`
- 早期未激活环境时的数据集：`/home/kk/.cache/huggingface/lerobot/local`
- 模型缓存：`/home/kk/kk_data/lerobot/hf_cache`
- 输出：`/home/kk/kk_data/lerobot/outputs`
- 日志：`/home/kk/kk_data/lerobot/logs`
- W&B 默认 `offline`，不会上传训练指标。
- 当前没有 Hugging Face 登录信息，也不会自动上传数据或模型。

后续需要 Hub 时，先激活环境，再手动执行 `hf auth login`。确认需要联网记录后，
可执行 `export WANDB_MODE=online` 并运行 `wandb login`。

## 12GB 显存起始策略

推荐先用 ACT 建立单任务基线；若 CUDA OOM，将 batch size 从 4 依次降到 2、1。
SmolVLA 从 batch size 1 开始，并保持冻结视觉编码器、只训练 action expert。
当前 LeRobot 0.6 训练 CLI 没有梯度累积参数，环境变量不会自动实现梯度累积。
完整命令见 [训练文档](docs/TRAINING.md)。

## 当前硬件链路

- 双 Firefly Y6 + GR80：`arm0/ee0` 为 leader，`arm1/ee1` 为 follower。
- `hex-hw-supervisor` + launcher/controller 负责 CAN 和安全状态机。
- `arm_teleop` 负责 leader-follower 控制。
- LeRobot 通过 Zenoh 只读采集 follower 实测状态和 follower 目标命令。
- D435 为 scene，D405 为 wrist；D405 在本机配置中软件旋转 180°。
- 已验证 `pyrealsense2 2.58.3.10794` 可同时读取两台相机。

这套自定义硬件不使用 LeRobot 通用串口电机初始化流程，不要运行：

```text
lerobot-setup-motors
lerobot-calibrate
```

尤其是 `lerobot-setup-motors` 会写电机持久配置。录制必须遵循启动文档，且不使用
`sudo`。当前 supervisor 的 `pc-always-clear` profile 不是实际急停保护；上电前必须
准备独立、可触及并已验证的断电或急停手段。
