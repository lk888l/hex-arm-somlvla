# 配置目录说明

本目录保存工作区的环境定义、版本清单和配置模板；Firefly 的录制、训练、部署 YAML
位于子模块的 `lerobot-robot-hexbase/configs/`。安装顺序见
[安装与环境恢复](../docs/INSTALL.md)，文件的 Git 保存范围见
[Git 使用与归档范围](../docs/GIT_WORKFLOW.md)。

## 文件用途与生效方式

| 文件 | 用途 | 如何生效 |
| --- | --- | --- |
| `lerobot-io.conda.yml` | 采集环境的最小定义：Python 3.12、FFmpeg 7.1.1、pip | 手动用 Conda 创建；不包含完整 Python 依赖 |
| `lerobot-train.conda.yml` | 训练环境的最小定义 | 同上；两个 YAML 都保留本机绝对 `prefix`，换路径时需显式覆盖 |
| `locks/lerobot-io-linux-64.explicit.txt` | 采集环境的精确 Conda 包 URL、构建和 SHA256 | `conda create --prefix ... --file ...`；只适用 Linux x86_64 |
| `locks/lerobot-train-linux-64.explicit.txt` | 训练环境的精确 Conda 包记录 | 同上；当前两份 Conda 锁相同，pip 包清单不同 |
| `lerobot-io.requirements.txt` | I/O 环境的已安装非 editable Python 包版本，含 Zenoh、RealSense 和测试依赖 | 先安装指定 cu128 的 PyTorch/torchvision，再 `python -m pip install -r ...`；最后单独安装本地插件 |
| `lerobot-train.requirements.txt` | 训练环境的 Python 包版本 | 按安装文档恢复；训练环境不安装 HexBase 插件 |
| `environment-snapshot-20261102.json` | 系统/GPU、安装源、插件提交、解释器与清单哈希、采集时的 `pip check` 结果 | 作为核对凭证；不是安装脚本，记录的是采集时的环境 |
| `udev/99-realsense-d405-d435.rules` | D435、D405 及 D400 恢复模式的 USB 权限规则 | 管理员安装到 `/etc/udev/rules.d/`、重载规则并重新插拔相机，见安装文档 |
| `hardware.env.example` | 通用设备信息填写模板 | 按需复制成 `hardware.env`，该副本被 Git 忽略；目前没有脚本自动读取它 |
| `smolvla-12gb.env` | 12GB 显存下的 SmolVLA 起始参数参考 | 目前没有脚本自动读取；需按 [训练文档](../docs/TRAINING.md) 转成实际 CLI/YAML 参数 |

Conda explicit 文件固定 Conda 包及其哈希；requirements 只固定 Python 包版本，
没有锁定 wheel/sdist 文件哈希。两者都需要保存，单独使用 `.conda.yml` 不能恢复
完整环境。插件的 editable 源码版本由父仓库的 submodule 指针固定。

## 终端环境变量

根目录的 `activate-io.sh` / `activate-train.sh` 先加载 `env.sh`，再激活对应 Conda
环境，必须用 `source` 执行。`env.sh` 的当前配置如下：

| 变量 | 当前值/含义 |
| --- | --- |
| `LEROBOT_WORKSPACE` | `/home/kk/kk_data/lerobot` |
| `LEROBOT_CONDA_BASE` | 工作区下的 `miniforge3` |
| `HF_HOME` / `HF_HUB_CACHE` | 工作区下的 `hf_cache` / `hf_cache/hub` |
| `HF_LEROBOT_HOME` | 工作区下的 `datasets`；显式填写的 YAML `dataset.root` 优先决定数据路径 |
| `PIP_CACHE_DIR` | 工作区下的 `pip_cache` |
| `WANDB_DIR` | 工作区下的 `logs/wandb` |
| `WANDB_MODE` | 默认 `offline`，保留终端中已设置的值 |
| `HF_HUB_DISABLE_TELEMETRY` | `1`，关闭 Hub 遥测 |
| `TOKENIZERS_PARALLELISM` | `false` |

直接调用环境内 Python 的绝对路径不会加载这些变量。W&B 离线和关闭遥测也不等于
禁止 Hub 下载；两份 `tools/smoke_*smolvla.py` 都会调用 `snapshot_download`。

`hardware.env.example` 中的 `ROBOT_*`、`TELEOP_*`、`CAMERA_*` 是记录模板。
当前 Firefly 链路通过 Zenoh/CAN 接入，相机使用 YAML 内的 RealSense 序列号；
填写或 `source hardware.env` 都不会自动替换录制 YAML 的值。
`smolvla-12gb.env` 中的 `SMOLVLA_*` 同样需要人工映射到训练参数；LeRobot 0.6
没有梯度累积 CLI 参数。

## 子模块里的机器配置

以下四份 `*.local.yaml` 已作为现机输入原样归档，属于子模块忽略规则的明确例外。
下表路径均相对 `lerobot-robot-hexbase/`：

| 文件 | 用途与依赖 |
| --- | --- |
| `configs/record_firefly_y6_fullcycle_v4.local.yaml` | fullcycle-v4 录制输入；由 `scripts/record_firefly_y6_fullcycle_v4.py` 校验并启动，固定任务、数据路径与合同 |
| `configs/launch_firefly_y6_follower_only.local.yaml` | 只启动 `arm1/ee1` 的 launcher 配置；依赖工作区外的 controller、CAN 和 URDF |
| `configs/deploy_firefly_y6_hold_fullcycle_v4_100k.local.yaml` | 100k 模型的部署配置，依赖原始 v4 数据和 schema-5 视觉审计 |
| `configs/deploy_firefly_y6_hold_fullcycle_v4_arm6_20k.local.yaml` | arm6 run2 20k 模型配置，依赖 arm-only 数据、原始 v4 数据和 schema-6 视觉审计 |

录制及两份部署配置引用 `outputs/firefly_diagnostics/` 下的
`controller_contract_validated_v2_arm0435_ee0165.json`；两份视觉审计分别为
`firefly_y6_tip_servo_open_only_fullcycle_v4_schema5_exact.json` 和
`firefly_y6_tip_servo_open_only_fullcycle_v4_schema6_exact.json`，三份 JSON 也在子模块
Git 中。数据和 checkpoint 仍需独立恢复。两份部署 YAML 的默认 `robot.mode` 为
`shadow`，当前任务阶段是接近后张爪的 `open_only`，完整夹取搬运尚未接通。

换设备时需核对 controller ID、Zenoh endpoint、CAN、相机序列号与腕部旋转方向、
URDF、数据/模型路径及合同/审计哈希。旧 arm6 wrapper 还绑定训练 Python 的路径和
可执行文件 SHA256。修改绝对路径或重算文件哈希需要同步验证其引用和来源凭证，
不能据此直接认定旧实验已在新机复现。当前模板不能覆盖替代现机配置。

## 更新环境清单

在选定的环境中先执行 `python -m pip check`，再把导出结果写到临时文件供比对。
例如采集环境：

```bash
source /home/kk/kk_data/lerobot/activate-io.sh
conda list --explicit --sha256 > /tmp/lerobot-io-linux-64.explicit.txt
python -m pip freeze --all --exclude-editable > /tmp/lerobot-io.requirements.txt
```

训练环境按相同方式单独导出。比对后更新对应锁和 requirements，保留清单头部的
安装源说明；editable 插件继续按仓库路径单独安装。检查导出中的本机路径与私有源
凭据，并生成新的日期快照，记录系统、驱动、插件提交、解释器/清单/udev 哈希及
验证结果。历史日期快照保留其原始记录。
