# 安装与环境恢复

本文对应 2026-11-02 的本机环境清单。记录见
[`environment-snapshot-20261102.json`](../configs/environment-snapshot-20261102.json)，
各文件的用途见 [配置目录说明](../configs/README.md)。锁文件与清单已从现机导出，
本指南尚未在空白电脑完成安装验收。

## 1. 准备代码、系统与安装路径

快照环境为 Ubuntu 24.04.5 LTS、Linux x86_64、RTX 5070 Ti Laptop GPU，驱动为
595.99.02。Conda explicit 锁的目标平台是 `linux-64`，不适用于 macOS、Windows
或 ARM64。更换 GPU/驱动组合需要重新验证 CUDA 与 BF16。

先按 [Git 使用文档](GIT_WORKFLOW.md) 获取所选工作区版本及其固定的子模块提交。
确认该版本包含 `configs/locks/`、更新后的 requirements、环境快照和 udev 规则；
最初的归档提交 `e21547a` 尚不包含这些后续补充。

以下命令在仓库根目录执行，假定 Miniforge 已安装到工作区的 `miniforge3/`，
当前路径为 `/home/kk/kk_data/lerobot`。环境目录本身被 Git 忽略，需要另行安装。

系统依赖可按快照准备：

```bash
sudo apt update
sudo apt install git curl ca-certificates build-essential pkg-config \
  libgl1 libglib2.0-0t64 libusb-1.0-0 usbutils iproute2 can-utils
```

上述命令不安装 NVIDIA 驱动或 Miniforge。先确认 `nvidia-smi` 可用。
纯离线训练不需要启动 supervisor、controller、遥操作或相机。

现有 `env.sh`、activate 脚本、部分 YAML、资产配置与训练 wrapper 固定了本机绝对
路径。恢复原实验时优先保持原路径；使用新路径时，除下文变量外还需逐项调整这些
引用，并重新验证来源凭证。仅修改环境变量不能迁移全部路径。

## 2. 用 Conda 锁创建两套新环境

以下步骤用于目标环境尚不存在时；已有环境先核对版本，不要覆盖现机环境。

```bash
export LEROBOT_RESTORE_ROOT=/home/kk/kk_data/lerobot
cd "$LEROBOT_RESTORE_ROOT"
source "$LEROBOT_RESTORE_ROOT/miniforge3/etc/profile.d/conda.sh"

conda create --prefix "$LEROBOT_RESTORE_ROOT/miniforge3/envs/lerobot-io" \
  --no-default-packages --file configs/locks/lerobot-io-linux-64.explicit.txt
conda create --prefix "$LEROBOT_RESTORE_ROOT/miniforge3/envs/lerobot-train" \
  --no-default-packages --file configs/locks/lerobot-train-linux-64.explicit.txt
```

锁文件固定 Python 3.12.13（`hd63d673_0_cpython`）、FFmpeg 7.1.1
（`gpl_hbbdf940_911`）等 Conda 包的 URL、构建与 SHA256，不含后续 pip 安装的包。
`configs/*.conda.yml` 只保存最小依赖定义，并保留原机器的绝对 `prefix`；
精确恢复应优先使用 explicit 锁。

## 3. 安装 CUDA wheel 与 Python 依赖

先从快照记录的官方 cu128 源安装 PyTorch 和 torchvision，再按对应清单安装其他
包。两套环境分别安装，不能用一个环境的 requirements 替代另一个。

```bash
for lerobot_restore_env in lerobot-io lerobot-train; do
  conda activate "$LEROBOT_RESTORE_ROOT/miniforge3/envs/$lerobot_restore_env" || break
  python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
    'torch==2.10.0+cu128' 'torchvision==0.25.0+cu128' || break
  python -m pip install --index-url https://pypi.org/simple \
    -r "$LEROBOT_RESTORE_ROOT/configs/$lerobot_restore_env.requirements.txt" || break
  python -m pip check || break
done
```

每一步均应成功后再继续；安装失败时循环停止，先处理报错，不要通过放宽固定版本
跳过。requirements 是已安装版本清单，没有 pip 制品哈希，不等于完整的离线安装包。
联网恢复还依赖原包源可用；离线恢复需要另行准备对应 Conda 包和 Python wheel。

只在 I/O 环境注册固定提交的本地插件：

```bash
conda activate "$LEROBOT_RESTORE_ROOT/miniforge3/envs/lerobot-io"
python -m pip install --no-deps \
  -e "$LEROBOT_RESTORE_ROOT/lerobot-robot-hexbase[realsense,dev]"
python -m pip check
python -c "import lerobot_robot_hexbase, pyrealsense2, zenoh; print('I/O imports OK')"
```

`--no-deps` 的前提是已完整安装 I/O requirements；其中包含插件及其 extras 的
依赖。插件源码由 submodule 提供，不能把旧机器的 `file:///home/kk/...` editable
路径作为下载地址。训练环境按当前 wrapper 的设计保持独立，不安装 HexBase。

## 4. 配置 RealSense USB 权限

需要采集时，在安装系统规则的步骤使用 `sudo`；Python、LeRobot 和相机采集由普通
用户运行。规则只授予 `plugdev` 组读写权限，覆盖 Intel USB `8086:0b07`、
`8086:0b5b` 以及 D400 恢复模式 `8086:0adb` / `8086:0adc`。

```bash
cd "$LEROBOT_RESTORE_ROOT"
sudo groupadd -f plugdev
sudo install -m 0644 configs/udev/99-realsense-d405-d435.rules \
  /etc/udev/rules.d/99-realsense-d405-d435.rules
sudo usermod -aG plugdev "$(id -un)"
sudo udevadm control --reload-rules
```

注销并重新登录，使组成员资格生效；在没有相机采集进程时重新插拔两台相机。
普通终端中检查组成员与 SDK 看到的序列号：

```bash
source /home/kk/kk_data/lerobot/activate-io.sh
id -nG
python - <<'PY'
import pyrealsense2 as rs

devices = list(rs.context().query_devices())
if not devices:
    raise SystemExit("No RealSense device found")
for device in devices:
    print(device.get_info(rs.camera_info.name), device.get_info(rs.camera_info.serial_number))
PY
```

规则没有序列号映射，也不配置 CAN/controller 权限。当前机器的 D435 scene 为
`030522071867`，D405 wrist 为 `323622270296`，wrist 软件旋转 180°；换相机后需
更新实际使用的 YAML。`hardware.env` 不会自动写入这些值。

## 5. 验证环境与路径

```bash
cd /home/kk/kk_data/lerobot
source ./activate-io.sh
python -m pip check
python tools/diagnose.py

source ./activate-train.sh
python -m pip check
python tools/diagnose.py
```

核对关键版本、FFmpeg、`cuda_available=True`、BF16 支持和 `HF_LEROBOT_HOME` 等
路径。预期为 LeRobot 0.6.0、PyTorch 2.10.0+cu128、torchvision 0.25.0+cu128、
TorchCodec 0.10.0。`diagnose.py` 打印诊断结果，不能只凭退出码判定 GPU 验收通过。

若要恢复旧 arm6 run2 的凭证，还需核对训练 Python：

```bash
sha256sum miniforge3/envs/lerobot-train/bin/python
```

旧 wrapper 要求的值是
`aa584ad7b853904b0ae7fb5a8f44d9d27960e05de1f4838855f93fa4359df869`。
版本相同不保证二进制哈希相同；不匹配时应调查包构建和路径，或为新运行时建立
新版本及凭证，不要修改旧校验来冒充原环境。

`tools/smoke_smolvla.py` 和 `tools/smoke_train_smolvla.py` 会请求未固定 revision 的
Hub snapshot，适合一般环境检查。严格复现需按固定 revision 恢复本地 bundle，
并使用插件内相应的来源验证流程。

## 6. 恢复运行输入

Git 保存了源码、小型配置、tokenizer、来源 lock 和两份归一化参数；数据集、训练
checkpoint、大权重、下载缓存及工作区外的硬件程序需要独立恢复。保存范围及
备份位置见 [Git 使用文档](GIT_WORKFLOW.md)，具体有效数据/模型链路见
[复现审查的恢复集清单](VERSION_CONTROL_AND_REPRODUCTION_20260910.md#5-数据与模型恢复集)。

训练/部署恢复时要保留数据完整目录、checkpoint 的 `pretrained_model/` 与
`training_state/`、`last` 链接及来源凭证，并核对配置引用的合同、审计和哈希。
只安装环境尚不能恢复完整实验。完成这些核对后，再按
[启动文档](STARTUP.md)、[录制文档](RECORDING.md) 或 [训练文档](TRAINING.md)
继续相应流程；当前部署范围仍是 `open_only`。
