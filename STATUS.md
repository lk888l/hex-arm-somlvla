# 安装与联调状态

最近离线复查：2026-09-09

- 当前存在 50 条、91,836 帧的 fullcycle_v4 数据及其 arm_only 派生集。
- 当前存在 v4 100k 模型，以及 arm_only run2 的 10k/20k 模型。
- 最新部署入口为 open_only：接近后张爪、保持终止，尚未接通完整夹取搬运。
- 保存的最近 v5 真机日志因 seeking_timeout 停止，未找到最新 arm6 配置对应的真机输出。
- 本次全量数值/起始窗口审计通过，相关离线测试 20 项通过；未验证新环境真机成功率。
- 已清理约 1.51 GiB 可再生缓存，录制、训练数据、模型与源码保留。

详情见 [2026-09-09 诊断报告](docs/AUDIT_AND_OPTIMIZATION_20260909.md)。

以下为 **2026-08-13 历史记录**，其中“尚未完成/不存在”不代表当前文件状态。

> 旧 `firefly_y6_manual_20260811_172346` 及其 ACT/SmolVLA checkpoint 已判定不安全，
> 禁止继续训练或真机部署。当前操作入口以
> `lerobot-robot-hexbase/README.md` 的 validated_v2 流程为准。

## 软件环境

- Miniforge 26.3.2-3。
- `lerobot-io` 与 `lerobot-train` 均为 LeRobot 0.6.0、Python 3.12、
  PyTorch 2.10.0+cu128、torchvision 0.25.0、TorchCodec 0.10.0、FFmpeg
  7.1.1。
- RTX 5070 Ti Laptop 的 CUDA 与 BF16 已在本机验证。
- `lerobot/smolvla_base` 和 `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`
  已缓存到工作区 `hf_cache`。
- W&B 默认关闭/离线，数据集和模型默认不上传 Hub。

## Firefly Y6 采集链路

- `lerobot-robot-hexbase` 已增加只读 `hex_zenoh_observer` 和
  `hex_zenoh_command_tap`。
- follower `arm1/ee1` 的 7 维实测 q 写入 `observation.state`。
- `arm_teleop` 已发布的 follower 7 维目标 q 写入 `action`。
- 插件不获取 controller session，不向机械臂发送控制命令。
- launcher 的 Zenoh endpoint 为 `tcp/127.0.0.1:7447`。
- D435 scene：`030522071867`。
- D405 wrist：`323622270296`，录制配置软件旋转 180°。
- `pyrealsense2 2.58.3.10794` 双相机端到端 60 帧 smoke test 已通过。
- 探针、配置反序列化、LeRobot `record_loop` 与只读约束测试已通过。

## 当前数据集

已核验目录：

```text
/home/kk/.cache/huggingface/lerobot/local/firefly_y6_manual_20260811_172346
```

- 50 个 episode，索引 0–49。
- 41,991 帧，总时长约 23 分 20 秒。
- state/action 均为 7 维，无 NaN、Inf 或空 episode。
- scene/wrist 均为 41,991 帧、640×480、30 FPS。
- 每路两个 MP4 是正常视频分片，不代表只有两条轨迹。
- LeRobot 数据加载器已成功读取首、中、末样本。

该数据在文件结构上完整，但 action 不安全：14,002/41,991 帧超出 controller 原始
位置范围，1,454 帧相邻 action 超出当前 30 Hz 步长合同，且任务文字错误。修改 task、
裁剪或复制数据都不能修复标签；仅保留作离线诊断，不得继续训练或真机部署。

## 尚未完成

- 尚未完成新的 `firefly_y6_manual_validated_v2_smoke` 真机录制与离线审计。
- 新 validated_v2 全量数据和从头训练的 SmolVLA checkpoint 尚不存在。
- 尚未建立固定成功标准并比较多个 checkpoint 的真实任务成功率。
- `hex_zenoh_follower` 的 shadow/execute、session、限位、跟踪和验收门已实现；但当前
  controller v0.3.7 不广告 arm `RAMP_STOP`，因此 replay/策略闭环仍被硬门禁止。
- `pc-always-clear.yaml` 不是实际急停保护；正式策略测试前必须补齐独立安全链路。

## 文档入口

1. [启动文档](docs/STARTUP.md)
2. [录制文档](docs/RECORDING.md)
3. [训练文档](docs/TRAINING.md)

## 常用软件验证

```bash
source /home/kk/kk_data/lerobot/activate-train.sh
python /home/kk/kk_data/lerobot/tools/diagnose.py
python /home/kk/kk_data/lerobot/tools/smoke_smolvla.py
python /home/kk/kk_data/lerobot/tools/smoke_train_smolvla.py
```
