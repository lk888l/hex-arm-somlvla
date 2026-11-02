# 红块抓取诊断与清理建议（2026-09-09）

你描述的现象是：机械臂大概下探、夹爪张开，然后没有后续动作。**最新部署入口只执行“接近后张爪验证”；张爪完成就是终点，尚未接通完整抓取流程。** 历史试验还存在视觉接管条件一直不满足的问题。

本次检查了 local 配置、控制代码、训练配置、全部 91,836 帧数值数据、历史执行日志，抽看 episode 0、25、49 和 8 月 20 日试验视频。没有启动机械臂、实时相机或训练。新环境具体改动尚未明确，因此旧场景审计不能代替新环境验证，也不能确定保存的哪次试验对应你描述的现象。

## 1. 张爪后为何停止

最新两份入口：[7 维 100k 配置](../lerobot-robot-hexbase/configs/deploy_firefly_y6_hold_fullcycle_v4_100k.local.yaml)、[6 维 20k 配置](../lerobot-robot-hexbase/configs/deploy_firefly_y6_hold_fullcycle_v4_arm6_20k.local.yaml)，都设置：

```yaml
visual_phase_latch:
  tip_servo:
    mode: execute
    probe_profile: open_only
    descent_step_m: 0.0
    max_total_descent_m: 0.0
    allow_close: false
```

`tip_servo.mode` 只是该模块模式，机器人配置默认仍是 `robot.mode: shadow`，实际真机入口另有阶段选择。文件名含 `fullcycle` 表示训练来源，不表示当前执行模式完成全流程。

```text
SEEKING：SmolVLA 移动六个臂关节，夹爪保持启动位置
   ↓ 红块位置、面积、速度等连续满足接管条件
OPENING：机械臂保持接管位置，视觉模块逐步张爪
   ↓ 确认张开
PROBE_COMPLETE：终止保持，执行循环返回
```

[firefly_tip_servo.py](../lerobot-robot-hexbase/lerobot_robot_hexbase/firefly_tip_servo.py) 的 `_advance_tip_servo_enabled()` 把 OPEN_ONLY 张爪完成转到 `PROBE_COMPLETE`；[run_firefly_y6.py](../lerobot-robot-hexbase/scripts/run_firefly_y6.py) 的 `run_tip_servo_probe_exclusive()` 及其调用者随后返回。相关现有离线测试 **20 项通过**，包含张爪后不调用 IK、机械臂保持和 20 秒 SEEKING 超时。

前段看起来“下探”可能是 SmolVLA 在 SEEKING 中输出的动作，并不代表张爪后的精细下探阶段已接通。直接把 `allow_close` 改成 true 也不能补齐流程：OPEN_ONLY 的配置校验和 exclusive 执行器都会拒绝。

6 维模型还指定 `policy_runtime_role: arm6_seeking`。它的 state/action 都去掉了夹爪数值，闭爪必须由另外的执行逻辑完成。

## 2. 其他已发现的问题

| 问题 | 本机证据 | 影响 |
| --- | --- | --- |
| 缺少红块相对两爪尖的横向误差闭环 | OPEN_ONLY 接管后固定机械臂；另一 `bounded_descent` 分支也固定接管点 X/Y，只减 Z | 偏左、偏右或前后未对准时，竖直下探不能主动纠偏 |
| 视觉阈值依赖旧场景 | 粗接管 X 范围 `[0.50, 0.545]`，640 宽图像中仅约 29 像素；还要面积 `[0.017, 0.029)`、连续 6 帧及低速度 | 看得到红块也可能不接管；相机、物体、夹爪或光照变化会改变检测和门限适配 |
| 历史试验一直留在 SEEKING | 最新保存的 v5 日志中 595 个 SEEKING 周期，随后 `external:seeking_timeout`；v1–v4 也未完成探针 | 部分停住实际是等不到接管条件后退出 |
| 推理期间反复保持 | v5 每块执行 10 个动作；推理中位 138 ms、P95 144 ms；596 个周期中 200 个为 `inference_hold` | 动作节奏比演示断续，需优化时延与衔接；它不是 OPEN_ONLY 终止的原因 |
| 缺少独立评价 | 当前无留出集、`eval_steps: 0`、图像增强关闭；已有审计注明 `hardware_completion_validated: false`、`closed_loop_convergence_claimed: false` | 检测通过、模型能加载或运行满时长都不能证明抓取成功 |
| 文档落后于实际文件 | 原 STATUS 仍说新数据/模型不存在，实际已经有 v4 100k 与 arm6 20k | 容易选错入口 |

v5 停止时红块中心约 `(0.476, 0.457)`、轮廓面积约 `0.0110`，不满足当前粗接管的 X/面积范围。这个日志里的夹爪一直保持启动位置，**不能认定就是你描述的已经张爪的那一次**。最新 arm6 配置对应的输出目录尚未找到。v3/v4 各有重复归档，不能按日志文件数计算独立试验次数。

旧静态检查 `firefly_static_alignment_v1_hold_pre_v5_adjusted7_final.json` 的 `alignment_ok` 仍为 false：腕部红块包围框高度偏差 -7 px，门限 3 px。文件名 final 不能代表检查通过；该检查也只是旧场景轮廓比较，不是新环境的手眼标定。

历史 pipelined 版本出现过关节跟踪误差停机，当前 synchronous_hold 有其背景。先测观测时间戳、动作预测跨度和队列衔接，再做异步改造；不能直接恢复旧开关。官方异步推理同样需要协调观测与动作队列。[官方说明](https://huggingface.co/blog/async-robot-inference)

## 3. 旧数据到底是否有用

原始录制集 `datasets/local/firefly_y6_fullcycle_v4`：**50 条、91,836 帧、30 FPS，共 51 分 1.2 秒**。单条 43.1–83.6 秒，中位 60.1 秒。两路 RGB 为 640×480，D405 软件旋转 180°，没有深度图。

用保存的 `controller_contract_validated_v2_arm0435_ee0165.json` 重新做全量审计：结构、任务文字、数值有限性、位置范围、连续动作步长全部通过，0 错误、0 警告；起始运动窗口也通过。这表示它符合这份历史合同，不等于新环境或新硬件已通过。

50 条数值轨迹都有“张开后闭合”。抽看的 3 条视频能看到接近、抓取过程和最终红块进入蓝盒，没有人工逐帧确认全部 50 条成功。回位并非严格一致：10/50 条的起终最大单关节差超过 0.1 rad，中位 0.066 rad、最大 0.185 rad。这个统计只是回位一致性诊断，不是成功或碰撞判据；后续应统一回位容差和稳定时长。

`firefly_y6_fullcycle_v4_arm_only_v1` 是同一批数据去掉第七维夹爪后的派生集，视频独立复制且内容相同。**两套仍然只有 50 次演示，不是 100 次；夹爪在视频里仍然可见。**

7 维模型训练 100k steps、batch size 1；6 维模型在它基础上再训练 20k steps、batch size 1。不能只凭步数判断训练充分或过拟合。按起始帧粗算，100k × 1 / 91,836 约 1.09 个采样轮次，但动作块覆盖未来帧，不能当成每帧真实使用次数。应按留出集和实机分阶段成功率选 checkpoint。

新环境复用建议：

- 主要改变背景、光照或物体摆放：保留旧 v4 作基线，补新场景演示并独立评价；先检查颜色检测与相机方向。
- 改变相机安装/角度、桌面高度、夹爪或红块尺寸：先重新确定相机、爪尖/TCP 与桌面的几何关系。新演示用全新 dataset root，不往旧 v4 续录。
- 当前 wrapper 和若干校验固定了 v4 路径、条数、指纹。应建立新的版本入口；删除 v4 再用同名路径录新场景会混淆数据来源。

## 4. 优化顺序

**先接通完整流程和每个阶段的控制权交接。** 保留 OPEN_ONLY 作为单独调试入口，为完整任务新增明确入口：

| 阶段 | 建议实现与通过条件 |
| --- | --- |
| 接近 | SmolVLA 到红块附近可观察位置；目标持续可见、速度降低后接管 |
| 张爪 | 保持机械臂，实测开度足够容纳红块 |
| 对准 | 用两爪尖的中点/夹持中心计算红块位置与朝向误差，依据标定关系小步修正平面位置，误差稳定收敛后继续 |
| 下探 | 保持横向位置与姿态，根据桌面和红块高度分步下探；不能只靠图像面积变大判断可抓取高度 |
| 闭爪 | 联合实测位置、目标差、可用力/电流反馈及图像判断接触，区分抓住与机构卡住 |
| 抬升 | 先短距离试抬，确认红块随爪移动、没有滑落 |
| 搬运、释放 | 用新观测交给后续策略或经过验证的动作段；容器上方释放，确认物体脱离 |
| 回位 | 到统一回位目标，在明确容差内稳定后结束 |

切换策略/视觉控制时丢弃剩余旧动作块，重置相关缓存和平滑状态，用当前观测重新规划。旧主 visual latch 已有部分抓取后阶段，但 OPEN_ONLY 路径不会进入；现有 exclusive 循环直接返回，需要补完整任务编排。

**再验证新环境的爪尖对准。** 核对 D405 方向、两爪尖可见性、张开量与实际间距、相机/TCP 外参和桌面高度。先做静态检测及离线误差核对，再验证小范围修正方向。当前使用 HSV/轮廓阈值，并未直接检测两爪尖。若采用深度，需要先验证近距离质量，也需要配套处理；只打开 `use_depth` 不会让旧 RGB 模型自动利用它。

**然后重新采集有覆盖的演示。** 先少量 smoke 确认图像、时间同步、夹爪方向与完整流程，再采集多个位置反复成功的演示，增加“已接近但偏左/偏右/偏前/偏后”的纠正示范，统一速度、回位和结束条件。可从约 50 条开始；官方示例采用 5 个位置各 10 条，这是起点而非成功保证。[官方采集建议](https://huggingface.co/docs/lerobot/main/smolvla)

**最后按评价结果调训练。** 按 episode/采集批次划分训练与验证，避免相邻帧泄漏；记录阶段成功率、对准误差、抓空/掉落、耗时及失败视频。小幅颜色增强要保留红/蓝语义，几何增强需与动作标签一致。先解决流程和新环境数据问题，再根据指标调整 batch size、训练步数或模型。

## 5. 删除清单与依赖

已清理的只有 pip 下载、Python 字节码和 pytest 缓存：**726 个文件，约 1.51 GiB**。现有环境不依赖这些缓存；以后安装可能要重新下载，Python 会自动重建字节码。[删除记录](audit_20260909/cleanup_manifest.json)

下面路径相对于 `/home/kk/kk_data/lerobot`。容量为近似分配空间，父子项不要重复相加。数据、模型和旧实验均未删除，供你决定：

| 内容与路径 | 容量 | 建议及删除影响 |
| --- | ---: | --- |
| 原始旧录制 `datasets/local/firefly_y6_fullcycle_v4` | 1.46 GiB | 暂留一份作对照；当前两个部署及 arm6 来源校验依赖它 |
| 六轴派生训练集 `datasets/local/firefly_y6_fullcycle_v4_arm_only_v1` | 1.46 GiB | 可重建，但 arm6 部署需要；放弃旧链路后可删 |
| 100k 模型 `outputs/train/firefly_y6_smolvla_fullcycle_v4_100k_attested_run1` | 1.50 GiB | 是 7 维部署模型及 arm6 的父模型，放弃旧链路前保留 |
| 六轴 10k/20k 模型 `outputs/train/firefly_y6_smolvla_fullcycle_v4_arm_only_v1_finetune_20k_attested_run2` | 3.00 GiB | 放弃旧 arm6 部署后可整体归档/删除；当前校验要求 010000、020000、last 同时存在，不能只删 10k |
| 旧 NAS 导入 `imports/nas_outputs_20260812` | 0.845 GiB | 优先考虑删除的旧实验资产；原审计判定其动作标签/输出不适合真机，不是当前 v4。建议保留 AUDIT.md，删除大模型前确认不再研究它 |
| 插件日志/试验视频 `lerobot-robot-hexbase/outputs` | 0.527 GiB | validated_v2/v3 等旧部署子目录约 217 MiB，可筛选归档；不能整目录删除，firefly_diagnostics 内有当前部署依赖 |
| 三份 `checkpoints/.../training_state` | 每份 0.384 GiB | 训练恢复状态，不是录制集；arm6 校验绑定 training_step.json，不能把整个目录当缓存删掉 |
| 16 份历史 local 配置、旧入口、7 份 .orig | 很小 | 可移出日常入口；7 份 .orig 都与当前源码不同，可能保留唯一历史实现，未删除。旧脚本还被测试/历史流程引用 |

保留 `miniforge3/`、`assets/firefly/`、`datasets/calibration/`、当前插件源码/local 配置和插件 `outputs/firefly_diagnostics/`。`hf_cache/` 暂留，里面有固定版本模型缓存及物化来源。环境和基础模型不是场景变化后就失效的录制数据。

早期 `/home/kk/.cache/huggingface/lerobot/local` 当前为空；旧 manual、validated_v2、validated_v3_trimmed 数据不在本次检查的现存数据目录里。`datasets/_archive`、`datasets/.trash` 也为空。未扫描 NAS 或所有外接盘，不能据此判断其他地方没有备份。

完整路径见 [清理候选清单](CLEANUP_INVENTORY_20260909.md) 和 [机器可读空间清单](audit_20260909/storage_inventory.json)。

## 6. 可复查证据

- [全量数值数据审计](audit_20260909/fullcycle_v4_dataset_audit.json)：按照保存的历史合同，通过，0 错误/警告。
- [起始运动窗口审计](audit_20260909/fullcycle_v4_start_motion_audit.json)：通过。
- [50 条数据统计](audit_20260909/dataset_summary.json)：长度、张闭爪位置、起终状态。
- [历史日志汇总](audit_20260909/execution_log_summary.json)：停止原因、阶段、时延；未去重，不能按文件数计算成功率。
- [离线测试](audit_20260909/open_only_tests.txt)：20 项通过，测试从插件目录运行。
- 录制抽帧：[episode 0](audit_20260909/dataset_episode_00.jpg)、[episode 25](audit_20260909/dataset_episode_25.jpg)、[episode 49](audit_20260909/dataset_episode_49.jpg)。
- [8 月 20 日 v5 失败试验抽帧](audit_20260909/deployment_aug20_probe_v5.jpg)。

此次只清理缓存、添加报告并纠正文档入口，未改变真机控制参数、执行模式、合同或训练/部署指纹。横向对准及完整流程编排仍需实现，并在新环境验证。
