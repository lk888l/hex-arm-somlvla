# GitHub 保存范围与 VLA 复现审查（2026-09-10）

2026-11-02 更新：已按本报告建议整理为工作区仓库与插件 submodule，并建立本地
归档分支 `archive/git-layout-20261102`。当前必要的四份机器配置、controller
contract 与 schema-5/schema-6 审计文件已纳入插件版本管理，原 Git 元数据和未提交
小文件已备份。当前布局、保存范围和提交顺序见 [Git 使用文档](GIT_WORKFLOW.md)。
工作区随后补充了 Conda explicit 锁、完整非 editable pip 清单、系统/环境快照和
RealSense udev 规则，恢复步骤见 [安装文档](INSTALL.md)；这些补充不在原始根归档
提交 `e21547a` 中。以下各节保留 2026-09-10 的检查记录；其中“待配置”“尚未提交”
和环境清单缺包等描述属于当时状态。远端发布、完整数据/模型备份和新机复现验收
仍需另行完成。

结论：只保存 `lerobot-robot-hexbase` 不足以恢复当前系统。建议保留插件仓库的独立历史，由 `hexfellow-lerobot` 保存工作区的环境、配置、文档与资产清单，通过 Git submodule 固定插件提交；数据和模型单独备份，用版本与 SHA256 关联。

本次检查本地目录、Git 元数据、已安装包元数据、源码和配置；新增本报告，没有整理/删除 `.git`、修改运行代码、提交或推送，也没有启动硬件或训练。未 fetch/查询 GitHub 分支，因此本地 `origin/master` 不代表远端实时状态。新机全流程尚未实测。

## 1. 当前 Git 状态

| 目录 | 本地实际情况 | 含义 |
| --- | --- | --- |
| 工作区根目录 `.git` | master 尚无提交，无已跟踪文件；origin 为 `git@github.com:lk888l/hexfellow-lerobot.git` | 外层环境脚本、文档等尚未进入本地版本历史 |
| `lerobot-robot-hexbase/.git` | HEAD 为 `21ef740`；origin 为 `https://github.com/lk888l/lerobot-robot-hexbase.git`；upstream 为 `hex-meow/lerobot-robot-hexbase` | 保留原仓库历史；9 个已跟踪文件修改、82 个未跟踪文件（含备份）尚未提交 |
| 两者关系 | 外层没有 `.gitmodules` 或 gitlink | 当前是嵌套独立仓库，尚未配置成 submodule |

外层 `.git` 约 563 MiB；`git count-objects -v` 报告约 559 MiB 的临时 pack `objects/pack/tmp_pack_DJ6Pxw` 为 garbage。不能据此确定原操作是什么，也不能把这个容量当作已经保存的版本历史。后续整理前先备份 Git 元数据和全部本地改动；不直接删除内层 `.git`。

外层缺少 `.gitignore`，会把环境、数据和缓存一起显示为未跟踪内容。不要直接在这里 `git add .`。外层提交也不会自动提交内层源码改动。

## 2. 应保存什么

| 内容 | 保存方式 | 具体要求 |
| --- | --- | --- |
| 插件 `lerobot_robot_hexbase/`、`scripts/`、`tests/`、`pyproject.toml`、README | 插件 Git 仓库 | 保存当前有效源码、测试和协议生成脚本；保留已生成的 `proto/*_pb2.py` |
| 根目录 README、STATUS、docs、tools、env/activate 脚本 | 工作区 Git 仓库 | 当前插件仓库没有覆盖这些复现入口 |
| 两套环境定义、依赖锁、安装源 | 工作区 Git 仓库 | 更新实际包清单，固定 Conda 包构建，去掉环境定义中不可移植的 prefix 或生成时覆盖 |
| 训练参数、录制参数、部署参数 | Git 中的具名实验配置和机器模板 | 将算法参数与相机序列号、控制器 ID、本机路径分开；机器实值另备份，不能只保存过时 example |
| controller contract、视觉审计、模型来源凭证、数据派生清单 | 小文件进 Git 或具名资产清单，原件随资产备份 | 当前运行会读取并校验它们，不是可随意丢弃的日志 |
| URDF、标定结果、CAN/launcher/supervisor 配置 | Git 中的配置或固定外部版本 | 同时记录实机安装、相机方向、关节顺序和夹爪/TCP 定义；当前 `datasets/calibration` 未发现文件 |
| 原始数据、派生数据、训练模型、恢复状态 | NAS/对象存储/模型数据仓库 | Git 保存可取回的位置、版本、文件清单、大小与 SHA256；不能只有本机路径 |
| 环境目录、下载缓存、普通日志、试验视频 | 忽略普通 Git；按需独立归档 | 重要失败视频和审计证据可筛选保存 |
| `.orig`、`.rej` | 先比对，再历史归档或忽略 | `.orig` 可能含唯一旧实现；不能在首次备份前一概删除 |

重点遗漏：插件 `.gitignore` 排除了 `configs/*.local.yaml` 和整个 `outputs/`。两份最新部署配置以及 `outputs/firefly_diagnostics/` 中实际被加载的合同/审计都在忽略范围。把所有源码 push 上去仍会缺这些运行输入。

例如 arm6 当前依赖：

- `configs/deploy_firefly_y6_hold_fullcycle_v4_arm6_20k.local.yaml`。
- `outputs/firefly_diagnostics/controller_contract_validated_v2_arm0435_ee0165.json`。
- `outputs/firefly_diagnostics/firefly_y6_tip_servo_open_only_fullcycle_v4_schema6_exact.json` 及配置中的对应哈希。
- 完整 arm6 run2、100k 父模型、原始及 arm-only 数据和相关来源凭证。

不要机械地移动这些文件：代码、配置和凭证可能绑定原路径或文件哈希。可先原样归档；迁移到专用目录时同步修改引用并重新验证。

## 3. 推荐仓库关系

```text
hexfellow-lerobot/                 # 整套框架的版本入口
├── .gitignore
├── .gitmodules                   # 待配置
├── README.md / STATUS.md
├── env.sh / activate-*.sh
├── configs/                      # 环境锁、配置模板
├── docs/ / tools/
├── manifests/                    # 待新增：外部软件/数据/模型位置与哈希
└── lerobot-robot-hexbase/         # submodule，固定一个已推送的插件 commit
```

submodule 保留独立历史，父仓库记录其 commit 和获取地址。这比直接留下一个没有声明的嵌套仓库明确。来源：[Git 官方说明](https://git-scm.com/docs/gitsubmodules)。

发布顺序：先审查并提交/推送插件源码，再由工作区提交对应插件指针、环境和资产清单。一次可复现版本至少绑定：工作区 commit、插件 commit、环境锁、数据版本、模型版本、硬件软件版本和生效配置。

如果以后确定插件不再单独维护，也可导入历史后合成一个仓库；当前已有 origin/upstream，保留独立插件更符合现状。整理现有仓库前应备份，再检查真实远端分支是否已有内容，避免覆盖远端历史。

## 4. 环境实测与缺口

| 项目 | 本机记录 |
| --- | --- |
| 系统 / 架构 | Ubuntu 24.04.4 LTS；当前内核 7.0.0-31-generic；软件包 linux-64 |
| GPU / 驱动 | RTX 5070 Ti Laptop，12227 MiB；595.84 |
| Python | 两环境均 3.12.13，Conda build `hd63d673_0_cpython` |
| LeRobot | 两环境均 0.6.0；各校验 500 个带 RECORD 哈希的包内文件，没有发现与安装记录不一致的文件 |
| PyTorch / torchvision / TorchCodec | 2.10.0+cu128 / 0.25.0+cu128 / 0.10.0 |
| FFmpeg | 7.1.1，Conda build `gpl_hbbdf940_911` |
| I/O 插件 | 本地 editable 安装 `lerobot-robot-hexbase==0.3.0` |
| 训练环境 | 没有安装插件或 Zenoh；现有离线 wrapper 特意绕开插件注册入口 |

现有 requirements 中已列出的包没有发现版本漂移；但 I/O 清单缺少后来安装的 `eclipse-zenoh==1.9.0`、`pyrealsense2==2.58.3.10794`、`protobuf==7.35.1`、`grpcio==1.83.0`、`grpcio-tools==1.83.0`、`pytest==8.4.2`、`iniconfig==2.3.0`、`pluggy==1.6.0` 和本地插件。pip 本身两环境均为 26.1.2，也未列入现有清单。

后续要导出完整 Conda explicit 锁与更新的 pip 清单；检查导出中的本机 editable 路径和任何私有源凭据，再提交。插件安装用仓库相对路径，不把 `file:///home/kk/...` 当作可下载依赖。PyTorch CUDA wheel 需要记录对应安装源，例如官方 cu128 wheel index；不能仅留下 `torch==...+cu128`。来源：[PyTorch 官方历史版本安装说明](https://pytorch.org/get-started/previous-versions/)。

`env.sh`、activate 脚本、Conda YAML、模型 config 和训练 wrapper 都有绝对路径。特别是 `train_firefly_smolvla_arm_only_v1_finetune_20k_run2.py` 固定 Python 路径、工作目录、脚本路径和可执行文件 SHA256：

```text
aa584ad7b853904b0ae7fb5a8f44d9d27960e05de1f4838855f93fa4359df869
```

仅安装“Python 3.12”或修改 env.sh 不能满足这个验证。迁移应选择：恢复原绝对目录和精确运行时，再验证旧凭证；或改造为可配置路径与新的来源验证，建立明确的新版本。不能删掉校验或手改哈希来声称旧训练仍已被验证。即使固定了软件版本，也不保证重新训练后的权重逐字节一致。

## 5. 数据与模型恢复集

当前建议优先备份以下有效链路，容量为近似磁盘占用：

| 相对路径 | 容量 | 说明 |
| --- | ---: | --- |
| `datasets/local/firefly_y6_fullcycle_v4` | 1.46 GiB | 原始 50 条、91,836 帧；完整保存 meta/data/videos 及其余来源文件 |
| `datasets/local/firefly_y6_fullcycle_v4_arm_only_v1` | 1.46 GiB | 同一批演示的六维派生集，当前 arm6 校验仍依赖 |
| `outputs/train/firefly_y6_smolvla_fullcycle_v4_100k_attested_run1` | 1.50 GiB | 100k 模型，包含 100000 和 last 链接，是 arm6 父模型 |
| `outputs/train/firefly_y6_smolvla_fullcycle_v4_arm_only_v1_finetune_20k_attested_run2` | 3.00 GiB | 同时保留 010000、020000、last、training_state 和 run/child 凭证 |
| `assets/firefly` | 约 870 MiB | 固定基础模型及 VLM tokenizer/config bundle，保留所有来源 lock |

上述约 8.3 GiB，再加当前配置、合同/审计和硬件发布包。它不是 Git 源码仓库应直接承载的全部内容。

基础资产来源版本已在本地目录及 provenance lock 中记录：

- `lerobot/smolvla_base`：`c83c3163b8ca9b7e67c509fffd9121e66cb96205`。
- `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`：`7b375e1b73b11138ff12fe22c8f2822d8fe03467`。

基础模型可按固定 revision 获取，再通过现有 materialize 脚本建立 bundle；这些脚本也需要迁移路径验证。基础 bundle 已有本机路径补丁，因此直接下载远端原始 config 不等于本机 bundle。训练好的本地模型必须有独立可取回备份，不能只靠基础模型仓库。

`miniforge3/` 约 18 GiB、`hf_cache/` 约 2.8 GiB、`imports/` 约 866 MiB 和普通 outputs 日志不进入常规 Git。缓存可在确认固定资产能恢复后重建；如需应急保全现机环境，可以额外制作独立环境归档，但归档不替代可重建安装清单。

## 6. 工作区外的硬件依赖

纯离线训练不需要启动下列程序；采集和真机部署需要。

| 位置 | 本地版本证据 | 应保存内容 |
| --- | --- | --- |
| `/home/kk/kk_data/git_clone/hex-hw-supervisor` | HEAD `5207f31e30acce18aff535e6f62a1c6f4516114f`；工作树干净 | 仓库地址、commit、构建方式、生效 profile、必要系统配置 |
| `/home/kk/kk_data/git_clone/hex-robot-proto` | HEAD `bf90247bf790df45577c46a892e35d00a60e02ab`；工作树干净 | 当前源码 revision；已提交 pb2 的实际生成来源需补证据 |
| `/home/kk/kk_data/hex-arm/remote-control` | launcher、arm_controller、ee_controller、arm_teleop 等二进制和 launch YAML | 对各实际使用二进制计算哈希，记录下载/构建来源；保存 launch 配置、URDF、CAN 与权限设置 |
| `remote-control/hex-controller-v0.3.7-x86_64/BUILD_INFO` | tag v0.3.7；controller `6e5c7a6fbe54c34ef7fa04abcf13dff48d0e853a`；target x86_64-unknown-linux-musl | 完整 BUILD_INFO 和原发布包 |

发布包 BUILD_INFO 中的 `hex-robot-proto` 为 `6848962383fbb62a379b4a2a1522a3e7efb2bafd`，与当前 proto 源码 HEAD 不同；不能假设重新生成当前 HEAD 的 pb2 就是原组合。顶层独立 `arm_teleop`/`ee_controller` 也不能仅凭旁边的 v0.3.7 文件夹就判定版本。

文档还有遥操作参数分歧：外层 STARTUP 写 MODE=dfr，插件 README 写 MODE=follow + gravity_comp。需要以选定有效 launcher 和实验流程统一成版本化入口，避免新机照着不同文档启动不同控制模式。

## 7. 空白电脑复现顺序

目前尚不具备“clone 后一条命令恢复”的条件。下面是整理完成后的复现合同，不是已经实测通过的一键安装脚本。

1. **准备系统。** 先用 Ubuntu 24.04 x86_64 和相近 GPU 建立复现基线；安装兼容驱动、Git、Miniforge。记录实际系统与驱动；若恢复旧 arm6 凭证，目录和精确解释器也必须满足现有约束。
2. **获取完整代码版本。** 在插件更新已推送、工作区已正确记录 submodule 后执行 `git clone --recurse-submodules https://github.com/lk888l/hexfellow-lerobot.git lerobot`。发布版本要固定工作区 tag/commit，不能一直追随分支最新状态。当前仓库还没有完成这些整理，今天直接照此命令不能保证恢复。
3. **重建两套环境。** 用补全后的 Conda 锁恢复系统库和 Python，用更新 pip 清单及固定 CUDA wheel 源安装。I/O 环境安装固定插件提交 `python -m pip install -e './lerobot-robot-hexbase[realsense,dev]'`，并用锁限制依赖版本；训练环境按现有 wrapper 设计保持独立。执行 pip check、检查关键包版本和 GPU 可用性。
4. **恢复资产。** 按 manifests 下载/复制上述原始和派生数据、父/子模型、基础 bundle、合同、审计、凭证和机器配置。保留目录结构及 last 链接；先核对完整清单与哈希。仅复制 MP4 或 model.safetensors 不够。
5. **处理路径与凭证。** 旧环境复刻要恢复精确绝对目录和 Python；可移植改造则需要同步处理脚本、YAML、模型 JSON、provenance 与验证逻辑。使用新的实验输出名和来源记录，不能覆盖已有 run1/run2 冒充相同实验。
6. **离线验收。** 先运行 tools/diagnose.py、适用离线测试和数据审计，再验证指定 checkpoint 加载、输入维度、两路相机键、预后处理器及有限 action。若目标是继续训练，还要验证 optimizer/scheduler/训练步恢复。现有 smoke_smolvla.py 和 smoke_train_smolvla.py 使用未固定 revision 的 Hub 下载，需改为固定快照后才能作为严格复现验收。
7. **恢复硬件链路。** 还原 supervisor、controller、teleop、协议、URDF、CAN 配置和 RealSense 权限；重新识别两台相机序列号、安装方向、controller ID。先只读状态/采集，再验证 shadow，最后按项目已有条件验证真机执行。历史合同和审计不能替代新机器/新场景验证。

当前“复现现在的框架”包含一个明确限制：最新 7 维/6 维部署配置是接近后张爪的 `open_only`，张爪完成即结束；完整抓取搬运没有接通。成功安装和加载模型不代表复现完整抓取成功率。

## 8. 整理优先级

1. 备份未提交源码、被忽略的当前配置/必需审计及 Git 元数据，保证现机状态可恢复。
2. 补全插件提交与工作区忽略规则，明确 submodule 关系；核对远端后发布。
3. 导出真实环境锁，保存硬件依赖版本与完整资产清单。
4. 处理本机路径和 arm6 运行时绑定，统一当前录制/训练/部署文档入口。
5. 在新目录或空白机器按上述流程验收后，才把该工作区 commit 标为“可复现版本”。
