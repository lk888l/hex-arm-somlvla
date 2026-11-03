# Git 使用与归档范围

根仓库 `hexfellow-lerobot` 是整套工程的版本入口；`lerobot-robot-hexbase/` 是拥有
独立历史的正式 submodule。环境配置说明见 [configs/README.md](../configs/README.md)，
安装步骤见 [INSTALL.md](INSTALL.md)。

## 当前归档基线

2026-11-02 的本地归档分支为 `archive/git-layout-20261102`：

| 仓库 | 归档提交 | 内容 |
| --- | --- | --- |
| 根仓库 | `e21547a` | 环境定义、脚本、工具、文档、基础资产的小文件与子模块指针 |
| HexBase 子模块 | `287f7388a45babfeb31be40e6c0557d6a4181f1f` | 采集、训练、部署源码及现机必要配置/合同/视觉审计 |

归档后工作区补充了 Conda explicit 锁、更新后的 requirements、环境快照和
RealSense udev 规则；这些不在原始根归档提交 `e21547a` 中。本文及安装指南也需
随后续版本保存。文件存在于工作区、已暂存、已提交和已推送是四种不同状态，
本次文档整理没有核验远端是否已包含归档提交。

父仓库用 `.gitmodules` 记录子模块地址，用 gitlink 固定子模块 commit。
插件 Git 元数据位于 `.git/modules/lerobot-robot-hexbase/`，插件目录内的 `.git`
是指向该位置的文本文件；保留这一布局。

## 保存范围和忽略例外

| 内容 | 保存位置 / 规则 |
| --- | --- |
| 根目录文档、`env.sh`、`activate-*.sh`、`configs/`、`tools/` | 根仓库 Git；`configs/hardware.env` 为本机副本，单独忽略 |
| 插件源码、脚本、测试、训练配置、example YAML | 子模块 Git；父仓库只保存其提交指针 |
| 四份 fullcycle-v4 `*.local.yaml` | 子模块白名单，文件清单见 [配置目录说明](../configs/README.md#子模块里的机器配置) |
| controller contract 和 schema-5/schema-6 视觉审计 JSON | 子模块 `outputs/firefly_diagnostics/` 下的三份白名单文件，是运行输入 |
| 基础 bundle 的配置、tokenizer、来源 lock | 根仓库 `assets/firefly/`；两个固定路径的 640 字节归一化 `.safetensors` 明确放行 |
| `datasets/calibration/` | 允许加入 Git，仍受通用文件后缀规则约束；允许跟踪不代表已有标定文件 |
| 原始/派生数据、训练输出、大模型权重 | 独立备份；根目录 `/datasets/*`、`/outputs/` 以及权重后缀被忽略 |
| Miniforge、Hub/pip 缓存、日志、imports、本地整理备份 | 根目录对应路径被忽略，按需独立归档 |
| 普通 `*.local.yaml`、历史输出、`.orig` / `.rej`、密钥、`.env` | 按各层 `.gitignore` 留在本地；重要旧副本需独立保存 |

根目录的 `/outputs/` 只匹配根级输出；子模块自行管理它的 `outputs/`。
`.gitignore` 不会取消已经跟踪的文件。新增必要机器配置或审计时，应在所属仓库
增加精确的例外，并核对文件引用、哈希和提交内容。

在根目录查看两层状态及实际跟踪情况：

```bash
git status --short
git submodule status
git diff --submodule=log
git -C lerobot-robot-hexbase status --short
git -C lerobot-robot-hexbase diff --stat
git ls-files configs assets/firefly
git -C lerobot-robot-hexbase ls-files configs outputs/firefly_diagnostics
```

`git submodule status` 开头为空格表示与父仓库记录一致，`+` 表示子模块 HEAD
与 gitlink 不同，`-` 表示尚未初始化，`U` 表示冲突。子模块内部未提交的修改还需
查看其 `status`，仅凭提交号一致不能认定工作树干净。

排查文件为何被忽略时使用所属仓库：

```bash
git check-ignore -v --no-index configs/hardware.env
git -C lerobot-robot-hexbase check-ignore -v --no-index configs/example.local.yaml
```

`--no-index` 也能检查已跟踪路径对应的规则；输出以 `!` 开头的匹配规则表示放行。
最终以 `git ls-files` 判断文件是否已经纳入当前索引。

## 克隆与切换版本

先确认所选父仓库版本和它引用的子模块提交都可从远端取得。当前配置地址为：

- 根仓库 origin：`git@github.com:lk888l/hexfellow-lerobot.git`。
- `.gitmodules` 中的插件地址：`https://github.com/lk888l/lerobot-robot-hexbase.git`。

可用 HTTPS 获取工作区，以下示例只获取代码，环境和数据按安装文档恢复：

```bash
git clone --recurse-submodules https://github.com/lk888l/hexfellow-lerobot.git lerobot
cd lerobot
```

按选定的发布 tag/commit 切换父仓库，例如使用 `git switch --detach <tag-or-commit>`
并替换占位值；然后恢复父仓库固定的子模块版本：

```bash
git submodule update --init --recursive
git submodule status
```

已有 clone 在切换父版本或拉取更新后也执行上述命令，先处理两层工作树中的未提交
修改。常规恢复按 gitlink 固定提交执行；`git submodule update --remote` 会改为
跟随远端分支，只适用于明确升级插件的场景。

## 修改、提交和发布顺序

只改根目录文档或环境文件时，在根仓库审查并按具体路径暂存、提交即可。插件
内部的修改必须先在插件仓库形成提交，再由父仓库记录新的 gitlink。
submodule 在恢复后通常处于 detached HEAD，开始修改前先建立工作分支。

例如在插件中更新录制配置说明，先审查两层远端与分支：

```bash
git remote -v
git -C lerobot-robot-hexbase remote -v
git branch --show-current
git -C lerobot-robot-hexbase branch --show-current
git -C lerobot-robot-hexbase switch -c work/firefly-config
```

编辑并审查具体文件后，先保存并发布插件提交：

```bash
git -C lerobot-robot-hexbase diff
git -C lerobot-robot-hexbase add -- configs/record_firefly_y6_fullcycle_v4.local.yaml
git -C lerobot-robot-hexbase diff --cached
git -C lerobot-robot-hexbase commit -m "Clarify fullcycle-v4 configuration documentation"
git -C lerobot-robot-hexbase push -u origin HEAD
```

确认插件提交可以通过 `.gitmodules` 的地址获取后，在父仓库的目标工作分支保存
指针；根仓库文档、环境清单等改动也按实际路径加入同次或对应提交：

```bash
git diff --submodule=log
git add -- lerobot-robot-hexbase
git diff --cached --submodule=log
git commit -m "Update pinned HexBase revision"
git push -u origin HEAD
```

发布前核对远端已有历史和目标分支，使用正常推送保存历史。父仓库 commit 不会
自动提交或上传插件内部修改；仅推送父仓库也不会发布它所引用的新插件 commit。

## 独立备份与复现边界

整理前的根仓库元数据及小文件快照位于
`.local-archive/git-layout-20261102-120215/`；其中 `README.txt` 记录范围，
`worktree-before.sha256.json` 记录校验和。该目录被忽略，只存在于本地，需要
独立保管；它不包含完整数据集、模型或环境备份。

数据/模型与外部硬件依赖清单见
[复现审查](VERSION_CONTROL_AND_REPRODUCTION_20260910.md)。归档应同时记录工作区
commit、插件 commit、环境快照、数据/模型位置与 SHA256、硬件版本和生效配置。
远端可获取性、完整备份和新机验收需分别核实，Git 布局整理本身不代表实验已可复现。
