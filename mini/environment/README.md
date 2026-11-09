# Ubuntu / WSL CPU 环境

CPU 主机负责数据准备、真实 Wikipedia 检索、训练调度和本地损失计算。
模型的前向、反向、生成与权重存储由 PyTRIO 执行；不需要在 CPU 主机下载完整模型。
训练后端见 [后端说明](../pytrio_backend/README.md)。

## Ubuntu 准备

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip ripgrep grep gawk sed \
  coreutils findutils git rsync ca-certificates curl bubblewrap util-linux build-essential
python3 -m venv .venv
.venv/bin/python -m pip install -r mini/environment/requirements.txt
.venv/bin/python -m pip install -r mini/pytrio_backend/requirements.txt
.venv/bin/python -m pip check
```

依赖固定PyTRIO版本；CPU Torch用于custom loss。系统沙箱需要支持bubblewrap创建隔离环境。
Ubuntu24.04可能需要管理员配置bwrap专用AppArmor策略；不要关闭整个系统的保护。

## Windows / WSL

可使用独立Ubuntu发行版和普通项目用户。`bootstrap_ubuntu.sh`、`setup_python.sh`
提供默认用户 `grepseek`、项目目录 `/home/grepseek/grepseek` 的初始化脚本。
从挂载的仓库目录以root执行bootstrap，再以普通用户执行setup；源码路径按本机调整。
脚本同步不会使用 `--delete`，但仍应保留运行目录和虚拟环境的独立备份。
`wslconfig.example` 仅为内存配置示例，调整会影响整个WSL虚拟机。

## 语料与搜索

```bash
bash mini/environment/download_corpus.sh
.venv/bin/python -m mini.environment.check_search
```

来源为 `PeterJinGo/wiki-18-corpus`，固定源版本与LFS SHA256在 `corpus-source.json`。
下载支持续传；先核验压缩包，再流式提取唯一JSONL，不把压缩包内路径当作目标路径。
完成后将hash、字节数与行数写入语料manifest。压缩包和完整语料需预留足够磁盘。
搜索检查使用真实passage、无匹配扫描、输出截断与超时，不调用模型。

PACE构建、包导出与验收见 [PACE说明](../pytrio_backend/PACE.md)。
`pace_semantic_fingerprint.py` 与配套C程序用于验证词条ID重排后倒排语义仍一致。

## 登录与本地文件

```bash
source .venv/bin/activate
trio login
swanlab login
python -m mini.environment.inventory --check-account
```

账户检查只读模型/容量元数据。密钥仅放SDK私有凭据存储，不写进仓库或命令脚本。
SSH连接文件、凭据迁移脚本、主机配置和一次性实验队列不属于通用核心源码。
数据、运行日志、checkpoint引用与生成报告保存在忽略的本地目录。
