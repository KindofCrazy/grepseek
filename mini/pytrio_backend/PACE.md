# PACE 构建与启用

PyTRIO 后端的 RL / eval 默认使用 PACE，结构目录为 `data/pace-full`。
索引需单独构建并验收，不随源码提交；本地验收报告写入 `mini/reports/pace/`。
`GREPSEEK_PACE_DIR` 是官方 inference 入口的开关；此后端通过配置字段选择，不依赖该环境变量。

```bash
# 无计费请求的配置检查：应显示 search_backend=pace。
python -m mini.pytrio_backend eval --out mini/runs/pace-plan
# 显式选择时，可在原训练命令中加：
# --config mini/pytrio_backend/pace_config.json
# 比较顺序基线时加：--search-backend sequential
```

推荐临时 CPU 主机：64 GB 内存、8 个或更多 CPU、Ubuntu Linux、Python ≥3.12，
至少 60 GB 可用 SSD。无需 GPU，也不需要 PyTRIO API key。
32 GB 机器接近官方约28 GB峰值，64 GB有更充足余量。
官方21分钟是作者机器的构建耗时；下载、校验、实际CPU速度及回传另计，先预留1–2小时，按进度停止实例。

在新机器把 `pace-build-job.tar.gz` 解压到一个新目录，然后从该目录运行：

```bash
sudo apt-get update
sudo apt-get install -y python3-venv build-essential curl
bash mini/pytrio_backend/pace_cloud_build.sh
```

脚本固定源码包及依赖，下载并验证官方完整语料，构建原生C库和PACE结构，
输出 `pace-full.tar.gz`、SHA256和逐文件完整性清单；不会访问模型服务或训练账户。
需要仅回传这两个输出文件，不必回传14 GB原语料、虚拟环境或编译后的 `.so`。
本机编译原生库，以匹配本机 glibc / CPU。

结构回传到本地 `GrepSeekUbuntu` 后：

1. 先核对压缩包SHA256，解压到新的目录，保留构建来源清单。
2. 安装本机编译依赖并运行 `bash inference/pace/_native/build.sh`。
3. 运行下列绑定命令，完整核对14 GB语料与PACE数据文件哈希之后才迁移路径和mtime：

```bash
python -m mini.pytrio_backend.pace_artifacts bind \
  --corpus data/wiki_18_corpus/wiki_corpus.jsonl --out data/pace-full
```

接入采用只读 PACE 规划和原生固定字符串核验；不支持或超过预算的命令回到原有搜索沙箱。
云端以真实训练命令验证原始字节，再比对冻结SFT轨迹中的七个工具字段，通过后才启用。
至少记录 PACE served/fallback、候选行数/字节、检索耗时、RSS及总工具时间。
运行时数组采用只读内存映射，28 GB构建峰值不等于运行时必须有28 GB内存；
当前2 GB WSL是否足够、需要调到多大，以完整结构上的实测为准，不预先承诺77倍加速。

若只能在本机完成，可以另做分批/磁盘外存构建器；这需要额外实现和一致性验收，
不是给官方构建命令增加一个低内存参数即可实现。

来源：[官方PACE说明](https://github.com/alirezasalemi7/grepseek/blob/bb6ccc73ecb6c17ab0c7360f9187cbde2a4b866a/inference/README.md#pace--pruned-adaptive-command-execution-optional-speedup)。
