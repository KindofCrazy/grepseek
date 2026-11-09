# GrepSeek Mini：Qwen3.5-2B 训练前检查

当前 4B/PyTRIO 的并发续跑、PACE 调优和“训练结束后评测完整500题”设置见
[ACCELERATION.md](ACCELERATION.md)，云端记录见 [EXPERIMENT_LOGGING.md](EXPERIMENT_LOGGING.md)。
Ubuntu/WSL 的通用准备步骤见 [运行环境](environment/README.md)。
主机连接配置、凭据、运行目录和生成的实验报告不提交 Git。

Windows + PyTRIO 路线的 WSL/CPU 环境、完整语料和真实搜索准备见
[environment/README.md](environment/README.md)。该环境独立于下文原始 verl GPU 容器路线。

PyTRIO 可用模型 Qwen3.5-4B 的冻结数据适配、模板设置与复现命令见 [QWEN35_4B.md](QWEN35_4B.md)。

已实现的 PyTRIO SFT / 真实搜索 RL、dynamic sampling、预算控制与训练记录见
[pytrio_backend/README.md](pytrio_backend/README.md)。以下保留原始 2B / verl 路线说明。
实际轨迹、模型 checkpoint 引用与验收记录写入本地 `mini/runs/`、`mini/reports/`。

新增的数据筛选、难度预采样与训练诊断流程见 [CURATION.md](CURATION.md)。
精确阈值、12k/8k预算区别和五数据集各30题评测见 [FILTERS_AND_EVAL.md](FILTERS_AND_EVAL.md)。
冻结的抽样 ID、配置与输入 Hash 同步保存在 [reports/frozen](reports/frozen/)；完整数据文件按文档另行准备。

本地关联仓库：https://github.com/KindofCrazy/grepseek 。官方源码版本见
[UPSTREAM.json](UPSTREAM.json)。新增文件均位于 `mini/`，官方论文配置和工具、Reward、GRPO 实现保持原样。
这里的 Base 指初始化模型 `Qwen/Qwen3.5-2B`（官方 post-trained checkpoint，并非 `-Base` 型号）。
下文 2B / verl 路线仍是训练前适配；4B / PyTRIO 的真实训练记录见上面的后端验收报告。

## 环境

GPU 训练使用 Linux + NVIDIA CUDA，优先复用官方镜像：

```bash
docker pull ghcr.io/alirezasalemi7/grepseek:v1-slim
bash mini/container.sh python -c 'import torch, transformers, vllm; print(torch.__version__, transformers.__version__, vllm.__version__); print(torch.cuda.get_device_name())'
```

`mini/container.sh` 挂载当前源码到 `/workspace`、缓存到 `/cache`；命令内路径以容器为准。
设置 `GREPSEEK_IMAGE` 可固定镜像 digest；记录 `docker image inspect` 的 RepoDigests 和 `pip freeze`。
容器不可用时严格按官方 `TRAINING_ENV.md` 的安装顺序操作，不能直接安装最新 transformers/vLLM 替代。
CPU 测试环境可独立安装，不用于训练：

```bash
python -m venv .venv-mini
source .venv-mini/bin/activate
pip install -r mini/requirements-cpu.txt
python -m pytest mini/tests -q
```

Windows 可用 `.venv-mini/Scripts/python`。真实 Bash 执行测试在 Windows 跳过；GPU 训练在 Windows 主动报错。

## 数据

```bash
python -m mini.prepare_data --out mini/data --model Qwen/Qwen3.5-2B \
  --seed 42 --sft-count 400 --train-count 300 --dev-count 100 --test-count 200 --max-length 9216
python -m mini.preflight --model Qwen/Qwen3.5-2B --data-dir mini/data
MINI_TOKENIZER=Qwen/Qwen3.5-2B python -m pytest mini/tests -q
# 使用官方完整 Wikipedia；约 5 GB 下载、14 GB 解压。此步骤未在本次本机执行。
python sft/data_generation/download_corpus.py --dest data/wiki_18_corpus
```

`prepare_data` 从指定的 SFT 和 FlashRAG 仓库加载。线上模式解析并记录实际 dataset commit；
`--sft-revision/--qa-revision/--model-revision` 可固定版本。离线模式：
`--local-input DIR --model LOCAL_TOKENIZER`，目录含 `sft.jsonl` 或 `sft-train.parquet`，
以及 `nq_train.jsonl/nq_dev.jsonl/hotpotqa_train.jsonl/hotpotqa_dev.jsonl`。
离线来源以输入内容 SHA256 为依据，不伪造 Hub commit。

train 来自官方 train；dev 和最终 mini test 是官方 dev 的两个不重叠、固定保留分区，
每项任务 NQ/HotpotQA 各半。这里不是完整官方 benchmark test，汇报时必须标注规模与来源。
整个原始 dev 从 SFT/RL 训练候选中排除；按规范化问题去重，SFT 和 RL train 也不重叠。
抽样使用 seed + 内容哈希排序；输入顺序不影响选择，输入版本/长度阈值变化会改变样本。
结构非法、Reward Format Gate 不通过、超过 SFT 长度的轨迹写入 rejection 文件。
保留 `manifest.json`、样本 ID 与文件 SHA256；禁止覆盖已存在的 manifest。
没有向训练 parquet 添加 tools 列，避免原生模板注入另一套 XML Schema。

## CPU 配置检查与训练

```bash
python -m mini.launch sft --dry-run
python -m mini.launch rl --dry-run
# 在完整训练容器中，检查真实入口的 Hydra 配置，不执行训练：
bash mini/container.sh python -m mini.launch sft --compose-only
bash mini/container.sh python -m mini.launch rl --compose-only
```

默认两步。所有 Mini 参数可由 CLI 或同名大写环境变量设置（连字符换下划线），
例如 `GPUS/STEPS/GROUP/PROMPT_BATCH/PPO_MINI_BATCH/RESPONSE_LENGTH`；模型是 `MODEL_PATH`，
数据是 `DATA_DIR`，语料是 `CORPUS_DIR`，输出是 `OUTPUT_DIR`。
附加 Hydra overrides 放最后，可调整任意官方配置；自行修改底层 batch/并行 overrides 时须重新校验约束。
启动器禁止复用非空运行目录，避免旧日志冒充本次 Smoke Test。

```bash
# 单卡：真实 SFT forward/backward/optimizer updates；最初只运行 2–5 步。
DOCKER_GPUS='device=0' bash mini/container.sh python -m mini.gpu_smoke sft \
  --gpus 1 --steps 2 --output mini/runs/sft-smoke --data-dir mini/data
# smoke 输出为 <output>/sft；使用官方 merger 导出完整 HF 权重（不能只指向 tokenizer/config 目录）。
SFT_CKPT_DIR=mini/runs/sft-smoke/sft/global_step_2 HF_DIR=mini/exports/sft \
  bash containers/merge_sft_checkpoint.sh

# 单卡：真正调用官方 agentic GRPO，2 个 trainer step，每步 4 questions × 4 rollouts。
DOCKER_GPUS='device=0' bash mini/container.sh python -m mini.gpu_smoke rl \
  --model /workspace/mini/exports/sft --gpus 1 --sp 1 --tp 1 --steps 2 \
  --output mini/runs/rl-smoke --corpus data/wiki_18_corpus

# 实际 SFT 小规模训练：400 样本、global batch=4，一轮 100 steps。
# 仅在 smoke 成功且预算允许后由使用者手动执行。
bash mini/container.sh python -m mini.launch sft --steps 100 --output mini/runs/sft400
# 导出最终 SFT checkpoint 后才能启动 20–50 step RL。
bash mini/container.sh python -m mini.launch rl --model /workspace/mini/exports/sft \
  --steps 30 --output mini/runs/rl30 --eval-freq 10 --save-freq 10
CKPT_DIR=mini/runs/rl30/rl/global_step_30 HF_DIR=mini/exports/grpo \
  bash containers/merge_rl_checkpoint.sh
```

Mini SFT 使用官方 Dataset 的子类，按完整对话前缀生成每轮 token，要求拼接与完整模板逐 token 一致，
assistant header 与所有 system/user/tool token 的 loss 为零。训练循环、优化器和 GRPO 不重新实现。
RL、SFT、评测均显式启用 thinking；保持官方 Hermes JSON `<tool_call>` 协议。

## 观测与 Smoke 验收

每个进程在 `<output>/<stage>/observations/PID.jsonl` 写实际事件：

- `cuda_forward`、含非空梯度的 `optimizer_step`：forward/backward/step 证据及 CUDA allocator 显存峰值。
- `training_metrics`：原始 Loss、grad norm、KL/clip 等官方输出、step 间壁钟时间。首条观测时间没有前一步，不能用来估算稳态吞吐。
- `grpo_groups`：在官方 advantage 计算后按实际 UID 聚合 sample reward std、reward 均值和全零 advantage group 比例，支持 batch 重排。
- 官方 `train_rollouts/`、`val_rollouts/` 保留轨迹与 Reward 分项。

KL loss 与 reward KL 默认关闭，不能将缺失 reference KL 报成 0；使用官方 actor 的 PPO approximate KL/clip 指标。
进程显存峰值是 PyTorch allocator 的峰值；`mini.gpu_smoke` 另自动采集每秒 nvidia-smi device memory，
保存 `<output>/<stage>-gpu-memory.csv`，把各设备采样峰值写入验收 JSON。这包含 vLLM/驱动分配，
也可能含同一设备的其他进程，采样峰值可能遗漏不足一秒的瞬时峰值。
Smoke Test 只有收集到 CUDA forward、带梯度 optimizer step、目标 trainer step 和真实 GRPO 分组/actor 更新日志才写
`smoke_verified.json`。它不要求 advantage 非零（小组可能全同分），也不宣称模型质量改善。

## 统一评测

复制 `mini/checkpoints.example.json`，填写实际 HF 导出目录和 endpoint。
单卡优先加 `--serve-locally`：脚本依次启动、评测并卸载 Base/SFT/GRPO，三者共享一张 GPU；
此模式忽略清单中的 base_url，需在官方 vLLM 容器内执行，无需对外映射端口。
也可不加此参数，用 CPU 客户端连接三个已有 endpoint。
用官方 `containers/serve_vllm.sh` 的 `MODEL_PATH/PORT/TP_SIZE` 启动服务，保持 text-only、eager、
`--reasoning-parser qwen3`，并将 `MAX_MODEL_LEN=9216`。官方 wrapper 默认 served name 是 `grepseek`，
必须显式设置 `SERVED_MODEL_NAME` 与 checkpoints.json 的 model 完全一致。例如（此服务命令需要独立终端）：

```bash
MODEL_PATH=Qwen/Qwen3.5-2B SERVED_MODEL_NAME=Qwen/Qwen3.5-2B \
  PORT=8000 TP_SIZE=1 MAX_MODEL_LEN=9216 MAX_NUM_SEQS=4 GPU_UTIL=0.8 \
  bash containers/serve_vllm.sh
```

官方 serve 脚本未加 eager。若 GPU 上出现与 RL 同类的 graph 崩溃，用官方镜像内直接启动：

```bash
# 在容器内（使用 docker -p 8000:8000 映射端口），或已有相同环境的 Linux 主机中：
vllm serve Qwen/Qwen3.5-2B --served-model-name Qwen/Qwen3.5-2B \
  --port 8000 --tensor-parallel-size 1 --max-model-len 9216 --max-num-seqs 4 \
  --language-model-only --reasoning-parser qwen3 --enforce-eager
```
评测使用官方 `run_agent_on_example`；客户端显式发送 thinking=true，不修改推理的纠错/工具行为。

```bash
python -m mini.evaluate --checkpoints mini/checkpoints.json --data-dir mini/data \
  --split dev --corpus data/wiki_18_corpus --out mini/runs/eval-dev --serve-locally
# 用 dev 选择 SFT/GRPO checkpoint，手动冻结 {"base": model, "sft": path, "grpo": path}。
# 检查 selection 与 checkpoints 完全一致后，最后只评测一次 mini test：
python -m mini.evaluate --checkpoints mini/checkpoints.json --selection mini/selection.json \
  --split test --corpus data/wiki_18_corpus --out mini/runs/eval-test --serve-locally
```

报告总体与逐数据集 EM/Token-F1、工具执行数/均值、非法请求及命令比例、无答案率、context 截断率、API 错误率。
非法率分母是包含 `<tool_call` 的 assistant 请求次数，非法 JSON/name/arguments 和实际工具 exit_code=-2 计非法；
工具超时不算非法。Context 指任一 assistant `finish_reason=length` 或 context overflow API 错误，
这是推理侧可观察定义，与 RL 的 `hit_max_context` 长度标记分别报告，工具 stdout 截断不算 context 截断。
默认不加每轮 token 上限（0），由服务的 9216 总上下文限制；使用同一服务参数比较三种 checkpoint。
最终 test 不能选择 checkpoint；脚本要求 frozen selection，但不能替代实验者遵守这一规则。

## 双卡备选和定位

先用 `--gpus 2 --sp 1 --tp 1`：训练 FSDP 两卡分片，rollout 每卡一个 TP=1 replica。
若 forward activation OOM，再试 `--gpus 2 --sp 2 --tp 1`；必须实际验证 Qwen3.5 混合注意力的 Ulysses 内核。
只有 vLLM 权重/cache 加载失败时试 `--tp 2`，并重新核对 2B 的 attention/KV/linear head 可整除性。
双卡不保证解决单个 embedding/layer all-gather、host RAM 或 hybrid weight-transfer 故障。

按日志定位：模型类/权重名错误→固定官方依赖与完整 HF 导出；wrap class 不存在→查 transformers build；
vLLM CUDA graph 崩溃→保留 eager；memory pool 错误→清除 expandable_segments；
rollout OOM→先减 `max_num_seqs`/memory utilization，仍失败再缩短 response；
训练 OOM→微批保持 1、offload，尝试双卡；全零 advantage 高→查看 format gate、无答案、命令错误、组内同分，不能改 Reward 掩盖。
把 response 改为 4096 时启动器自动同步 prompt/response、总 context、actor/ref/rollout token budgets 和 Reward L_max；
SFT 长度另用 `--sft-length`，必须重做数据长度检查。tool stdout 2048 与 turn budget 保持官方值。

仍需人工 GPU 验证的项目及静态审查结论见 [AUDIT.md](AUDIT.md)。
