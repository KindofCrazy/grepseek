# GrepSeek Mini：数据筛选、难度估计和诊断

本轮仅运行 CPU 数据工作。未运行训练、GPU Rollout、付费 LLM API，也没有租赁 GPU。
附件末尾在“三”处截断；这里按 Training Rollout / Validation / Final Evaluation 三类分开记录。

## 实际来源与复现

实际 Schema 和真实样本摘要见 [SOURCE_SCHEMA_SAMPLES.json](reports/SOURCE_SCHEMA_SAMPLES.json)。
SFT parquet 是 `id: string, messages: list<struct<role,content>>`，并没有独立 question/source/gold/正确性标签。
Question 从初始 `Query:` 消息提取；Source 按规范化问题连接 FlashRAG 原始记录，只有明确 `nq_` ID 时才允许前缀推断。
`train_...` 不能直接等同 HotpotQA。歧义与未匹配来源保持 unknown。
所有 SFT 样本 `answer_correctness/task_success=unknown`，不因为格式通过、没有工具报错或数据集卡片宣称质量而改成成功。

NQ train/dev 含 id/question/golden_answers；HotpotQA 还有 metadata。
训练文件仅使用 QA 字段，不把 HotpotQA metadata 中原始 supporting context 作为 prompt、工具 observation 或难度证据。
官方工具的 Wikipedia 仍是 `PeterJinGo/wiki-18-corpus`，不以 benchmark metadata 替代。
本轮另通过 HTTP Range 只取官方压缩文件前 2 MiB，读取了三个真实 passage：字段为 `id: str, contents: str`。
这验证了实际样本结构；完整约 14 GB 语料和检索性能仍未验证。样本、tar member 路径与 partial SHA256 也已记录。

优先复用已缓存的真实数据。可重复下载入口（只下数据与 tokenizer，不下模型权重）：

```bash
python -m mini.fetch_data --out mini/.cache
python -m mini.curation --input-dir mini/.cache --model mini/.cache/tokenizer \
  --out mini/curated --seed 42 --context-budget 12288 \
  --train-count 800 --val-count 100 --rl-pool-count 600
```

下载失败时明确报错并保留完整缓存，不会自动生成替代数据。离线目录需包含：

```text
sft-train.parquet                       # 或 --sft-file 指定 JSONL/parquet
nq_train.jsonl / nq_dev.jsonl
hotpotqa_train.jsonl / hotpotqa_dev.jsonl
tokenizer/{tokenizer.json,tokenizer_config.json,chat_template.jinja,config.json}
```

线上下载解析并记录 Hub commit。现有缓存由镜像取得，没有可验证的原始 revision，因此使用输入 SHA256，
明确记为 revision unknown。`sft_manifest.json` 记录输入/代码/tokenizer/输出 hash、全部选中 ID、seed 与过滤参数。
同一输入和配置可重跑；输入或参数改变须用新的 `--out`。不要在重跑过程中并行读取尚未写完的输出。

## SFT 审计、Token 与采样

每条 `sft_audit.jsonl` 区分原始字段、解析字段、unknown 字段；保存消息数、assistant/tool turns、name/command 类别、
observation 数、工具错误/输出截断、最终答案/结构终止、模板 token 分区、格式/过滤原因与来源连接。
静态调用官方 command validator 和 Format Gate；不执行来自数据的 shell 命令。
`normal_termination` 是结构终止；没有导出的 generation finish reason，不能证明从未发生内部截断。
工具 stdout 的 `truncated` 与整条 trajectory/context 截断分别统计，默认不因 stdout 截断删除完整轨迹；
可通过 `--reject-observation-truncation` 改变策略。

Token 使用真实 Qwen3.5 tokenizer + 模板，thinking=true、tools=[]，无按字符估算。
prompt/assistant/tool-observation 是互斥消息段且包含各自模板开销；assistant tokens 中单列 trainable tokens。
assistant header 与 system/user/tool 均 mask，边界 BPE 合并的处理与 Mini SFT adapter 一致。
所有 segment 拼接必须和完整模板 token 完全相同。报告 P50/P75/P90/P95/P99/Max 与五档 budget 保留率。

筛选顺序保守：缺失/解析/协议/终止/长度 → 问题与 assistant 轨迹精确去重 → 词面近重复/保留集近重复隔离。
近重复采用确定性 MinHash 候选 + word-set Jaccard；另检查 question+commands+answer 的 action trajectory 近重复。
它不能穷尽语义等价，可能有漏检或误报；`manual_review.jsonl` 供人工复核，不能宣称语义泄漏已完全排除。
全体官方 dev 禁止进入任何训练；SFT validation 也禁止进入 RL。最终 mini test 继续使用上一轮固定 dev 分区，
不会重新抽题挑选模型。不存在原始官方 test 的完整评测承诺。

SFT validation 用来源比例与稳定 hash 保留；训练按 source、tool turns、length、行为 proxy 分层。
来源比例贴近可用数据，各来源内部以 sqrt(stratum_size) 调整覆盖；命令类别用于审计，不机械均分命令数量。
多跳 proxy 仅指多个不同搜索命令，不证明真实多跳推理。修复指观测到 error→新调用→成功工具退出，不代表任务答对。
300/800/1200 条训练候选嵌套共享同一独立 validation；不足时报告实际数量，不重复补齐。

主要输出：

- `sft_selected.jsonl`、`sft_validation.jsonl`、`sft_train.parquet`、`sft_val.parquet`。
- `sft_candidate_{300,800,1200}.jsonl` 与对应配置/ID。
- `sft_curation_report.json/.md`、`sft_curation_charts.png/.svg`、`sft_manifest.json`。
- `qa_audit.jsonl`、`rl_candidates.jsonl`、`rl_dev.jsonl`、`rl_test.jsonl`、`manual_review.jsonl`。

RL 候选池为 600 题，NQ/HotpotQA 各 300。所有原始 QA 都做字段/精确去重检查；
昂贵的词面近重复检查仅用于即将选入的候选，未被检查的非选中 QA 明确标为 not_assessed。
默认排除选中 SFT train 的精确/词面近重复；可用 `--allow-sft-rl-overlap` 允许精确监督重叠并记录 ID。
validation/最终保留集的重叠始终禁止，不能用此参数放开。

## 难度预采样：本轮只运行 plan

```bash
python -m mini.difficulty plan --candidates mini/curated/rl_candidates.jsonl \
  --limit 100 --seed 42 --out mini/profile-plan
```

默认只计划约 100×4 次采样，保存 ID 和总次数，不会导入或启动模型服务器。
扩大范围需显式 `--allow-expand`，应先根据首批真实耗时、token 数、有效桶比例和剩余 200 元总预算决定。
没有 GPU 型号/吞吐/价格实测时不生成费用或训练时长承诺。

未来在有 frozen SFT checkpoint 和本机 vLLM endpoint 后才手动执行：

```bash
# 本轮没有执行。server 必须已经启动、仅 loopback、模型名必须是 checkpoint 的绝对路径。
python -m mini.difficulty run --execute-local-rollouts \
  --checkpoint /workspace/mini/exports/sft --base-url http://127.0.0.1:8000/v1 \
  --corpus data/wiki_18_corpus --candidates mini/curated/rl_candidates.jsonl \
  --limit 100 --temperature 1.0 --seed 42 --out mini/profile-first100
```

每题使用独立 seed 的四条轨迹，checkpoint 仅服务、无优化器更新，记录权重 SHA256。
复用官方 inference agent 与官方 F1 Reward 公式（format gate、length penalty off），
不引入新 teacher、不重实现 GRPO。此 profiler 使用官方推理的一次纠错语义，**不等同 RL 的 terminal-invalid loop**；
报告同时记录语义差异。Response Tokens 定义为 assistant+tool 内容 BPE 数（不含 chat header），
不能冒充 RL 的 `response_ids` 数。统计 seed、EM/F1/reward、连续方差、格式/工具合法率、失败/截断、tool 数和 wall time。
基础设施 API 错误直接中止，不能被记成难题。

Easy=4 次 EM 命中，Medium=1–3 次命中。主要格式/协议失败优先归 Protocol Failure。
4 次 EM=0 只有在协议完整且 `retrievable=true`、有 `retrievability_evidence` 时才归 Hard；
其余保留待复核桶，不能因为有 Gold Answer 就声称在该 Wikipedia 上可检索。
人工确认可检索后，在一份新候选文件中添加这两个字段并记录证据，使用新输出目录。
四次样本只是粗估，不是精确成功率；F1/reward 方差始终保留。

```bash
# 从真实测量的 raw traces 重新分析，不做新采样：
python -m mini.difficulty analyze --candidates mini/curated/rl_candidates.jsonl \
  --rollouts mini/profile-first100/profiling_rollouts.jsonl --out mini/profile-analysis
# 用已确认 profiles 调整桶比例或 seed，uniform 使用同一 eligible profiled population：
python -m mini.difficulty sample --candidates mini/curated/rl_candidates.jsonl \
  --profiles mini/profile-first100/rl_profile.jsonl --count 300 --ratios .15 .70 .15 \
  --out mini/rl-sampling
```

桶不足时从剩余可用桶不重复借用，仍不足则留 shortfall，报告实际比例。
预采样前的 `rl_uniform.jsonl` 是 candidate-pool baseline；预采样后的 uniform 才是难度采样的可比 baseline。
本轮 `rl_profile.jsonl`、`rl_selected.jsonl` 保持空，report/manifest 标记 pending，不填虚构难度。
采样输出只包含 QA 信息，GRPO 训练必须重新生成 on-policy trajectories。
Mini launcher 也拒绝把含 `rollouts/raw_record/messages/output` 的 profile cache 作为 RL 训练输入。

## 训练器输入（仅准备命令）

旧版默认路径保持不变，新导出路径可显式指定：

```bash
python -m mini.launch sft --dry-run --sft-train-file mini/curated/sft_train.parquet \
  --sft-val-file mini/curated/sft_val.parquet --sft-length 12288 --eval-freq 10
python -m mini.launch rl --dry-run --model mini/exports/sft \
  --rl-train-file mini/rl-sampling/rl_selected.jsonl --rl-dev-file mini/curated/rl_dev.jsonl
```

训练时把 --dry-run 去掉属于未来 GPU 阶段，本轮未执行。长度 12288 用于静态覆盖；
低预算时可以另开 8192 budget 筛选配置，不能直接截断已选完整长轨迹。
新路径同时支持 SFT_TRAIN_FILE/SFT_VAL_FILE/RL_TRAIN_FILE/RL_DEV_FILE 环境变量。

## 诊断定义与边界

`mini.observability` 只包裹原始入口、读取返回结果，不替换 Loss/Reward/Tool Protocol 或重新计算 Advantage。

| 指标 | 来源与限制 |
|---|---|
| Reward / EM / F1 / Format / Failed | 官方 reward_extra_info；raw EM/F1 与 format-gated Reward EM/F1 分列 |
| 无答案、tool 数、assistant turns、tool errors | 从真实 response transcript 静态解析；无错误不代表成功 |
| Tool latency、generation time | 官方 AgentLoopMetrics timer，含调度/工具处理开销，不伪称纯 subprocess 用时 |
| Group mean/std、flat/all-zero/non-flat | 原始 compute_advantage **返回后**读取 token_level_rewards.sum(-1)，按 UID 分组；包括真正训练使用的 Reward/KL |
| All-success EM group | 独立使用实际 EM 数组；缺失即 null，不替代 flat 判断 |
| Advantage 均值/std、正/负/零比例 | 读取原始 advantages，并用 response_mask 排除 Observation 与 padding |
| Loss/grad/clip/LR/PPO approximate KL | 保留官方 metric key；未提供则 unavailable，不填 0 |
| Reference KL loss | 官方默认关闭，`actor/kl_loss` 初始化的 0 不当作测量值；只显示实际 PPO approximate KL |
| Step time | 官方 timing_s/step；inter-log wall time 单独命名，首条 null，不当作官方 step timer |
| 显存 | 现有 GPU Smoke Test 的 allocator peak 与 nvidia-smi samples；本轮无 GPU 数据 |

`agent_trajectory` 中的 Reward 是 rollout score（训练 KL 之前）；`grpo_groups` 才基于实际进入 Advantage 的训练 Reward。
标准差均为 sample std，flat 判断 max-min<=1e-12，阈值在结果中记录。
`scope=training_rollout/validation` 由官方 validate 标志确定；optimization 日志单独 scope。
Final Evaluation 需要明确指定，不能混入训练曲线。

```bash
# 未来已有实际 rollout dumps 后：
python -m mini.diagnostics --scope training_rollout \
  --input mini/runs/rl30/rl/train_rollouts/1.jsonl --out mini/analysis/train
python -m mini.diagnostics --scope validation \
  --input mini/runs/rl30/rl/val_rollouts/10.jsonl --out mini/analysis/validation
python -m mini.diagnostics --scope final_evaluation \
  --input mini/runs/eval-test/grpo.jsonl --out mini/analysis/test
# 也可显式传入 observations 下各 PID.jsonl；scope 不符的事件不会混合。
```

文本 dumps 没有足够 UID/advantage/mask 信息时，GRPO 指标为空，不根据 EM 或输出顺序猜分组。
训练容器的 hook 集成仍须未来 GPU smoke；本轮 CPU fixture 只验证公式统计、scoping 和原函数返回值保持不变。

## CPU 检查

```bash
MINI_TOKENIZER=mini/.cache/tokenizer python -m pytest mini/tests -q
python -m mini.check_curation --data-dir mini/curated --model mini/.cache/tokenizer
```

覆盖异常记录/消息角色/工具 JSON、静态命令、模板 token 与实际 mask 一致性、截断区别、来源缺失与泄漏、
确定性/不重复采样、独立四采样检查、可检索性 unknown、桶不足、Reward 与 EM 的 flatness 差别、mask padding、
未启用 KL 的占位值、runtime scope 与保持原返回值。测试用的小 fixture 不进入真实数据审计或报告。
