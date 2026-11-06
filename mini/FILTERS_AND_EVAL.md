# 具体筛选规则、长度预算与 5×30 评测

此文描述当前代码实际行为。数据筛选验证的是结构、长度、重复和覆盖度；SFT 答案正确性尚未验证。

## 1. SFT 的确定性流程

输入是官方 10,000 条 id/messages。以下统计取自已生成的 `curated/sft_audit.jsonl`：

| 步骤 | 判定条件 | 本次排除 |
|---|---|---:|
| 原始结构 | JSON 可解析；id 非缺失；messages 为字典列表；所有 content 是文本 | 0 |
| 问题与输出 | 有非空 question 或首个 user/Query；至少有非空 assistant 输出 | 0 |
| 角色顺序 | 初始 system→user，之后 assistant 调用→tool observation 配对；不能留下 pending call | 0 |
| Tool Schema | 每个 assistant 最多一个 action；JSON 可解析，name=shell，arguments 恰有非空字符串 command | 0 |
| 静态命令 | 官方 `_validate_pipeline(command)` 接受；没有执行命令 | 0 |
| Observation | JSON 包含 command/stdout/stderr/exit_code/timed_out/truncated/information_lines | 0 |
| 结构终止 | 最后一条是 assistant，含非空 answer，末尾是 `</answer>`；原始 finish_reason 若为 length 则拒绝 | 0 |
| 格式门控 | 官方 `_is_valid_sequence` 接受 think/reason→tool_call→tool_response→最终 answer 的结构 | 0 |
| 长度 | 实际 tokenizer+template 的完整序列长度 ≤12,288 | 0 |
| 精确重复 | 不能重复原始 ID、规范化问题或完整 assistant 文本指纹 | 0 |
| 与保留集近重复 | 问题 word-set Jaccard≥0.90 的 MinHash 候选保守隔离 | 15 |
| SFT 内部近重复 | 同样的 question Jaccard≥0.90；或 question+commands+answer action trajectory Jaccard≥0.97 | 12（均为问题规则） |

问题规范化是 `casefold → re.findall(r'\w+') → 空格拼接`；精确轨迹指纹对拼接后的 assistant 文本做同类规范化再 SHA256。
近重复是 MinHash 的 12 个签名、6 个 band 候选，再算真实词集合 Jaccard：`|A∩B|/|A∪B|`。
LSH 会漏检语义改写；这些阈值是词面筛查策略，不能当作语义等价证明。

因此漏斗为：`10000 → 10000（结构/协议/长度）→ 9985（保留集近重复）→ 9973（内部近重复）`。
27 条是隔离待审，不意味着已证明它们错误或应永久删除。
目前 Observation 只检查字段存在，未逐项验证字段类型，也没有验证 observation 事实能否支持最终答案。
工具非零退出会计数；默认不单独删除这些轨迹，以便保留修复行为。stdout.truncated 不等于轨迹截断。
Format Pass 不验证事实正确，SFT 没有逐条导出的 gold/judge label，因此没有 EM/F1/Reward 门槛。

## 2. 9973 条怎样选成 800+100

先用 seed=42 按 source 比例和 ID 哈希保留 100 条 SFT validation。
剩余样本用 seed=43 排序，stratum 是：

```text
(source,
 min(tool_calls, 4),
 length_bin ∈ {≤4096, (4096,6144], (6144,8192], (8192,12288], (12288,16384]},
 multi_hop_proxy,
 repair_observed)
```

`multi_hop_proxy = tool_calls≥2 且 distinct_command_count≥2`，只描述搜索行为。
`repair_observed` 指工具 error 后出现新的合法调用，随后工具退出为 0；不证明最终答对。
来源之间按可用量 N_s 的比例选择；每次挑 `(已选_s+1)/N_s` 最小的 source。
来源内部按 `sqrt(N_h)` 对分层桶分配覆盖；每次挑 `(已选_h+1)/sqrt(N_h)` 最小的桶，桶内按 SHA256(seed,id) 排序。
命令类别没有单独等量配额；分层代表的是覆盖，不是模型难度。

当前 800 条实际构成：

| 维度 | 数量 |
|---|---:|
| HotpotQA / NQ | 401 / 399 |
| 1 / 2 / 3 次工具调用 | 209 / 435 / 156 |
| ≤4096 / 4096–6144 / 6144–8192 / 8192–12288 tokens | 477 / 240 / 68 / 15 |
| 多次不同搜索 proxy | 521 |
| 可观察修复行为 | 0 |

300/800/1200 候选是同一训练顺序的前缀，共享独立 validation；没有复制样本补数量。

RL 先从 NQ train 79,168、HotpotQA train 90,447 条中检查问题/gold 有效性、精确去重和已知训练/保留集重叠。
对即将入选的问题再做词面近重复检查，确定性选择 NQ/HotpotQA 各 300。没有按模型成绩筛题。
Easy/Medium/Hard 仍需冻结 SFT 的 4 次真实采样；Protocol Failure 与无可检索性证据的问题不当作普通 Hard。

## 3. 12,288 的含义和依据

12,288 是静态 SFT **完整序列**上限，用于保留长度尾部。
这批数据最大长度为 11,956，所以选了覆盖它的 12 Ki-token 档位；不是显存、速度或效果实验证明的最优长度。

| 完整序列 budget | 原始 10k 的长度保留率 |
|---|---:|
| 4096 | 76.53% |
| 6144 | 97.78% |
| 8192 | 99.84% |
| 12288 / 16384 | 100% |

总长度 P50=3182、P95=5498、P99≈6749、Max=11956。
Assistant 段 Max=838，而 Tool Observation 段 Max=10865；长轨迹主要由检索内容占用。
筛到 800 条时长轨迹有意被覆盖为 15 条，不能用原始 0.16% 的长尾比例推算此子集。

| 环节 | 当前限制 | 解释 |
|---|---|---|
| SFT curation | total≤12288 | system/user/assistant/tool/template 的完整序列 |
| SFT launcher | 默认 total=9216，可显式 `--sft-length 12288` | SFT 没有独立的生成 Response cap；用当前 12k 导出训练必须显式匹配预算 |
| GRPO prompt | ≤1024 | 初始 system+question+generation prefix |
| GRPO response | 8192 | 初始 prompt 之后的整条多轮延伸，含生成与工具 observation |
| GRPO total | 1024+8192=9216 | 由启动器同步 actor/ref/rollout budgets |
| 工具 stdout | 每次最多2048 tokens | 不是整个 observation JSON 的2048；JSON 还包含 information_lines 等字段 |
| GRPO 工具/Assistant 轮次 | 最多5次工具、6次 Assistant | 保留官方 turn budget |

8192 是最初 Mini 任务指定的起始 Response 值，尚未经过 GPU 调优。它不是每轮都能生成8192，也不只统计思考文本。
官方 observation JSON 同时含 stdout 与 information_lines，后者可能重复结果，因此2048 stdout并不意味着2048完整观察。
SFT 的12k与 RL 的9216并非完全对齐；若单卡预算紧，可另建 SFT total=8192 的筛选快照，避免直接截断。
不在本轮擅自覆盖已冻结的12k数据。Response 是否降为4096，需在真实 smoke 后看截断率、组内方差和显存。

## 4. 五个数据集各30：固定 Pilot 评测

适合当前预算下先验证完整链路、查看大幅退化/提升及失败模式。单集30题每答对一题就是3.33个百分点；
15/30 的 EM=50% 时 Wilson 95% 区间约[33.2%,66.8%]，不能以几个百分点宣称稳定提升。
建议按同一150题配对比较 Base/SFT/GRPO，总计450条单次轨迹，temperature=0，固定seed。
可用每集≥100作为之后的扩展检查，但这不是本轮已执行或已承诺的GPU工作。

本轮已核对实际文件并固定：

| Dataset | 实际 split | 原始行数 | 本轮评测 |
|---|---|---:|---:|
| NQ | test | 3610 | 30 |
| HotpotQA | dev | 7405 | 30 |
| TriviaQA | test | 11313 | 30 |
| 2WikiMultiHopQA | dev | 12576 | 30 |
| Bamboogle | test | 125 | 30 |

先排除已知全部 SFT10k、RL候选、SFT validation、RL dev和既有mini test中的规范化问题，再按 seed+source+id+question 哈希选题；
也检查词面近重复及五个数据集之间的重复。共有10,866个不同的已知问题参与排除。
没有按 gold 内容、模型输出、Reward 或预采样难度选择。所有输入有SHA256，样本ID固定。
这150题作为最终 pilot set，不用它选checkpoint；checkpoint选择仍只用既有Dev。
Wikipedia-2018对这些问题的覆盖尚未实测，不能在模型答不出后换掉题目。

## 5. 评测指标的精确定义

| 指标 | 分子、分母与处理 |
|---|---|
| EM | 官方归一化与 max-over-gold；包括全部30题，错误/超时不静默删除 |
| Token-F1 | 官方 token F1，取 gold aliases 中最大值，再对全部30题平均 |
| Format Pass | 官方 Format Gate，检查已记录的完整 assistant/tool transcript，沿用缺失起始think补回规则 |
| Avg Tool Calls | 实际工具 executor 调用次数之和 / 30；解析失败而未执行的 request 另记 invalid |
| 成功轨迹 | EM=1、Format Pass=1、有非空最终答案；经过修复的工具错误仍可成功 |
| 成功生成长度 | 对成功轨迹统计服务API的 completion_tokens，含thinking，排除工具观察；报告 mean/median/P95/observed_n |
| 成功 context 增长长度 | 成功轨迹最终完整模板 token 数减初始 system+user模板长度；包括工具观察和后续模板开销 |
| 异常率 | 至少出现一种异常的轨迹数 / 30；不是分项异常率的相加 |
| 异常分项 | 格式失败、非法工具、工具非零退出、超时、无最终答案、context/response截断、turn budget耗尽、API错误等 |

API usage 缺失时生成长度是 null；content-only BPE 另存字段，不冒充真实生成token。
零成功时成功长度为 null，不能填0。工具 exit=1 可能只是无匹配，非零退出标记不等于答错。
异常可以与任务成功同时出现：比如先工具出错、后恢复并答对。

总体报告 pooled micro 和五数据集等权 macro。每集都30时，EM/F1/Format/平均工具数/轨迹异常率的 macro=micro；
成功长度按实际成功轨迹数汇总，invalid tool rate按实际request数汇总，不能机械平均五个均值。
EM、Format Pass、异常率另给Wilson区间。完整分项、分母和可用测量数量保留在JSON。

统一评测仍调用官方 inference agent。现在客户端在每次生成前核算累计 response budget：
`remaining = 8192 - (当前templated prompt tokens - 首轮templated prompt tokens)`，并与总context剩余量取min。
工具行为保持原样：最后一次工具Observation可能把预算占满，此时不再发送下一次模型请求；
若最后一次无后续生成的Observation溢出，单列 observation_budget_overflow，不伪称模型被截断。
推理的一次纠错语义与 RL terminal-invalid 语义仍不同，所有 checkpoint 使用同一评测策略。
官方 inference 只限制6次 Assistant，未单独强制5次工具；没有最终答案时可能出现第6次工具调用，这会如实计入。

## 6. 可执行命令（本轮仅准备与CPU检查）

```bash
python -m mini.fetch_data --include-benchmark --out mini/.cache
python -m mini.benchmark30 --input-dir mini/.cache --model mini/.cache/tokenizer --out mini/benchmark30
python -m mini.benchmark30 --out mini/benchmark30 --check-training \
  mini/curated/sft_train.parquet mini/curated/rl_candidates.jsonl
```

实际输出：`mini/benchmark30/test.jsonl`（150题）、逐数据集文件、`source_audit.json`、`benchmark_manifest.json`。
本轮未运行下面的模型评测：

```bash
python -m mini.evaluate --data-dir mini/benchmark30 --split test \
  --checkpoints mini/checkpoints.json --selection mini/selection.json \
  --corpus data/wiki_18_corpus --out mini/runs/benchmark30-final \
  --prompt-length 1024 --response-length 8192 --context-length 9216 --seed 42 --serve-locally
```

`summary.json` 和 `summary.md` 分别报告 Base/SFT/GRPO 的五个数据集、pooled总体、macro和上述指标。
三个checkpoint顺序加载，单GPU复用。使用外部server时，它本身的max_model_len也须一致。
