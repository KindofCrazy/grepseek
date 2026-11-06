# 源码审查与验证边界

审查日期：2026-11-05；源码与来源 hash 见 `UPSTREAM.json`。所有官方文件与下载 archive 字节对比一致。

| 范围 | 静态结论 | Mini 处理 / 仍待验证 |
|---|---|---|
| `sft/run_sft.sh` | torchrun 无 >=2 硬性限制；SP 默认 NPROC，batch 必须能被 DP/micro 整除 | Mini 显式 GPU=1、SP=1、global batch=4；FSDP 单卡初始化/offload 仍待 GPU |
| `rl/run_rl.sh` / `grpo_trainer.yaml` | 默认 GPU=4、SP=2，继承 rollout TP 默认=2；直接 NPROC=1 不成立 | 同步设 GPU/FSDP=1、SP=1、TP=1；真实 Ray/weight transfer 待 GPU |
| 模型加载 | 2B config 是 `qwen3_5` / `Qwen3_5ForConditionalGeneration`，不能简单假定 AutoModelForCausalLM | 官方 FSDP loader 按 AutoModel mapping 选类，vendored veRL 有 Qwen3.5 monkey patch；用官方固定 transformers build |
| FSDP Wrap | RL 已指定 `Qwen3_5DecoderLayer`；SFT 原始脚本依赖默认 policy | Mini SFT 显式指定 decoder wrap；单卡 NO_SHARD/hybrid、host offload 待真实执行 |
| vLLM | 官方 eager + language_model_only=true；text-only agent 跳过 vision processing | 原样复用；2B GatedDeltaNet cache、tied weights、传权仍待 GPU，不能靠参数量推断兼容 |
| Chat Template | 官方 helper 已有 dummy-user fallback，可处理单消息模板异常；原生 tool schema 则注入 `<function=...>`，与 Hermes JSON 不同 | 禁止 tools 列/原生 schema 注入；Mini 完整前缀渲染严格检查 token 等价；实际 tokenizer CPU 验证；thinking 显式 true |
| Loss Mask | SFT 仅 assistant target 监督，header masked；RL `response_mask` assistant=1、tool/user=0 | Mini adapter 屏蔽同类 token；真实 tensor/collator/反传集成留 GPU 容器验证 |
| Tool Protocol | `shell(arguments.command)`，logical corpus.jsonl→wiki_corpus.jsonl；JSON observation payload、2048 stdout tokens、单 pipeline whitelist | 官方代码原样复用；不要把 validator 当完整安全沙箱，运行时用只读语料挂载可限制文件访问 |
| 不一致之处 | RL 非法请求直接终止；inference 允许一次纠错。命令级验证返回 exit_code=-2，不等同 request-invalid 标志 | 保留原始语义；评测把两类错误都计入 invalid rate，日志分别保留 |
| 输出截断 | RL `_truncate_tokens` 有 chars<=3×tokens 的快捷路径，该假设并非任意 Unicode/BPE 的严格上界；工具输出与上下文截断是两回事 | 不修改官方工具行为，记录此限制；context metric 不混用 stdout.truncated |
| Reward | 实际为 format-gated F1、length decay=false，而非文件开头旧注释的 EM×decay；缺失首个 `<think>` 会补回 | 不改函数/参数；格式非法即使 EM=1 也为0。无答案统计独立检查 `<answer>`/prediction，因为 reward 的 legacy fallback 不完全等价 |
| `prepare_rl_data.py` | seed 可复现但不处理 SFT/QA 跨 split 重复；某些 count>=len 返回全部，不是严格目标数 | 新增严格计数、规范化问题去重、dev/test 保留、manifest 和 fail-fast；不改官方数据加载接口 |
| Context | response budget 含工具 observation；减少 response 会改变可用检索轮数 | 同步 data 与 rollout prompt/response、max_model_len、actor/ref/rollout token budgets、Reward L_max；SFT 单独重筛 |
| GRPO batch | actor worker 将 PPO mini batch 乘 rollout.n 再除 DP；真实 train batch 必须除 GPU count | 4 prompts×4=16 trajectories；PPO 4 prompts=16 trajectories；1卡每 DP16、2卡SP1每 DP8，micro1 均整除；group>=2 |
| `inference/` | 官方 agent/scoring/tool 执行原样调用；需要 Linux fcntl/bash | 统一客户端统计六项指标并保留完整轨迹；端点模型名核对，test 要求 frozen selection |

实际模型定义来源：[Qwen3.5-2B config](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json)、
[Chat Template](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/chat_template.jinja)、
[vLLM 官方 recipe](https://github.com/vllm-project/recipes/blob/main/Qwen/Qwen3.5.md)。
`qwen35_chat_template.jinja` 是实际模板的本地测试副本，通过镜像下载并与官方页面逻辑核对；真实训练使用 checkpoint 自带模板。

CPU 通过只能证明数据、模板、配置和纯函数路径；没有执行模型 forward/backward、FSDP 分布式、vLLM rollout 或 GRPO 更新。
完整 Wikipedia 未下载，真实大语料 shell 性能与 timeout 尚未验证。
GPU 验收要求：先 SFT 2–5 steps 并导出/重新加载 HF 权重，再 GRPO 2 steps，检查 observations 与原始日志、
显存（训练 allocator + nvidia-smi）、step wall、reward/std/zero advantage、loss/approx KL/clip；确认实际工具调用过官方 corpus。
如模型自行回答而没有任何工具调用，更新可算 GRPO 更新，但尚不能认定检索链路已验证；需另用固定真实命令验证语料工具路径。
未观察到这些证据前，不得称“单卡已跑通”或填任何训练结果。
