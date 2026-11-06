# 本轮源码与真实数据结构审查

审查范围：官方 `sft/data_generation/to_parquet.py`、`sft/run_sft.sh`、数据生成 pipeline、
`rl/prepare_rl_data.py`、QA loader、GrepSeekDataset、Reward、agent loop、GRPO config、`inference/`。
官方版本仍为 `mini/UPSTREAM.json` 中的固定提交；本轮只改 Mini 包。

- SFT parquet 实际为 10,000 行、`id/messages` 两列，messages 是 role/content struct 列表。已读多个真实样本，并全量审计。
  `train_...` ID 本身不能证明来自 HotpotQA；实际问题连接结果为 5000 HotpotQA、4999 NQ、1 unknown。
- 原始 Question、Source、Gold Answer、生成结束标志、正确性标签不在 SFT 导出中。Question 与来源另列解析 provenance；
  所有 SFT 答案正确性保持 unknown。数据生成代码中的 judge 不等于当前导出保留了逐样本证据。
- NQ 实际字段为 id/question/golden_answers；HotpotQA 另有 metadata，含 type/level/supporting_facts/context。
  QA 全量读取时不把这些 benchmark metadata 当 retrieval observation，也不据此宣称 corpus retrievable。
- 官方 parquet 转换器只保留 id/messages，可选 tools 列。本 Mini 保留 schema 在 system prose，不加 tools 列，避免另一套 native XML。
  原 SFT trainer 监督 assistant，mask system/user/tool；Mini adapter 继续做严格 token 拼接与 mask 校验。
- 官方 RL QA loader 接受 QA JSONL；额外 source 字段不进入策略 prompt。对 malformed/empty 基础字段，Mini 输出明确审计结果。
- 官方 Reward 实际配置为格式门控 F1、length decay=false、无 KL reward；不依据文件顶部旧注释解释为 EM×decay。
  GRPO group flatness 必须依据实际 token_level_rewards，而非 EM。
- 官方 agent loop 记录 response_token_count/hit_max_context，并对结构非法 request 终止；官方 inference 允许一次纠错。
  profiling 复用 inference，明确记录这一区别；没有冒充与训练循环逐步等价。
- vLLM/Qwen3.5、FSDP、真实优化更新仍待 GPU；本轮没有运行。
- Wikipedia 实际 preview 来自官方 wiki-18.jsonl.gz 前 2 MiB HTTP Range，解出三个真实 passage，字段为 id(str)/contents(str)。
  未下载完整语料、未执行搜索命令。

真实 Schema、样本和 Wikipedia preview 见 [SOURCE_SCHEMA_SAMPLES.json](SOURCE_SCHEMA_SAMPLES.json)。
输入 revision 无法从既有镜像缓存可靠恢复，使用 SHA256 留证；在线 fetch 入口会解析实际 Hub commit。
词面近重复筛查不保证覆盖所有语义等价问题，人工复核列表为必要的后续工作。
