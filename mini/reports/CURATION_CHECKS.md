# 本轮 CPU 检查结果

- 68 项测试通过，1 项 Windows 不适用的 /bin/bash 执行测试跳过。
- 真实 SFT：10,000 条全量审计，静态筛选后 9,973 条；选中 800 train + 100 独立 validation。
- 900 条导出轨迹通过实际 Qwen tokenizer 的完整 token 拼接、Assistant/Observation mask 与 budget 检查。
- 600 条 RL 候选：NQ/HotpotQA 各 300；冻结 dev=100、final mini test=200。
- 各导出划分规范化问题精确重叠为 0；300/800/1200 训练候选嵌套，且都不含独立 SFT validation。
- 导出 hash 检查全部通过。官方 630 个源文件与原始 archive 字节一致。
- 27 对词面近重复待人工复核；没有宣称语义等价已全部排除。
- 首批 100×4 难度采样只生成计划；Profile/Selected 为空并标注 pending。
- 没有执行训练、GPU Rollout、付费 LLM API 或租用 GPU；没有训练曲线/难度分布的虚构结果。

实际 QA 样本、corpus 三条 preview、数据结构见 SOURCE_SCHEMA_SAMPLES.json；完整数据结果见 ../curated/sft_curation_report.md。
