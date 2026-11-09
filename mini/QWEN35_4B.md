# Qwen3.5-4B：冻结数据适配

本阶段只做 CPU 数据适配，不训练、不采样、不执行语料工具。模型为账户已列为可用的
`Qwen/Qwen3.5-4B`，官方 tokenizer/config 固定版本为
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`；官方元数据见 `qwen35_4b_source.json`。

## 快照内容

输出为 `mini/curated-qwen35-4b/`。使用既有冻结选择：SFT train=800、val=100，
保留嵌套的 300/800/1200 候选，共验证 1300 条唯一 SFT 轨迹。
保留 RL candidates=600、dev=100、mini test=200、uniform=300，以及原五数据集各30题的 pilot set。
原 `mini/curated/`、`mini/benchmark30/` 和冻结选择继续作为来源留存。

采用实际 4B tokenizer：`tools=[]`、`enable_thinking=True`。
逐条检查完整模板 token 与每轮 prefix 拼接一致；system/user/tool 和 assistant header 的 mask 为0；
每个 assistant 有实际监督目标；自回归 input/target/mask 使用一致的一位 shift。
每条轨迹另与实际 2B tokenizer 的 token IDs 和完整 mask 数组比较。

词表文件相同，但模板和 tokenizer_config 不同。模板差异是：未传 `enable_thinking` 时，
4B 模板默认开启 thinking，而既有2B模板默认关闭。这里所有调用都显式传 true，不能依赖默认值。
SFT 和 QA 原始文字均保持原样；没有为凑长度截断或重新抽样。

快照中：

- `sft_train.parquet` / `sft_val.parquet`：原 id/messages 内容，已用4B tokenizer验证。
- `token_audit.jsonl`：1300条轨迹的新旧 token 统计与逐token/逐mask比较。
- `prompt_audit.jsonl`：RL候选、dev、mini test、uniform与150题 pilot 的实际4B初始prompt长度。
- `validation_report.json`：统计与边界；`sft_manifest.json`：来源、固定模型版本、tokenizer、代码与产物Hash。
- `benchmark30/test.jsonl`：原始冻结文件逐字节保留；新的4B prompt长度单独记录，旧manifest只用作原始选题来源。

## 长度与使用边界

SFT 完整序列预算为 12288，RL初始prompt为1024，后续完整延伸（含工具观察）为8192，
RL总上下文为9216。它们是数据适配阶段的历史检查阈值。
后续确认的 PyTRIO 后端将 SFT / RL 总上下文统一为12288、RL延伸提高到11264（包含工具观察）。
学习率、LoRA rank、batch/group size与真实运行记录见 [后端文档](pytrio_backend/README.md)。
本页的QA检查只覆盖初始prompt；多轮预算由后端逐请求核验。

800条SFT主训练集与600条RL候选、各保留集保持隔离。
1200条SFT扩展候选中有1个问题与既有RL候选重叠；若采用1200条方案，后续必须重新构建对应的RL排除集。
本次保留所有原ID，将此项明确报告，不更改冻结的主方案。
正确性、语料可检索性和粗难度仍不由token验证证明；SFT答案正确性保持unknown。
按后续决定，不实施离线难度profile，改用训练时dynamic sampling。

## 重现

在准备好的 `GrepSeekUbuntu` 中，激活 `.venv`。下载只允许tokenizer文件，不下载权重：

```bash
HF_ENDPOINT=https://hf-mirror.com python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    'Qwen/Qwen3.5-4B', revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
    local_dir='mini/.cache/qwen35-4b-tokenizer',
    allow_patterns=['tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'config.json'],
)
PY
python -m mini.retokenize --out mini/curated-qwen35-4b
```

脚本按官方元数据的LFS SHA256或Git blob SHA1核验每个文件；只能使用本地tokenizer，不会网络回退。
输出目录必须不存在，所有数据检查完成后才导出；最后写manifest作为完成标记。
再次重现须使用新目录（例如 `--out mini/curated-qwen35-4b-repeat`）。

```bash
MINI_TOKENIZER=mini/.cache/qwen35-4b-tokenizer python -m pytest mini/tests -q -rs
python -m mini.check_curation --data-dir mini/curated-qwen35-4b \
  --model mini/.cache/qwen35-4b-tokenizer --out mini/reports/qwen35-4b-checks.json
python -m mini.benchmark30 --out mini/curated-qwen35-4b/benchmark30 --check-training \
  mini/curated-qwen35-4b/sft_train.parquet mini/curated-qwen35-4b/rl_candidates.jsonl
```

PyTRIO SFT/RL入口、保存/重载与真实搜索小验证现已完成，见
[真实验证报告](reports/pytrio-smoke/README.md)。本页保留数据快照的来源与适配过程。
