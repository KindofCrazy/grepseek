# 并发、搜索加速与评测续跑

所有模型调用共用一个有界调度器，避免同题分支与跨题并发相乘导致失控。

- `question_concurrency` 控制同时推进的评测问题或RL候选组。
- `sample_concurrency` 控制整个run内的远端采样RPC，包括首轮和后续分支。
- `search_concurrency` 控制本地检索，`concurrency` 控制同题分支线程数。
- 评测补充已完成问题的空位；RL按预定题序接纳完整组，避免只挑快题。
- 无效RL组可以在慢题完成前触发所需补采；保持同一策略和确定的题目/seed顺序。
- 异常时等待在途任务结束，再记录预算和保存，不把基础设施错误当作答错。

默认是4题、4个采样RPC、1路搜索。资源充足时可单独验证更高上限；
并发阈值不是服务端能力保证，也不能直接推断整轮训练的加速倍数。

## PACE / SPCS

默认使用PACE，支持保守的SPCS分片回退及原生沙箱。缺少索引或绑定时明确报错。
head命令可单独扩大候选行、字节、倒排与suffix-array预算；其他命令保持保守限制。
LRU缓存有数量与字节上限，相同在途命令会合并等待。索引和语料必须经过hash绑定，
语料发生变化立即停止，定期对照原生命令的完整payload，差异视为环境错误。
缓存命中和命令合并不冒充PACE命中。详见 [PACE](pytrio_backend/PACE.md)。

## 评测断点续跑

每条完整轨迹持久化后flush+fsync。恢复可追溯多段日志链，验证日志hash、
问题/答案、原serial/seed、推理设置和源码。只容忍崩溃尾部的一条不完整JSON。
已完成题不重复采样，未结算请求按原预留工作量保守计入，预算不因重启清零。

```bash
python -m mini.pytrio_backend.continue_eval \
  --prior mini/runs/previous-eval \
  --schedule mini/curated-formal/benchmark500.jsonl \
  --out mini/runs/continued-eval --execute
```

跨版本恢复需要显式验证过的源码迁移收据，不能忽略源代码差异。
服务端数值运算与运行负载变化可能影响生成结果，不承诺逐位复现。

训练期间不自动评测。阶段末或最终模型使用明确指定的冻结题集，
开发集与多数据集benchmark分别报告；主机队列脚本和本次实验产物留在本地。
