# 实验记录

`RunLog`同时保存本地JSONL与可选SwanLab标量。RL CLI未指定时使用私有online实验，
SFT/eval默认offline；可通过 `--swanlab online|offline|disabled` 配置。
联网初始化失败允许回退offline。认证来自SDK凭据存储，不把密钥写进配置或源码。

## 保存内容

- 配置、依赖版本、源代码与数据manifest哈希、平台能力快照。
- 完整轨迹的token/logprob、assistant mask、工具调用与结果、结束原因。
- 原始EM/F1、格式有效率、实际奖励及各奖励分量、未完成率、工具效率诊断。
- 候选/有效组数、零方差组、补采及排队耗时、优势分布、PPO ratio/KL/clip统计。
- 采样/搜索并发峰值、缓存和检索路由、原生一致性抽查、步骤各阶段耗时。
- 累计与本次新增token消耗、费用估算、checkpoint和优化器提交/完成状态。

梯度范数、完整策略熵、GPU显存、实际账单仅在平台确实返回时记录，未知时为null。
采样token的surprisal单独命名，不当作完整策略熵。被丢弃的候选同样计入费用。
完整指标定义见 [后端指标](pytrio_backend/METRICS.md)。

`trainer_created` 表示client已创建，不等于惰性恢复完成；`optimizer_state_loaded`
表示显式完整状态加载已返回，`trainer_ready` 在采样器真正就绪后记录。

## 使用

```bash
python -m mini.pytrio_backend rl --config local-rl-config.json \
  --out mini/runs/new-rl --swanlab online --execute
# 已有离线日志可通过SDK补传：
swanlab sync mini/runs/previous-run/swanlab
```

只自动上传标量，完整命令、observation和轨迹保存在本地JSONL。
生成报告与运行目录默认不入Git；checkpoint JSON通常是远程权重引用，并非权重本体备份。
费用是按未缓存token估算，不是账单；CPU租金与控制API无法由token日志完整覆盖。
