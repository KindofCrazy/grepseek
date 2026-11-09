# Agent监控口径

对应用户补充要求，2026-11-07。数据层级为step聚合、完整trajectory、逐次tool_call事件。
SFT/eval默认offline；RL CLI未显式指定时默认online，只上传标量。
JSONL保存分子、分母、关联ID和原始结果。具体设置见 [实验记录设置](../EXPERIMENT_LOGGING.md)。

- 工具执行成功：执行调用中无超时/诊断stderr、退出码0，或搜索命令无结果的退出码1。
  管道的最终退出码可能掩盖上游错误，因此stderr非空仍作为诊断错误记录。
- 空搜索：检索命令正常执行但stdout空；`wc -l`的0也为空，不把字符串"0"当命中文档。
- 非空检索：正常的非count-only检索返回非空内容；这是检索结果存在，不证明有用证据。
- 工具错误率：实际执行错误+执行前非法请求，禁止命令返回-2与解析器标记不重复计数。
- 调用决策率：含工具调用tag的assistant轮数/全部assistant轮数；不是从完整概率分布推导的行动概率。
- 查询去重：shell token/空白规范化后的完全相同命令；不声称识别同义搜索。
- 工具类型：管道中的rg/grep/head等程序出现次数，分母是程序总出现次数；不把一个管道当多次shell调用。
- 正确回答：EM=1、官方format gate通过、存在最终答案；按工具调用次数统计EM/F1时保留每组题数。
- 长度：生成token与上下文增长分开；后者包含observation。成功条件均值和全部开销/成功数分开。
- 延迟：从同题首轮采样开始到分支结束，包含采样、排队和工具时间；每次工具额外保存wall与p50/p95。
- 奖励：保留原始F1、format、实际total；效率奖励未启用为null/disabled。
- RL：保存advantage标准差/正负比例、零方差组、补采、丢弃与全部候选成本，PPO ratio/approxKL/clip fraction。
- Response diversity仅作为规范化最终assistant文本的exact uniqueness，非语义多样性证明。
- Policy/action entropy：SDK0.2.9未返回完整策略分布，值为null并记录不可用原因。
  sampled-token surprisal独立命名，明确不是精确entropy；温度0时也不把它当熵估计。
- Gradient norm/GPU显存/实际账单：仅服务端实际返回时记录，否则null；不计算虚假的0。
- Evidence hit rate：缺乏可靠证据标注，暂为null；不使用答案字符串包含作为替代。
- PACE：每20次实际native加速结果抽查一次原生命令；比较完整payload，不一致即报告环境错误。
  聚合比较次数、差异次数和比率，分母0时比率null。缓存与coalescing不冒充PACE命中。

云端PACE可配置已字节校验的 `shards`：PACE拒绝后，仅符合保守条件的命令交给SPCS；
其他命令原生执行。日志 `search_execution.execution_route` 区分pace/sharded/sequential/cache/coalesced。
PACE与SPCS共用实际加速执行抽查计数，每20次对照一次原生完整payload，任何差异立即失败。
SPCS记录为 `sharded_native_audit`，PACE仍为 `pace_native_audit`。
RL的 `search/*` 是run累计路由/抽查计数，cache_entries/cache_bytes是当时缓存占用。

提前补采样：只为已按预定题序确认无效的完整组分配替代组；分配、serial与逻辑轮次和
原逐轮方案一致，固定同一policy，不额外超采。`dynamic/refill_groups` 是分配的替代组数，
`advanced_refill_groups` 是父轮尚未按序接纳完时已分配的替代组数；
`overlapped_refill_groups` 才是启动时父轮仍有未完成生成的替代组数。
`group_queue_wait_s` 是所有组等待启动时间之和，`group_compute_s` 是所有组执行时间之和；
并行组会重叠，这两个累计量都不能与整步wall相加。逐组started/finished事件保留关联ID。
RL新增 `time/sampler_refresh_s`、`rollout_s`、`batch_prepare_s`、`forward_backward_s`、
`optimizer_s`，分别记录同一步的串行阶段耗时；checkpoint耗时不含在optimizer_s内。

SFT teacher-forcing更新记录loss、学习率、阶段、样本/来源/调用桶、输入与监督token、验证及保存/重载。
Agent行为指标来自真实生成评测或RL rollout，不从静态SFT示范伪造模型的搜索能力。

2026-11-08 独立 RL 实验 `format_f1_v2`：

- 用户指定独立权重 format=0.2、F1=1.0，不重新归一化权重。
- `valid_format` 要求官方格式检查通过、正常以非空最终答案结束且无非法工具请求。
  `total = 0.2 * valid_format + 1.0 * valid_format * raw_f1`，范围为0至1.2。
  不叠加失败、长度、工具调用次数或检索命中惩罚/奖励。默认 `outcome_v1` 保持原分数。
- `agent/reward_format_component`、`reward_f1_component`、`reward_total` 为实际加权贡献，
  `reward_answer_f1`、`reward_valid_format_rate`、`strict_success_count` 分别记录原始质量。
- `dynamic/auxiliary_only_effective_groups` 记录原答案奖励相同、仅格式分产生差异的有效组数。
  advantage 分母为 `max(group_sample_std, 0.1) + 1e-6`；权重不直接等于归一化后的学习信号比例。
- `continuous` 模式达到每20个候选组仅记录 `dynamic_soft_limit`，保留有效组并在同一策略下继续补采，
  凑齐4个完整有效组才更新。`soft_limit_warnings` 计数；费用、token、运行时长与人工 STOP 保护仍生效。
- 新实验从 SFT 最终权重重新开始，重置 RL 优化器。`budget_parent_loaded` 绑定原结果文件哈希，
  继承所有此前 RL 请求的 token 费用。新实验的增量费用与150元累计估算上限分别记录。
