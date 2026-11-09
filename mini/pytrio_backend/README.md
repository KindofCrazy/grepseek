# Windows / Ubuntu 上运行 PyTRIO

本地 Ubuntu 负责数据、真实 Wikipedia 搜索、组内优势和 CPU 上的 PPO 损失；
PyTRIO 负责 4B 模型的采样与训练。使用用户指定的
[PyTRIO skill](https://github.com/SwanHubX/pytrio-skill)，固定 SDK 0.2.9。
原始 verl 入口、工具实现和官方 reward 文件保持不变。

评测跨题滚动并发，RL按题序接纳完整组；整个run共享采样和搜索限流。
默认4个采样RPC/1路搜索，可在验证主机资源与平台限流后调整。
支持连续补采、版本化格式/F1奖励、完整LoRA/Adam恢复与独立预算保护。
训练期间不运行benchmark；评测只读取明确选择的 sampler checkpoint。
运行目录、凭据、语料、模型和生成报告不入库。源码迁移与断点续跑见 [加速设置](../ACCELERATION.md)。

在 Windows PowerShell 进入已准备好的环境：

```powershell
wsl -d GrepSeekUbuntu -u grepseek --cd /home/grepseek/grepseek
```

随后所有示例均在 Ubuntu 项目根目录执行。源码以 Windows 仓库为准，修改后按
[环境文档](../environment/README.md) 同步，避免覆盖 Linux 的运行结果。

## 依赖与本地检查

```bash
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0
python -m pip install -r mini/pytrio_backend/requirements.txt
python -m pip check
python -m pytest mini/pytrio_backend/test_backend.py -q
python -m mini.pytrio_backend sft --out mini/runs/plan
```

系统依赖 bubblewrap、prlimit、rg 和 Bash；CPU Torch 用于 SDK custom loss 的 autograd，
不在本机加载模型。数据及 tokenizer 必须已经离线准备，执行前校验冻结清单中的全部哈希。
最后一条只校验并打印配置，不创建云端 client；添加 `--execute` 才提交任务。

## 默认小验证参数

| 项目 | 设置 |
|---|---|
| 模型 / LoRA | Qwen/Qwen3.5-4B；rank 32；attention + MLP；不训练输出 embedding |
| SFT | LR 5e-5；逻辑 batch 8；前 ceil(计划步数 × 5%) 线性 warmup 后恒定 |
| RL | LR 1e-6 恒定；每更新 4 个有效题组 × 每题 4 条轨迹 |
| 优化器 | AdamW betas (0.9,0.95)，eps 1e-8，weight decay 0，grad clip 1 |
| 上下文 | SFT / RL 都是 12,288；RL 初始 prompt ≤1,024，延伸 ≤11,264，含 observation 与模板 |
| 多轮预算 | 单轮最多 2,048 个采样 token；最多 6 轮 assistant / 5 次工具；工具 stdout 上限 2,048 |
| 采样 | thinking 开启，训练 temperature=1，top_p=1；评测 temperature=0；seed=42 |
| Reward / advantage | 官方格式门控 token-F1；组内减均值除样本标准差加 1e-6；无长度惩罚 |
| PPO | 对称 clip 0.2；全部 assistant token 求均值；单次优化；无 KL penalty |
| Dynamic sampling | 默认 bounded：最多5轮/20个候选题组；continuous 模式保留有效组并持续补齐，候选数量仅告警 |
| checkpoint | 默认每 20 次成功更新保存；结束及异常额外尝试保存；RL 最多 100 次成功更新 |

上下文上限一致不表示每条 SFT 数据都被填满。RL 预算按真实 sampled token 加环境后缀计算，
prompt、工具 observation 和人工补入的模板 token 不进入 loss，保留模型原始 token IDs 与旧 logprob。
这套方案是带 dynamic sampling 的组相对 PPO；没有采用完整 DAPO 的 clip-higher / soft-overlong。

默认数据路径指向800条SFT/600题RL的小验证快照；可通过 `--data` 指定其他已校验快照。
`mini.sample_curriculum` 支持分阶段抽样；SFT 的 `--curriculum`、`--lr-schedule cosine`
支持同一训练计划内连续学习率；`mini.expand_rl_pool` 可扩展QA池，不做难度预采样。
CLI 默认2步，仅适合冒烟；
100 是 RL 的硬上限，需要显式 `--steps 100`。不会自动启动正式训练。

## 运行与恢复

```bash
python -m mini.pytrio_backend sft --out mini/runs/sft-smoke --steps 2 --val-count 2 --execute
python -m mini.pytrio_backend rl --out mini/runs/rl-smoke --steps 2 \
  --initial-checkpoint mini/runs/sft-smoke/checkpoint.json --execute
# 验证选定 checkpoint 的真实 dev 推理；冒烟不使用冻结最终测试集。
python -m mini.pytrio_backend eval --out mini/runs/dev-smoke --count 2 \
  --initial-checkpoint mini/runs/rl-smoke/checkpoint.json --execute
```

每个输出目录必须是新的。可通过 `--config file.json` 传 Config 同名字段，CLI 覆盖 JSON。
SFT → RL 的 `--initial-checkpoint` 只加载模型、重置优化器。
同阶段恢复用 `--resume path/checkpoint.json`，显式提供原来的算法配置和计划总步数；
加载优化器、数据 cursor、epoch、题目抽取次数和累计 token 消耗。
种子由 policy 版本和候选序号确定，服务端数值运算不保证逐位复现。
若优化器请求超时导致是否提交成功不确定，checkpoint 标记 `resume_safe=false`，拒绝自动精确恢复。

SDK 的恢复工厂存在惰性初始化：创建 client 不等于权重和优化器已经加载。
`trainer_ready` 只在采样器真正准备完成后记录。需要显式完整恢复时，使用
`--resume-load-mode explicit`，先创建同形状的LoRA，再等待公开接口
`load_state_with_optimizer` 完成，之后才采样和更新；不叠加第二套LoRA。

完成的恒定学习率RL计划可通过 `--resume-migration receipt.json` 延长到最多100步。
收据类型为 `rl_step_limit_extension`，绑定 `checkpoint_sha256`、`checkpoint_step`、
`train_state`、`steps`（before/after）、`data_manifest_sha256` 和 `protocol_sha256`。
协议哈希由 `mini.pytrio_backend.pending.protocol_hashes()` 生成。改变奖励、学习率、
数据或组大小仍被精确恢复检查拒绝，原始checkpoint文件不会被改写。

## 连续补采与奖励版本

`--dynamic-sampling-mode continuous` 在同一固定策略下保留有效组，遇到平分组就换题，
直到凑齐完整batch才更新。`max_candidate_groups` 在此模式中是重复告警阈值，
`sampling_rounds` 不终止补采；费用、token、墙钟与人工STOP保护始终生效。
默认 `bounded` 模式仍执行轮数/候选硬上限。

默认奖励 `outcome_v1` 保持官方格式门控F1。可选 `--reward-version format_f1_v2`：

```text
reward = format_reward_weight * valid_format
       + f1_reward_weight * valid_format * answer_f1
```

两项权重独立，默认0.2和1.0；有效但答错为0.2，全对为1.2，格式无效或未完成为0。
`--advantage-std-floor 0.1` 可限制微小奖励差经组内归一化被放大的程度。
分别记录原始EM/F1、格式和F1加权贡献，以及仅因格式分形成差异的有效组数。
此版本未添加工具次数或长度奖励。评分设置不代表其效果已优于SFT，须单独评测。

SFT 小验证还会实际重新创建训练 client，核对相同验证样本 NLL，并从 sampler checkpoint 生成。
本地 `checkpoint.json` 同时包含远程训练状态 URI 和 sampler URI；远程可用性受平台保留策略约束。

## 消耗与停止

默认每次运行上限：prefill 1.5M、sample 100k、训练输入计算量 500k、
未命中缓存的 token 费用估算 20 元、墙钟 1,800 秒；可用同名 CLI 参数收紧。
这些是小验证保护值，不代表已确定正式实验总预算。
每个请求提交前预留最坏 token 消耗，成功后按返回长度核销，失败或结果不明不退款；
丢弃组、验证前向和 custom loss 的两次计算都计入本地账本。
墙钟限制在请求间检查，不能取消已经提交的远程请求；工具有独立超时和字节捕获上限。
checkpoint / 初始化本身的服务消耗和真实账单没有 SDK token 数据，估算不能作为发票或绝对费用保证。
平台欠费错误会终止后续提交，不依赖欠费自动停止来控制预算。
如需主动结束，在对应运行目录创建空文件 `STOP`；程序会在下一次计费请求前停止，
等待已提交任务结束后尝试保存。断点恢复携带累计消耗，同时单独记录本次新增消耗，避免汇总重复计费。

## 记录与搜索隔离

每次运行保留 `config.json`、依赖与代码哈希 `provenance.json`、`capabilities.json`、
`events.jsonl`、`checkpoint-*.json`、`checkpoint.json` 和 `result.json`。
事件中包括实际轨迹、生成 token/logprob、工具结果、官方 reward、分组接受/丢弃原因、优势、
候选及有效组分别的 EM/F1/格式/搜索/截断指标、补采比例、token、耗时及优化指标。
没有服务端返回的梯度范数、GPU 显存和账单保留未知，不填造数值。
SFT/eval 默认 SwanLab offline；RL CLI 未指定时默认 online（私有实验，初始化联网失败时回退 offline）。
旧配置显式写有 offline 时用 `--swanlab online` 覆盖；完整轨迹留在 JSONL，本地 SwanLab 日志可补传。
云端上传、历史曲线合并和查看说明见 [实验记录设置](../EXPERIMENT_LOGGING.md)。
不自动上传轨迹、登录信息或给他人发送消息。

搜索复用官方命令白名单及结果字段，在 Linux bubblewrap 中执行：仅只读挂载语料与
系统程序、隔离网络和 PID、不挂载主机 home / Windows 盘、清空环境变量。
对不合法命令、无匹配和超时保留真实工具反馈；基础设施故障中止实验，避免当成答错奖励。

## 工具加速

默认启用最多256条/32 MiB的LRU结果缓存；同一条正在执行的命令会合并等待。
缓存只复用同一完整语料、同一命令的成功或无匹配结果，不复用超时、错误、截断输出。
语料大小、inode 或修改时间变化立即停止，避免使用过期证据。
`search_execution` 记录真实执行、缓存命中、合并等待、后端路由及耗时；
`--search-cache-entries 0` 关闭结果留存。

已接入官方 SPCS 引擎，仍在相同的只读、无网络沙箱中执行。
只加速保守支持的固定字符串过滤加 `head` / `wc -l`；行号、正则、排序和其他管道走顺序路径。
新分片在构建时核验按顺序拼接的 SHA256 与完整原语料完全相同，执行时再次检查文件状态。
在本机实测前不要假设论文的 32 核 / RAM 常驻语料收益可以复现。

```bash
# 独立离线性能测试；临时构建 2/4 分片，严格比较完整返回 payload，默认自动清理临时大文件。
python -m mini.pytrio_backend.benchmark_search --work data/search-benchmark-new \
  --out mini/reports/search-benchmark-new.json
# 要保留单一分片配置时，在新的输出目录构建：
python -m mini.pytrio_backend.acceleration --out data/shards-2 --shards 2
# 在已有训练命令后添加：--search-backend sharded --shards data/shards-2
```

SPCS与顺序搜索的实测记录保存在本地报告目录。
RL / eval 默认使用PACE加有界缓存；运行前需单独完成索引构建与绑定验收。
不支持的管道仍在相同的只读沙箱顺序执行；缺少结构或绑定时明确报错。
本机2 GB WSL只负责运行；高内存构建在临时CPU主机完成。
构建、导入及配置见 [PACE.md](PACE.md)，完整证据见 [PACE验收报告](../reports/pace/README.md)。
