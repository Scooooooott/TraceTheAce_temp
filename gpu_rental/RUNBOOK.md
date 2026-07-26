# GPU 租卡 Runbook

本文件是"当天做什么"的完整命令清单——现场只复制粘贴,不现场翻文档、不现场拼命令。
配套文件:`setup_instance.sh`(装环境)、`sync_caches.sh`(本地定时回传备份)。

**第三版**:prompt 已经重构完成(转录前置 + 两特征共享前缀),两个独立的 precompute 脚本已合并成一个,main.py 同步改造。这是当前唯一有效版本——第二版里"现有 prompt / 重排 mastery / 完全 merge"三档并列的表述已经不适用,merge 已经是唯一在用的结构,不再是待选项。

## 已经确认的数字(prompt 重构后,2026-07-25 重测)

| 项目 | 数值 | 状态 |
|---|---|---|
| 训练集 session 数 | 22,821 | 实测 |
| 训练集 response 数(= mastery_check 调用次数) | 35,072 | 实测 |
| 竞赛测试集 response 数 | 10,509 | 实测(`data/raw/submission_format_ZQLcKx7.csv` 行数) |
| 竞赛测试集 session 数 | **未知,不可本地获得** | 无测试集转录,无法验证;~6,838 是按训练集比例(1.537)外推的估计,容器里跑的时候才有真数 |
| 共享前缀(转录+统一 system prompt)token 长度 | mean=6170, p50=6146, p95=8299, max=9544 | 实测(300 session 抽样,新 prompt) |
| strategy_tags 后缀长度 | 固定 268 token | 实测 |
| strategy_tags 完整 prompt | mean=6438, max=9812 | 实测 |
| mastery_check 后缀长度(随 LO 文本变化) | mean=254, p95=263, max=270 | 实测(300 response 抽样) |
| mastery_check 完整 prompt | mean=6552, p95=8745, max=9877 | 实测 |
| 超过 `PRODUCTION_MAX_PROMPT_TOKENS=10240` 的比例 | 0/300(本次抽样) | 实测,过去(旧8192阈值)约10%,新阈值下抽样未见超限 |
| 输出 token(thinking=True,strategy_tags) | mean=610, p95=1214, max=1535(n=8) | 实测,小样本,新 MAX_NEW_TOKENS=2048 覆盖 |
| 输出 token(thinking=True,mastery_check) | mean=562.5, max=774(n=8) | 实测,小样本,远低于当前预算 |
| data/raw 总大小 / 转录部分 | 1.2GB / 619MB | 实测 |
| 转录 gzip 压缩后 | 160MB(压缩率 74%) | 实测 |
| 竞赛容器锁定版本 | `torch==2.11.0+cu129` + 特定 vLLM wheel(见 `setup_instance.sh`)+ Python 3.12 | 实测 |

## 已经落地的代码改动(2026-07-25)

1. **共享前缀**:`src/features/llm_config.py::build_shared_prefix()` 产出转录+统一 system prompt,`llm_strategy_tags.py`/`llm_mastery_check.py` 的 `build_prompt` 都改成调用它 + 各自的 `TASK_SUFFIX`——两者的 prompt 现在逐字节共享前缀(已用 `startswith` 断言验证)。两个模块的 `PROMPT_VERSION` 都 bump 到 `v2`。
2. **合并成单脚本**:`scripts/precompute_llm_strategy_tags.py`/`precompute_llm_mastery_check.py` 已删除,替换为 `scripts/precompute_llm_annotations.py`(`--tasks strategy,mastery`,默认两个都跑并交错提交)。缓存文件仍是两个独立的 `.pkl`(各自 model_id+prompt_version 命名),下游 join 逻辑零改动。断点续存按 **session 为单位**:一个 session 只有 strategy + 全部 mastery 都落盘才算完成,否则整 session 重做。
3. **调度逻辑**:新增 `src/features/llm_annotate.py::score_combined()`,按 session 分组构造交错的 flat 请求列表(每个 session 的 strategy 请求后紧跟它全部的 mastery 请求),一次 `generate_batch_fn` 调用喂完,失败样本再统一重试一次,结果按 kind 拆回两个 dict。`scripts/precompute_llm_annotations.py` 和 `submission_src/main.py` 调同一份逻辑,不是各写一遍。
4. **重试机制**:`score_sessions_batch`/`score_responses_batch`/`score_combined` 都对首轮解析失败的样本做**一次**更高预算重试(strategy 2048→3072,mastery 768/1400→2800),仍失败落 `FALLBACK_TAGS`。已用 mock 验证逻辑正确。
5. **MAX_NEW_TOKENS 上调 + 重跑验证**:strategy_tags 768→2048(基于实测输出分布)。重新用同一 seed 复测过撞 cap 的那条样本,确认它确实是纯退化漂移(无 JSON 痕迹),不是"差一点写完"——重试机制对这类样本无害但也不保证救回,冒烟时看**重试后仍失败率**判断退化 vs 预算不够的比例。
6. **`MAX_MODEL_LEN` 修正**:`14336`,覆盖 `PRODUCTION_MAX_PROMPT_TOKENS(10240) + 最大 RETRY_MAX_NEW_TOKENS(3072)` 加缓冲——上一版的 12288 会在 strategy 重试时溢出,已修。

## 已验证:新 prompt 没有退化,和旧 prompt 输出可比

用 `--backend hf --limit 5` 跑通新合并脚本,5 个 session(与本轮之前验证过的 4 个 session 重合)+ 13 个 response,**18/18 一次解析成功**,无需触发重试。抽查同一批 session 新旧 prompt 的输出:

| session | 旧 prompt(v1) | 新 prompt(v2,转录前置) |
|---|---|---|
| bcaufvc | scaffolding=3, giveaway=2, follow_up=1, confusion=2, self_correct=1 | scaffolding=3, giveaway=1, follow_up=2, confusion=2, self_correct=1 |
| eyutanf | scaffolding=3, giveaway=0, follow_up=3, confusion=2, self_correct=1 | scaffolding=3, giveaway=0, follow_up=3, confusion=2, self_correct=1 |
| ntwkcfj | scaffolding=2, giveaway=1, follow_up=2, confusion=2, self_correct=1 | scaffolding=3, giveaway=1, follow_up=2, confusion=2, self_correct=1 |

个别字段有小幅漂移(结构变了,贪心解码路径本就不该逐字节复现),但量级和分布形态一致,不是坍缩成单一值。判定:**新 prompt 可用,不需要额外调整就能冒烟**。

## 预算可行性(merge 后,唯一有效的场景)

用重测的精确均值(不用估算):容器侧(测试集,S_te≈6,838 估计 / R_te=10,509 实测)prefill 总量约 **4,693 万 token**(shared 6170 × 6838 sessions,strategy 后缀 268 × 6838,mastery 后缀 254 × 10509),decode 总量约 **1,008 万 token**。

| 模型 | prefill 吞吐假设 | 容器侧合计(prefill+decode,不含加载/传统特征) |
|---|---|---|
| 8B-AWQ | 6-9k tok/s(prefill)/ 3-5k(decode) | **2.01-3.11h** |
| 32B-AWQ(dense) | 2-3k / 1-2k | 5.75-9.32h(仍死) |
| 30B-A3B(MoE,3.3B激活) | 未测,理论上界接近 8B 档 | 需实测才能填 |

8B 在悲观端也有接近一倍的余量(3.11h vs 6h 上限,还没算传统特征管线和模型加载,但即使加 1h 也远在安全区)。32B dense 即使 merge 完成仍然出局。**30B-A3B 是唯一还留着悬念的候选**,必须靠冒烟实测吞吐才能判断,不能假设。

## 生产路径保险丝(已实现,2026-07-25)

S_te 是外推不是实测,容器超时=整次提交作废+烧一次周配额。`submission_src/main.py` 现在通过 `score_combined_chunked`(`src/features/llm_annotate.py`)按 chunk 处理测试集,每个 chunk 之间用已耗时/已完成 session 数外推剩余耗时,一旦预测超过 `llm_config.LLM_TIME_BUDGET_SECONDS`(4.5h,留 1.5h 给其他步骤+余量),剩余 session 直接落 `FALLBACK_TAGS`,不再调用模型——保证提交仍然完整、按时写出,只是部分行退化成兜底特征。已用 mock 验证:正常预算下行为和不加保险丝完全一致(0 条 fallback);预算不够时正确触发,且每个 id 仍有结果(真实或 fallback,不会缺行)。**只挂在 main.py,不影响 precompute 脚本**(标注侧不受墙钟约束)。

## 阶段 0 — 开机检查(~10 分钟)

```bash
ssh <instance>
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
df -h /workspace   # 确认 network volume 确实挂在这里、容量是申请的那个数(RunPod: network volume 固定挂载点是 /workspace)
tmux new -s gpu_run
```

**磁盘位置铁律(RunPod Network Volume vs Container Disk)**:Network Volume 挂载在 `/workspace`,Pod 终止后数据仍在;Container Disk(这次开的 60GB)跟 Pod 生命周期绑定,是临时的。下面所有步骤把仓库克隆到 `/workspace/trace-the-ace`,不是 `~`(`~` = `/root`,落在 60GB 的 Container Disk 上)——实测体积:8B-AWQ 6.11GB + 30B-A3B-AWQ 主选 16.83GB + data 0.8GB + venv(torch/vllm cu129,估计15-25GB)已经 43-49GB,60GB 里剩不了多少余量,一旦要换备胎 checkpoint(再 +16.83GB)直接爆盘。180GB 的 network volume 才是这些东西该待的地方。

## 阶段 1 — 上传数据 + 装环境(并行,~20-40 分钟)

代码走 git(2026-07-25 定案,不再用 rsync 传代码 -- 已把 precompute 脚本真正 import 到的最小依赖闭包推到
https://github.com/Scooooooott/TraceTheAce_temp.git:src/data.py、src/features/session_stats.py、
llm_config/llm_strategy_tags/llm_mastery_check/llm_backend/llm_annotate.py、
scripts/precompute_llm_annotations.py、gpu_rental/、pyproject.toml、uv.lock。不含 data/raw -- 比赛数据不进
第三方托管的 git 仓库,数据仍走下面的 tar 包)。

本地(只需传数据):
```bash
cd /home/scott/workspace/trace-the-ace
tar --exclude=raw/train_transcripts.zip -czf /tmp/trace-the-ace-raw.tar.gz -C data raw
rsync -avz /tmp/trace-the-ace-raw.tar.gz <instance>:/workspace/trace-the-ace-raw.tar.gz
```

远程(tmux 里):
```bash
git clone https://github.com/Scooooooott/TraceTheAce_temp.git /workspace/trace-the-ace
cd /workspace/trace-the-ace
mkdir -p data
tar -xzf /workspace/trace-the-ace-raw.tar.gz -C data
bash gpu_rental/setup_instance.sh
```

装完核对打印出来的 torch/vllm 版本和 `tutoring-outcomes-runtime/runtime/pyproject.toml` 完全一致。

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 uv run huggingface-cli download Qwen/Qwen3-8B-AWQ --local-dir ./models/Qwen3-8B-AWQ
# 30B-A3B 主选(2026-07-25 定死,不要当天现挑):QuixiAI/Qwen3-30B-A3B-AWQ
# 月下载量 50k、明确写 vllm>=0.8.5、基于原始 hybrid-thinking 的 Qwen/Qwen3-30B-A3B
# 量化(不是后来拆分成 Instruct-2507/Thinking-2507 那批 -- 那些不支持
# make_vllm_batch_generate_fn 依赖的 enable_thinking 切换,选错等于重演
# enable_thinking=False 那次内容坍缩故障)。
HF_HUB_ENABLE_HF_TRANSFER=1 uv run huggingface-cli download QuixiAI/Qwen3-30B-A3B-AWQ --local-dir ./models/Qwen3-30B-A3B-AWQ
# 备胎(主选加载失败/输出异常时换这个,只试一次,不行就 30B-A3B 出局):
# ELVISIO/Qwen3-30B-A3B-AWQ -- 月下载量 12k,同样基于原始 hybrid-thinking 基座,
# model card 明确示例代码带 enable_thinking=True。
# HF_HUB_ENABLE_HF_TRANSFER=1 uv run huggingface-cli download ELVISIO/Qwen3-30B-A3B-AWQ --local-dir ./models/Qwen3-30B-A3B-AWQ-backup
```

本地(另开终端):
```bash
bash gpu_rental/sync_caches.sh <instance> /workspace/trace-the-ace 600
```

## 阶段 2a — 采样配置 A/B(必测项,~20-30 分钟,8B 上跑,在阶段2主冒烟之前)

**背景(2026-07-25 定案)**:现在 `llm_config.py` 冻结的是 `temperature=0(greedy)+repetition_penalty=1.3`——1.3 是当时为了压住一次真实撞见的贪心解码重复陷阱(见 experiments.md 2026-07-24)硬加的。但 Qwen3 官方 model card 明确**不建议**thinking 模式用贪心解码,推荐 `temperature=0.6, top_p=0.95, top_k=20, min_p=0`,重复控制用 `presence_penalty`(0-2 区间),并直接警告贪心会导致无尽重复和性能退化——也就是说 1.3+greedy 很可能是给一个反模式打的补丁,而不是修复了根因。1.3 本身也偏高(常规区间 1.05-1.15),对 thinking 质量和 JSON 里合法的重复 token(键名、重复数字)有潜在隐性损伤,本地小样本验证看不出来。

**这条不能跳过,也不能现在盲改**——1.3+greedy 是目前唯一有本地验证背书的配置,deploy 前必须用真实模型对比一次,不能凭 model card 的通用建议直接切换。

**关键:A/B 两组都必须用一次性 `--model-id`,不能用正式的 `qwen3-8b-awq`**——缓存文件名只烤了 `model_id + prompt_version`,不含采样参数,用正式 model-id 跑 A/B 会把不同采样配置的结果混进同一份正式缓存,写完就再也分不清哪 30 条是哪组配置产出的。

```bash
# A 组(现配置,基线)
uv run python -m scripts.precompute_llm_annotations ./models/Qwen3-8B-AWQ \
  --model-id qwen3-8b-awq__ab-a --backend vllm --limit 30 2>&1 | tee logs/ab_a.log

# B 组(Qwen3 官方推荐,rep_penalty 归 1.0 只用 presence_penalty,二选一别叠加)
uv run python -m scripts.precompute_llm_annotations ./models/Qwen3-8B-AWQ \
  --model-id qwen3-8b-awq__ab-b --backend vllm --limit 30 \
  --temperature 0.6 --top-p 0.95 --top-k 20 --repetition-penalty 1.0 --presence-penalty 1.5 \
  2>&1 | tee logs/ab_b.log
```

对比四项 + 肉眼抽查:首轮解析失败率、撞 cap 率(退化)、两任务 thinking 长度分布(`_summarize_lengths` 那行)、标签分布是否有区分度(不是全坍缩成同一个值)。

判定:B 组解析率不差且退化消失 → **切 B**,把 `src/features/llm_config.py` 的 `TEMPERATURE/TOP_P/TOP_K/PRESENCE_PENALTY/REPETITION_PENALTY` 改成 B 组的值(`main.py` 和 precompute 脚本都从这里读,改一处两边同步冻结)。B 组反而不稳/未见改善 → 留 1.3+greedy,至少是被数据背书过的选择,`llm_config.py` 不用动。

**决定之后**,用正式 `--model-id qwen3-8b-awq` 再跑一次 `--limit 30` 冒烟(不带任何 A/B 覆盖参数,让它读 `llm_config.py` 刚冻结的值)——这次才是计入全量、真正验收的冒烟(阶段2的第一条命令就是它),A/B 那两个一次性缓存文件事后直接删掉。

## 阶段 2 — 冒烟(~40-60 分钟)

```bash
uv run python -m scripts.precompute_llm_annotations ./models/Qwen3-8B-AWQ --model-id qwen3-8b-awq --backend vllm --limit 30 2>&1 | tee logs/smoke_8b.log
uv run python -m scripts.precompute_llm_annotations ./models/Qwen3-30B-A3B-AWQ --model-id qwen3-30b-a3b-awq --backend vllm --limit 30 2>&1 | tee logs/smoke_30b.log
```

检查清单:
- [ ] 两个模型都能加载,无版本/内核报错(**8B 先跑,已知能跑,用它排查环境问题最快**)
- [ ] 显存 + 磁盘水位(另开一个 tmux 窗口常驻,冒烟和后面的全量都用它盯):
  ```bash
  watch -n 30 'nvidia-smi; echo ---; df -h /workspace /; echo ---; du -sh /workspace/hf 2>/dev/null'
  ```
  同时看 GPU、Network Volume(`/workspace`)、**Container Disk 根分区(`/`)** 三个水位——`/` 如果开始异常增长,说明还有哪个工具的默认写入点没钉在 `/workspace`,这条命令能当场抓到,不用等爆盘才发现
- [ ] `chunk` 打印的耗时/ETA——记录 cold prefill+decode 吞吐,替换预算表里的假设区间
- [ ] **`score_combined` 打印的重试统计**(`N/total needed retry, M/N still failed`)——首轮失败率 > 3-5% 就先回去改 prompt 措辞(退化对指令措辞敏感),不要指望重试硬扛
- [ ] **vLLM 的 prefix cache 命中率**(日志里 vLLM 自己周期性打印的吞吐/命中率行,不是我们代码打的,格式以实际跑出来的为准)——理论上界约 `前缀占比×(session内平均请求数-1)/该请求数`;如果显著低于这个数,两个旋钮按序试:调小 `--chunk-size`(默认300,先试100)、把每个 session 的 strategy 请求提前一个 chunk
- [ ] 抽查缓存内容,标签有真实差异,不是全部退化成同一个值
- [ ] 两任务输出 token 长度分布(脚本每个 chunk 后自动打印 `output tokens (cumulative): ...`,按 strategy/mastery 分开)
- [ ] 用实测吞吐重算一遍训练侧(全量)和容器侧(测试集)总耗时
- [ ] **决策点**:30B-A3B 若容器侧估时 > 4.5h,出局,本次只跑 8B 全量

## 阶段 3 — 8B 全量过夜(7-12h,保底落袋)

启动前过一遍(按顺序,过完再回车):
1. 阶段2a 的 A/B 结论已经写进 `llm_config.py`(或者确认维持 1.3+greedy)
2. A/B 那两个一次性缓存文件已经删掉
3. 确认正式 `--model-id qwen3-8b-awq` 冒烟是用冻结后的配置跑的
4. **磁盘水位**:`/workspace` 剩余 >30GB 且 `/`(Container Disk 根分区)剩余 >20GB——过夜跑的是 7-12 小时无人值守,爆盘是这里代价最高的失败模式(比容器超时更难现场补救,补救得等醒来),启动前这一步不能省

tmux 里 `nohup` 是冗余的(tmux 本身就防挂断)但无害,保留:

```bash
nohup uv run python -m scripts.precompute_llm_annotations ./models/Qwen3-8B-AWQ --model-id qwen3-8b-awq --backend vllm > annotations_8b.log 2>&1 &
wait
```

`sync_caches.sh` 本地持续跑着,每 10 分钟自动回传。阶段2那条 `watch -n 30 '...'` 磁盘监控窗口**整晚留着别关**——过夜期间才是慢性磁盘泄漏真正有时间累积到爆盘的窗口,不是只在冒烟那几十分钟看一眼。盯前 20 分钟确认吞吐、checkpoint 落盘、显存/磁盘稳定,然后可以离开去睡。每半小时(醒着的话)`tail -f *.log` 瞄一眼 ETA。

## 次日 — 30B-A3B 视判定跑子集

阶段2决策点 GO 或边缘区才跑,否则整个跳过,只有 8B 落袋:

```bash
nohup uv run python -m scripts.precompute_llm_annotations ./models/Qwen3-30B-A3B-AWQ --model-id qwen3-30b-a3b-awq --backend vllm --limit 5000 > annotations_30b.log 2>&1 &
wait
```

只跑 5k 子集(2-4h),不跑全量——8B 已经是全量,配对比较不需要 30B 也全量。`--limit` 现在是固定种子(42)shuffle 过的 session 顺序取前 N(2026-07-25 修复,见 precompute 脚本内注释),不是原始文件序——原始文件按 response_id 排序,response 数越多的 session 系统性排得越靠前(实测:原始序前5000个 session 平均每 session 多 35% response、正确率高2个百分点),shuffle 前直接用 `--limit 5000` 会给 CV 判定带来有偏子集。这个修复已经在推到临时仓库的代码里,`git clone` 下来就是修好的版本。

## 阶段 4 — 收尾(~15 分钟)

```bash
uv run python -c "
import pandas as pd
for f in ['llm_strategy_tags_cache__qwen3-8b-awq__prompt-v2',
          'llm_mastery_check_cache__qwen3-8b-awq__prompt-v2']:
    d = pd.read_pickle(f'gpu_rental/backup/{f}.pkl')
    print(f, len(d))
# 仅当 30B-A3B 跑了 5k 子集:
for f in ['llm_strategy_tags_cache__qwen3-30b-a3b-awq__prompt-v2',
          'llm_mastery_check_cache__qwen3-30b-a3b-awq__prompt-v2']:
    d = pd.read_pickle(f'gpu_rental/backup/{f}.pkl')
    print(f, len(d))
"
```

8B 应分别等于 22,821 / 35,072(全量)。30B-A3B(若跑了子集)strategy 应为 5,000,mastery 约 7,000-8,000(取决于这 5000 个 session 各自的 response 数,不是固定值)。核对无误后再关实例,本地单独下载一份选定模型权重:

```bash
uv run huggingface-cli download Qwen/Qwen3-8B-AWQ --local-dir ./models/Qwen3-8B-AWQ
```

## 之后(本地,不再需要 GPU)

1. 跑 CV 对比:baseline / +8B(全量)/ +30B-A3B(若跑了)。
2. 把最终选定模型的 cache 复制成规范文件名:
   ```bash
   cp gpu_rental/backup/llm_strategy_tags_cache__<选定model-id>__prompt-v2.pkl data/processed/llm_strategy_tags_cache.pkl
   cp gpu_rental/backup/llm_mastery_check_cache__<选定model-id>__prompt-v2.pkl data/processed/llm_mastery_check_cache.pkl
   ```
3. `uv run python scripts/fit_ensemble_production_model.py`
4. `uv run python scripts/build_submission.py level4_ensemble_lr_lightgbm ./models/<选定模型目录>`
5. `tutoring-outcomes-runtime` 里 `just pack-submission` / `just test-submission`。
6. 本地 `test-submission` 依然测不了 vLLM 推理路径(WSL2 老问题)——首次带 LLM 特征的提交仍是这条代码路径的最终真实测试,按"验证性提交"的心态对待。
