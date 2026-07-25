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
df -h
tmux new -s gpu_run
```

## 阶段 1 — 上传数据 + 装环境(并行,~20-40 分钟)

本地:
```bash
cd /home/scott/workspace/trace-the-ace
tar -czf /tmp/trace-the-ace-raw.tar.gz -C data raw
rsync -avz --exclude .venv --exclude data/processed ./ <instance>:~/trace-the-ace/
rsync -avz /tmp/trace-the-ace-raw.tar.gz <instance>:~/trace-the-ace/
```

远程(tmux 里):
```bash
cd ~/trace-the-ace
tar -xzf trace-the-ace-raw.tar.gz -C data
bash gpu_rental/setup_instance.sh
```

装完核对打印出来的 torch/vllm 版本和 `tutoring-outcomes-runtime/runtime/pyproject.toml` 完全一致。

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 uv run huggingface-cli download Qwen/Qwen3-8B-AWQ --local-dir ./models/Qwen3-8B-AWQ
# 30B-A3B 候选(社区 AWQ checkpoint,注意挑口碑好的来源,如 QuixiAI/ELVISIO):
HF_HUB_ENABLE_HF_TRANSFER=1 uv run huggingface-cli download <30B-A3B-AWQ repo> --local-dir ./models/Qwen3-30B-A3B-AWQ
```

本地(另开终端):
```bash
bash gpu_rental/sync_caches.sh <instance> ~/trace-the-ace 600
```

## 阶段 2 — 冒烟(~40-60 分钟,比之前多一项)

```bash
uv run python -m scripts.precompute_llm_annotations ./models/Qwen3-8B-AWQ --model-id qwen3-8b-awq --backend vllm --limit 30
uv run python -m scripts.precompute_llm_annotations ./models/Qwen3-30B-A3B-AWQ --model-id qwen3-30b-a3b-awq --backend vllm --limit 30
```

检查清单:
- [ ] 两个模型都能加载,无版本/内核报错(**8B 先跑,已知能跑,用它排查环境问题最快**)
- [ ] 显存峰值(`watch nvidia-smi`)
- [ ] `chunk` 打印的耗时/ETA——记录 cold prefill+decode 吞吐,替换预算表里的假设区间
- [ ] **`score_combined` 打印的重试统计**(`N/total needed retry, M/N still failed`)——首轮失败率 > 3-5% 就先回去改 prompt 措辞(退化对指令措辞敏感),不要指望重试硬扛
- [ ] **vLLM 的 prefix cache 命中率**(日志 `gpu_prefix_cache_hit_rate`,或用实测吞吐反推)——理论上界约 `前缀占比×(session内平均请求数-1)/该请求数`;如果显著低于这个数,两个旋钮按序试:调小 `--chunk-size`(默认300,先试100)、把每个 session 的 strategy 请求提前一个 chunk
- [ ] 抽查缓存内容,标签有真实差异,不是全部退化成同一个值
- [ ] (可选,冒烟顺手做)`repetition_penalty` 从 1.3 试到 1.05 一次,看对退化尾部和 JSON 合法重复键名的影响,不要盲改 `llm_config.py` 里的默认值
- [ ] 用实测吞吐重算一遍训练侧(全量)和容器侧(测试集)总耗时
- [ ] **决策点**:30B-A3B 若容器侧估时 > 4.5h,出局,本次只跑 8B 全量

## 阶段 3 — 8B 全量(+ 30B-A3B 若通过决策点)

```bash
nohup uv run python -m scripts.precompute_llm_annotations ./models/Qwen3-8B-AWQ --model-id qwen3-8b-awq --backend vllm > annotations_8b.log 2>&1 &
wait

# 仅当阶段2决策点通过:
nohup uv run python -m scripts.precompute_llm_annotations ./models/Qwen3-30B-A3B-AWQ --model-id qwen3-30b-a3b-awq --backend vllm > annotations_30b.log 2>&1 &
wait
```

`sync_caches.sh` 本地持续跑着,每 10 分钟自动回传。期间每半小时 `tail -f *.log` 瞄一眼 ETA。

## 阶段 4 — 收尾(~15 分钟)

```bash
uv run python -c "
import pandas as pd
for f in ['llm_strategy_tags_cache__qwen3-8b-awq__prompt-v2',
          'llm_mastery_check_cache__qwen3-8b-awq__prompt-v2']:
    d = pd.read_pickle(f'gpu_rental/backup/{f}.pkl')
    print(f, len(d))
"
```

应分别等于 22,821 / 35,072。核对无误后再关实例,本地单独下载一份选定模型权重:

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
