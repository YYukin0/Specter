# Overnight 进展 — 2026-08-28

执行 `notes/overnight-任务队列_2026-08-28.md` 的 4 个任务。全部各自 commit + push 到 `main`，工作区干净，`pytest tests/ -q` 36 passed。

执行顺序：Task 1 → Task 2 → Task 4 → Task 3（Task 3 的实验在后台跑，期间先做了 Task 4）。

| commit | 任务 |
|---|---|
| `1e350e0` | Task 1 — M4 P5.3 batch 熔断器（完成 M4） |
| `68e9887` | Task 2 — P5.4 HF 双基线（部分：BanditSpec 待云端） |
| `a6c867e` | Task 4 — M1 AWQ P2.0 + P2.1 第一版 |
| `7cae173` | Task 3 — 探索性 side experiment（更差匹配模型对） |

---

## Task 1 — P5.3 Batch-aware 熔断器 —— ✅ 完成（M4 收官）

**产出**：`src/circuit_breaker.py` / `tests/test_circuit_breaker.py`（8 测试）/ `src/verify_circuit_breaker.py` / `results/p5_3_circuit_breaker.json`；新坑 15 进 `notes/project_plan_v9.md` §9.2；`TASKS.md` M4 标完成。

**一句话结论**：按任务决策点 6 口径的主指标 **Metric A 判熔断器负**（circuit-breaker 0.247±0.008 < always-spec 0.280±0.010，c=7，区间不重叠），根因是新坑 15——Metric A 成本模型没有任何 batch 依赖项、本地 batch 信号是合成的且不反馈进 α，于是 always-spec 是结构性上界。补的 `sat_tax` 敏感性分析（高 batch 段 draft 前向记 `sat_tax` 单位/次）里 `sat_tax≥2` 熔断器追平、`sat_tax=3` 两条信号都反超（≈复现 Nightjar 30% 倒退量级，坑 7）。机制正确性与主指标判负解耦看：降级滞后 0 轮、恢复滞后 1 轮、切换开销代理量 ~11.5ms（占墙钟 0.1%），机制都对。附带发现：单轮 γ=3 重探测不够估 α（量子化到 {0,0.5,1.0}，绝对偏差 0.2–0.75），生产要 pool 多轮。

**留给用户的决策点**：真实 batch 吞吐曲线（M5[A] / 云端）落地前，熔断器"有用与否"取决于 `sat_tax` 取多少——这个数只能靠带 KV cache 的真实高并发墙钟测。本地能做的到此为止。

---

## Task 2 — P5.4 对标 HF 内置 assisted generation —— 🟡 部分完成

**产出**：`src/verify_hf_baseline.py` / `results/p5_4_hf_baseline.json`；`TASKS.md` M8 P5.4 行标"部分完成"。

**一句话结论**：HF 两个自适应调度（`num_assistant_tokens_schedule="heuristic"` + `assistant_confidence_threshold`）跑通并与我们的固定 γ / GammaTune 在"每次 target 前向 token 产出"上并排。排名 hf_heuristic 4.96±0.80 > hf_confidence_0.4(static) 3.78±0.17 > ours_fixed_γ5 3.55 > ours_gammatune 3.19 > hf_constant_g5 3.04 > ours_fixed_γ3 2.85。**α 对齐验证通过**（ours α_exact 0.77–0.79 vs HF α_est 0.69–0.77，hf_constant_g3 0.771 ≈ ours_γ3 0.781）——手写 rejection sampler 与 HF 参考实现一致，无正确性红旗。

**踩到的 API 坑**（已写进结果文件 + TASKS.md）：transformers 5.16.1 里 `num_assistant_tokens` / `_schedule` / `assistant_confidence_threshold` 必须设在 `draft.generation_config` 上，传给 `target.generate()` 会被静默忽略（4 个配置输出逐 bit 相同才发现）。sklearn 未装 → `assistant_confidence_threshold` 是静态 0.4，非 Intel/HF DSL 的在线 ROC 重调。坑 14 caveat：主指标无 draft 代价项，hf_heuristic 靠把窗口冲到 10–15 登顶，成本模型排名需要 HF 逐轮窗口大小（本轮未采集）。

**没做的部分**：Task 2b BanditSpec（`github.com/sail-sg/BanditSpec`）本地跑不通——`flash_attn` 硬 import 且 CUDA-only、通篇 `.cuda()` / `torch.cuda.synchronize()`、需要训练好的 EAGLE draft heads。已克隆到仓库外临时目录检查过、**未提交进本仓库**，结果文件里记为"本地未跑通，留云端 M7/M8"。

**留给用户的决策点**：BanditSpec 作为 P5.4 第三基线只能在云端 GPU 阶段跑；要不要为它单独分配云端时间，用户定。

---

## Task 4 — M1 AWQ 从零实现：P2.0 + P2.1 第一版 —— ✅ 完成（第一版）

**产出**：`src/awq_activation_stats.py`（+6 测试）/ `src/awq_scaling.py`（+10 测试）/ `results/p2_0_activation_stats.{pt,json}` / `results/p2_1_scaling_demo.json`；`TASKS.md` M1 P2.0、P2.1 打勾（P2.2/P2.3 待）。

**P2.0 — 激活统计**：对 Qwen2.5-1.5B-Instruct 的 196 个目标 Linear（28 层 × q/k/v/o/gate/up/down）挂 forward-pre-hook，24 条 prose+code 校准 prompt（截断 512 tok），逐输入通道 fp32 累加 `abs_mean` + `abs_max`。**关键发现**：早层 `mlp.down_proj` 的 per-channel `abs_mean` max/median 比高达 ~7750（layer 1）/ ~3880（layer 2）/ ~1160（layer 26），少数输入通道极端主导——正是 AWQ 要保护的激活离群通道现象，坐实"逐通道缩放有东西可保护"。

**P2.1 — 逐通道缩放搜索（第一版）**：`fake_quantize_groupwise`（4-bit / group=128 / 非对称 / **round→+zero→clamp→dequant** 顺序，单测钉死 clamp 在 round 之后）；`s = act_scale**α / weight_scale**(1-α)`（clamp + `/sqrt(max·min)` 归一）；`search_scale` 在 α∈{0,0.1,…,1.0} ∪ {不缩放} 网格上按"该层输出 MSE"搜，保留 best-of{缩放, 不缩放}。代表性 6 层（early/mid/late × v_proj/down_proj）：输出 MSE 改善 **+14.8% ~ +71.0%**（late 层最大），**layer 14 mlp.down_proj 网格打不过不缩放 → 回退 α=none（+0.0%）**（诚实记录，非 bug）；缩放变换数学恒等误差 ≤1.4e-4。附录 A.3 两通道玩具算例单测通过。

**没做**：perplexity / 端到端、GPTQ 对比、跨分布（P2.2）、校准集大小消融（P2.3）、196 层全扫、真实 int4 打包（云端，风险 B：PyTorch quantized 后端未移植 MPS）。

**留给用户的决策点**：无卡点。第一版能采统计、能搜缩放因子、fake-quant 数值对、玩具算例过。下一步 P2.2（跨分布鲁棒性）/ P2.3（校准集消融）可直接接着做，都不依赖用户拍板。P5.2（量化×γ 耦合）现在前置解锁了。

---

## Task 3 — 探索性 side experiment：更差匹配的 draft/target 对 —— ✅ 完成（负结果，有信息量）

**产出**：`src/explore_worse_pair.py`（给 `verify_gammatune.py` 加 `--draft/--target/--results-path` 后复用其函数）/ `results/explore_worse_pair_pair{1,2}_{alpha,gammatune}.json`；`TASKS.md` M4 段加 side-experiment 行；`notes/project_plan_v9.md` §9.2 坑 9 补记。主线模型对**未动**。

**动机**：直接验证 P5.0/P5.1 的归因"主线对 α≈0.79 太稳、方差不足，GammaTune 没东西可利用（坑 9）"。

**探索对 1** — draft = `Qwen2.5-0.5B`（**BASE，非 instruct**）+ target = `Qwen2.5-1.5B-Instruct`：
- α(γ=3) 从 ~0.79 → **0.700±0.027**（降了 ~0.09）
- accept 长度 pooled std（γ=3/5）= 1.28/1.91 vs 主线 1.24/1.95 —— **基本没变**
- Phase 2：GammaTune 2.482 vs 最优固定 γ=7 的 3.199（主指标 -22.4%，区间不重叠）；成本模型 c∈{4,7,10} 落最优簇内（-1.5% / -1.9% / -2.1%），稳优于极端 γ

**探索对 2** — draft = `Qwen2.5-0.5B-Instruct` + target = `Qwen2.5-3B-Instruct`（能力差距 6×）：
- α(γ=3) = **0.726±0.029**（比对 1 还高，接近主线）
- accept 长度 pooled std（γ=3/5）= 1.26/1.90 —— 主线水平
- Phase 2：GammaTune 2.513 vs 最优固定 γ=7 的 3.201（-21.5%，区间不重叠）；成本模型 -1.1% / -1.5% / -3.4%，优于极端 γ

**一句话结论**：在 Qwen2.5-Instruct **同族内换模型对造不出**坑 9 假设需要的"α≈0.5–0.65 + 明显更大方差"regime。共享 tokenizer / 同族 / instruct-tuning 主导了 draft↔target 分布对齐——base draft 只压低"接受率"不放大"方差"，拉大能力差距对两者都几乎不动。**好处**：P5.0 的 null 在 3 个模型对上稳健，不是单对偶然。**局限**：这一步没真正检验到坑 9 假设本身。

**留给用户的决策点**：真正制造高方差 regime 需要**跨族 / 非 Qwen 的 draft**（会破坏坑 1 的共享 tokenizer 前提，需要 logits 对齐或 token 映射层）。这是要用户拍板的方向——要不要为"精确刻画 GammaTune 在什么方差水平开始有用"投入做一个跨族 draft 的对齐层。不做也行，P5.0/P5.1/本实验三处一致的 null 已经是可报告的结论。

---

## 其它

- **文件命名 bug 已修**：`explore_worse_pair.py` 初版两次运行写同一组文件名，pair 2 跑时覆盖了 pair 1 的 alpha JSON。已改成 `explore_worse_pair_<pairN>_*.json`；pair 1 的 alpha JSON 从运行日志重建（数字与日志核对过），commit message 里写明了。
- `git status` 里 4 个 `??` 未跟踪文件（`notes/` 下的几个 prompt md + 1 个 pdf）是会话开始前就有的，不是本次产物，未动。
- `pytest tests/ -q`：36 passed（M2 之前 12 + P1 若干 + 本次新增 P5.3 的 8 + AWQ 的 16）。

## 建议的下一步（按可独立推进排序）

1. **P2.2 跨分布鲁棒性**（AWQ vs GPTQ 矩阵，坑 5）—— 不依赖用户拍板，P2.0/P2.1 已就位，直接能做。
2. **P2.3 校准集大小消融**（坑 6）—— 同上。
3. **M5[A] 投机解码 batch 吞吐曲线（本地小 batch 版）** —— 给 Task 1 熔断器的 `sat_tax` 提供真实锚点。
4. （需用户拍板）跨族 draft 对齐层 → 才能真正检验坑 9；云端 GPU 阶段跑 BanditSpec + GPTQ + P5.2 BnB 对照臂。
