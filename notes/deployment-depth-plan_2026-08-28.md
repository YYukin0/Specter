# 方案：把 Specter 做成"能部署的推理服务"（方向 B — 工程深度）

> 2026-08-28。用户决定走方向 B。这份是全面评估 + 分阶段方案。
> **v2（同日）**：按最新文献逐条 challenge 过一遍，改动见每阶段的「文献核对」小节 + §7「先行工作定位」。
> 目标岗位假设：推理 / 基础设施 / 工程 craft 类。不是研究岗。
> 全程 Mac 本地、不花云端预算（P5.2 也不再建议上云，理由见 §7）。

---

## 0. 为什么是这个方向

现状：三根支柱（投机解码 / 自研 AWQ / 自适应 γ）都是"从零实现 + 严格验证"，但**没有一条真实 serving 路径**，也没有一个能引用的真实加速数字。三个硬伤：

1. **全项目刻意不用 KV cache**（"for obvious correctness"）。每个 draft step 重新前向整个前缀。所有墙钟数字都标着 "indicative only, no KV cache"。
2. **熔断器的 batch 信号是合成注入的**（`circuit_breaker.py` docstring + 坑15）。"always-spec 是结构性上界"就是因为没有真实 batch 成本反馈进 α。
3. **AWQ 是 fake-quant**（fp16 浮点落在 4-bit 网格），没有真实内存节省、没有真实速度。

方向 B 就是补掉这三条，把项目从"我复现了三篇论文"抬成"我从算法层造了一个带 4-bit 权重、带自适应控制、能扛并发的本地投机解码推理服务，每个部件都对着 bit-exact / 输出等价参照验过"。

**诚实边界**：P6.0（KV-cache 单序列投机解码）本身不 novel —— HF assisted-generation、gpt-fast 早就有。它的价值是 portfolio / 工程 craft（能造出来 + 能证明输出等价 + 能讲清难在哪），不是新发现。P6.1（输出等价的批量投机解码）在生态里确实还没解决好（见 §7），那部分有实打实的工程含量。

---

## 1. 现有资产盘点（能直接复用的）

| 文件 | 复用点 |
|---|---|
| `src/rejection_sampling.py` | `speculative_step` / `speculative_generate` / `target_only_generate`；纯函数 `adjusted_distribution` / `acceptance_probability` / `_sample` / `dist_from_logits` / `collect_eos_ids` / `encode_prompt`。**接受判据的数学不重写**，KV 版直接调这些。 |
| `src/spec_batch.py` | 左 padding + attention_mask + position_ids 的批量骨架；batch=1 与单序列 token 一致的 parity 契约写法；ragged EOS 处理。**注意**：它现在用的正是「masking + 逐序列 clamp」路线，文献说这条路线跨轮次会累积 padding、position-id 断裂（§7 C3）——P6.1 会换掉。 |
| `src/circuit_breaker.py` | 纯状态机 `circuit_breaker_decide` / `advance_state` / `simulate_decisions`（骨架保留，trip 条件要改，见 P6.1 文献核对）；`measure_switch_cost`（P6.1 换成真实 KV 重建成本）；周期性重探测机制（坑11）——与 vLLM"禁用后无法重新启用、必须持续跑 drafter"的生产事实一致，保留。 |
| `src/gammatune.py` | `gammatune_update` + `carry_state`。 |
| `src/awq_quantize_model.py` | `capture_all_layer_inputs` + `quantize_model` 搜出的逐通道 α/scale。 |
| `src/awq_perplexity.py` | `eval_perplexity` / `load_eval_corpus`，P6.2 验真实 int4 模型 ppl 用同一口径。 |
| `tests/` FakeModel 模式 | position-one-hot 的确定性假模型（`test_spec_batch.py` / `test_verify_p3_1_alpha.py` 那套），P6.0/6.1 的 parity 与回滚测试直接照搬。 |
| `src/model_loader.py` | Qwen2.5 0.5B/1.5B，MPS，fp16。 |

**已探明的关键事实**（`transformers 5.16.1`，本地实测）：
- `DynamicCache.crop(n)`：`n` 为**负**= 删掉这么多 token（新 API）；`n` 为正 = 旧的"裁到绝对长度"语义（deprecated，5.18 移除，会 warn）；正数 ≥ 当前长度 = no-op。P6.0 封一个 `_crop_to(cache, target_len)` 薄封装，内部用负数形式，锁死 5.18 前行为。
- `DynamicCache.batch_select_indices()` 存在 —— P6.1 序列完成后收缩 batch 用。

---

## 2. 分阶段方案

命名沿用 plan 的 P 编号习惯，归到新的 **支柱6：可部署推理服务**，阶段 P6.0–P6.4。每个小块一个 commit + push，先 `--smoke` 再放大，进报告的数字 ≥3 seed。

---

### P6.0 —— KV-cache 正确的单序列投机解码（keystone，~2 个工作单元）

**产出**：
- `src/spec_kv.py`：
  - `speculative_generate_kv(prompt, draft, target, tokenizer, *, gamma, max_new_tokens, temperature, seed, apply_chat_template) -> GenResult`（复用现有 `GenResult`，签名对齐 `speculative_generate`）。
  - `target_only_generate_kv(...)` —— KV-cached 的 target-only 基线（现在的 `target_only_generate` 也没 cache，基线不公平）。
  - draft、target 各持一个 `DynamicCache`。
- **核心难点：部分接受后的回滚**。一轮提出 γ token、接受 k 个（k ≤ γ）后：
  - **target cache**：单次前向覆盖了 γ+1 个位置，只有 k+1 个 token 被提交（k 接受 + 1 重采样/bonus）。`_crop_to(target_cache, prefix_len + k + 1)`。
  - **draft cache**：γ 次自回归前向留了 γ 个新条目。被接受的前 k 个位置的 KV 是在真实前缀上算的、有效；第 k 个位置往后是基于"提议 token"算的，而实际提交的第 k+1 个 token（重采样值）≠ draft 的提议值，stale。`_crop_to(draft_cache, prefix_len + k)`，下一轮把重采样 token 当新"最后一个 token" 喂 draft（1-token 前向补到 prefix_len+k+1）。
  - **bonus 分支（k==γ）**：全部接受，bonus 来自 target。draft/target KV 都到 prefix_len+γ-1 有效，下一轮两模型都从 bonus token 起步。
- **正确性契约（= 验收测试）**：
  - `speculative_generate_kv(temp=0)` 与 `speculative_generate(temp=0)` 与 `target_only_generate` **逐 token 完全一致**。
  - 采样模式、同 seed 下与 `speculative_generate` **逐 token 一致**（generator 消费顺序必须一模一样 —— KV 只改性能、不改任何可观测输出）。
- **测试 `tests/test_spec_kv.py`**：
  1. FakeModel（position-one-hot，确定性）：每轮结束后断言两个 cache 的 `get_seq_length()` **精确等于**已提交 token 数。
  2. 贪心 parity：FakeModel 上对 `speculative_generate` 和 `target_only_generate` bit-exact。
  3. 采样 parity：与 `speculative_generate` 共享 seed，bit-exact（钉死"KV 不改变任何输出"）。
  4. 回滚压力：接受率 ~0.5 的 FakeModel → 大量部分回滚 → 仍 parity + cache 长度精确。
- **verify `src/verify_spec_kv.py` → `results/p6_0_kv_cache_speculative.json`**：γ∈{1,3,5}、seed{0,1,2}、8 prompts。真实墙钟 tok/s：`speculative_generate_kv` vs 旧无 cache `speculative_generate` vs `target_only_generate_kv`。**项目第一个能引用的真实加速数字。** 附每轮 draft 前向次数（应从 γ 次「整前缀」降到 γ 次「单 token」）。

**文献核对（v2）**：
- 回滚设计与文献一致 —— "target 在验证位覆写 provisional KV，在首个 reject 位从调整分布重采样，之后的 KV 全丢；drafter 写 provisional KV，reject 时裁掉"（多篇实现描述，见 §7）。**无改动**。
- MPS fp16 在「cached 前向」和「单次大前向」之间的数值不一致 → parity 契约放在 FakeModel（确定性）；真实模型 parity 用「前 N token 一致，后段偶发 flip 记为 MPS fp 噪声」，这个 caveat `spec_batch.py` 已在用。文献里 EQSPEC/EXSPEC 的真实模型也只做到 ~95% exact match（fp 非确定性 + argmax 平局），不是 100% —— 对齐这个预期，别追 bit-exact。

---

### P6.1 —— 输出等价的批量投机解码 + serving loop + 熔断器接真信号（~2 个工作单元）

**产出**：
- `src/spec_kv_batch.py`：批量 + KV cache + **逐序列 ragged 回滚，每轮结束重新同步成矩形已提交状态**（EQSPEC 路线，见文献核对）：每轮 verify 后，把每条序列的 cache 裁到它各自的已提交长度，重建一份干净连续的 `position_ids` / `attention_mask`；序列完成用 `batch_select_indices()` 从 batch 移除。**不走** `spec_batch.py` 现在的「跨轮次累积 padding + mask」路线。
- `src/serving_loop.py`：`class SpecServer` —— `submit(prompt) -> req_id` / `step()` 推进整个活跃 batch 一轮 / `poll(req_id)`。单线程确定性事件循环。**连续批处理**：某条序列完成就把队列里的新请求塞进空槽，不等整个 batch 排空。
- **熔断器接真信号 + trip 条件改造**（见文献核对）：`SpecServer` 把三个量喂 `circuit_breaker_decide` —— 真实并发数 `len(scheduler.active)`、**滚动 α**、以及**周期性 target-only 延迟探测**。trip 条件从"batch ≥ 阈值"改成"**滚动 α < ~0.5** 或 **γ-tolerance 类的实测延迟判据不满足**"；batch size 只当输入之一，不再是规则本身。`measure_switch_cost` 换成真实 KV cache 丢弃/重建成本。**坑15 的"合成信号"caveat 到此消除。**
- **测试**：连续批处理能中途接纳新请求；单请求流与 `speculative_generate_kv` 逐 token 一致；**每个 batch size 下**整批输出与"逐条单独跑 `speculative_generate_kv`"输出等价（~95%+ exact，fp 噪声 caveat）；短上下文高并发 → 熔断器 trip；长上下文高并发 → 熔断器**不** trip（文献说这个 regime 仍 memory-bound）。
- **verify `src/verify_serving_loop.py` → `results/p6_1_serving_throughput.json`**：聚合 tok/s vs 并发请求数 {1,2,4,8}，熔断器 ON vs OFF；**外加 realignment 开销占比**（EQSPEC 路线的"正确性税"，文献测到 BS=8 时约 40% —— 我们本地测一遍，作为 finding 展示："朴素批处理看着简单其实是坑，这是实测的正确性代价"）。这也是 M5 `[A+B 联合]` 一直卡着的真实交叉点，Mac 小规模能画出来了。

**文献核对（v2，改动较大）**：
- **C3 — 朴素 masking 路线是已知会坏的**。arXiv:2510.22876《Batch Speculative Decoding Done Right》：所有现有批量投机解码实现都破坏输出等价（DSD、BSP 输出损坏到重复/乱码），根因是 ragged tensor —— 同 batch 内各序列接受数不同，position ids / attention mask / KV cache 跨轮次失同步。三条朴素路线（Masking / Rollback / Dynamic Padding）都有问题；BSP 的 masking 就是"跨轮次累积 padding、position-id 变不连续"坏掉的。**正确解**：EQSPEC（每轮 verify 后强制同步不变量，正确但 BS=8 时 realignment 吃掉约 40% 算力，BS>8 负 scaling）或 EXSPEC（SequencePool 逐条持有 ragged 状态、按等长分组，BS=8 拿到 3×）。
  - **我们的选择**：走 **EQSPEC 风格（每轮重同步）**——比 EXSPEC 简单、可证正确，40% 开销本身当 finding 展示。BS 上限 ~8 现在是**有依据的**（对齐 EQSPEC 负 scaling），不只是 Mac 内存限制。
  - 该论文参考实现要 `transformers==4.51.3`（"更高版本重构了 KV cache API，跑不了"）。**我们的环境是 5.16.1** —— 所以我们是在它自己代码都不支持的新 API 上做输出等价的批量投机解码，这本身是卖点。引用它、把它形式化的"同步不变量"写成测试。
- **C4 — 熔断器"batch 高就降级"的前提已被 2025 工作部分推翻**。Together AI：长上下文 + 大 batch → KV cache 大 → 仍 memory-bound → 投机解码给到 2×、且 batch 越大加速越多。"Rethinking High-Throughput..."（ICLR 2026 投稿，后撤稿）提 γ-tolerance 延迟判据、论证大 batch 下带宽仍是主瓶颈。MagicDec 同向。生产经验值：短上下文 BS>32 才 compute-bound；**α < 0.5 投机解码才真的净亏**。→ trip 条件按上面改造，并在 verify 里同时测短/长上下文两个 regime。

---

### P6.2 —— 真实 int4（走 mlx-lm）：自研 AWQ vs mlx-lm 学习型量化套件（~1 个工作单元）

**产出**：
- `src/awq_to_mlx.py` + `src/verify_p6_2_real_int4.py` → `results/p6_2_awq_int4_real.json`。
- **四方对比，全部 4-bit / group=128、本地 Metal、同一 `eval_perplexity` 口径 + `mlx_lm.evaluate` 下游任务**：
  1. 我的 from-scratch AWQ（fake-quant → 灌进真实 int4 打包）
  2. `mlx_lm.awq`（`--bits 4 --num-samples 32 --n-grid 10`，黑盒基线）
  3. `mlx_lm.gptq`（**本地就能跑**，见文献核对 —— 原计划以为要云端）
  4. `mlx_lm.dwq`（可选，蒸馏型，通常最好，作为"上限"参照）
- 报：真实磁盘大小 (GB)、推理峰值 RSS、mlx Metal 生成 tok/s、wikitext2 ppl（对比 P2.1 的 fake-quant 数字 —— 应接近，**验证 fake-quant 当初没撒谎**）。
- **接回投机侧**：用真实 int4 的 1.5B 当 target 跑 `speculative_generate_kv` —— 4-bit target 会不会改变最优 γ？（这是 P5.2 那个问题本地免费的一半，但注意 §7 C1：SpecKV 已大致回答了。）

**文献核对（v2，改动较大）**：
- **C2 — mlx-lm 现在自带 AWQ / GPTQ / DWQ / dynamic-quant 全套，都是本地 Metal**（`LEARNED_QUANTS.md`，Casper Hansen ~2025-04 起）。
  - **"GPTQ 臂留云端"这条作废** —— `mlx_lm.gptq` 本地能跑。P6.2 顺手把 GPTQ 对比也做了。
  - `mlx_lm.awq` **没有**公开的"灌自己的 scale"接口（搜索是内部的，只有 `--n-grid` 之类）。要注入我搜的逐通道 scale 得 patch `mlx_lm/awq.py` 源码 —— **fallback**：把 `mlx_lm.awq` 当纯黑盒基线，我的 from-scratch 走"fake-quant → 自己实现 int4 group 打包"路径，两者对比仍给出真实内存/速度数字，只是少了"同框架内换 scale 搜索"的干净 head-to-head。这个风险在结果文件里写明。

---

### P6.3 —— live demo（~1 个工作单元）

**产出**：
- `src/demo/`：终端实时视图（优先 stdlib + ANSI；`rich` 若已在环境里则用，**不为 demo 装新依赖**；Web 版要另问用户是否允许加 FastAPI）。基于 P6.1 的 `SpecServer`。展示：流式 token + 每轮 γ/接受长度/滚动 α/tok-s/并发数/熔断器状态 + 三个开关（投机、自适应 γ、熔断器）现场看加速差。
- 录 asciinema / GIF 放 README。

**文献核对**：无冲突。timebox，纯 ANSI 也能交付。

---

### P6.4 —— 打包 + 写成工程故事（~1 个工作单元）

**产出**：
- README 重写：定位改成"从零实现的本地投机解码推理服务，4-bit AWQ 权重 + 自适应步长控制器，每个部件对着 bit-exact / 输出等价参照验证"。
- 工程故事（是工程故事不是论文，见记忆 `project_specter_demo_polish_idea`）：
  1. **"那条平坦曲线其实是个 bug"** —— 坑16，复现里的确认偏误。
  2. **"把投机解码的 KV cache 做对"** —— P6.0 回滚问题、三向 parity 契约、FakeModel 测试怎么钉死它。
  3. **"批量投机解码的正确性税"** —— P6.1，ragged tensor 问题、为什么 masking 会坏、EQSPEC 每轮重同步的 40% 开销实测、在参考论文都不支持的新 transformers KV API 上做输出等价。
  4. **"给熔断器一个真信号，然后发现前提本身过时了"** —— 坑15 → P6.1，α<0.5 判据 vs batch 阈值，2025 长上下文工作对"大 batch 投机没用"的反驳。
  5. **"fake-quant vs 真 int4：我那个 perplexity 数字诚实吗"** —— P6.2 验证 + 与 mlx-lm awq/gptq/dwq 的四方对比。
- 坑表（现 17 条，会涨到 ~19）作为一等公民产物。

---

## 3. 依赖顺序

```
P6.0 (keystone) ──> P6.1 ──> P6.3 ──> P6.4
      │                              ↑
      └──> P6.2 ─────────────────────┘   (P6.2 与 0/1 独立，可穿插)
```

P6.0 先做。P6.2 任何时候能插。P6.4 最后。

---

## 4. 工作量 / 成本

| 阶段 | 估计 | 云端 $ |
|---|---|---|
| P6.0 KV-cache 单序列 | ~2 单元（难点集中在这） | 0 |
| P6.1 EQSPEC 风格批量 + serving loop + 熔断器改造 | ~2 单元 | 0 |
| P6.2 真实 int4：from-scratch vs mlx-lm awq/gptq/dwq | ~1 单元 | 0 |
| P6.3 live demo | ~1 单元 | 0 |
| P6.4 打包 + 写作 | ~1 单元 | 0 |
| **合计** | **~7 个专注单元，全本地** | **$0** |

---

## 5. 这个方案买到什么

- **一个能引用的真实 tok/s 加速数字**（KV cache）—— 现在根本给不出。
- 一个**输出等价**的批量投机解码实现，在参考论文自己代码都不支持的 transformers 版本上，带把论文"同步不变量"写死的测试；外加实测的"正确性税"数字。
- 熔断器从"合成信号、结构上赢不了"变成"接真实负载 + 用 α/延迟判据、并诚实说明 batch 阈值这套已过时"。
- 自研 AWQ 的**真实内存/速度数字** + 与 mlx-lm awq/gptq/dwq 的四方对比（含本地 GPTQ）。
- 一个 live demo。
- 5 篇以坑表为骨架的工程故事。
- 项目定性：从"复现三篇论文" → "从零造了个能用的加速推理服务，且对每个难点都诚实"。

---

## 6. 开工前要同步的事

- 方案落地时把支柱6 / P6.0–P6.4 加进 `TASKS.md` 和 `notes/project_plan_v9.md` §7。
- P6.0 的三向 parity 契约、P6.1 的输出等价不变量写进 `project_plan_v9.md` §9.2 作为新验收纪律。
- §7 里 challenge 出来的三条（C1 SpecKV 抢先、C3 ragged tensor 正确性、C4 熔断器前提过时）在 P6 真正动手撞上时，按 repo 规则7 转成 §9.2/§9.3 的坑18/19/20；现在先留在本文档。
- **`AGENTS.md` / `contracts/` 这套本仓库当前没有**（早期实验版已随那条废弃 `main` 丢弃，旧 ADR 引用已全指向 `project_plan_v9.md`）。solo 仓库不值得为流程而流程重建 ADR 体系 —— 决策继续记进 `project_plan_v9.md` §7/§8 和 agent 记忆即可。除非之后真有跨人协作需求，否则不建。
- Web 版 demo → 需要用户批准加 FastAPI 依赖。

---

## 7. 先行工作定位（v2 新增，逐条 challenge 的结果）

### C1 — P5.2 的"novel 发现"基本被 SpecKV 抢先了
**SpecKV: Adaptive Speculative Decoding with Compression-Aware Gamma Selection**（arXiv:2605.02888）已经做了"压缩 × 最优 γ"：4 类任务 × 4 个 γ × 3 个压缩档（FP16 / INT8 / NF4），5112 条 step 级记录。结论：
- 各压缩档平均接受率相当（FP16 0.70 / INT8 0.69 / NF4 0.70），但**高 γ 处接受率掉得更陡**。
- **最优 γ 随压缩档漂移**：FP16 → 低（2 或 4）；INT8 → 6 或 8；NF4 → 4 或 6。
- 归因：INT8（BitsAndBytes）的 dequant kernel **计算开销大**，使更长的投机序列相对更划算 —— 是**算子开销**驱动的，不是分布漂移。
- 自称"第一个用 draft 模型自身信号自适应选 γ 的系统"（与 GammaTune 空间重叠）。

**对我们的影响**：
- "AWQ vs BnB 改变最优 γ"这个 headline **不再 novel**，降级成"把 SpecKV 的表扩到它没测的 AWQ"。
- 而且结果基本可从 SpecKV 的归因预测：AWQ 的 int4 算子比 BnB 快 → 最优 γ 大概率**贴近 FP16**（不怎么漂）。低惊喜。
- **决定**：P5.2 不再建议花云端钱。P6.2 里用 `mlx_lm` 本地顺手测一下 AWQ / GPTQ 下的最优 γ 漂移，作为 SpecKV 的一个小确认性扩展，够了。

### C2 — mlx-lm 已有完整本地学习型量化套件
`mlx_lm.awq` / `mlx_lm.gptq` / `mlx_lm.dwq` / `mlx_lm.dynamic_quant` 都在 Metal 上本地跑（`LEARNED_QUANTS.md`）。**"GPTQ 臂留云端"作废**。`mlx_lm.awq` 无自定义 scale 注入接口（内部搜索，`--n-grid` 控粒度）。`mlx_lm.evaluate` 提供下游任务评测。→ P6.2 重构成四方本地对比。

### C3 — 朴素批量投机解码的正确性是个坑
**Batch Speculative Decoding Done Right**（arXiv:2510.22876）：所有现有批量实现破坏输出等价（ragged tensor：接受数不同 → position id / mask / KV cache 失同步）。Masking / Rollback / Dynamic Padding 三条朴素路线都有问题。正确解 EQSPEC（每轮重同步，BS=8 约 40% 开销，BS>8 负 scaling）/ EXSPEC（SequencePool，BS=8 拿 3×）。参考实现要 `transformers==4.51.3`（更高版本 KV API 重构后跑不了）。真实模型 ~95% exact match。→ P6.1 走 EQSPEC 风格、在 5.16.1 上做、把开销当 finding、BS 上限 ~8 有依据。

### C4 — "大 batch 投机解码没用"的前提已被 2025 工作部分推翻
Together AI（长上下文 + 大 batch 仍 memory-bound，投机给 2×、batch 越大越快）、"Rethinking High-Throughput LLM Inference"（ICLR 2026 投稿→撤稿，提 γ-tolerance 延迟判据）、MagicDec（长上下文大 batch KV 越大加速越多）。生产经验值：短上下文 BS>32 才 compute-bound；**α<0.5 才真净亏**。→ 熔断器 trip 条件改成滚动 α / 实测延迟判据，batch size 只当输入之一；verify 里测短/长上下文两 regime。

### C5 — GammaTune 的 P5.0 null 不完整
**Token-Driven GammaTune**（arXiv:2504.00030，Gautam et al.）：SpecBench 上 GammaTune 平均 +15%±5%、GammaTune+ +16%±3%。论文自己承认"某些 pair 上 GammaTune 不如 AssistantThreshold，此时 GammaTune+（logit-based early stopping）反超两者"。初始 γ 集 [1,2,3,4,5,6,7,8,12,16,20,24]。**我们只实现了 GammaTune、没实现 GammaTune+**。→ 我们的 null（Qwen 0.5B/1.5B 上 GammaTune ≤ 固定 γ）与论文一致但不完整；如要补，实现 GammaTune+ 是个便宜的本地跟进（不在 P6 主线，记进 TASKS/记忆）。

### 已被文献佐证、无需改的部分
- P6.0 KV-cache 回滚设计（target 覆写 verified 位 KV / 首个 reject 位重采样 / 之后 KV 全丢 / drafter provisional KV 裁剪）—— 与多篇实现描述一致。
- 熔断器周期性重探测（坑11）—— 与 vLLM 生产事实一致："禁用后无法重新启用投机，必须持续跑 drafter，均摊 2–3% 开销"。
- HF `generate()` 的 assisted generation **至今不支持 batched 输入**（最新文档确认）—— P6.1 不是重复造轮子。

---

### v3（2026-08-28 深挖轮）—— 又搜了一轮找"漏网的 novel 角度"，结论：没有干净的"推翻某篇"的角度，2026 上半年这块封得很死

再搜的方向 + 已被占的格子（别再造轮子）：

| 想过的角度 | 已经有人做了 | 出处 |
|---|---|---|
| 从零实现 + 3 级分布等价验证 + Apple Silicon + **单独隔离"量化 Metal 后端把并行 verify 串行执行"** + 建议报告 draft/target 延迟比和 verify-batch scaling 曲线 | **几乎就是 P6.0 + P6.2 动机的原文**，2026-07 发表，同一类硬件、同样"from scratch"框架 | *Lossless but Not Free: An Empirical Anatomy of Speculative Decoding on Consumer Hardware*, arXiv:2607.17283 |
| 批量投机解码输出等价 / ragged tensor / EQSPEC / 40% realignment tax | 见 C3 | arXiv:2510.22876 |
| 压缩档 × 最优 γ 耦合 | 见 C1 | SpecKV arXiv:2605.02888 |
| 量化 drafter（AWQ/GPTQ/RTN）几乎不动接受长度 | 竞赛 writeup 直接对比过 RTN/AWQ/GPTQ 量化 drafter，结论"INT4 对 drafter 平均接受长度只有极小影响，三种 PTQ 相当" | arXiv:2607.04244 |
| 结构化 / tool-call 输出接受率近 100%、双峰、按 slot-local EMA 调 γ | ToolSpec（schema-aware）、SimpleTool（>93% 接受率）、AgentSpec（且明确发现"基于接受率的预算分配对 agent 批量推理无效"）、Stateful Inference（prompt-lookup + acceptance gating）全是 2026 | arXiv:2604.13519 / 2603.00030 / 2608.24004 / 2605.26289 |
| 各任务加速不均 + 只微调 drafter 做公平性 | Disparate Impacts：under-fit / 低资源任务系统性拿更少加速；按 divergence 加权只更新 drafter | arXiv:2510.02128 |
| 自适应 γ（EMA） | GammaTune / GammaTune+，见 C5 | arXiv:2504.00030 |
| MLX 上连续批处理 + 投机解码 | vLLM-mlx（EuroMLSys '26，M4 Max 525 tok/s）、mlxcel（Rust，continuous batching + spec + KV 压缩）、dflash-mlx（MLX 上从零 exact spec decode） | 多个开源 / EuroMLSys '26 |
| "光看接受率不能预测加速，要报 cost ratio" | 多篇，且 2607.17283 把它当头条建议 | — |

**活下来的缝（都不是"推翻"，是"没人系统做过"）**，按推荐度排：

- **缝 B（推荐当命名交付物）——投机解码实现的一致性/一致性测试套件（conformance + fault-injection test harness）。**
  `Batch Spec Decoding Done Right` 是靠对着真实 repo 做差分测试抓 bug，但**没放出可复用的套件**；没人发表过"这是一套 property / pytest 插件，拿去测你自己的 spec-decode 实现"。Specter 已经有的东西正好是这个：`injection=` 故障注入钩子 + FakeModel 确定性 parity oracle + KV 回滚压力测试 + ragged-EOS 用例 + argmax-tie / fp 非确定性容差带。**这条零"被抢"风险、正好是项目最强的地方、且不需要打赢任何人的数字**——对推理/基础设施岗是对味的 craft 交付物。做法：把 P6.0/P6.1 的测试基建提升成一个独立命名模块（`tests/spec_conformance/` + 一页 README「拿去测你的实现」），故障注入清单覆盖：position-id 错位、KV 裁剪差 1、mask 泄漏、residual 分布采样错、EOS 早停不同步。

- **缝 A（并进 P6.1 当头条测量）——unified memory 上、verify batch 维（部分）串行执行时的批量投机解码正确性税。**
  2607.17283 明确只做 BS=1，且写了"server-side batching 完全改变经济性——不在本文范围"。2510.22876 是服务器 GPU（CUDA、transformers 4.51.3）。**没人把两者合起来**：如果 Metal 把 verify 的 batch 维串行跑，那 EQSPEC 那 ~40% realignment 开销是叠加在一个几乎拿不到 batch 并行的 verify step 上——可推出"在某个上下文长度以下，Apple Silicon 上批量投机解码净负"这个具体、可证伪的假设。P6.1 正是量它的仪器。风险：可能只是复现"Mac 上别batch投机解码"，是个温和的负结果；vLLM-mlx 团队可能已有内部数据。

- **缝 C（本地免费侧验，别当头条）——标定数据分布 → draft 接受率的耦合。**
  搜索明确说这"是个 open gap"：大家都用一个 calib set 量化 drafter、报告接受率几乎不动；没人变过 calib 分布（C4 vs WikiText vs 领域/对话 vs 评测分布本身）再测传导到各领域接受率的漂移。Specter 的 P2.2 跨分布 AWQ 模型已经建好了。风险：效应量可能极小（竞赛结果说 INT4 drafter 接受率本来就几乎不动），大概率又落地成"确认很小"的 null。

- **缝 D（有意思但贴着已占区）——把自适应 γ 的评价重新框成消费级（对齐差的）pair 上的尾延迟 / SLO 问题。**
  P1.4 已测到接受长度 std 随 γ 涨 0.42→4.12，比 AdaEDL 调优 pair（→2.35）陡。GammaTune/AdaEDL 优化的是平均吞吐；Disparate Impacts 说平均掩盖了 per-task 分散。没人针对"主导本地/消费场景的对齐差 pair"围绕 p99 / inter-token latency 方差来评自适应 γ。风险：和 GammaTune+（logit 早停本身就是方差压缩器）+ Disparate Impacts 重叠，定位要小心。

**决定**：方向 B 主线不变。把 **缝 B 显式抬成命名交付物**（`tests/spec_conformance/`，见 P6.0/P6.1 与 P6.4 写作提纲），**缝 A 并进 P6.1 当头条测量**（unified-memory 批量正确性税），**缝 C/D 列为可选本地侧验、不作 headline 主张**。没有"推翻论文"的路，项目定位继续是 portfolio / 工程 craft（见 [[project_specter_direction_b_deployment]]）。

---

## 8. 参考文献

- Leviathan et al. 2023, *Fast Inference from Transformers via Speculative Decoding* — 基础算法（已在 `rejection_sampling.py` 引用）。
- SpecKV, arXiv:2605.02888 — 压缩 × 最优 γ 耦合（C1）。
- *Batch Speculative Decoding Done Right*, arXiv:2510.22876 / github.com/eBay/spec_dec — ragged tensor 正确性、EQSPEC/EXSPEC（C3）。
- *Rethinking the High-Throughput LLM Inference*, OpenReview 59OJOgKLzN（ICLR 2026 投稿，已撤）— γ-tolerance（C4）。
- Together AI, *Speculative decoding for high-throughput long-context inference* — 长上下文大 batch 反例（C4）。
- MagicDec, arXiv:2408.11049 — latency-throughput tradeoff，长上下文（C4）。
- Token-Driven GammaTune, arXiv:2504.00030 — GammaTune / GammaTune+（C5）。
- *The Synergy of Speculative Decoding and Batching*, arXiv:2310.18813 — batch × 投机解码早期分析。
- mlx-lm `LEARNED_QUANTS.md` — 本地 AWQ/GPTQ/DWQ/dynamic（C2）。
- AWQ, arXiv:2306.00978 / GPTQ, arXiv:2210.17323 — 量化基线。
- *Lossless but Not Free: An Empirical Anatomy of Speculative Decoding on Consumer Hardware*, arXiv:2607.17283 — 从零实现 + Apple Silicon + 隔离"量化 Metal 串行 verify"（v3 缝 A / B）。
- *The Disparate Impacts of Speculative Decoding*, arXiv:2510.02128 — per-task 加速不均、drafter-only 公平性微调（v3 缝 D）。
- ToolSpec arXiv:2604.13519 / SimpleTool arXiv:2603.00030 / AgentSpec arXiv:2608.24004 — 结构化 / tool-call 高接受率、slot-local EMA gating（v3）。
- *Speculative Decoding and Beyond: An In-Depth Survey*, arXiv:2502.19732 — 最新综述（v3 深挖起点）。
