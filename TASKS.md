# Specter 两人开发任务表

> 派生自 [`notes/two_developer_plan_v2.md`](notes/two_developer_plan_v2.md) —— 完整依据、并行窗口分析、里程碑验收标准都在那份文档里，这里只放**可勾选的操作清单**，人或 agent 逐条打勾、提交时更新这个文件即可。owner 标签：**[A]** = Track A（投机解码 + 自适应控制）／**[B]** = Track B（量化 + Agent 评测）／**[A+B]** = 需要双方在场或双方各自产出后合并。

## 待确认事项（建议在 M0 之前定下来，详见 plan §5/§7/§8）

- [ ] 云端 GPU 租几个实例 —— 1个：预算不变但云端阶段排队执行；2个：真并行但大概率突破 $50 硬顶（plan §5）
- [ ] 云端账号/计费主体由谁持有，$50 预算两人怎么同步记账
- [ ] Track A/B 之间的模型加载接口形状（B 的 AWQ 模型如何被 A 的引擎调用）
- [ ] Git 分支命名约定（建议 `a/`、`b/` 前缀区分 owner）
- [ ] `contracts/` 维护责任 —— 跨轨道决策变化（如中途换 draft/target 模型对）由谁同步更新 ADR/坑表

## M0 — 迁移 + 复核 P1.0（阶段0，已完成，待收尾）

- [x] **[A+B]** 把 `src/gate_p1_0.py`、`gate_p1_0_mlx_crosscheck.py`、`prompts.py`、`results/*.json` 迁移进本仓库 —— 结果文件改名为 `results/p1_0_gate_result_b_track.json`、`results/p1_0_mlx_crosscheck_result.json`（顶层 `results/p1_0_gate_result.json` 是 Michael 用 `scripts/p1_acceptance.py` 等脚本产出的另一份 P1.0 证据，target 用的是 Qwen2.5-1.5B、alpha=0.7685，和这里 B 用 3B target 跑出的 0.7024 是两份独立结果，不能同名覆盖）
- [x] **[B]** 通读 `gate_p1_0.py` 核心逻辑，确认无假阳性风险（如两条推理路径共享缓存导致"殊途同归"）—— draft_model/target_model 是两个独立加载的 `AutoModelForCausalLM` 实例（不同尺寸，不可能共享权重），target 侧每轮对 `candidate`（context+draft_tokens）做一次完整前向重新计算 logits，没有向 draft_model 传入或复用任何 `past_key_values`，两条路径之间没有缓存共享，未发现假阳性风险
- [ ] **[A+B]** 双方确认 P1.0 结果（`overall_alpha=0.7024`, PASS）可作为阶段0完成依据，正式进入阶段1/2

## M1 — AWQ 量化基础（支柱2）【B】

- [ ] P2.0 激活统计采集
- [ ] P2.1 逐通道缩放校准算法
- [ ] P2.2 跨分布鲁棒性实验（AWQ vs GPTQ 矩阵）
- [ ] P2.3 校准集大小消融
- [x] P2.4（Mac部分）`mlx_lm.awq` 交叉验证（可提前，不依赖 P1）——PR见下，云端GPTQ对比部分不在本次范围内

## M2 — 投机解码核心（支柱1）【A】

> 用户 2026-08-28 起接手 Track A（Michael 停更，见记忆 `project_specter_solo_pivot`），`[A]` 标签保留。模型对沿用 Michael P1.0 gate 的 draft=Qwen2.5-0.5B-Instruct / target=Qwen2.5-1.5B-Instruct（`src/model_loader.py` 默认值，1.5B 在 24GB Mac 上成本低）。分支 `a/p1-1-to-1-4-spec-decode-core`。

- [x] P1.1 Rejection Sampling 核心算法 —— `src/rejection_sampling.py`（`speculative_step`/`speculative_generate`，temperature=0 时 one-hot 使贪心成为同一码路径的特例）；单测 `tests/test_rejection_sampling.py` 复现附录 A.1 手算算例（接受概率 0.571、拒绝后重采样分布 A 清零为 {A:0,B:1/3,C:2/3}）+ FakeModel 端到端验证 bonus token 来自目标模型（坑2）
- [x] P1.2 贪心模式验证器 —— `src/verify_greedy.py`，结果 `results/p1_2_greedy_verifier.json`。正向：8/8 prompt 与 target-only 贪心逐 token 完全一致（另与中和了 repetition_penalty 的 HF `generate` 独立交叉验证）。反向（§9.6风险3）：注入 bonus-from-draft(坑2) 和 force-accept 两个已知 bug，验证器分别在 5/8、7/8 prompt 上报出分叉 → 非盲
- [x] P1.3 采样模式验证器（含验证器故障注入测试，§9.6风险3）—— `src/verify_sampling.py`，结果 `results/p1_3_sampling_verifier.json`。①统计：实测 α=0.788 vs 理论 E[min(p,q)]=0.798，z=−0.95（Poisson-binomial SE 内吻合）；②坑2 bonus 溯源：Δ=mean(log p_TM−log p_DM)，正确实现 +0.38 / bug −0.41 清晰分离（贪心模式下坑2常不可见，故采样模式补这条分布层面的注入检查）；③下游 parity：ROUGE-L（LCS-F1 内联实现，避免 HumanEval 依赖/沙箱）spec 29.55 vs target-only 31.25，差 1.70 点（<2）
- [x] P1.4 γ 扫描 —— `src/gamma_sweep.py`，结果 `results/p1_4_gamma_sweep.json`。γ∈{1,3,5,7,10}，每 γ 跑 3 seed（采样模式，使重复真有方差）。接受长度 std 随 γ 单调递增：0.42/1.24/1.95/2.55/3.24（AdaEDL "DL 越大方差越大" 的实证基础，支撑支柱5自适应γ）。墙钟加速比 1.22x→0.78x（MPS 无 KV cache，仅指示性；真实吞吐曲线是 P4）；γ=1 vs γ=3 差异不显著，其余相邻对显著。
- 顺带记录新坑：`notes/project_plan_v9.md` §9.2 坑13（块内 EOS 停止逻辑）

## M3 — AgentBench 评测（支柱3）【B】

- [ ] P3.0 任务集改编（15-20个任务，含3-5个 held-out 标记；可提前，不依赖 P1）
- [ ] P3.1 接受率对比实验（需 P1 完成）

## M4 — 自适应控制器 Mac 部分（支柱5前半）【A】（部分完成：P5.0 + P5.1 done，P5.3 未做）

> 用户 2026-08-28 继续 Track A（Michael 停更）。直接 commit 到 `main`，不开分支/PR。模型对沿用 draft=Qwen2.5-0.5B-Instruct / target=Qwen2.5-1.5B-Instruct。GammaTune 超参数用论文默认值写死不调参（`GammaTuneConfig`，§9.6 风险1）。

- [x] P5.0 GammaTune 算法实现 —— `src/gammatune.py`：`gammatune_update` 纯函数复现附录 A.2 三轮算例（`A=[3,2,3] → (γ,γ̄)=(5,3.0)/(3,2.7)/(5,2.7)`，单测 `tests/test_gammatune.py`）；`gammatune_generate` 复用 `speculative_step` 不重写 rejection sampling，块内 EOS 截断照坑13，`carry_state=(γ,γ̄)` 支持跨 prompt 传状态。实验 `src/verify_gammatune.py`，结果 `results/p5_0_gammatune.json`：8 prompt × seed{0,1,2}，采样 temp=1.0，GammaTune vs 固定 γ∈{1,3,5,7}。**主指标 `mean_emitted_per_round`：GammaTune 2.94±0.00 < 最优固定 γ=7 的 3.67±0.01（区间不重叠）——在这对模型/这批 prompt 上 GammaTune 未跑赢固定 γ**，归因坑9（α≈0.79 稳定、accept 长度方差不足）+ 新发现坑14（该指标对 γ 单调、无 draft 代价项，结构上偏袒大 γ）。γ 轨迹健康（均值 3.41，落 γ_max 仅 1.4%）。成本模型次级分析（`sum(emitted)/(n·c+Σγ)`，实测 c≈1.26 + 文献 c∈{4,7,10}）：GammaTune 落最优簇内（差 0.8%/3.0%/5.9%）、稳优于极端 γ=1/γ=7。
- [x] P5.1 波动场景鲁棒性测试 —— `src/nonstationary_prompts.py`：段 A 8 条代码/结构化 + 段 B 8 条开放聊天，序列 `A_to_B` / `A_to_B_to_A` / `ABAB`，控制器状态用 `carry_state` 跨 prompt 连续。实验 `src/verify_nonstationary.py`，结果 `results/p5_1_nonstationary.json`，≥3 seed。**三条序列主指标 GammaTune 均落后最优固定 γ=7**（A→B 2.86±0.03 vs 3.81、A→B→A 3.09±0.07 vs 4.09、ABAB 2.82±0.32 vs 3.81，区间不重叠）——**复现坑9 / 论文 Limitations**：历史依赖的控制器在这种切换频率下收益打折，不算失败，是 P5.3 需要更强场景感知的证据。段内行为符合预期：段 A（代码，高 α）emitted≈3.5–3.8、段 B（聊天，低 α）≈2.3；ABAB 高频切换使 GammaTune 方差明显放大（std 0.32 vs 其它序列 <0.07）。γ 轨迹逐 seed 存进 JSON。
- [ ] P5.3 Batch-aware 熔断器（含周期性重探测机制）—— 本次未做，见 `notes/M4-P5.3-Batch熔断器任务prompt_2026-08-28.md`（决策：P5.3 与 GammaTune 关注点正交，且依赖 P4 batch 曲线，单独会话做）
- 顺带记录新坑：`notes/project_plan_v9.md` §9.2 坑14（硬件无关主指标 `emitted_per_round` 对 γ 单调、无 draft 代价项，评自适应控制/熔断收益必须用成本模型或带 KV cache 墙钟）

## M5 — Batch 交叉点 Mac 部分（支柱4前半）【A+B 分曲线】

- [ ] **[A]** 投机解码吞吐曲线（batch ∈ {1,4,8,16,32,64}）
- [x] **[B]** 量化吞吐曲线（同 batch 范围，用 B 自己的 AWQ 模型）—— `src/p4_quant_throughput_curve.py`，结果见 `src/results/p4_quant_throughput_result.json`；batch=1/4/8/16/32/64 全部测通，未触发 OOM
- [x] **[A+B 各自]** 显存占用记录（各记自己那条曲线）—— 仅 B 这一半完成（随上一条一并记录）；A 那一半仍未开始（A 的 P1 尚未完成）
- [ ] **[A+B 联合]** 合并两条曲线，标出交叉点，写 P4.1 记录 —— 阻塞于 A 的曲线未产出，不在此提交范围内

## M6 — 云端启动会【A+B 联合】

- [ ] 确认"待确认事项"里的 GPU 实例数量决定
- [ ] 确认预算记账方式
- [ ] 云端脚本本地 dummy dry-run 最后确认（§10 纪律）

## M7 — 云端规模化验证（支柱2补全 + 支柱4后半 + 支柱5中段）

- [ ] **[B]** P2.4 云端补全：LLM Compressor GPTQ 真实速度对比
- [ ] **[A]** P4.2 投机解码曲线（云端大模型）
- [ ] **[B]** P4.2 量化曲线（云端大模型）
- [ ] **[A]** P5.2 AWQ 臂（量化-γ耦合）
- [ ] **[B]** P5.2 BnB 对照臂（同源复现 SpecKV 设置，ADR-008）
- [ ] **[A+B 联合]** 合并 P4.2 两曲线 + P5.2 两臂结果

## M8 — 生产基线对比 + held-out 最终确认（支柱5收尾）

- [ ] **[A]** P5.4 HF 双基线（`num_assistant_tokens_schedule` + `assistant_confidence_threshold`）
- [ ] **[B]** P5.4 BanditSpec 公开代码克隆运行
- [ ] **[A+B 联合，必须双人在场]** held-out 任务集最终跑一次（§9.6风险1，只能跑这一次，不能回头调整）

## M9 — 产出（阶段7）【A+B 联合】

- [ ] GitHub repo 定稿（代码 + README + 引用 + 诚实边界说明）
- [ ] 简历 bullet 定稿

---

详细依据、任务级依赖分析、并行窗口表、里程碑验收标准见 [`notes/two_developer_plan_v2.md`](notes/two_developer_plan_v2.md)。此文件只负责"打勾"，不重复论证——如果某条任务的归属或顺序需要改，先改那份文档，再回来同步这里。
