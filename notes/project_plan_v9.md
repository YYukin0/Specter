# Specter 项目计划 v9 —— 资深架构师评审后的可行性修正版

用户要求以资深架构师视角，从可行性和是否已有成熟方案两个角度评审v8。评审基于实证调研，发现一处实验设计问题、一处工作量估计偏乐观，以及若干未被计划采用的现成方案。v9据此修改了相关章节，架构方向和支柱1-5的整体设计保持不变；具体改动见下方§0版本信息表，正文中标注"v9新增"或"v9修正"的段落是改动落点。

---

## 0. 版本信息

| 版本 | 内容 |
|---|---|
| v1 | 三支柱构想，基于摘要级调研 |
| v2 | 精读论文原文后修正三处理论风险 |
| v3 | 生产复盘后新增支柱4（batch size 交叉点） |
| v4 | 发现平台兼容性硬约束，新增支柱5（自适应控制器），补充12个具体已知坑 |
| v5 | 按工程设计文档骨架重组，补齐目标/非目标、考虑过的其他方案、依赖假设、里程碑判据、悬而未决问题 |
| v6 | 对标参考文档前18页的密度：三层指标体系、机制级拆分（P1.0-P5.4）、端到端走读、叙事收尾、三份附录 |
| v7 | 读完参考文档全部28页后补四处颗粒度缺口：走读加入失败/恢复场景、公式配数值算例、真实文献列表内嵌（而非外链）、附录C加可执行的诊断分析代码；新增"研究诚信护栏"一节 |
| v8 | 精读5篇自适应控制论文原文，更正一处错误引用(SpecKV"41.2%"实为误归因，真实数字2→8/4倍)，重新措辞BanditSpec"太复杂"的不准确判断，补充若干可精确溯源的新引用 |
| **v9**（本版本） | 资深架构师视角评审：发现P5.2实验用AWQ对标BnB实验结果存在方法论错配风险并修正、AWQ工作量估计从2天调整为3-4天、补齐mlx-lm原生AWQ/BanditSpec公开代码/HF双重内置基线三处"已有成熟方案未利用"的信息差、§8补充self-speculative作为遗漏的备选方案、新增云端预算的本地dry-run硬性约束 |

---

## 1. 摘要

Specter 是一个面向 Agent 场景的本地推理加速引擎：手写实现投机解码（含严格正确性证明）和 AWQ 风格量化，用一套系统原理（内存带宽优化 vs 计算瓶颈切换点）统一两者的失效边界，再加一个训练-free 的自适应投机步长控制器。目标硬件：个人 24GB 统一内存 Mac（开发/调试，$0）+ 短期云端 GPU 租用（规模化验证，$30-50）。当前状态：计划已经过两轮论文精读（含一轮针对5篇自适应控制论文的原文核对）、一轮生产实践调研、一轮针对性踩坑调研、一轮资深架构师可行性评审，尚未开始任何代码实现。

---

## 2. 背景与问题陈述

### 2.1 为什么现在做这件事有意义（不只是"学新技术"）

一个残酷的招聘现状,恰好也是 Time x Mart 那份文档里引用过的数据(Gartner 预测到 2028 年全球四分之一的候选人档案将是伪造的; LinkedIn 每分钟吞下约11,000份几乎一模一样的申请):当 AI 能替任何人生成一份看起来专业的项目描述时,简历上的一行 bullet 本身已经不构成信号。真正稀缺的是**可验证的技术判断力痕迹**——你在实现之前有没有读懂边界条件、有没有踩过真实的坑、有没有诚实地说清楚"我为什么没选那个更优的方案"。

Specter 的定位是一份**留下判断痕迹的工程记录**：12个具体的、有来源的已知坑，8处"考虑过但拒绝"的替代方案及其理由，以及一个从平台兼容性调研中反推出来的架构决定。这些内容 AI 可以帮忙写代码，但编不出"调研后发现 vLLM 在 Mac 上跑不动，因此把方案拆成两个阶段"这类只有真正做过调研才会有的具体细节。

**差异化的诚实边界（v9新增）**：GitHub上已经有多个成熟的"手写投机解码"公开实现(romsto/Speculative-Decoding、suryavanshi/speculative_decoding、dilab-zju/self-speculative-decoding等)，单纯"我手写了投机解码"这句话本身，作为简历差异化程度已经不高。这个项目的差异化重点在于三处组合：量化-投机步长耦合实验、batch交叉点的系统性刻画、AgentBench结构化输出场景验证，而不是"从零实现"这个动作本身——文献里很少有人把这三件事放在同一个项目里系统性地测过，故事重心应该放在这里。

### 2.2 技术现状与缺口

投机解码和量化在 2025-2026 已经是主流推理框架的标配(vLLM/SGLang 一个 flag 就能开)——这意味着"会调库"本身已经没有技术含量。缺口在于:几乎没有人系统性地验证过——(a) 手写实现能否达到和调库一致的正确性保证,(b) 这些技巧在什么条件下失效(batch size、压缩程度),(c) 主流生产库还没内置的"自适应控制"能做到什么程度。这个项目要填的就是这个"调库拿不到"的部分。

---

## 3. 目标与非目标

### 目标
1. 手写实现投机解码核心采样算法,贪心模式逐 token 精确验证正确性,采样模式统计层面验证分布等价。
2. 手写实现 AWQ 风格激活感知量化校准,诚实区分"自己写的算法"和"复用的现成 kernel"。
3. 用统一系统原理(内存带宽 vs 计算瓶颈)解释投机解码和量化各自的失效边界,实测出具体交叉点。
4. 实现训练-free 自适应投机步长控制器,验证其在压缩模型和生产基线(HF Transformers 内置动态投机)面前的表现。
5. 用 AgentBench 方法论改编的 agent 工具调用评测集,验证结构化输出场景下投机解码的优势。

### 非目标(显式声明,防止范围蔓延)
1. 不重新实现 GPTQ——太复杂,用 LLM Compressor 做对比基线。BitsAndBytes NF4 例外：HF Transformers内置几行配置即可开启，不算"重新实现"，P5.2的对照臂会用到（见坑10）。
2. 不做完整 AgentBench 八个环境——只改编 OS/bash 子环境。
3. 不重新实现 BanditSpec 的 bandit 理论方法或 AdaEDL 的熵基提前停止——只作对比引用；BanditSpec 有公开代码(`github.com/sail-sg/BanditSpec`)，P5.4 允许直接**跑**这份代码做基线对比，但不允许把它的算法搬进自己的P5实现里（"跑现成代码做对比"和"重新实现算法"是两回事，见§7 P5.4）。
4. 不训练专用 speculative head(EAGLE/Medusa 那种)——没有多卡训练资源,经典两模型方案是预算约束下的合理取舍。
5. 不追求在 Mac 上跑出真实速度数字——但量化速度例外：mlx-lm 有原生 AWQ 支持(`mlx_lm.awq`)，可以在 Mac 上产出真实的、Metal加速的AWQ速度数字，见§7 P2.4；此例外仅限"借用现成工具产出一个交叉验证数字"，不改变"投机解码真实速度只能在云端测"这条主约束（原因见§9.1风险B的具体技术依据）。
6. 不做多租户/高并发服务化——关注单用户/低 batch 的本地 agent 场景。
7. 不采用self-speculative/layer-skip架构（v9新增，见§8）——两模型架构本身是项目故事的核心，self-speculative会规避掉正想展示排查能力的那类失效模式(tokenizer不一致、bonus token跨模型采样)。

---

## 4. 指导原则(Tenets)

1. **能在免费环境验证的,绝不烧云端预算**——决定 Mac/云端两阶段分工；云端阶段的脚本必须先在本地用极小dummy模型跑通全流程再上云（v9新增，见§10）。
2. **诚实边界优于假装SOTA**——每处方案选择都要写清楚"生产界最优解是什么、我们为什么没选它"，给出的理由应是实际的技术判断依据。
3. **正确性验证要有可复现的数学/统计依据**——贪心逐token精确比对、采样统计检验+任务指标双重验证。
4. **每个技巧不仅要证明"有效",还要测出"什么时候失效"**——batch size交叉点实验和量化-步长耦合实验的存在理由。
5. **基线出来之前不编造目标数字**(借用 Time x Mart 文档的原则)——第5节的所有阈值都标注"预期区间"而非"承诺值",实际数字以阶段0-6跑出来的结果为准。
6. **测量本身也要接受怀疑**——一个"符合预期"的实验结果不会自动免检;必须先证明测量工具本身是可靠的(测试的测试),再采信结果。
7. **引用别人的数字之前,先确认自己真的读对了**——转述/摘要会丢失或扭曲归因关系；任何要写进计划或最终报告的具体数字，引用前必须能标注清楚"这句话在原文哪一节、是谁的实验结果、不是谁引用谁的背景数字"。
8. **对标一个实验结果之前,先确认自己的方法和对方是不是同一件事(v9新增)**——不同的量化方法/不同的实验设置测出来的现象，即使"名字一样"（都叫"量化程度影响最优γ"），底层机制也可能不同，直接拿数字对标而不核对方法论前提，会把"方法不同导致的差异"误判为"实验失败"（P5.2的BnB/AWQ错配是这条原则的具体触发案例）。
9. **别人已经做过的部分,优先用现成工具交叉验证(v9新增)**——mlx-lm的原生AWQ、BanditSpec的公开代码都是可以直接拿来当参照系的现成资源，用好这些能省下时间和预算去打磨差异化的部分。

---

## 5. 指标体系(北极星 / 伴随 / 护栏)

> 沿用 Time x Mart 文档 §6 的三层指标框架。**以下数字均来自已读论文的原始实验结果，代表"这个方法在别人的实验里做到过什么量级"，并非 Specter 自身的实测承诺**——Specter 自己跑出来的数字以最终 README 为准,可能高于也可能低于这些参考区间,这正是实验要回答的问题。

### 北极星指标
**正确性优先于速度**:贪心模式下投机解码输出与目标模型直接推理的 token 级一致率——目标 100%(允许可忽略的浮点误差),这是唯一不允许"这次没做到但下次再说"的指标。

### 伴随指标(每个支柱一个,参考区间来自文献)

| 指标 | 参考区间(来自文献,非承诺值) | 来源 |
|---|---|---|
| 投机解码接受率 α | ≥0.65 才有净加速(低于则收益递减) | mlx-lm 生产实践阈值 |
| 投机解码整体加速比 | 2-3x(生产环境实测) | IBM/PyTorch 官方生产博客 |
| 4-bit量化显存降低 | ~70%,perplexity涨幅<5%(同分布校准) | AWQ论文 |
| 跨分布校准 perplexity 涨幅差异 | AWQ +0.5-0.6 vs GPTQ +2.3-4.9 | AWQ论文实验 |
| GammaTune 风格自适应控制器 vs 固定γ | 15%±5% | GammaTune论文 SpecBench结果 |
| 量化程度对最优γ的偏移幅度 | SpecKV在BnB压缩下最高4倍(FP16下γ=2 → INT8下γ=8)；**AWQ下的偏移幅度是本项目要回答的开放问题,不预设应该复现同等量级(v9更正，见坑10)** | SpecKV论文 Table 1 |
| 高负载下投机解码相对无投机的倒退幅度 | 最高30.25% | Nightjar论文生产实测(RTX 4090, DeepSeek-R1-Distill-Qwen-7B, ShareGPT) |

### 护栏指标(不能突破的底线)
- 采样模式下游任务指标差距(HumanEval pass@1 / ROUGE)必须 <2 个点,否则视为"分布等价"验证失败,需要回头检查采样实现而不是接受结果。
- 云端预算硬顶 $50,超支即停止规模化验证,用已有结果收尾,不追加开支。
- 任何"自适应控制器让情况变得更差"的场景(GammaTune论文承认的边界)必须被记录进报告,不能因为不好看就不提——诚实边界本身是护栏。

---

## 6. 现有方案与文献综述

条件性结论(完整分类文献列表见**附录E**,不再用外链指针代替内容): 投机解码/量化已是主流框架标配, 真正区分度在手写实现+正确性验证+边界刻画; IBM/PyTorch生产复盘证实代码/结构化输出场景效果最好, batch>64开始吞吐下降; AWQ相对GPTQ对校准集分布更鲁棒; 2024-2026自适应投机控制是活跃前沿方向(至少7-8篇独立工作，v9评审时新发现的AdaSD是又一例，方向不重叠于P5.2，见附录E), 且动态γ调整已是HF Transformers默认行为(4.45.0起，且v9评审发现该默认行为其实包含两套不同机制，见§7 P5.4)。

---

## 7. 方案与架构(机制级拆分,对标 R0-R17 的颗粒度)

### 架构总览

```
Mac 开发阶段（阶段 0-4, $0）：HF Transformers (PyTorch, MPS backend)
云端 GPU 阶段（阶段 5-6, $30-50）：vLLM 对比基线 + LLM Compressor
```

### 支柱1：手写投机解码 + 正确性验证

**P1.0 — 前置存活性 gate（S，半天）**
断言 `draft_tokenizer.get_vocab() == target_tokenizer.get_vocab()`；测 α，两级门槛：<0.4 换模型对；0.4-0.65 降低预期继续；≥0.65 正常推进。用 mlx-lm 跑同一模型对交叉验证 α 数量级。

**P1.1 — Rejection Sampling 核心算法（M，2-3天）**
标准算法：草稿模型自回归生成 γ 个候选 token，目标模型单次前向对全部候选打分；接受判据 $p_{TM}(x) \geq p_{DM}(x)$ 时接受，否则以 $1 - p_{TM}(x)/p_{DM}(x)$ 概率拒绝，从调整分布 $p'_{TM}(x) = \text{norm}(\max(0, p_{TM}(x) - p_{DM}(x)))$ 重采样。**关键实现检查点**：bonus token 必须来自目标模型分布，不是草稿模型（坑2）。数值算例见**附录A.1**。

**P1.2 — 贪心模式验证器（S，1天）**
逐 token 严格比对投机解码输出 vs 目标模型直接推理输出，要求 100% 一致（或仅浮点误差）。

**P1.3 — 采样模式验证器（M，2天）**
统计层面验证实测 α 与理论公式 $\alpha = E[\min(p,q)]$ 吻合；下游任务指标 parity(HumanEval pass@1 / ROUGE，差距<2点)。

**P1.4 — γ 扫描（S，1天）**
γ ∈ {1,3,5,7,10}，记录接受长度分布（预期形状参考 AdaEDL 论文 Figure 1 / Appendix Fig 7c：Dolly-15k创作数据集,目标模型Llama2-7B,草稿模型是对齐微调过的115M模型,不同 DL 下方差递增，DL=3→std≈1.2，DL=7→std≈1.92，DL=16→std≈2.35，这是"为什么需要自适应γ"的实证基础，支柱5会复现类似分布）。

### 支柱2：AWQ 量化

**P2.0 — 激活统计采集（S，1天）**
在校准集上跑前向，收集每层激活值的逐通道统计量。

**P2.1 — 逐通道缩放校准算法（M，3-4天，v9调整：原估计2天偏乐观）**
AWQ 核心思想：找到一组逐通道缩放因子，使量化后误差最小化，同时保护"显著性权重通道"（由激活值幅度决定）不被粗暴量化。直觉性数值算例见**附录A.3**。**工作量修正说明**：实现难点主要在于fake-quantize模拟(round→clamp→dequantize)的实现顺序调试，而非缩放因子的数学步骤本身——顺序或数值范围出错会导致perplexity大幅偏离预期，且往往需要逐层排查才能定位问题所在，加上激活值统计采集时机(hook挂在模块输入还是输出)和缩放因子网格搜索范围也需要反复试验。2天足以写出初版实现，3-4天是让perplexity数字可信的现实估计。

**P2.2 — 跨分布鲁棒性实验（S，1天，坑5）**
两个分布明显不同的校准集（代码语料 vs 自然语言）分别校准，交叉在两种评测集上测 perplexity，验证"AWQ 跨分布涨幅远小于 GPTQ"这一 AWQ 论文核心论点是否在自己的模型上复现。

**P2.3 — 校准集大小消融（S，半天，坑6）**
"校准集大小 vs perplexity"曲线，验证过小样本(几十条量级)是否出现过拟合迹象。

**P2.4 — 速度对比（M，v9调整：Mac可先出一部分，云端补全）**
云端阶段用 LLM Compressor 跑 GPTQ 基线，测真实加速比——这个数字 Mac 阶段无法产出（平台约束，具体技术原因见§9.1风险B：PyTorch旧版量化API从未移植到MPS后端）。**v9新增**：Mac阶段可以先用 `mlx_lm.awq`（mlx-lm 原生支持的AWQ量化路径，不是自己写的实现）跑一个真实的、Metal加速的Mac端AWQ速度/内存数字，作为免费的交叉验证点——不能替代云端阶段和GPTQ的对比，但能在花云端预算之前先建立一个"AWQ大概应该长什么样"的参照系，也能减少云端阶段专门为"要一个AWQ真实速度数字"分配的时间。

### 支柱3：AgentBench-OS 子集评测

**P3.0 — 任务集改编（M，2天）**
15-20个 bash/工具调用任务，改编自 AgentBench OS 子环境，具体任务清单见附录B。

**P3.1 — 接受率对比实验（S，1天）**
同一模型对，工具调用/结构化输出场景 vs 自由文本对话场景，对比 α 差异，验证"结构化输出接受率更高"这一生产界共识。

### 支柱4：Batch Size 交叉点实验

**P4.0 — Mac 小模型扫描（S，1天）**
batch ∈ {1,4,8,16,32,64}，7B级别模型，投机解码和量化各测一条吞吐提升曲线。

**P4.1 — 显存占用记录（S，与P4.0同步）**
每个 batch size 下草稿模型权重和 KV cache 各自估算显存占用(坑8)。

**P4.2 — 云端大模型验证（M，云端阶段）**
更大模型/更高 batch 验证交叉点是否随模型尺寸偏移，对标 Nightjar 30.25% 倒退量级(坑7)。

### 支柱5：自适应投机步长控制器

**P5.0 — GammaTune 算法实现（M，2天）**
Algorithm 1：接受数 A 与当前 γ 相等时扩窗（$\gamma \leftarrow A + \delta$）；否则 EMA 更新 $\bar\gamma \leftarrow \text{clip}(\gamma_{\min},\gamma_{\max},(1-\eta)\bar\gamma + \eta A)$，$\gamma \leftarrow \lceil \bar\gamma \rceil$。三轮数值算例见**附录A.2**。

**P5.1 — 波动场景鲁棒性测试（S，1天，坑9）**
混合代码生成+开放聊天的 prompt 序列，中途切换任务类型，验证 GammaTune 在非稳态场景下是否还有优势，还是像论文承认的那样收益打折。

**P5.2 — 量化-γ耦合实验（M，云端阶段，坑10，v9重新设计）**
复用 P2 的量化模型(FP16 vs 自己的4-bit AWQ版本)，对同一 draft/target 组合分别扫描最优γ。**v9修正**：SpecKV的2→8偏移是在BitsAndBytes压缩下测出来的，AWQ的逐通道缩放本身是为了保护输出分布不被大幅扰动而设计的，两者不是同一件事，直接拿AWQ的结果去对标BnB测出的具体倍数存在方法论错配风险。因此：(a) 增加一个可选的BitsAndBytes NF4对照臂（HF Transformers内置`load_in_4bit=True`几行配置，不算重新实现GPTQ，云端阶段顺手跑一遍即可）,直接复现SpecKV的原始设置作为"同源对照"；(b) 自己AWQ模型的实验成功判据从"复现4倍量级"改写为"验证方向一致(压缩程度越高最优γ越小)即可，具体偏移量级本身就是可报告的发现——如果AWQ下偏移明显小于BnB，这本身是一个值得写进报告的结论(AWQ对投机解码更友好)，不是实验失败"。

**P5.3 — Batch-aware 熔断器（M，2天，坑11）**
高 batch 时自动降级/禁用投机解码；**必须包含**周期性重探测机制(避免 DSD 式"重激活难题")；**必须实测**切换开销(KV cache重建成本，避免 BanditSpec 式"假设切换免费"的问题；Nightjar 实测同类切换开销量级为17.87-102.03ms，随输入长度/batch size变化，可作为自己实测数字是否合理的参考量级)。

**P5.4 — 生产基线对比（M，1-2天，v9拆分为三个明确基线）**
不再笼统写"对标HF内置"，明确对比三个基线：(a) HF Transformers `num_assistant_tokens_schedule="heuristic"`(简单启发式，4.45.0起默认)；(b) HF Transformers `assistant_confidence_threshold`(基于置信度阈值的无监督自适应机制，是Intel/HF"Dynamic Speculation Lookahead"论文的落地版本，官方数据显示最高2.7x提速，方法论上比(a)更接近AdaEDL的熵/置信度思路，是更强的基线，不应该被漏掉)；(c) **v9新增**：BanditSpec公开代码(`github.com/sail-sg/BanditSpec`，兼容LLaMA/Qwen2架构)直接克隆跑一遍作为第三个对比点——用现成代码做对比，不是重新实现它的bandit理论，不违反§3非目标3。给出定量对比结论(不要求超越，要求诚实数字)。

---

### 支柱6：部署工程深度（方向 B，2026-08-28 新增，Mac 本地 $0）

**背景**：Mac 侧研究（支柱1-5 的本地部分）基本做完，没有推翻任何论文结论——全是成功复现或复现论文自承认的局限。用户据此决定：不再追新研究发现，改把工程/部署侧做深。完整论证 + 逐条文献 challenge + 四轮 novelty 复查见 [`notes/deployment-depth-plan_2026-08-28.md`](deployment-depth-plan_2026-08-28.md)。三个硬伤要补：全项目刻意不用 KV cache（所有墙钟数字都标"indicative only"）、熔断器的 batch 信号是合成注入的（坑15）、AWQ 是 fake-quant（无真实内存/速度节省）。

**P6.0 — KV-cache 正确的单序列投机解码（keystone，~2 单元）— DONE**（5a352c9 + verify 3e7f83f）
`src/spec_kv.py`。难点=部分接受回滚。**实现纠正了 plan 的裁剪公式（→ 坑18）**：plan 说 target cache 裁到 `prefix+k+1`、draft 裁到 `prefix+k`，实测**两个 cache 都裁到 `prefix+n_accepted`**（满接受时 `prefix+k`）——resample/bonus token 是模型的**输出**、从未作为**输入**喂过任何前向，其 KV 不存在，多裁/少裁都错。对齐 HF `generation/utils.py`（`n = candidate_length - n_matches; cache.crop(-n)`）+ 第一性原理。`_crop_to(cache, target_len)` 薄封装锁 `DynamicCache.crop` 负数语义（正数走 deprecated 的"裁到绝对长度"、5.18 移除）。三向 token-exact parity 契约（写进 §9.2 新验收纪律）成立：`speculative_generate_kv(temp=0)` == `speculative_generate` == `target_only_generate_kv`；采样同 seed 长共同前缀。`tests/test_spec_kv.py` 34 tests（FakeModel cache 长度不变量 == `len(committed)-1`、三向 greedy parity、采样 parity、回滚压力 phase=2.4、EOS mid-block、injection）。`results/p6_0_kv_cache_speculative.json`（978s，8 prompt × γ∈{1,3,5} × seed∈{0,1,2}）：**greedy parity 8/8 prompt bit-exact（全 γ）**；采样共同前缀 0.92–0.99；draft tokens/gen-token 从 ~23–60（无 cache）塌到 ~1.8–2.3（KV）；KV cache 本身给 target-only 2.70×。**诚实结论**：spec-KV vs **KV-cached** target-only 只有 ~0.93–1.01×（打平）——0.5B/1.5B 这对在这台 Mac 上 α≈0.77、accept_len≈1.85(γ=3)，投机相对公平基线不赚；相对旧的无 cache 路径 2.1–2.4×。**不 novel**（HF assisted-gen / gpt-fast 早有），价值是 portfolio / craft + 这个"把公平基线也加上 KV cache 后加速比塌回 1×"的诚实数字。

**P6.5 — `specdiff`：投机解码故障注入 + 差分调试器（头条交付物，~2.5 单元）— Parts 1–5 DONE + O2 DONE；剩 Part 6 工程故事**（a143f9c + b5d2ae9 + 本次）
与 P6.0 并行起步（O1/O4 + M-SAMPLE/M-CTRL 只依赖 `rejection_sampling.py`）。四轮文献复查（7+ 检索角度，见 deployment-depth-plan §7 C6）确认：**没有公开发表的、可复用的、投机解码专用的主动种障式测试方法学**——相邻工作（vLLM in-tree e2e、《Batch Spec Decoding Done Right》的 batch 不变量 + EQSPEC/EXSPEC 实现、Ekka arXiv:2606.04594 的 agentic 静默错误 root-cause、DiFR/LLM-42/MarginGate 的 trace 自洽验证器）各差一步。
- `src/spec_oracles.py` —— O1（FakeModel 符号级 exact，唯一精确 oracle；position-one-hot、cached forward 与全量前向 bit-identical）/ O3（采样匹配 seed vs clean spec 参照，超出即杀）/ O4（结构不变量常开断言：KV 长度、pos 连续、概率守恒 sum-to-one）/ **O5（Part 5，批量路径等价保持）：同一变异算子同时开在 `spec_kv_batch.run_round` 批量跑和 N 个单序列 `speculative_generate_kv` 上，两边必须逐 token bit-exact —— 不是变异-vs-干净（那是 O1/O3），问的是"共享代码被打断时批量驱动是否还等于单序列循环"。17 个算子全部 `broken=[]`，即 P6.1 的输出等价在对抗变异下仍由构造成立**。
- **O2（真实模型 CPU fp32 greedy exact）DONE**（本次）—— `src/verify_p6_5_o2.py` → `results/p6_5_o2_real_model.json`（3:12，真实 Qwen2.5 0.5B/1.5B CPU fp32）。四部分：**A** 干净 greedy spec == greedy target-only 逐 token bit-exact，3/3 prompt 通过（`speculative_generate_kv` docstring 只保证 MPS fp16 长公共前缀，CPU fp32 上真的精确成立）；**B** 对 O1 杀的 9 个算子重跑——O2 抓到 5 个（accept-logic 3 + adjusted_abs_not_relu + bonus_token_from_draft@33tok），**漏掉全部 3 个 M-POS（cache_position ±1 / 冻结）**：真实 RoPE 模型 greedy 下对这些 prompt 把小的/塌缩的位置误差直接吃掉，而 position-one-hot FakeModel 不能 → **真实模型输出 oracle 不是 FakeModel 的超集，FakeModel 过度放大位置敏感度**；也漏 `eos_ignored_midblock`（没 prompt 真的 mid-block 撞 EOS）。M-POS 仍要靠 O4 + specdiff 的 `UPSTREAM_KV_POS`（直接读 cache_position 向量），任何输出等价检查都抓不到。**C** 真实模型单序列 vs 左 padding 批量 row0 的末 token logits **bitwise 相同（delta 0.0）**——正确 attention mask 下 padding 位不贡献、reduction 序不变，arXiv:2607.17283 的 5.8e-3 是量化 Metal 后端 artefact，参考算法路径没有 batch 不确定性。**D** 构造"结构签名全同、只前缀 hash 不同"的 trace 对喂 `specdiff.classify` → 确实回 `BACKEND_NONDETERMINISM`。`tests/test_verify_p6_5_o2.py`（3：hermetic 契约 + budget-overshoot 容差 + 有权重时 smoke）。
- `src/spec_faultlib.py` —— **17 个活跃算子** + 5 个 `DEFERRED`（每个带"为什么还不激活"的理由串，P6.1 落地后更新）。M-SAMPLE 8（resample_from_target / adjusted_no_renormalize / adjusted_abs_not_relu / accept_ratio_inverted / accept_always / accept_strict / leniency_injected l=1.05 / bonus_token_from_draft）、M-CTRL 2（eos_ignored_midblock / force_accept_first）、M-KV 4（crop ±1 / 绝对vs相对 / 不裁）、M-POS 3（pos-id ±1 / 冻结）。`_patch` / `_wrap_math`（打 `_rs`+`_kv`+`_kvb` 三个 binding）/ `_inject`（走 `Injection`，同时 patch `_kv` 和 `_kvb` 的 `speculative_step_kv` binding，否则 force-accept/bonus 变异在批量路径静默 no-op）三种原语 + `apply()` ExitStack 组合。`DEFERRED` 里 `mask_leak_future` / `mask_left_pad_drift` 在每序列 cache 架构上**不可表达**（没有共享 ragged tensor / mask / 左 padding）——这个不可表达性本身就是 §7 C3 / 坑19；`mask_left_pad_drift`（arXiv:2510.22876 BSP 签名）只能打 `spec_batch.py` 的 masking 路线且需 RoPE 真实模型 → 转 P6.4。
- `src/specdiff.py` —— 差分调试器：lockstep 重跑 suspect + trusted 参照（采样模式）→ 每轮结构签名（draft 提案、喂给每个 cached forward 的 pos-id 向量、KV 长度、accept/reject 向量、n_accepted、emitted、前缀 hash）→ `bisect` 找首个签名不同的轮 R → 规则序分类器给机制（`UPSTREAM_KV_POS` / `SAMPLING_MATH` / `CONTROL_DESYNC` / `BACKEND_NONDETERMINISM` / `NO_DIVERGENCE`）。**规则序是关键**：上游 KV/pos 损坏会污染所有下游信号（decisions→n_accepted→回滚后 cache 长度），故先查上游（pos 向量、cached verify argmax vs 从头重算、draft 提案）、裸 cache 长度差摆最后一条 KV 规则，否则 M-KV bug 被误判成采样数学。instrumentation 全部走 module object（`_rs.collect_eos_ids` / `_kv.speculative_step_kv` / `_kv._cache_position` / `_kv.dist_from_logits`）否则 faultlib 的 monkeypatch 不生效。和 Ekka 同形状但规则式、本地、免费、投机解码专用。
- `results/p6_5_mutation_adequacy.json` —— mutation-adequacy 矩阵（≥3 seed）。**可引用 finding**：any-oracle kill 1.0，但 O1(greedy exact)=0.5625 / O3(采样)=0.75 / O4(不变量)=0.25；**每个 M-KV cache-管理 bug 对 greedy AND 采样输出等价都不可见，只有 O4 结构不变量断言能抓**；`adjusted_no_renormalize` 对所有输出 oracle 不可见（`torch.multinomial` 静默重归一化），只有 O4 显式 sum-to-one 断言能抓。
- `results/p6_5_specdiff_blind_hunt.json`（n=60）—— accuracy 0.933、**manifested 时机制精度 1.0（56/56，从不报错机制）**；漏的 4 个都是 leniency l=1.05 太温和当轮没触发。
- `results/p6_5_batch_invariance.json`（Part 5）+ `results/p6_5_o2_real_model.json`（O2 Part C）—— batch-invariance 分类基线：无 mutant 时 FakeModel 批量 == 单序列 max token delta 0（180 对），specdiff 干净路径全 `NO_DIVERGENCE`；**O2 实测**真实 Qwen2.5-1.5B CPU fp32 单序列 vs 左 padding 批量 row0 末 token logits delta = **0.0（bitwise）**——参考算法路径本就 batch 不变。arXiv:2607.17283 的 ~5.8e-3 是**量化 Metal 后端**的 batch-variant kernel reduction artefact（非算法 bug），specdiff 把"结构签名全同、只 token 不同"归 `BACKEND_NONDETERMINISM`（O2 Part D 契约测试钉死）。
- `tests/test_spec_faultlib.py` 32→52（+O5 参数化 17 + faultlib 触达批量 binding + clean O5）+ `tests/test_specdiff.py` 28 + `tests/test_spec_kv_batch.py` +1（batch-invariance smoke）（全套 pytest 59→153→184）。
- 剩余：2510.22876 BSP 签名复现（转 P6.4 真实模型，架构上不可表达）、Part 6 工程故事。O2 已 DONE（见上）；O2 的 batch-invariance 实测 delta = 0.0（CPU fp32 参考路径本就 batch 不变，非算法问题）。
- **诚实边界**：mutation testing 元思想是 1970 年代的；新的是投机解码专用算子目录 + O1/O3/O4 oracle 栈 + 规则式机制分类器，不是元思想。

**P6.1 — 输出等价的批量投机解码 + serving loop（~2 单元）DONE**（commit e0f8b58 + dtype 修 fda65fe）
`src/spec_kv_batch.py`（`SeqState` + `run_round`）+ `src/serving_loop.py`（`SpecServer`：请求队列、continuous batching、真信号熔断器）。**设计选择：每序列独立 KV cache** —— 把 EQSPEC 的矩形不变量推到"每序列"粒度：每个活跃序列自带 draft/target `DynamicCache` + 自己的 committed 列表，一"批量轮"= 对每序列各跑一次 `speculative_step_kv`。没有共享 ragged tensor 就没有东西会漂，**输出等价由构造保证**（`tests/test_spec_kv_batch.py` 钉死 batched round == N 个单序列 `speculative_generate_kv`，greedy + 采样 bit-exact；`assert_rectangular_invariant` 每轮断言每序列 cache == `len(committed)-1`）。**代价 + 换来的测量**：没有 kernel 级 ragged verify 批处理 → 吞吐随并发几乎平（**这个平坦就是 finding**：真正的 batch 加速需要把共享 ragged tensor 加回来，即需要付 EQSPEC 的 realignment tax）。`ragged_realignment_overhead(work) = 1 - Σwork/(n·max(work))` 把那个 tax 对"矩形 padded batch 反事实"**解析算出、不付**。熔断器按滚动 α<`alpha_floor` + 可选 target-only 延迟探针 trip；`len(active)` 只是输入、从不是判据。degraded 时每 `reprobe_every` 轮强制一次 spec 探针（坑11）。测试：`tests/test_spec_kv_batch.py`（6）+ `tests/test_serving_loop.py`（6），全套 pytest 165。
`src/verify_serving_loop.py` → `results/p6_1_serving_throughput.json`（319s；8 prompt、width∈{1,2,4,8}、breaker on/off、short/long 两 regime、real Qwen2.5 0.5B/1.5B MPS fp16）：
- **吞吐随 width 几乎平**：`speedup_vs_width1` 落在 0.94–1.03，没有 batch 加速（预期——每序列 cache = 无 kernel batching）。
- **spec-KV vs KV-cached target-only ~0.96–1.03×**（打平，和 P6.0 一致：α≈0.77、accept_len≈1.9）。
- **realignment tax（正确性税，算出不付）**：mean 随 width 1→4 涨到 0.073（short）/ 0.096（long），p90 达 0.68–0.74 —— 把 ragged verify 折成矩形 padded batch 会浪费 ~7–10% 均值 / ~70% p90 的 target 前向。w=8 反降到 0.007–0.02（8 序列每轮全跑 → work 更均匀 → spread 小、padding 少）；tax 在**中等 width** 最大。
- **熔断器**：16 个 run 里只 `long / w=2 / br=on` 触发一次（degrade 20 轮 + 1 次强制探针后恢复 —— 正是坑11 的"不重探测就永不恢复"机制在动），从不因 batch size 触发（w=8 从不 degrade）——坑15 的点验到了。
撞上 §7 C3（ragged tensor 正确性）→ **§9.2 坑19**；撞上 §7 C4（熔断器"大 batch → 没用"前提）→ **§9.2 坑20**。P6.5 扩到 batch 见 §7 C6 尾。

**P6.2 — 真实 int4（走 mlx-lm，~1 单元）DONE**（2026-08-28）
`src/verify_p6_2_real_int4.py`（子进程分臂 bench，克隆 `p2_4_mlx_awq_crosscheck.py` 模式）+ `results/p6_2_awq_int4_real.json`。目标 Qwen2.5-1.5B-Instruct，四方本地对比全 4-bit / g128：
- **fp16_mlx**（mlx-lm 加载 bf16 权重）：wikitext2 ppl **11.54**，decode **31 tok/s**，权重 **3.09 GB**
- **mlx_lm.awq**（`--num-samples 16 --seq 256 --n-grid 10`）：ppl **13.14（Δ+1.60）**，decode **104 tok/s（3.3×）**，权重 **0.84 GB（3.7× 更小）**，RSS 1.65 GB
- **mlx_lm.gptq**：**退化** —— bits=4 g=128 产出常数 `!` 输出 / wikitext2 nll = nan，`--num-samples` 16(seq256) 与 64(seq512) 两次都复现，所以不是标定数据不足；同 model/config 下 AWQ + RTN 都正常。未继续二分（group size？坏 fallback 层？mlx-lm 该架构 bug？—— 算力预算）。**结论：Apple silicon 上 `mlx_lm.awq` 可用，`mlx_lm.gptq`（0.31.3）对这个架构不可用。**
- **mlx_rtn_int4_g128**（`mlx_lm.convert -q`，仿射 RTN，无标定）：ppl **13.81（Δ+2.26）** —— 即 **AWQ 标定比朴素 RTN 多买 +0.66 ppl**，非 null 结果，标定确实有用。
未做**自研 AWQ scales → 真 int4 打包**臂：`mlx_lm.awq` 无自定义 scale 注入 API，P2.x 手写 AWQ 是 torch fake-quant 路径无 int4 打包；RTN 臂替代"无标定"参照。
ppl 口径：wikitext-2-raw-v1 test，非重叠 512-tok 块前 32 个（`awq_perplexity.load_eval_corpus`），mlx tokenizer，所有臂共享同一 token 数组 → Δ 内部自洽，但**不可**与 P2.x 的 torch 滑窗口径对比。交叉参照：P2.2 torch fake-quant Δ ≈ **+1.2**，真 MLX int4 AWQ Δ **+1.60** —— 同量级，**真打包比 fake-quant 差约 0.4 ppl，即 fake-quant 曾略乐观**（诚实性验证的答案）。
P5.2 的"AWQ vs BnB 改变最优 γ"已基本被 SpecKV（arXiv:2605.02888）抢先；本地 spec-decode 联动未跑（投机栈是 torch/MPS，MLX int4 模型进不去，且无 torch/MPS int4 运行时，§9.1 Risk B）。`tests/test_verify_p6_2_real_int4.py` 3 个（纯 helper + 有模型时跑 smoke）。撞上 mlx_lm.gptq 退化 → **§9.2 坑21**。

**P6.3 — live demo（~1 单元）DONE**（本次）。`src/demo/live.py`：stdlib + ANSI 终端 dashboard，零新依赖，基于 `SpecServer`。每轮重绘每序列流式文本 + round/mode/γ/accept-len/滚动α/realign-tax/agg tok-s/并发/队列/熔断器原因。三开关 `--no-spec`（每轮 degraded 纯 target 解码）/ `--gammatune`（`ServeConfig.gammatune_on`，每轮 batch-mean 接受长度跑 `gammatune_update`，demo 级非 per-stream）/ `--no-breaker`；`--compare` 打印 vs spec-off 的 speedup；`--fake` 用 FakeModel 零下载。`serving_loop.py` 加 `spec_enabled`/`gammatune_on`/`gamma_min`/`gamma_max` + `RoundInfo.round_gamma`。真实 smoke（3 prompt×32 tok）：spec+breaker 22.6 tok/s / 16 轮 vs spec-off 24.5 tok/s / 51 轮 = 0.92×（与 P6.0/P6.1 一致：这对模型在这台机器投机打平/微亏，但"轮数少 3×、墙钟持平"看得见）。`tests/test_demo.py` 5 个。asciinema/GIF 是用户手动步骤。

**P6.4 — 打包 + 工程故事（~1 单元）DONE**（本次）。`README.md` 全量重写（"规划阶段尚未开始" → 真实系统清单 + 195 tests + 诚实 headline：KV cache 2.7× 是大杠杆 / spec-decode 在这对模型这台机器打平 0.93–1.0× / 批量吞吐 flat / 真 int4 AWQ 3.7× 小 + 3.3× 快 + ppl+1.60 / adaptive-γ 无收益 / 输出等价是栈里最弱的测试）。6 篇工程故事 `docs/engineering-notes/0N-*.md`：01 确认偏误（坑16）/ 02 把 KV cache 做对（坑18，回滚锚点=喂过前向的 token 数，不是本轮碰过的）/ 03 批量正确性税（每序列 cache 等价由构造成立，realignment tax 中等并发最痛 p90 0.68–0.74@w4、w8 塌回 0.02）/ 04 熔断器真信号（坑15/20，batch-blind 指标 + 真 rolling α 重建，16 跑只 trip 1 次且从不因 batch size）/ 05 fake-quant vs 真 int4（runtime 决定你能测什么；坑21 gptq 退化当一等 finding；跨 harness Δ 不可相减）/ 06 测投机解码器（O1/O3/O4/O5/O2 lattice；真实模型输出 oracle **不是** FakeModel 超集——FakeModel 位置一热过度放大位置敏感度，所以 O1 杀 M-POS 而 O2 漏；batch 不变性在参考路径 bitwise，5.8e-3 是量化 Metal artefact）。坑表独立成 `docs/pitfalls.md`（坑1–21，动手撞上的 13–21 在前，literature 派生的 1–12 压缩成表）。pytest 195 不变（纯文档）。

依赖：P6.0 与 P6.5 并行起步 → P6.1 → P6.3 → P6.4；P6.2 任何时候可插。合计 ~9.5 单元、全本地、$0。**优先级高于 §11 路线图里 M6-M8 的云端条目。** 进度：P6.0/P6.1/P6.2/P6.3/P6.4 DONE；P6.5 Parts 1–5 + O2 DONE，剩 Part 6 工程故事（与 P6.4 故事 06 重叠，实质已覆盖）+ 2510.22876 BSP 签名复现（架构上不可表达，转真实模型演示，未做）。

**P6.6 —（支柱7 Bullet 3）自研 AWQ 接下游 eval（GSM8K + IFEval）** —— 归属支柱7（source of truth `notes/简历定稿计划-Specter_2026-08-28.md`），代码沿 `verify_p6_*` 家族放。`src/build_self_awq_hf.py` 把 P2.1/P2.2 的 fake-quant AWQ 管线（4-bit g128、逐层 α 搜、非改进层回退 fp16；calib 锁 P2.2 主格 = wikitext-2 NL / n_calib=32 / seed 0）产物写成 HF safetensors checkpoint（fp16 权重落在 4-bit 网格上 → 服务端按 fp16 跑，下游精度 == fake-quant 精度）。`src/verify_p6_6_downstream_eval.py` 编排：每臂（fp16 / self_awq / mlx_awq_int4[=P6.2 那份]）起 `mlx_lm.server`，用 EleutherAI lm-evaluation-harness `local-chat-completions` 跑 gsm8k + ifeval，收 metric + 对 fp16 的 delta。三臂同一 eval 配置、只权重不同：greedy / `max_gen_toks=768` / `--apply_chat_template` / `--fewshot_as_multiturn` / `num_concurrent=1` / 无 system prompt。MMLU/HellaSwag 排除（loglikelihood 任务过不了 chat API，Bullet 3 坑1）。harness 装在隔离的 `.venv-lmeval`（lm-eval[api] 0.4.12 + langdetect/immutabledict/nltk punkt），项目 `.venv`（195 tests）不动。测试 `tests/test_verify_p6_6_downstream_eval.py`（6 hermetic）。

**结果（limit=400、greedy、seed 0；全量跑 4.19 h）**：

| 臂 | wikitext-2 ppl（同 P6.2 harness） | Δppl | GSM8K flex-extract | Δ | GSM8K strict | IFEval prompt-strict | Δ | IFEval prompt-loose |
|----|---:|---:|---:|---:|---:|---:|---:|---:|
| fp16 | 11.54 | — | 0.648 | — | 0.378 | 0.418 | — | 0.448 |
| self_awq | 12.94 | +1.39 | 0.553 | −0.095 | 0.158 | 0.393 | −0.025 | 0.418 |
| mlx_awq_int4 | 13.14 | +1.60 | 0.608 | −0.040 | 0.208 | 0.408 | −0.010 | 0.440 |

**头条**：ppl 把两个 4-bit 实现排**反**了——self_awq ppl 低 0.2（"更好"），GSM8K 却低 5.5 点（更差）。4-bit 伤推理（GSM8K −9.5 / −4.0）远多于伤指令遵循（IFEval ≤ −2.5）。两臂差异全在 scale/clip 搜索 + 校准集 + fp16 回退（详见 note 07），不是 fake-vs-real kernel 的锅。

**契约 / 复现**：`self_awq` ppl 在**同一** P6.2 mlx harness 上重算（12.9368），ppl→下游对比无跨-harness caveat；`--ppl-only` flag 可单独复现三臂 ppl。每臂 server 日志实收 801 `POST /v1/chat/completions`（400+400+1 warm-up），空补全/掉请求已排除（Bullet 3 坑3）。`strict-match` 在 chat 模型上是格式检查不是算术检查（fp16 自己只有 0.378）——主指标用 `flexible-extract`（坑22）。产物 `results/p6_6_downstream_eval.json`（含 `perplexity_same_harness` / `headline` / `known_config_differences_between_awq_arms`）。write-up = `docs/engineering-notes/07-perplexity-is-not-accuracy.md`；坑22/坑23 见 §9.2。

**P6.7 —（支柱7 可选）融合 Metal kernel + roofline 案例研究** —— 挑投机解码里**唯一**自有的算子（其余 matmul/norm/RoPE/KV cache/softmax/采样全和普通自回归共用、已被 `mx.fast` / `mlx-lm` 覆盖）：拒绝采样的**验证步** —— 给定 γ 个提议位置上 target/draft 的下一 token 分布，判定接受几个 draft，并在首个拒绝处构造调整分布 `p'(x)=norm(max(0, p_t−p_d))` 供重采样。`src/rejection_sampling.py` 里这是逐 γ 的 Python 标量循环 + 逐行 `torch.softmax`（显然正确、所有正确性测试走的路径）。P6.7 问：把它当批量 GPU 算子，代价多大、单个融合 kernel 能否打过朴素 MLX 图？`src/metal_accept_kernel.py`：`fused_accept` = 一个 `mx.fast.metal_kernel`，单 threadgroup 1024 线程，逐行**一趟 online softmax**（running max + running sum-of-exp 一次扫 V，再 threadgroup 归并）、thread 0 上跑接受扫描、只写**一行**调整分布、原地归一化 —— **不落任何 `[*,V]` 中间量**（朴素 MLX 图落好几个）。三个 MLX 对照臂：`reference_sync`（host sync 拿 n_accepted 再建行）/ `reference_branchless`（纯 lazy 图）/ `reference_compiled`（`mx.compile` 后者）。`src/verify_p6_7_metal_roofline.py` 编排：正确性网格（γ × 接受率 × seed）+ 进程内实测 peak（streaming BW、fp32 GFLOP/s）+ 延迟中位数 + roofline 放置。测试 `tests/test_verify_p6_7_metal_roofline.py`（9：byte/flop 账 + 纯 MLX 参考逻辑 hermetic 常跑，2 个真启 Metal kernel 的按 GPU 门控）。

**结果（V=151936、fp32、本机实测 peak ~84 GB/s BW、~3050 fp32 GFLOP/s、roofline ridge ≈ 33–36 flop/byte；算子小，每个数 ±3–5% 抖动）**：

| γ | reference_compiled µs | fused_accept µs | fused ÷ best-ref | fused 字节 / 朴素模型字节 |
|--:|---:|---:|---:|---:|
| 2 | 469 | 405 | 1.16× | 6.7M / 17.6M |
| 4 | 741 | 729 | 1.02× | 11.5M / 29.8M |
| 8 | 1100 | 1058 | 0.97× | 21.3M / 54.1M |

正确性：`n_accepted` 36/36 精确；调整行与参考差 < 1e-9（fp32 归约顺序噪声）。**头条 / 负结果**：融合 kernel 搬**少 2.6× 字节**却只跑 **~1.0×** 速度。原因：roofline 说 memory-bound（AI 0.4–1.5 « ridge 36），但 roofline 的"memory-bound"假设你**吃满**带宽；单 threadgroup kernel 只用一个 GPU core → ~16 GB/s = 84 峰值的 **~19%**，MLX 多 kernel 图扇出全部 core → ~40 GB/s = **~48%**。少搬 × 占用率差 2.5× = 打平。而且**整个步骤只占一次 target 前向的 ~2.3%**（fp16 1.5B 解码 31 tok/s → ~32 ms/前向，接受步 0.73 ms），greedy 下更是零（argmax、无 softmax）。**结论**：本栈上 pointwise / 小归约算子，`mx.compile` 干净图就是正解，手写 Metal kernel 是校准练习不是优化。真要赢得把 V 拆到多 threadgroup + 两级归约 —— 那就是重写 MLX kernel 生成器已经在做的事。坑24（V < threadgroup 时 online-softmax 归并 `exp(−∞−(−∞))=NaN`，只在边界 vocab 触发；kernel 测试特意用 V=256）见 §9.2。产物 `results/p6_7_metal_roofline.json`；write-up = `docs/engineering-notes/08-a-fused-metal-kernel-and-the-roofline.md`。

---

**P6.8 —（支柱7，工程落地物证）Demo：能当纯静态站的单文件 lab 页，回放录制好的真实 run** —— 之前的"落地"只有 P6.3 的终端 dashboard，README 里没提、也没 HTTP 面。**用户反馈**（初版把"跑 `python -m src.serve_http` 并一直挂着 + 两个模型常驻"当成 demo）：没人会在自己电脑上一直挂一个加载了两个模型的 Python server —— demo 要"打开就看"。于是最终形态是 **recording-first**：`docs/site/index.html` 单文件，`<script src="sample_run.js">` 内嵌一次录制好的真实 run（`window.SPECTER_RUN`，无 fetch、`file://` 双击即可、GitHub Pages 直接发、`python3 -m http.server` 也行），点 ▶ Replay 按录制里每轮的 `wall_ms` 等比配速（~8s @1×，带 1/2/4× 档）逐帧重放。可视化：masthead → **LAB（第一节，"Watch it decode"）**：6 格遥测（round / mode / γ / α / tok·s / active+queued）+ **round strip**（每轮一根竖条，色=mode[spec 绿 / degraded 红 / probe 黄 / idle 灰]、高=emitted/n_active，spec pass 43 根后一道 pass-break 缝隙再接 baseline pass 128 根——一眼看出 baseline 每轮只出 1 token）+ throughput sparkline（canvas，dpr-aware，带当前值）+ 4 个 prompt 卡片（label = prompt 原文，流式填 speculative pass 的输出，`.live` 带闪烁光标）+ 结尾 headline 行 + 诚实注解 GLOSS（"parity on this pair … token-exact and honestly measured — see note 03"）→ THE SYSTEM（11 行 build 表当卡片）→ NUMBERS（诚实 headline）→ NOTES（8 篇 finding-first 摘要卡，链到 GitHub 上的 .md）→ footer（坑1–25 链接）。载入即静态显示录制的成品输出（从 `done` summary 的 `final_texts`，`rid.replace("B","A")` 映射）+ 静态 headline，不点也不空。**没有**"backend offline"道歉 banner —— 静态就是设计本身。`src/serve_http.py` 降级为两个用途：(a) 生成录制 —— `--capture docs/site/sample_run.json` 真跑一次（Qwen2.5 0.5B/1.5B，`model_loader` + `spec_kv._new_cache`），同时写 `.json`（`/sample` 路由 + 工具用）和 `sample_run.js`（`window.SPECTER_RUN = {...};`，页面靠 `<script src>` 加载、绕开 `file://` 的 fetch/CORS 限制）；只剔页面不读的 `tps_series`，**保留** `final_texts`。(b) 可选 live 后端 —— `python -m src.serve_http` 起同一页 + `POST /generate` SSE 真推理（事件 `start` → `round`×N → `done` →（可选）`compare_done`，每轮 `{pass,index,mode,gamma,rolling_alpha,realign_tax,n_active,n_queued,emitted,wall_ms,tok_per_s,breaker,texts}`）；单 Mac GPU → 非阻塞 `threading.Lock` 串行化，并发 POST 返回 429；`Connection: close` + `self.close_connection = True`（坑25）。页面探到 `fetch('/health')` 返回 `backend === "qwen"` 才显示 `#livebox`（"run your own prompt" textarea + 三 toggle + Run），否则完全不提 server。`--fake` 用 `make_fake_pair` + `LengthOnlyCache`（无下载，供 CI）。`docs/site/sample_run.js` / `.json` = 提交进仓库的真实 4-prompt batch（demo_batch + compare，max_tokens=128）：**spec agg 25.9 tok/s vs baseline 24.6 = 1.05×**，mean accept 2.5，~188 KB、174 事件（43 spec 轮 + 128 baseline 轮）。和 README「0.93–1.0× / parity」一致（单条 code prompt 能到 1.3×，聚合回到平手——没有 cherry-pick）。测试 `tests/test_serve_http.py`（10 hermetic：起真 `ThreadingHTTPServer` 于端口 0 跑 FakeModel，断言 `/health`、`GET /` 发页、SSE 线格式（start/round/done 字段、compare 加 baseline+speedup+baseline 无 spec 轮）、max_tokens 双端夹到 8..160、demo_batch 跑满 4 prompt、并发 429、未知路由 404、`capture()` 同时写 json + js sidecar 且 js 前缀 `window.SPECTER_RUN = ` 且剔掉 `tps_series`）。pytest 210 → **220**。README「Demo」节改成「打开 `docs/site/index.html`」打头（"No server, no model download, nothing to keep running"），live 后端 + `--capture` 当次要项。坑25（HTTP/1.1 keep-alive + 无 Content-Length 的 SSE 流把客户端挂死；修 = `Connection: close` + `self.close_connection = True`，用 socket EOF 当流结束信号）见 §9.2。write-up：暂不单独成篇（当 README「Demo」节 + 站点本身）。

**P6.8 追加（交互式多场景切换）** —— 单一录制 run 换成可切换的 4 个场景，理由：一次录制只能证明"这一对模型这一个 prompt 组合打平"，看不出投机解码在不同工况下的行为差异；有对比才有说服力。`src/serve_http.py` 新增 `SCENARIOS` 有序字典（`batch`/`codegen`/`prose`/`breaker`，每项 `{label, caption, body}`），`_cfg()` 把此前硬编码的 `alpha_floor=0.5, warmup_rounds=3, reprobe_every=10` 改成从 body 透传（默认值不变，不影响旧场景）。`capture()` 遍历全部场景：`<path>`（`sample_run.json`）只落 `DEFAULT_SCENARIO`，给 `/sample` 路由和工具用；新增 `<path.parent>/sample_runs.js` 落 `window.SPECTER_RUNS = {key:{label,caption,captured,backend,events}}` 全量，页面靠 `<script src="sample_runs.js">` 加载（不 fetch，`file://` 双击照样能用）。四个场景的 body：`batch`＝原来的 4-prompt demo_batch；`codegen`＝单条 Fibonacci 函数 prompt；`prose`＝单条开放式续写 prompt；`breaker`＝同 prose prompt 但 `alpha_floor=0.6, warmup_rounds=2, reprobe_every=4`——这个数字不是编的：先用 `alpha_floor=0.92` 试过，rolling α 在这台机器这对 prompt 上从没真正恢复过（一直卡在 0.5–0.75 之间），breaker 触发后再也回不到绿色；降到 0.6 才是这对模型这个 prompt 真实 α 分布会跨越的门槛，能反复触发 + 自愈（跑出来 spec 53 轮 / degraded 12 轮 / probe 3 轮，序列上真出现多次 spec→degraded→probe→spec 循环）。四场景真实结果（Qwen2.5 0.5B/1.5B，非 cherry-pick）：batch **1.04×**（parity，README headline 数字不变）、codegen **1.25×**（可预测续写→长接受串→spec 赢）、prose **0.80×**（不可预测续写→短接受串→开销吃不回来）、breaker **0.85×**。页面 `docs/site/index.html`：`#lab` 面板加 `.seg`「场景切换条」+ `#runcaption`（每场景一句「看什么」提示）；`selectScenario(key)` 用自增的 `PLAY_TOKEN` 让旧 `replay()` 循环在下一次 `await` 恢复后发现 token 不匹配就自行 `return`，防止快速连点场景按钮时叠出多条 `setTimeout` 链；切场景即 `paintStatic()`（刷新 runmeta/caption/卡片/静态 headline）+ 自动 `replay()`（不留手动触发，"自动播放更耐玩"选一个模式后全场一致，不是有的场景自动有的不自动）。`loadRuns()` 换成 `window.SPECTER_RUNS`，无嵌入文件时 fallback fetch `/sample` 包一层 `{default:{label:"Recorded run",caption:"",...}}` 保持接口一致；file:// 下 `/health` CORS 失败照旧静默降级（live 后端探测功能不受影响）。用 Playwright 在 `file://` 页面上验证：连续快速切 4 个场景不叠 timer（每次切完 strip 根数精确等于该场景 round 数，不多不少）、breaker 场景 strip 里 spec（绿）/degraded（红）/probe（黄）三色全出现。`tests/test_serve_http.py`：`test_capture_writes_json_and_js_sidecar` 改断言 `sample_runs.js` 解析出的 `window.SPECTER_RUNS` 有 `SCENARIOS` 的全部 key，每个 key 的 label/caption/events 齐全、`events` 首尾仍是 `start`/`compare_done`、不含 `tps_series`；新增 `test_capture_scenarios_hit_max_tokens_cap_and_floor`（monkeypatch 两个合成场景，body 的 `max_tokens` 分别给 99999 和 1，断言 capture 出来的 `start` 事件真的被夹到 160 和 8——踩「烟雾测试要打边界值，不能只等比例缩小参数」这条经验）。pytest 220 → **221**。README「Demo」节从单一数字改写成四条场景列表，每条一句人话解释。

---

## 8. 考虑过的其他方案

| 决策点 | 选择的方案 | 考虑过的替代方案 | 拒绝理由 |
|---|---|---|---|
| 投机解码架构 | 经典独立草稿模型 | 训练专用speculative head(EAGLE/Medusa) | 需要多卡训练资源(IBM用FSDP两阶段训练)，个人项目没有这个预算 |
| 投机解码架构 | 经典独立草稿模型 | **self-speculative decoding(LayerSkip/Draft & Verify等，v9新增)** —— 同一个模型跳过中间层生成草稿，不需要第二个模型 | 不需要第二个模型意味着天然规避tokenizer不一致(坑1)和bonus token跨模型采样(坑2)这两类失效模式，但这两类失效模式正是本项目想展示排查能力的内容；采用self-speculative会使"两模型架构"这条串联量化与投机解码的主线失去意义。拒绝原因是它会规避掉项目想展示的失效模式，与实现难度无关 |
| 量化自研对象 | AWQ | GPTQ | 实现复杂度(Hessian/Cholesky分解)和时间线不匹配；AWQ对校准集分布更鲁棒 |
| 量化对比基线工具 | LLM Compressor(云端) + mlx_lm.awq(Mac，v9新增) | 分别装AutoGPTQ+AutoAWQ | 行业已转向统一工具；AutoGPTQ/AutoAWQ在Mac上都不可用；mlx_lm.awq可以在Mac上免费产出一个真实的AWQ速度交叉验证点，不必等到云端阶段 |
| Mac开发阶段主框架 | HF Transformers(MPS backend) | (a)vLLM CPU (b)vLLM-metal (c)MLX主实现 | (a)比llama.cpp慢20-30倍 (b)2026年仍是早期版本 (c)HF Transformers有论文先例(SpecKV方法论)，MLX降级为交叉验证工具 |
| Agent评测范围 | AgentBench OS子环境 | 完整八个环境 | 完整版对时间线太重；OS子环境恰好是投机解码效果最好的场景 |
| 支柱5控制器算法 | GammaTune(EMA) | BanditSpec(bandit理论)/AdaEDL(熵基早停) | BanditSpec的UCBSpec核心算法本身并不复杂（论文自称是最简单的一类UCB算法，且有公开代码可直接跑做对比，见P5.4），未采用的原因是它的核心目标(regret-bound理论保证)和AdaEDL的核心目标(熵阈值提前停止)与Specter支柱5的核心目标(自适应步长)只有部分重叠，引入完整框架的抽象成本和项目目标不成比例；两者保留为文献对比对象，BanditSpec另可直接跑其公开代码做基线对比 |
| 量化-γ耦合实验的压缩方式 | AWQ(主) + BitsAndBytes NF4(对照，v9新增) | 只用AWQ单一压缩方式 | 只用AWQ会导致对标SpecKV数字时出现方法论错配(AWQ和BnB对输出分布的扰动机制不同)；加一个BnB对照臂几乎零成本(HF内置配置)，能直接复现SpecKV的原始设置作为同源参照 |
| 正确性验证标准 | 贪心逐token严格比对+采样统计检验 | 对采样模式也要求KL散度严格为0 | 不同设备/框架间不保证bit-exact输出，严格要求KL=0不现实且无必要 |

---

## 9. 已知风险与应对

### 9.1 平台兼容性硬约束

**风险A：vLLM 在 Mac 上不能用**——vLLM CPU模式实测 3-5 tokens/s(M5 Max, Llama-3.1-8B, batch=4)，对比 llama.cpp Metal 是 92 tokens/s，差距20-30倍；`vllm-metal`插件截至目前仍是早期版本，模型覆盖不全。来源：[vLLM Issue #1441](https://github.com/vllm-project/vllm/issues/1441)、[vllm-metal GitHub](https://github.com/vllm-project/vllm-metal)。**应对**：Mac阶段用HF Transformers手写实现(SpecKV论文方法论原文如此)，云端阶段才引入vLLM做对比基线。

**风险B：AutoAWQ/AutoGPTQ/LLM Compressor 全部要求CUDA**——AutoAWQ要求"Compute Capability≥7.5, CUDA≥11.8"；AutoGPTQ维护者明确需要专门写MPS kernel(未实现)；LLM Compressor底层依赖和vLLM生态一致。**v9补充更精确的技术原因**：PyTorch的旧版量化API(`torch.quantize_per_tensor`等)用的是独立的"Quantized"后端，从未被移植到MPS，直接报错`"QuantizedMPS" backend not implemented`——这是Mac上做不了真实int4/int8张量运算的根本原因，不只是第三方库懒得适配。来源：[AutoGPTQ Issue #223](https://github.com/PanQiWei/AutoGPTQ/issues/223)。**应对**：支柱2拆两半，Mac阶段只产出正确性/压缩率数字(fake-quantize模拟,普通float运算,不受此限制)，云端阶段才测真实速度；例外见P2.4，mlx-lm走的是自己的Metal kernel路径，不受PyTorch这个限制。

**参照系：MLX 生态经验阈值**——`mlx-lm`文档：接受率α>0.65才有净加速。**应对**：阶段0 α-gate改两级判断(见P1.0)。

### 9.2 支柱1已知坑

**坑1**：tokenizer/vocab不一致使接受率静默归零，不报错。同一模型家族内部也可能不一致(Qwen2 1.5B vocab_size=151936, 72B=152064)。**应对**：P1.0显式断言检查。

**坑2**：bonus token采样源搞错——已知真实bug(DSD)：错误地从草稿模型分布采样bonus token，违反正确性保证但不会崩溃。**应对**：单元测试专门检查bonus token采样代码路径来自哪个模型的logits。

**坑3**：batch>1时ragged tensor同步问题——不同序列接受的草稿token数不同，position id/attention mask/KV cache长度参差不齐。**应对**：支柱4设计里手动维护per-sequence状态，不依赖框架自动padding。

**坑4**：draft/target尺寸"死亡区间"——尺寸差距不够大(2-3倍)可能比不用还慢。**应对**：诊断标准——α尚可但墙钟时间变慢，先查草稿模型自身延迟占比。独立佐证：AdaEDL论文的对照实验里，用未经微调的TinyLlama-1B给Llama2-7B做草稿模型、固定投机长度=7时，静态投机解码(Base-SPD)反而比不用投机解码的自回归基线慢16%；而同样的模型对，只要加上自适应提前停止(AdaEDL/Max-Confidence-SPD)，就变成比自回归快43%——同一组模型，"静态步长"和"自适应步长"的差别决定了投机解码是帮倒忙还是真加速，这是坑4现象的一个独立数据点（AdaEDL论文本身不复现，只作为佐证引用，见§3非目标3）。

**坑13（2026-08-28，M2 P1.2 实现时踩到）**：一轮投机会一次性提交"已接受前缀 + 1 个 bonus/重采样 token"这一整块，EOS 可能落在这块的**中间**而不是末尾。如果生成循环只检查 `new_tokens[-1] == eos` 来决定是否停止（很自然的写法），当 EOS 出现在块中间时循环不会停，会越过 EOS 继续生成——而逐 token 的目标模型自回归基线会恰好停在 EOS。表现是：贪心模式下前缀逐 token 完全一致、`first_divergence_index` 为空，但两条路径长度不同，P1.2 正向检查判为不一致。不会报错、不会崩溃，容易误判成"浮点误差导致的分叉"。**应对**：`speculative_generate` 提交每一块前先扫描块内首个 EOS，截断到该 EOS（含）并停止；P1.2 正向检查里把"EOS 提前停止位置不同"和"token 内容不同"分成两类分别判定。已在 `src/rejection_sampling.py` 修复并加进 P1.2 判据。

**坑14（2026-08-28，M4 P5.0/P5.1 实现时踩到）**：P5.0 按计划把"每轮提交 token 数" `mean(emitted_per_round) = mean(n_accepted)+1`（等价于"每次 target 前向的 token 产出"）作为**硬件无关主指标**（理由充分：每轮一次 target 前向、是投机解码主成本；本地 MPS 无 KV cache 使墙钟不可靠）。但这个量对 γ **单调非减**——每轮无论浪费了多少次 draft 前向，只要 target 接受了前缀就照单全收，指标里没有任何一项惩罚 draft 前向的浪费。实测固定 γ=1/3/5/7 → 1.77/2.80/3.49/3.67（随 α≈0.79 饱和而增速放缓），一个把 γ 控制在中间值的自适应控制器（GammaTune 实测均值 γ≈3.4、落 γ_max 仅 1.4%，轨迹健康）在这个指标上**结构性地不可能跑赢大 γ**，最多打平中等固定 γ（实测 GammaTune 2.94 < 固定 γ=7 的 3.67，区间不重叠）。GammaTune 论文优化的是附录 A.2 成本模型里 `(c+γ)` 那一项（`c=T_target/T_draft`），γ 项正是 draft 浪费的代价；只看 `emitted_per_round` 等于把这项抹掉。**表现**：主指标判定"GammaTune 无效"，但墙钟 tok/s 上 GammaTune 1.19x 反而追平最优、固定 γ=7 掉到 0.89x（比不投机还慢）——两个指标对 γ 的排序相反。**应对**：(a) 主指标判定保留不动、不改口径（§9.6 风险1/2），但补一个成本模型加权的次级分析 `throughput = sum(emitted) / (n_rounds·c + sum(γ))`，`c` 用 `src/verify_gammatune.py:measure_c()` 实测（本机 0.5B/1.5B on MPS ≈ 1.3，被 kernel-launch 开销主导、低估真实算力比 ~3x，故另按文献典型 `c∈{4,7,10}` 各算一遍）；`c∈{4,7,10}` 下 GammaTune 落在最优簇内（差 0.8%/3.0%/5.9%）、稳定优于两个极端 γ=1/γ=7。(b) **评"要不要投机 / 投机划不划算 / 自适应控制有没有用"这类问题，必须用带 draft 代价的指标（成本模型 `(c+γ)` 或带 KV cache 的墙钟），不能只看每 target 前向的 token 产出**。P5.3（batch 熔断器）判据同理，真实吞吐确认留 M5[A]/云端。已在 `src/verify_gammatune.py` / `src/verify_nonstationary.py` 落实为 `cost_model_supplement` 字段。

**坑15（2026-08-28，M4 P5.3 实现时踩到，是坑14 在熔断器上的延续）**：P5.3 熔断器的成本模型按计划写成 `total_emitted / total_cost_units`，投机轮 `c+γ`、降级步 `c`（任务决策点6）。但这个式子**没有任何一项依赖 batch size**——一个投机轮无论 batch=1 还是 batch=1000 都记 `c+γ`。于是 always-spec 在这个指标上是熔断器**结构上无法跑赢**的上界：降级到纯 target 每单位算力做的有用功严格更少（`emitted/round` 从 ~3.4 掉到 1），指标里又没有"高 batch 下 draft 前向变贵"的项来补偿。实测 always-spec 0.280±0.010 vs circuit-breaker 0.247±0.008（c=7，区间不重叠），主指标判"熔断无用"。**根因**：本地 batch 信号是合成的、**不反馈进 α**（真实高 batch 会让 draft 前向争抢饱和的加速器、投机相对无投机倒退最多 30.25%，坑7），所以合成信号下"投机在高 batch 变差"这件事在数字上根本没发生。**应对**：(a) 主指标口径不动、不改（§9.6 风险1），如实写它判负；(b) 补一个 `sat_tax` 敏感性分析——高 batch 段里投机轮的 γ 次 draft 前向每次记 `sat_tax` 单位而非 1（round cost = `c + γ·sat_tax`），`sat_tax∈{1,2,3}`，`sat_tax≈3` 时对 c∈{4,7,10} 大致复现 Nightjar 的 30% 倒退量级；`sat_tax≥2` 下 circuit-breaker 追平或反超两个对照。(c) 熔断器的机制正确性与主指标判负**解耦**看：降级滞后 0 轮、恢复滞后 1 轮（那 1 轮是恢复前的强制重探测）、切换开销代理量 ~11.5ms（重 encode 前缀过 0.5B+1.5B，低于 Nightjar RTX4090+7B 的 17.87–102ms，量级合理）、占墙钟 0.1%——机制都对，是评价指标口径的问题，不是熔断器坏。(d) **附带发现**：单次 `γ=3` 的重探测轮不足以估计 α——一轮只有 2–3 次接受判定，测出的 α 量子化到 {0, 0.5, 1.0}，对参考 α 的绝对偏差 0.2–0.75。生产环境要 pool 多轮重探测再取均值，不能靠单轮。已在 `src/verify_circuit_breaker.py` 落实为 Metric A / Metric B(`sat_tax`) 双报 + `circuit_breaker_secondary` 字段。

**坑18（2026-08-28，M6/支柱6 P6.0 实现时踩到）**：plan §7 P6.0 把部分接受回滚的裁剪公式写成"target cache 裁到 `prefix+k+1`、draft 裁到 `prefix+k`"（两者不同、target 多留一格给 bonus/重采样 token）。实现下来**这个公式是错的**：一轮里 target 前向只吃了 `[pending_target + γ 个 draft token]`，产出 `γ+1` 行 logits；第 `k` 个位置的 argmax（无论是被接受的 draft token、还是拒绝时的重采样、还是满接受时的 bonus）都是这次前向的**输出**，从来没有作为**输入**喂过任何前向，所以它的 KV **不存在**。正确做法是 **draft 和 target 两个 cache 都裁到 `prefix + n_accepted`**（满接受时即 `prefix + γ`）。多裁一格会丢掉一个已 committed token 的 KV、下一轮 pos-id 断裂；`prefix+k+1` 那版会把一段不存在的 KV"保留"下来、`crop` 到比实际 seq_length 还大的目标（`_crop_to` 里 `target_len >= cur` 直接 return，静默变成不裁）。**为什么容易误判**：`prefix+k+1` 在**满接受**轮恰好等于 `cur`（没有 rejected token 要裁），`_crop_to` 判 `>= cur` return，行为看起来"对"；只有出现 rejection 的轮才暴露。**佐证**：HF `transformers/generation/utils.py` 的 `_speculative_sampling` 后处理是 `n_matches = ...; num_tokens_to_crop = candidate_length - n_matches; past_key_values.crop(-num_tokens_to_crop)`——裁到 `prefix + n_matches`，和第一性原理一致。**应对**：(a) `src/spec_kv.py` 按 `prefix + n_accepted` 实现、`_crop_to(cache, target_len)` 薄封装只接受 `target_len <= cur` 并锁 `DynamicCache.crop` 负数语义；(b) `tests/test_spec_kv.py` 加"每轮结束后 `cache.get_seq_length() == len(committed) - 1`"的常开不变量（34 tests 全过）；(c) plan §7 P6.0 正文已改，此坑记为公式勘误。**教训**：回滚长度这类"差一"公式不要照抄脑内推导，回滚的锚点是"喂过前向的 token 数"不是"这一轮涉及的 token 数"，且要有一个 rejection-heavy 的压力测试（`phase=2.4`）逼出非满接受轮。

**坑19（2026-08-28，§7 C3 在 P6.1 动手时撞上 → 转坑）**：§7 C3 说每条 naive 批量投机路线（Masking / Rollback / Dynamic Padding）都会破坏输出等价——一批里每序列接受的 draft 数不同 → position id / attention mask / KV-cache 长度跨轮次累积漂移（EQSPEC / arXiv:2510.22876）。P6.1 动手时的选择：**不修 `spec_batch.py` 的 masking 路线，而是绕开整个问题** —— 每序列独立 KV cache（`src/spec_kv_batch.py`），一"批量轮"对每序列各跑一次已验过的单序列 `speculative_step_kv`，没有共享 ragged tensor 就没有东西会漂，输出等价**由构造成立**（`tests/test_spec_kv_batch.py` 钉死 batched == N×单序列，greedy+采样 bit-exact；`assert_rectangular_invariant` 每轮断言）。**代价**：放弃 kernel 级 ragged verify 批处理 → 吞吐随并发几乎平（`results/p6_1_serving_throughput.json`：`speedup_vs_width1` 0.94–1.03）。**把没付的税测出来**：`ragged_realignment_overhead(work) = 1 - Σwork/(n·max(work))` 对"矩形 padded batch 反事实"解析算——mean 随 width 1→4 涨到 0.073–0.096、p90 0.68–0.74（即 EQSPEC 那条路要付的 padding 浪费），w=8 反降（work 更均匀）。**教训**：批量投机的"正确性 vs 吞吐"是真实取舍；能证明输出等价的实现拿不到 batch 加速，能拿加速的实现要付 realignment tax。报告里把 tax 当一等测量量，而不是假装 batch 加速免费。已在 `src/spec_kv_batch.py` docstring + 结果文件 caveats 写明。

**坑20（2026-08-28，§7 C4 在 P6.1 动手时撞上 → 转坑）**：§7 C4 说 P5.3 熔断器的 batch 信号是合成注入的（坑15），且"大 batch 让投机变差"这个前提已被 2025 长上下文工作部分推翻，要求 P6.1 的熔断器接**真信号**。P6.1 落实：熔断器 trip 判据 = 滚动 α（跨所有序列、最近 `alpha_window` 个接受判定的窗口均值）< `alpha_floor`，OR 可选的 target-only 延迟探针显示投机轮墙钟反而更慢；`len(active)` 只作为输入、**从不作为规则**。degraded 时每 `reprobe_every` 轮强制一次 spec 探针让 α 恢复可见（坑11）。`results/p6_1_serving_throughput.json`（16 个 run）：**熔断器只在 `long / w=2 / br=on` 触发一次**（degrade 20 轮 → 强制探针 → 恢复，正是坑11 机制），**w=8 满批从不 degrade** —— 证明判据真的不看 batch size。**与坑15 的对比**：坑15 里 always-spec 是熔断器结构上无法跑赢的上界（合成信号不反馈进 α）；P6.1 里 α 是真实测出来的，熔断器不再是纯负担，但在这对 0.5B/1.5B + 这台 Mac 上 α≈0.77 稳定高于 floor，所以大多数轮它就是个 no-op。**教训**：熔断器的价值只有在 α 真的会掉的工作负载上才显现；健康 α 下它应该基本不动，"从不 degrade"不是 bug。真实验证要故意找会掉 α 的 pair / 分布（留 P6.2 int4 target 或跨分布 prompt）。

**坑21（2026-08-28，P6.2 动手时撞上）**：P6.2 计划四方对比里包含 `mlx_lm.gptq`（mlx-lm 0.31.3）。实测 `mlx_lm.gptq -m Qwen2.5-1.5B-Instruct --bits 4 --group-size 128` 产出的模型**退化**：`generate` 出常数 `!`，wikitext2 前向 nll = nan。第一反应是标定不足（首跑 `--num-samples 16 --sequence-length 256` 只喂了 2 个 Hessian batch），于是加到 `--num-samples 64 --sequence-length 512` 重跑 —— **仍然退化**，同样常数 `!`。同一模型、同一 bits/group-size 下 `mlx_lm.awq`（ppl 13.14）和 `mlx_lm.convert -q` RTN（ppl 13.81）都正常。没继续二分（可能是 group-size、某个 fallback 层、或 mlx-lm 对 Qwen2 架构的 GPTQ 实现 bug）—— 算力预算，且 AWQ 已是主 real-int4 臂。**应对**：`verify_p6_2_real_int4.py` 的 `_eval_ppl` 把非有限 ppl 落成 `null` + `degenerate_forward` 标志，`run()` 里 `_sane()` 判退化臂、`deltas_vs_fp16_mlx` 该臂记 `None`、headline 印 `GPTQ DEGENERATE`，JSON 保持严格（无 NaN token）。结果文件 `quant_config.gptq` 完整写明复现步骤。**教训**：(a) 借第三方量化工具当对比臂时，产出模型必须过一个"生成一句话看是不是人话 + 一次前向看 nll 有限"的哨兵检查，不能只看它有没有跑完 / 权重文件在不在；(b) 退化臂不要从结果里删掉——"这个工具在这个架构上不能用"本身就是给读者的有用信息，如实记为 finding。**工程故事**：P6.4 的"fake-quant vs 真 int4"那篇顺带收这条（Apple silicon 上 AWQ 能用、GPTQ 这版不能用）。

**坑22（2026-08-29，P6.6 / 支柱7 Bullet 3）**：lm-eval 的 GSM8K 报两个数——`strict-match`（答案要以 `#### <数字>` 出现在固定位置）和 `flexible-extract`（回复里最后一个数字）。套 Qwen2.5 chat template 后模型出的是对话式 CoT、几乎不吐 `####` 锚点，**fp16 基线自己** strict 只有 0.378、flexible 0.648。于是量化 vs 基线的 `strict-match` delta（−22、−17 点）大部分在测"命中格式的频率"而不是"算对的频率"。**应对**：GSM8K 主指标用 `flexible-extract`，`strict-match` 在这套栈上当噪声（两个都留在结果 JSON、write-up 里以 flexible 打头）。是 lm-eval #1841（chat template 悄悄挪分数）的表亲。**教训**：引用某个 metric 变体的 delta 前先搞清它到底奖励什么；基线本身就因为与你的改动无关的原因失分的 metric，不是对你改动的测量。

**坑23（2026-08-29，P6.6 / 支柱7 Bullet 3）**：同一 wikitext-2 harness 上，自研 AWQ 比 fp16 涨 **+1.39 ppl**、`mlx_lm.awq` int4 涨 **+1.60** —— ppl 说自研的是更好的量化器。GSM8K flexible-extract 上顺序**反转**：自研掉 **9.5 点**、`mlx_lm.awq` 掉 **4.0**。prose 上 0.2 ppl 的"优势"对应小学数学上 5.5 点的**劣势**。IFEval 两个都几乎不动（−2.5 / −1.0 点）——伤害专门落在多步推理上（逐权重舍入误差在 CoT 步骤间累积）。两个臂差在 scale/clip 搜索（`mlx_lm.awq` 做完整 AWQ 权重裁剪搜索，自研只做 scale 搜索）+ 校准集 + fp16 回退策略——这些差异在 ppl 上几乎看不出、在 GSM8K 上值 ~5 点。**应对/实践**：ppl 是筛选指标不是验收指标；量化模型上线前至少跑一个需要正确多步输出的任务，eval 配置（chat template、few-shot 格式、解码参数）与基线**逐字一致**。**教训**：ppl 和下游准确率不是同一个测量，这里连符号都不一致；"AWQ 4-bit g128"不是一个数——搜索和校准细节决定模型还能不能推理。

**坑24（2026-08-29，P6.7 / 支柱7 可选）**：`src/metal_accept_kernel.py` 的融合 kernel 逐行做一趟 online softmax，线程各持 running `(m, s)`，再 threadgroup 归并 `mM=max(mA,mB); sM=sA·exp(mA−mM)+sB·exp(mB−mM)`。1024 线程 + 真 vocab V=151936 时每个线程都干活，没事。V < 1024（smoke 配置、V=256 单测）时多余线程从不进扫描、带着 identity `(m,s)=(−∞,0)`，两个这样的线程归并出 `mM=−∞`，`exp(−∞−(−∞))=exp(NaN)=NaN`，污染整行。**应对**：归并加护栏 `sM = (mM > −INFINITY) ? (…) : 0`。**教训**：这正是"按比例缩小的 smoke 测试"该抓、"直接用真实尺寸"会漏的 bug —— 失效模式住在 `n_threads > vocab` 这条全尺寸跑永远跨不过的边界上；kernel 测试特意用 V=256。

**坑25（2026-08-29，P6.8 / 支柱7）**：`src/serve_http.py` 第一次冒烟 `/generate` 端点，客户端 2 分钟超时都不返回、服务器进程还活着。`BaseHTTPRequestHandler` + `protocol_version = "HTTP/1.1"` 默认 keep-alive。SSE 响应没有 `Content-Length`（流长度事前未知）也没有 `Transfer-Encoding: chunked`，于是最后一个 `event: done\n\n` 之后客户端一直等——等更多字节 / 等一个永远拿不到的长度 / 等一个永远不来的关闭。**应对**：发 `Connection: close` header + 在 handler 上设 `self.close_connection = True`，让 socket EOF **就是**流结束信号；`fetch()` 的 reader 和 `curl -N` 都在 EOF 上干净终止。（另一条路——手写 chunked 编码——这里没收益：一次生成独占连接。）**教训**："没有 `Content-Length`" 只有在服务器**真的关连接**时才等于"读到 EOF 为止"；keep-alive 下它等于"挂死"。测客户端的流结束路径，别只测字节到没到。

### 9.3 支柱2已知坑

**坑5**：校准集分布不匹配使GPTQ严重过拟合，AWQ相对稳健——跨分布(PubMed→Enron)时AWQ涨0.5-0.6，GPTQ涨2.3-4.9。**应对**：P2.2交叉验证实验复现这个矩阵。

**坑6**：校准集太小会过拟合——标准实践C4数据集、group size 128。**应对**：P2.3做校准集大小消融。

**坑16（2026-08-28，M1 P2.3 实现时踩到）**：自研 AWQ 的 `capture_all_layer_inputs` 一次前向就把全部 196 个目标 Linear 的输入激活捕获进内存（不是逐层量化-释放），因此必须有一个 `max_tokens_per_layer` 上限防 OOM。它的捕获循环写成"每层都攒够 `max_tokens_per_layer` 就 `break`"。P2.3 校准集大小消融复用 P2.2 的 `_quantize_fresh`，后者把这个上限**写死成 512**、每条校准序列也截断到 512 token——于是**第一条 512-token 的 wikitext 行就把每层的池子填满，捕获立即停止**。结果：`n_calib ∈ {8,16,32,64,128}` 喂进去的是**逐 bit 相同**的"前 512 token"池，量化结果、perplexity 全都一模一样（seed 0：n_calib=4/8/16/32/64 都是 13.6015）。**为什么容易误判**：这条平坦曲线**正好符合预期**——AWQ 论文说"小校准集就够"，Michael 3B 那份也几乎平（11.22→11.27），会让人直接写"复现成功"。是 §9.6 风险4（符合预期的结果要反向复核）的教科书案例：真正的复核问题是"这个 n_calib 旋钮到底有没有拧动"，而不是"曲线平不平"。**应对**：(a) `_quantize_fresh` 增开 `max_tokens_per_layer` / `max_seq_len` 参数（默认仍 512，P2.2 行为字节级不变）；(b) P2.3 把每行截到 64 token、上限设成 `n_calib × 64`，让池子真正随 n_calib 线性增长；(c) 结果 JSON 记录每个 n_calib 实际捕获到的 `captured_tokens_per_layer`，并加一个 `capture_knob_actually_moved` 布尔——`max/min ≤ 1.5×` 时 verdict 字符串自动标注"cap 仍在生效，曲线不作数"；(d) `n_calib=128` 从网格里去掉（8192 token/层 × ~509k 汇总 in-features × 4B ≈ 17GB CPU 侧捕获，24GB 机器会爆），更大规模需要把 `capture_all_layer_inputs` 改成逐层量化即释放（留给用户拍板）。

**坑17（2026-08-28，M1 P2.2 语料替换）**：记忆/plan 里写"本地已缓存 `codeparrot/codeparrot-clean-valid` 和 `allenai/c4`"，实际 `~/.cache/huggingface/datasets/` 下只有这两个的 README snapshot、没有数据分片，`HF_HUB_OFFLINE=1` 下取不到。P2.2 跨分布实验的"代码"分布只能退回用 `google-research-datasets/mbpp` 的 `code` 字段（全 split 拼起来 ~176k 字符）。**为什么容易误判**：mbpp 的题解都是 3–8 行、结构高度模板化的小函数，比真实代码语料窄得多——用它当校准集会让"跨分布涨幅"被**放大**（P2.2 实测 calib=code→eval=NL 比 calib=NL→eval=NL 多涨 +0.56 ppl，看起来像"AWQ 跨分布不稳"，但部分是 mbpp 太窄造成的假象），用它当评测集则 ppl 低得不正常（fp16 baseline 才 3.1）。**应对**：P2.2 结果文件里 `code_corpus_note` 显式写明用的是 mbpp 替代、fp16 baseline 数字、以及"跨分布涨幅含 mbpp 窄语料贡献、非纯 AWQ 效应"；GPTQ 对照臂和真正的 code 语料（codeparrot/c4）留云端阶段，届时重跑 P2.2。

### 9.4 支柱4已知坑

**坑7**：Nightjar实测高负载下投机解码相对无投机倒退最多30.25%，比"batch>64开始下降"更极端更具体。**应对**：P4.2对标这个数字。

**坑8**：草稿模型权重和KV cache抢显存，独立于计算瓶颈之外的第二个原因——与我们自己24GB Mac的资源约束是同一原理的缩影。**应对**：P4.1记录显存占用维度。

### 9.5 支柱5已知坑

**坑9**：GammaTune自己承认的边界——draft/target匹配度高(方差小)时收益有限，依赖历史接受率使其在对抗性/剧烈波动场景下退化。**应对**：P5.1设计波动场景测试。

**坑9 补记（2026-08-28，探索性 side experiment 结果）**：P5.0/P5.1 把 GammaTune 的 null 归因于"主线对 α≈0.79 太稳、方差不足"。做了一步验证——换"更差匹配"的 draft/target 对，看方差变大后 GammaTune 是否就有用（`src/explore_worse_pair.py`，结果 `results/explore_worse_pair_pair{1,2}_*.json`，主线模型对不动）。**探索对 1**（draft 换成 `Qwen2.5-0.5B` **BASE** 版）：α(γ=3) 从 ~0.79 掉到 0.700±0.027，但 accept 长度 pooled std 各 γ 与主线几乎相同（1.28/1.91 vs 1.24/1.95）。**探索对 2**（target 换成 `Qwen2.5-3B-Instruct`，能力差距 6×）：α(γ=3) 反而是 0.726±0.029（比对 1 还高），方差同样与主线持平。两对上 GammaTune 结论与 P5.0 一致（主指标落后最优固定 γ 约 -21~-22%，成本模型 c∈{4,7,10} 落最优簇内差 1~3.4%）。**教训**：在 Qwen2.5-Instruct 同族内换模型对**造不出**检验坑9 假设所需的"α≈0.5–0.65 + 明显更大方差"regime——共享 tokenizer / 同族 / instruct-tuning 主导了 draft↔target 分布对齐，base draft 只压低"接受率"不放大"方差"，拉大能力差距对两者都几乎不动。**好处**：P5.0 的 null 因此在 3 个模型对上稳健，不是单对偶然。**遗留**：真正检验坑9 需要跨族 / 非 Qwen draft（会破坏坑1 的共享 tokenizer 前提，需 logits 对齐或 token 映射），属于要用户拍板的方向，未做。

**坑10（v9补充方法论注记）**：SpecKV自己的实验(Table 1)发现最优γ随目标模型压缩程度(BitsAndBytes的FP16/INT8/NF4，不是AWQ)从FP16下的2偏移到INT8下的8，偏移量级4倍——量化和自适应控制不是独立技巧。**v9新增方法论注记**：这个4倍数字是在BnB压缩下测出来的，AWQ的逐通道缩放设计目的是最小化量化对输出分布的扰动，两者机制不同，直接拿自己的AWQ模型去对标这个具体倍数存在方法论错配风险——如果AWQ下测出的偏移明显小于4倍，可以将其视为"AWQ对投机解码更友好"的一个值得报告的发现，无需归结为实验错误。**应对**：P5.2复用支柱2的量化模型验证偏移方向；同时加一个BitsAndBytes NF4对照臂直接复现SpecKV原始设置作为同源参照，避免把"方法不同导致的数字差异"误判为"结果异常"。

**坑11**：Nightjar批评DSD"禁用后不再收集观测数据，难以重新启用"，批评BanditSpec"忽视KV cache重建开销"。**应对**：P5.3熔断器必须含周期性重探测机制+实测切换开销；Nightjar自己实测的同类开销在RTX 4090+DeepSeek-R1-Distill-Qwen-7B配置下是17.87ms(短输入/小batch)到102.03ms(长输入/大batch)，可作为自己实测数字量级是否合理的参照。

**坑12（v9再次更新）**：BanditSpec论文自己在附录B.2说UCBSpec是"最简单的一类UCB算法之一"(只维护经验均值+置信半径)，因此不复现其完整bandit理论的原因并非实现复杂度，而是：(a) 它的K-armed框架没有把batch size当作决策的上下文特征(自己在"未来工作"章节列为Contextual Bandits方向)；(b) 它的regret-bound理论分析没有建模KV cache重建的切换成本，这正是P5.3要弥补的地方。**v9补充**：BanditSpec有公开代码(`github.com/sail-sg/BanditSpec`)，可以直接克隆跑起来做P5.4的第三个对比基线，省去重新实现的时间。**应对**：只作对比引用，不重新实现算法本身；P5.4直接跑其公开代码做基线；P5.3的切换开销实测吸收了这个批评里最相关的部分。

### 9.6 研究诚信：防止自己欺骗自己

> 参考文档有一份"风险台账与反刷设计"，防止平台被用户系统性钻空子。Specter 没有外部用户，对应的风险是**评测/调参过程中无意识地让结果看起来比实际更好**——这在没有同行评审、只有自己既是研究者又是评审者的个人项目里尤其容易发生，需要显式设计对策来应对，不能假设自己足够客观。

**风险1：评测集/校准集泄漏到调参过程**——如果 P5.0 的 GammaTune 超参数(η, δ, γ_min/max)是在 P3.0 的 AgentBench-OS 任务集上反复试出来的，那么 P5.1/P5.4 用同一套任务集"验证"控制器有效，验证的只是过拟合。**应对**：P3.0 的 15-20 个任务在设计阶段就固定切出一部分(建议 3-5 个)作为 held-out set，只在所有超参数定下来之后跑一次，不回头调整。

**风险2：Best-of-N 挑好看的一次运行报告**——batch 扫描(P4)、γ 扫描(P1.4)、量化对比(P2.4)这类实验都有随机性(采样温度、校准集shuffle顺序)，如果每个数字都是"跑了几次挑最好的一次"，报告出来的加速比会系统性偏高。**应对**：凡是进最终报告的数字，至少跑3次取均值±标准差；如果标准差大到影响结论(比如GammaTune 15%提速的置信区间跨过了0)，如实写出"这次没有统计显著性"，而不是只报均值。

**风险3：正确性验证器本身有bug却一直"通过"**——P1.2的贪心比对器如果实现有误(比如比较逻辑写反、或者两条推理路径共享了同一份缓存导致"殊途同归"式的假阳性)，会一直静默通过，给人"正确性已验证"的错觉，比不验证更危险。**应对**：故意注入已知会破坏正确性的bug(比如手动让bonus token从错误的模型采样、或者手动打乱一个draft token)，确认验证器**必须报错**——"测试验证器本身"是P1.2完成判据的一部分，不是事后可选项。

**风险4：符合预期的结果比违反预期的结果少一道复核**——如果P2.2真的复现出"AWQ比GPTQ更抗跨分布过拟合"，因为这和AWQ论文的结论一致，很容易不假思索地直接采信；但如果结果恰好和论文数字接近到可疑的程度，反而应该多查一遍(比如检查是不是校准代码不小心复用了同一份数据、或者perplexity计算窗口和论文不一致导致巧合)。**应对**：任何"和文献高度吻合"的结果，都过一遍"如果这是bug造成的假象，最可能是哪个环节"的反向检查清单，再写进报告。

**风险5：引用链条比原文长两层以上时,归因关系会悄悄失真**——多手转述是最不容易被自己发现的错误类型，因为读起来完全通顺、不会触发"这里有问题"的直觉。SpecKV的41.2%误归因就是这样进入v6/v7的。**应对**：任何要写进计划或最终报告、且来自"精读论文"这个动作的具体数字，标注时至少包含"论文名+这是第几节/哪张表+这是该论文自己的结果还是它转述的背景引用"三项，缺一项就视为未完成溯源，不能直接采信。

**风险6：拿别人的实验结果当对标时,没核对方法论前提是否一致(v9新增)**——风险5针对"数字本身记错了归属"，这条针对另一种更隐蔽的错误："数字归属没有记错，但拿来对标的自己的实验与原实验条件并不相同"（P5.2的AWQ vs BnB就是这类风险的具体触发案例：SpecKV的2→8确实是SpecKV自己的结果，没有归因错误，但直接拿这个数字给AWQ实验定"应该复现的量级"是方法论层面的误用）。**应对**：任何"对标XX论文的YY数字"的表述，写之前先确认自己的实验设置(压缩方法/模型规模/数据集)和对方是否真的可比，不可比时要么补一个同源对照组，要么把"应该复现同等量级"的预期改写成"方向应该一致，量级差异本身可以是发现"。

---

## 10. 依赖与假设

### 硬件/环境
- 开发机：Mac,24GB统一内存,无独立GPU
- 云端：Vast.ai/RunPod按需租用NVIDIA GPU实例

### 软件依赖(按阶段)
| 阶段 | 依赖 | 约束 |
|---|---|---|
| Mac开发 | HuggingFace Transformers, PyTorch(MPS) | 确认所选模型对MPS算子支持完整；MPS不支持真实int4/int8量化张量运算(见§9.1风险B)，量化正确性验证只能走fake-quantize模拟 |
| Mac交叉验证 | mlx-lm(含`mlx_lm.awq`原生AWQ路径，v9新增) | α数量级sanity check；`mlx_lm.awq`额外可产出真实的Mac端AWQ速度/内存数字，见P2.4 |
| 云端验证 | vLLM | CUDA≥11.8 |
| 云端量化对比 | LLM Compressor(GPTQ) + BitsAndBytes(NF4，v9新增，P5.2对照臂用) | CUDA环境,不支持Mac；BnB配置简单(`load_in_4bit=True`)，不算重新实现GPTQ |
| 云端基线对比 | BanditSpec公开代码(`github.com/sail-sg/BanditSpec`，v9新增) | 需要`torch transformers fairscale flash-attn`，flash-attn需针对CUDA/PyTorch版本编译；只用来跑对比，不修改其算法逻辑 |

### 模型/数据假设
- draft/target模型对暂定Qwen2.5系列,具体尺寸待P1.0结果确定
- 假设模型对共享一致tokenizer/vocab(P1.0显式验证,不假设)
- 量化校准：C4 + 至少一个分布明显不同的第二数据集(P2.2)
- Agent评测：改编自AgentBench OS子环境的15-20个任务(附录B)，其中3-5个held-out(9.6风险1)

### 云端预算执行纪律(v9新增)
所有计划在云端跑的脚本(LLM Compressor校准、vLLM基线、BnB对照臂、BanditSpec代码、P4.2大模型batch扫描)必须先在本地用极小的dummy模型(比如2-4层的toy transformer,几秒钟能跑完一次完整流程)验证脚本逻辑无误，再上云端跑真实规模。理由：云端调试比本地调试烧钱快得多，$50预算经不起"边跑边改bug"的消耗；云端时间只应该用来产出真实规模的数字，不用来定位代码逻辑错误。

---

## 11. 路线图与工作量(对标参考文档 §6 格式)

| 阶段 | 上线内容 | 工作量 | 环境/预算 | 完成判据 |
|---|---|---|---|---|
| 0 | P1.0前置gate | S(0.5天) | Mac,$0 | vocab断言通过;α≥0.4;mlx-lm交叉验证一致 |
| 1 | P1.1-P1.4投机解码核心+验证 | M(7天) | Mac,$0 | 贪心100%一致;采样α与理论公式吻合;bonus token测试通过;验证器故障注入测试通过(9.6风险3);γ扫描曲线产出 |
| 2 | P2.0-P2.3 AWQ校准+正确性 | M(5-6天，v9调整：P2.1从2天上调到3-4天) | Mac,$0 | perplexity涨幅数字产出;跨分布矩阵产出;校准集消融产出 |
| 3 | P3.0-P3.1 AgentBench-OS | S(3天) | Mac,$0 | 15-20任务跑通(含3-5个held-out);结构化vs自由文本α对比数字产出 |
| 4 | P5.0-P5.1+P5.3自适应控制器(Mac部分) | M(4天) | Mac,$0 | GammaTune复现论文量级提速(3次跑均值±标准差);波动场景测试产出;熔断器重探测机制实现 |
| 5 | P4.0-P4.1 batch交叉点(Mac部分) + P2.4 mlx_lm.awq交叉验证(v9新增) | S(2天) | Mac,$0 | 交叉点初步曲线产出;显存占用数据产出;mlx_lm.awq真实Mac AWQ速度数字产出 |
| 6 | P2.4(云端补全)+P4.2+P5.2(含BnB对照臂)+P5.4(三个基线) 云端规模化验证 | M(4天，v9调整：新增BnB对照臂+BanditSpec代码运行) | 云端,$30-50 | LLM Compressor对比数字;vLLM对比数字;AWQ量化-γ偏移结论+BnB同源对照结论;HF双基线+BanditSpec代码对比数字;held-out集跑一次最终确认 |
| 7 | 产出 | S(2天) | $0 | GitHub repo+README定稿;简历bullet定稿 |

**成本注记**：Mac阶段全程$0；云端阶段预算硬顶$50(护栏指标)，超支即停止规模化验证用已有结果收尾，且所有云端脚本必须先本地dry-run(见§10)。**总时长**：约27-28天(v9从25-26天上调，主要因P2.1和P5.4/P5.2的额外对照实验)。

---

## 12. 验证/测试计划

- **贪心模式**：逐token严格比对，目标模型直接推理 vs 投机解码输出必须完全一致(或仅可忽略浮点误差)。
- **采样模式**：(a)统计层面验证实测α与理论公式$E[\min(p,q)]$吻合;(b)下游任务指标parity(差距<2点)。
- **量化正确性**：perplexity涨幅、逐层量化误差分布。
- **自适应控制器**：稳定场景+故意设计的波动场景双重测试。
- **验证器自身**：故障注入测试(9.6风险3)——手动破坏正确性，确认验证器能抓到。
- **引用溯源**：任何写进最终报告的文献数字，附上"论文名+章节/表号+是否为该论文自己的结果"三项标注(9.6风险5)。
- **对标方法论前提核对(v9新增)**：任何"对标XX论文数字"的结论，先确认自己的实验设置和对方论文是否可比(9.6风险6)，不可比时补同源对照组或改写预期。
- **交叉验证**：mlx-lm跑同一模型对做α数量级对拍，`mlx_lm.awq`做AWQ速度交叉验证；参考[romsto/Speculative-Decoding](https://github.com/romsto/Speculative-Decoding)等开源实现做输出合理性sanity check；BanditSpec公开代码作为P5.4的第三个对比基线。

---

## 13. 端到端走读（计划中的产出格式，非已实测数字）

> 对标参考文档 §8"一天的旅程"。参考文档的故事里包含一次险些失败又恢复的交换过程，用叙事本身检验系统在不顺利情况下是否仍然可靠，而非只演示顺利路径。下面的走读同样包含一次"控制器一度做出错误决策、随后依靠自身补救机制恢复"的场景。

一个 agent 收到任务："读取 `repo/utils.py`，把其中的 `parse_config` 函数重构成支持嵌套 key 的版本，跑通已有单测。"

**第一幕：顺利的主路径**

1. **草稿模型先开口**。3B 级别的草稿模型看到任务描述 + 文件内容，自回归生成前 γ=5 个候选 token（比如先吐出 `def parse_config(`）。因为这是代码补全场景（经验依据来自 IBM/PyTorch 生产复盘：代码场景可以用更多 draft token），P5.0 的 GammaTune 控制器此时观察到最近几轮的高接受率，已经把 γ 从默认值扩到 7。

2. **目标模型单次前向验证**。7B 目标模型对这 7 个候选 token 做一次前向，逐位比较：前 5 个匹配，第 6 个不匹配——接受 5 个，从目标模型的调整分布里重采样第 6 个，丢弃第 7 个候选，本轮产出 6 个 token。P1.3 的统计验证器把这一步的接受数记录进 α 的滚动窗口。

3. **量化在背后悄悄生效**。目标模型本身是 P2 阶段自己校准出来的 4-bit AWQ 版本——如果这一步是在 Mac 上跑，验证的是"输出和 FP16 目标模型比 perplexity 涨幅是否可控"；如果是云端阶段用 LLM Compressor 的 GPTQ 版本做对照，验证的是"真实加速比"。

**第二幕：批量压力下控制器暂时判断失误——以及它怎么发现自己错了**

4. **batch 从 1 涨到 16。** 同时有 16 个类似的重构任务在跑。P4 的 batch 扫描告诉我们：从某个 batch 值开始，草稿模型的开销不再被目标模型省下来的前向次数覆盖，P5.3 的熔断器观察到吞吐下降信号，把 γ 降到 1（等效关闭投机解码）。

5. **与此同时，其中4个任务的性质发生了变化**——它们不再是"重构一个函数"这种代码模式规律的任务，而是转成了"读一段自然语言 changelog，判断要不要触发一次版本号升级"这种更接近自由文本推理的任务。如果熔断器是 DSD 那种"关了就不会再看"的设计（坑11），它会持续认为"投机解码在这批任务上不划算"，但这4个任务的α下降原因其实与"高batch"无关，是任务类型变了才需要重新评估。**这正是9.6风险4想防的陷阱**：一个表面符合预期的判断（高batch→该关）掩盖了另一个原因（任务类型变了）。

6. **P5.3设计的周期性重探测机制在第50步插入一次γ=3的试探**——探测到这4个任务的α回升到了0.6以上，原因是它们混进了原本就该受益的代码模式任务，与batch下降无关。按照P5.3的设计意图，控制器不会整体重新开启投机解码，而会把粒度下放到per-task级别记录这次探测结果，为后续同类任务重新评估γ提供依据。这一行为在实现前属于**设计待验证的假设**，而非已确认的行为：P5.1的波动场景测试专门用来检验控制器在"部分任务性质变化、部分没变"这种混合场景下是否按预期反应，还是像GammaTune论文承认的那样在非稳态场景下打折扣（坑9）。

**第三幕：落到评测**

7. **最终评测**。这个重构任务本身来自 P3 改编的 AgentBench-OS 任务集里的一条，最终是否 FINISHED（单测通过）决定了这条 trace 在下游任务指标里怎么计分。第二幕的那次"熔断器差点判断失误"不会出现在最终的加速比数字里——它会被记录进P5.3的实验日志(附录C)，作为"控制器设计是否达到预期"这一独立问题的证据，和"这个任务本身有没有做完"分开评估，不混为一谈。

这条走读连接了支柱1(接受率)、支柱2(量化正确性)、支柱4(batch行为)、支柱5(自适应控制，包括它可能犯错的地方)、支柱3(任务结果)。第二幕的设计是刻意的：只呈现成功场景的系统设计，通常说明设计者没有充分考虑失败模式。

---

## 14. 我们能讲的故事

一句话定位候选（面试/简历场景下用）：

1. "投机解码和量化是同一个系统原理的两个例证，我测出了这个原理失效的具体边界在哪。"
2. "调库打开投机解码只要一行 flag；我把这一行 flag 背后的正确性证明、失效条件、自适应控制全部手写实现并验证了一遍。"
3. "我知道生产界的最优解是训练专用 speculative head——我诚实地选择了预算约束下可行性更高的方案，并且把这个取舍写进了报告里。"

北极星论点：这个项目要证明的是对这些加速技巧失效条件的理解，以及设计一个能感知自身失效边界的系统的能力——这代表从"实现论文里的技巧"到"理解系统边界"的差距，也是新grad项目和有2-3年经验的工程师项目之间常见的差距。

---

## 15. 产出物

- GitHub repo(代码+README，含论文引用、prior art引用、诚实边界说明——包括平台兼容性踩坑过程本身)
- 简历bullet(见附录D)
- 本系列计划文档(v1-v9)作为项目思考过程的书面记录，可选择性放入repo的`docs/`目录

---

## 16. 悬而未决的问题

1. 具体draft/target模型尺寸对——暂定Qwen2.5系列，等P1.0的α-gate结果才能定。
2. P5.3熔断器是否和P4的batch扫描代码共享同一套实现——倾向共享，等P4实现完看接口是否好复用。
3. 云端预算在阶段5和阶段6之间的具体分配——等实际云端GPU租用单价和阶段4本地测试结果出来后细化；v9新增的BnB对照臂+BanditSpec代码运行会分走一部分预算,需要在阶段6开始前重新估算。
4. AgentBench任务集改编的具体任务数量(15个还是20个，扣除held-out后实际可调参的任务数会更少)——先按20个设计，如果太重再削减。
5. **P5.2的AWQ-BnB对照实验如果测出偏移量级差异很大，报告里要花多少篇幅讨论"为什么不同"(v9新增)**——这本身可能是一个值得深入的小发现，但也可能超出项目原定的时间预算，需要在跑出结果后再决定深挖还是简单记录。

---

## 17. 决策记录

v1-v6：均由用户在迭代过程中通过继续深化调研要求的方式确认，尚无正式的"批准进入实现阶段"信号。
v7：读完参考文档全部28页后补齐颗粒度缺口，内容决策不变，等待用户确认后再讨论是否进入阶段0。
v8：精读5篇自适应控制论文原文并更正一处错误引用(SpecKV"41.2%")，重新措辞BanditSpec不复现的理由；架构/技术决策不变。
v9(本版本)：用户要求"以资深架构师视角，从可行性和已有成熟方案角度"评审v8。评审发现P5.2实验设计存在AWQ/BnB方法论错配风险(已修正为方向性验证+BnB同源对照)、P2.1(AWQ)工作量估计偏乐观(2天→3-4天,总时长25-26天→27-28天)、以及mlx_lm.awq/BanditSpec公开代码/HF双重内置基线三处"已有成熟方案未被利用"的信息差(均已补入相应章节)；§8补充self-speculative作为此前遗漏的备选方案并写明拒绝理由；新增云端预算的本地dry-run执行纪律。架构方向不变，仍等待用户确认后再讨论是否进入阶段0。

---

## 附录A：核心公式、算法伪代码与数值算例

### A.1 Rejection Sampling（Leviathan et al. 2023）

```
for i in 1..γ:
    draft_token[i] ~ p_DM(x | context, draft_token[1..i-1])
target_probs = TargetModel.forward(context, draft_token[1..γ])  # 单次前向
n_accepted = 0
for i in 1..γ:
    r ~ Uniform(0,1)
    if r < min(1, p_TM(draft_token[i]) / p_DM(draft_token[i])):
        accept draft_token[i]; n_accepted += 1
    else:
        break
if n_accepted < γ:
    x_new ~ norm(max(0, p_TM(x) - p_DM(x)))  # 从调整分布重采样
else:
    x_new ~ p_TM(x | context, draft_token[1..γ])  # bonus token，必须来自目标模型
```

**数值算例**（γ=3，词表简化成3个候选token {A, B, C}，模拟一次verification）：

草稿模型对第1个位置的分布：`p_DM = {A:0.7, B:0.2, C:0.1}`，采样得到 `A`。
目标模型对同一位置的分布：`p_TM = {A:0.4, B:0.3, C:0.3}`。

- 接受概率 = `min(1, p_TM(A)/p_DM(A))` = `min(1, 0.4/0.7)` = `0.571`。
- 假设采样到的 `r=0.8 > 0.571` → **拒绝**，本轮到此为止，不再看后续候选。
- 从调整分布重采样：`p'_TM = norm(max(0, p_TM - p_DM))` = `norm({A:max(0,0.4-0.7)=0, B:max(0,0.3-0.2)=0.1, C:max(0,0.3-0.1)=0.2})` = `{A:0, B:0.33, C:0.67}`。
- 最终这一步吐出的 token 从 `{B:0.33, C:0.67}` 里采样，**绝不会是A**——这符合直觉：目标模型没那么偏爱A（0.4 vs 草稿模型的0.7），所以拒绝后重采样把A的概率清零，公平地把概率让给目标模型相对更青睐、又被草稿模型低估的B和C。

这个算例也直接演示了坑2要防的错误：如果重采样时不小心又从 `p_DM` 采样（而不是从上面算出的调整分布 `p'_TM` 采样），会系统性地偏向草稿模型的偏好（A），累积下来的输出分布就不再等于目标模型单独推理的分布——这是"看起来能跑、但正确性保证已经被破坏"的那类bug，P1.2的贪心验证器要能抓到,9.6风险3要求故意注入这类bug测试验证器本身。

### A.2 GammaTune 成本模型与Algorithm 1（Kim et al. 2025）

$$\text{cost} = \frac{N}{\alpha\gamma+1} \times (c+\gamma) \times T_{draft}, \quad c = T_{target}/T_{draft}$$

```
if A == γ:  # 上一轮全部接受，可能还有余量
    γ ← A + δ
else:
    γ̄ ← clip(γ_min, γ_max, (1-η)·γ̄ + η·A)
    γ ← ceil(γ̄)
```

**三轮数值算例**（η=0.3, δ=2, γ_min=1, γ_max=10, 初始 γ̄=3, γ=3）：

| 轮次 | 实际接受数 A | 触发分支 | 计算 | 新 γ̄ | 新 γ |
|---|---|---|---|---|---|
| 1 | 3（全部接受，A==γ） | 扩窗 | γ ← 3+2 | （不更新γ̄） | 5 |
| 2 | 2（部分接受，A<γ） | EMA | γ̄ ← clip(1,10, 0.7×3 + 0.3×2) = clip(1,10,2.7) | 2.7 | ceil(2.7)=3 |
| 3 | 3（全部接受，A==γ） | 扩窗 | γ ← 3+2 | （不更新γ̄） | 5 |

可以看到：只要连续命中"全部接受"，γ 会持续扩窗式增长（3→5→7→...直到γ_max）；一旦某一轮没有全部接受，立刻回落到EMA的保守估计，不会让γ在一次不理想的输出后继续野蛛式增长——这是"扩窗要快，收缩要稳"的设计意图在数字上的体现，也是P5.1波动场景测试要验证的核心行为。原论文还提出了一个GammaTune+变体，额外加入基于草稿模型top-1 logit概率的提前停止阈值τ，在4组模型对上平均提速从GammaTune的1.15±0.05x略微提升到1.16±0.03x——Specter不实现这个变体(不在P5.0范围内)，只作为"同一套EMA思路还能怎么扩展"的背景参考。

### A.3 AWQ 逐通道缩放——直觉性数值算例

AWQ 的核心洞察：并非所有权重通道都同等重要，激活值幅度大的输入通道对应的权重通道更"显著"，量化时应该被优先保护。一个简化的两通道玩具例子：

假设某一层有两个输入通道，激活值统计幅度分别为 `|X1|=10, |X2|=1`（通道1的激活值普遍大10倍，说明这个通道承载的信号更重要）；原始权重 `W1=0.5, W2=0.5`（两个通道权重原本一样大）。

- 如果直接对 `W1, W2` 统一做4-bit量化而不做任何缩放，两个通道的量化误差量级相近——但通道1的误差会被激活值放大10倍(`误差×|X1|`)，对最终输出的影响远大于通道2。
- AWQ 的做法：引入缩放因子 `s`(通常与激活值幅度正相关，比如 `s=|X1|^α` 的形式，α 是一个校准出来的超参数)，量化前把 `W1` 放大为 `W1×s`、把对应的激活值缩小为 `X1/s`，使数学上等价(`(W1×s)×(X1/s)=W1×X1`不变)，但量化误差被压缩到激活值更"迟钝"（即通道1本身激活值大、相对量化误差的百分比影响反而变小）的那一侧。
- 净效果：通道1(重要通道)的有效量化精度因缩放因子而提高，通道2(不重要通道)相应地牺牲一点精度——这组 `s` 值需要在P2.1中通过校准集统计量计算得出。**v9提醒**：这个玩具例子只涉及一个数学步骤；P2.1实现时最容易卡住的是fake-quantize的round→clamp→dequantize顺序和激活值统计hook的挂载时机，这两处debug往往比写这个数学步骤本身耗时得多(见§7 P2.1工作量修正)。

这个算例只是帮助建立直觉，P2.1实现时的具体缩放因子搜索方式（网格搜索最小化量化后输出误差）以AWQ原论文Section 3的公式为准，不是这里简化的启发式。

---

## 附录B：AgentBench-OS 子集任务清单（草案，实现时细化）

改编自 AgentBench 的 OS/bash 子环境，聚焦"生成结构化输出/工具调用"这一投机解码效果最好的场景，草案任务类别（每类2-4个具体任务，共15-20个，其中3-5个在设计阶段就标记为held-out，仅在超参数定稿后跑一次，见9.6风险1）：
1. 文件操作类（读取/修改/重命名指定文件）
2. 代码重构类（如 §13 走读示例：重构函数签名并跑通单测）
3. 命令行工具调用类（用 grep/find/awk 完成指定查询并输出结构化结果）
4. 多步骤依赖类（前一步的输出是后一步的输入，测试长上下文下的接受率变化）

每个任务的完成判据（FINISHED）：产出的文件/命令输出通过预先写好的校验脚本，不依赖人工判断。

---

## 附录C：实验日志 / Telemetry Schema + 诊断分析代码

> 对标参考文档里引用自己产品日志(`tx_demand_search_log`等)和附录A"只读查询"的做法。Specter没有生产数据库，日志落地为本地文件（jsonl/parquet），下面的"查询"是pandas风格的伪代码,不是真实SQL。

### C1. 日志字段定义

**每个投机解码步骤应记录**：`step_id, draft_tokens[], target_logits_hash, n_accepted, gamma_used, latency_ms, model_pair_id, quantization_level, quantization_method`（v9新增`quantization_method`字段，用于区分AWQ和BnB对照臂，避免P5.2分析时把两种压缩方式的数据混在一起算）

**每个γ调整决策应记录**（P5.0-P5.3）：`timestamp, prev_gamma, new_gamma, trigger_reason(EMA更新/扩窗/熔断/重探测), acceptance_history_window`

**每个batch扫描数据点应记录**（P4）：`batch_size, throughput_tokens_per_sec, draft_model_memory_mb, kv_cache_memory_mb, speedup_ratio_vs_no_spec`

**每个量化校准实验应记录**（P2.2/P2.3）：`calibration_dataset, eval_dataset, n_calibration_samples, perplexity, group_size, quantization_method`

### C2. 诊断分析代码（只读，实现阶段直接复用）

```python
# 滚动窗口接受率 α（对应P1.3、P5.0的EMA输入）
df_steps["alpha_rolling"] = (
    df_steps["n_accepted"] / df_steps["gamma_used"]
).rolling(window=50).mean()

# 加速比按 batch_size 分布（对应P4，定位交叉点）
speedup_by_batch = (
    df_batch.groupby("batch_size")["speedup_ratio_vs_no_spec"]
    .agg(["mean", "std", "count"])
)
# 交叉点 = speedup_ratio 均值首次跌破 1.0 对应的 batch_size

# γ调整触发原因分布（对应P5.3，检查熔断/重探测是否按预期触发）
trigger_counts = df_gamma_decisions["trigger_reason"].value_counts(normalize=True)

# 量化方法对最优γ的偏移（对应P5.2，坑10核心实验，v9：按quantization_method分开算，AWQ和BnB不混算）
optimal_gamma_by_quant = (
    df_steps.groupby(["quantization_method", "quantization_level"])
    .apply(lambda g: g.loc[g["latency_ms"].idxmin(), "gamma_used"])
)
shift_ratio_by_method = (
    optimal_gamma_by_quant.groupby("quantization_method")
    .agg(lambda s: s.max() / s.min())
)
# 预期：shift_ratio_by_method["bitsandbytes"] 应接近SpecKV的4倍(同源对照)
# shift_ratio_by_method["awq"] 可能明显不同——这本身是可报告的发现，不是异常

# 跨分布校准 perplexity 矩阵（对应P2.2，坑5核心实验）
ppl_matrix = df_calibration.pivot_table(
    index="calibration_dataset", columns="eval_dataset", values="perplexity"
)

# 验证器故障注入测试结果（对应9.6风险3）：注入已知bug后必须全部FAIL
assert df_fault_injection_tests["detected"].all(), "验证器漏检——先修验证器再信任其他结果"
```

---

## 附录D：简历 bullet 草稿

- 手写实现投机解码采样算法，贪心模式下逐 token 精确验证正确性，采样模式下通过统计检验和任务指标 parity 确认分布等价；诊断并规避 tokenizer 不一致、bonus token 误采样等已知实现陷阱
- 自研 AWQ 风格激活感知量化校准 pipeline，通过跨分布校准实验验证其相对 GPTQ 的抗过拟合优势
- 实测投机解码与量化在不同 batch size 下的吞吐增益曲线，定位出两者从"内存带宽优化"转为"计算瓶颈拖累"的交叉点，并验证不同量化方法(AWQ vs BitsAndBytes)对最优投机解码步长的系统性影响存在差异
- 实现训练-free 的自适应投机解码步长控制器（基于 EMA 的 GammaTune 算法），设计 batch-aware 熔断与周期性重探测机制，对标 Hugging Face Transformers 双重内置基线及 BanditSpec 开源实现
- 基于 AgentBench 方法论改编 agent 工具调用评测集，验证结构化输出场景下投机解码接受率显著高于自由文本生成

---

## 附录E：分类文献与信息来源

### 一、投机解码核心与SOTA方法
- Leviathan et al. 2023 (arXiv 2211.17192, ICML'23)，*Fast Inference from Transformers via Speculative Decoding* —— 奠基论文之一，rejection sampling 正确性证明的来源，附录A.1算例直接基于其算法描述。
- Chen et al. 2023 (arXiv 2302.01318, DeepMind)，*Accelerating LLM Decoding with Speculative Sampling* —— 另一篇奠基论文，70B Chinchilla 上 2-2.5x 加速的实证参考。
- Li et al. 2024 (arXiv 2401.15077)，*EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty* —— SOTA方案，特征级drafting而非独立小模型，Specter不复现但作为"生产界最优解"在§8/§14里对照引用。
- Li et al. 2024 (arXiv 2406.16858)，*EAGLE-2: Faster Inference with Dynamic Draft Trees* —— 树形草稿+动态调整，接受率更高，SGLang生产数据（batch=24掉速/batch=64仍有效）来自这条线。
- Cai et al. 2024 (arXiv 2401.10774)，*Medusa: Simple LLM Inference Acceleration with Multiple Decoding Heads* —— 多头并行解码对比基线，需要专门训练，是§8"投机解码架构"决策点的拒绝对象之一。
- Meta 2024 (arXiv 2404.16710, ACL'24)，*LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding* —— self-speculative路线的代表作(v9新增)，训练时对早期层用低dropout、后期层用高dropout+提前退出损失，推理时不需要单独草稿模型，LLaMA上最高2.16x提速；§8已把这条路线列为"考虑过但拒绝"的方案，拒绝理由是它会规避掉本项目想展示的tokenizer/bonus-token类失效模式，与方案本身的优劣无关。
- 综述 2025 (arXiv 2502.19732)，*Speculative Decoding and Beyond: An In-Depth Survey* —— 全景综述，含SpecInfer/Sequoia等树形方法背景。

### 二、量化方法
- Lin et al. 2023 (arXiv 2306.00978)，*AWQ: Activation-aware Weight Quantization* —— 支柱2的核心方法来源，跨分布perplexity数字(+0.5-0.6)、附录A.3算例均基于此文。
- Frantar et al. 2023 (arXiv 2210.17323)，*GPTQ: Accurate Post-Training Quantization* —— 对比基线，Hessian-based，跨分布perplexity涨幅(+2.3-4.9)数字来源，本身不重新实现（§3非目标1）。
- Xiao et al. 2023 (arXiv 2211.10438)，*SmoothQuant: Accurate and Efficient PTQ for LLMs* —— W8A8方案，仅作背景对比，说明W4A16（AWQ路线）更适合本项目的低batch/延迟优先场景。
- Dettmers et al.，*BitsAndBytes* —— NF4/INT8量化，HF Transformers内置(`load_in_4bit=True`)，v9新增作为P5.2的同源对照臂，直接复现SpecKV原始压缩设置，不算重新实现GPTQ（配置调用，非算法实现）。

### 三、服务层架构
- Kwon et al. 2023 (arXiv 2309.06180, SOSP'23)，*Efficient Memory Management for LLM Serving with PagedAttention* —— vLLM底层架构论文，决定"造轮子 vs 用轮子"边界：Specter不重新实现KV cache管理，vLLM云端阶段作为现成对比基线引入。

### 四、Agent评测方法论
- Liu et al. 2023 (arXiv 2308.03688, ICLR'24)，*AgentBench: Evaluating LLMs as Agents* —— 支柱3任务设计方法论来源，只改编其OS/bash子环境（§3非目标2）。
- Qin et al. 2023 (arXiv 2307.16789)，*ToolLLM: Facilitating LLMs to Master 16000+ APIs* —— 工具调用任务集设计参考。

### 五、自适应投机步长控制（2024-2026前沿，五篇均已精读原文）
- Kim et al. 2025 (arXiv 2504.00030, Texas A&M)，*Token-Driven GammaTune* —— P5.0核心算法来源，Algorithm 1（EMA更新+扩窗）直接实现，成本模型 cost=N/(αγ+1)×(c+γ)×T_draft 中 c=T_target/T_draft 典型取值4-10；4组模型对上平均提速1.15±0.05x，单张80GB H100实测；GammaTune+变体(logit提前停止)提速略升至1.16±0.03x(附录A.2已引，不实现)；Limitations章节明确承认低方差场景收益有限、对抗性场景会退化（坑9）。
- SpecKV 2026 (arXiv 2605.02888)，*Adaptive Speculative Decoding with Compression-Aware Gamma Selection* —— 最优γ随压缩程度(BitsAndBytes的FP16/INT8/NF4)从2偏移到8(4倍/300%)(Table 1)；headline结果是其MLP-16控制器让每步期望token数比固定γ=4的基线提升56.0%，控制器开销仅0.34ms(单步耗时<0.5%)。**v9提醒**：这个实验用的压缩方式是BnB，不是AWQ，P5.2对标这个数字时需要注意方法论前提是否一致（见坑10、9.6风险6）。
- Agrawal et al. 2024 (arXiv 2410.18351, Qualcomm AI Research)，*AdaEDL: Early Draft Stopping via Entropy-based Lower Bound* —— 熵基提前停止判据 $1-\sqrt{\gamma H_{DM}(x)}<\lambda$ 来源，注意此处γ是论文自定义的"熵因子"超参数(全部实验固定为0.2)，和投机解码通常语境下的"步长γ"是两个不同的量，命名冲突。DL=3/7/16下接受token数标准差递增(1.2/1.92/2.35)确认来源为其Figure 1/附录Fig 7c。独立佐证坑4的数据点：未微调的TinyLlama-1B给Llama2-7B做草稿、固定投机长度=7时，静态投机解码反而比自回归基线慢16%,自适应提前停止后变成快43%。只作文献对比不重新实现（§3非目标3）。
- BanditSpec 2025 (arXiv 2505.15141)，*Adaptive Speculative Decoding via Bandit Algorithms* —— UCB(UCBSpec)/EXP3(EXP3Spec)双算法框架，论文自己在附录B.2明确说UCBSpec"是最简单的一类UCB算法之一"（不复现的原因是它未建模batch size与切换成本，而非实现复杂度，见坑12）；附录B.4明确将自己定位为与SpecDec++"正交"的方案——训练-free vs 训练-based。**v9新增**：本论文有公开代码`github.com/sail-sg/BanditSpec`(兼容LLaMA/Qwen2架构)，P5.4直接克隆运行作为第三个对比基线，不修改其算法逻辑。
- Nightjar 2025 (arXiv 2512.22420)，*Dynamic Adaptive Speculative Decoding for LLM Serving* —— 高负载30.25%吞吐倒退的数字来源(坑7，原文Section 3.1)；批评DSD"重激活难题"和BanditSpec"忽视切换开销"(坑11)；自己实测的KV cache切换开销(Table 3, RTX 4090 + DeepSeek-R1-Distill-Qwen-7B)在17.87ms到102.03ms区间；headline结果是6个7B/13B基准设置上平均比无投机解码提速27.29%、比标准投机解码提速8.32%(最高14.76%)、比BanditSpec/DSD分别提速22.89%/19.76%。
- Lu, Hong, Liu et al. 2025-2026 (arXiv 2512.11280)，*AdaSD: Adaptive Speculative Decoding for Efficient Language Model Inference*（v9新增，评审时发现的更新文献）—— 双组件(熵判定终止生成 + Jensen-Shannon距离判定接受标准)、超参数-free的自适应方案，在Qwen系列模型上有实验(Table 2)，最高1.46x提速、精度损失<1.8%。**不涉及量化维度**，和P5.2/SpecKV不重叠，只作为"自适应投机解码是活跃前沿方向"的又一佐证收入文献综述，不影响现有实验设计。

### 六、生产实践 / 工程博客（"理论之后发生了什么"）
- [A Hitchhiker's Guide to Speculative Decoding](https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/)（PyTorch官方博客，2024，IBM Research生产环境实测）—— 全项目最重要的一手资料：Llama2-13B/Llama3-8B/Granite-7B 2x提速，Granite-20B代码模型3x提速；代码模型用更多draft token更划算（P1.1/P3设计依据）；batch>64开始吞吐下降（支柱4存在理由）；训练speculative head比经典两模型方案效果更好但需要多卡训练资源（§8架构决策拒绝理由的一手证据）；精确对比应用贪心而非采样（P1.2设计依据）；先测接受率再测吞吐的方法论（P1.0 gate设计依据）。
- [HF blog: Dynamic Speculation Lookahead](https://github.com/huggingface/blog/blob/main/dynamic_speculation_lookahead.md)（v9新增）—— HF Transformers内置的`assistant_confidence_threshold`机制的方法论来源，Intel labs与HF合作，最高2.7x提速；P5.4现在明确把这个机制和更简单的`num_assistant_tokens_schedule="heuristic"`分开对待，因为前者方法论上更接近AdaEDL的熵/置信度思路，是更强的基线。
- [HF docs: Assisted Decoding](https://huggingface.co/docs/transformers/en/assisted_decoding)（v9新增）—— `num_assistant_tokens_schedule`和`assistant_confidence_threshold`两个参数的官方文档，P5.4两个基线的具体配置依据。
- [vLLM Blog: Speculative Decoding](https://vllm.ai/blog/tags/speculative-decoding) —— 原生支持EAGLE/EAGLE-3/Medusa/n-gram草稿，一个flag开启，佐证"调库无技术含量"论点（§2.2）。
- [SGLang 文档](https://docs.sglang.ai/advanced_features/speculative_decoding.html) —— EAGLE在batch=24掉速、EAGLE-3在batch=64仍有效的具体数字来源（支柱4）。
- [llama.cpp docs/speculative.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md) —— 经典两模型方案+n-gram自投机的现状参考，EAGLE-3支持仍在讨论（Issue #15902）。
- [AutoAWQ GitHub](https://github.com/casper-hansen/AutoAWQ) —— 官方文档承认高batch场景下W4A16因反量化开销可能变慢（坑8/支柱4呼应），且明确CUDA-only（风险B来源之一）。
- [mlx-lm GitHub](https://github.com/ml-explore/mlx-lm)（v9新增）—— 原生支持`mlx_lm.awq`，Mac端真实AWQ量化+Metal加速推理，P2.4交叉验证来源，也是P1.0/P4的α与速度sanity check工具。
- [Medium: A practical guide to INT4 quantization for SLMs](https://medium.com/data-science-at-microsoft/a-practical-guide-to-int4-quantization-for-slms-gptq-vs-awq-olive-and-real-world-results-2f63d6963d1d) —— GPTQ vs AWQ真实场景对比，W4A16延迟优先/W8A8吞吐优先的场景划分依据。
- LLM Compressor（vLLM团队）—— 统一GPTQ/AWQ/FP8量化API，行业从分裂工具链（AutoGPTQ+AutoAWQ）走向统一工具的现状，决定P2.4对比基线工具选择（§8）。
- [vLLM Issue #1441](https://github.com/vllm-project/vllm/issues/1441)、[vllm-metal GitHub](https://github.com/vllm-project/vllm-metal) —— vLLM在Mac上不可用的直接证据（风险A）。
- [AutoGPTQ Issue #223](https://github.com/PanQiWei/AutoGPTQ/issues/223) —— AutoGPTQ需要专门MPS kernel（未实现）的维护者原话（风险B）。
- [bitsandbytes MPS backend PR #1853](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1853)（v9新增）—— 佐证PyTorch量化生态对MPS支持普遍不成熟这一现状(风险B)，同时说明BnB的MPS支持目前处于PR阶段未必稳定，P5.2的BnB对照臂建议仍放在云端CUDA环境跑，不依赖这个未合并的PR。

### 七、参考开源实现（对拍用，不直接抄）
- [romsto/Speculative-Decoding](https://github.com/romsto/Speculative-Decoding) —— 干净实现Leviathan et al. 2023算法，含经典自回归/beam search/投机解码三种对照，用于P1.2输出合理性sanity check。
- [foundation-model-stack/fms-extras](https://github.com/foundation-model-stack/fms-extras) —— IBM生产代码开源部分，展示PagedAttention kernel如何支持多head KV cache管理，帮助理解投机解码+KV cache管理结合的工程难点。
- [Pramodith Dissects 博客+notebook](https://pramodith.github.io/posts/speculative-decoding/) —— SmolLM2-360M/1.7B手写贪心投机解码，模型选择思路（同系列不同尺寸）与Qwen2.5系列思路一致，α值/加速比量级sanity check参考。
- [sail-sg/BanditSpec](https://github.com/sail-sg/BanditSpec)（v9新增）—— BanditSpec论文官方代码，P5.4直接克隆运行作为第三个对比基线，见§7/坑12。
