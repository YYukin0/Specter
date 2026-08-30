# Bullet 2 执行计划 —— vLLM / A100 EAGLE3 投机解码基准（~$15 一次性云）

> 写给**下一个 agent session** 执行，假设它对本次会话没有任何记忆。所有背景、
> 已完成的本地准备、坑表、命令都在本文件里，不要依赖别处的隐含上下文。
> 本文件是 `TASKS.md` 支柱7 Bullet 2（第 112 行，`[ ]` 未完成）的执行手册；
> 完整的定位/主线/资历叙事见 `notes/简历定稿计划-Specter_2026-08-28.md` 第 3
> 节（这份文件是 source of truth，本计划只管"怎么跑"）。
> **用户已确定用 Lambda Labs 租卡——先读 §1.5，那是专门针对 Lambda 的排查，
> 比下面 §4 步骤1里"Vast.ai 或 RunPod"的通用描述新，以 §1.5 为准。**

## 0. 一句话目标

在 vLLM + A100（或 RTX 4090）上系统评测投机解码（EAGLE3 / n-gram / draft-model
vs baseline），Llama-3.1-8B target，报**并发扫描**（c∈{1,4,16,32,64}）下的
speedup / TTFT-P99 / TPOT-P99 曲线，锁 config 保证跨臂可比。目的：拿到 CUDA +
生产 serving 栈的真实经验、一个正面加速数字、以及与本项目 Apple-silicon 从头
实现的**跨实现对照**（分离"根本限制" vs "我硬件的限制"）。这条实验结果会喂：
README 第 3 个 headline 数字、engineering note #7（Mac vs A100 对照）、demo
页的 "Cloud validation" 卡片（已搭好架子，见 §5）。

## 1. 读这个之前先读：预算与安全护栏

- **硬预算 ~$15–20，一个下午。** 原先设想的 Vast.ai/RunPod 价位（RTX 4090
  spot ~$0.25–0.35/hr、A100 ~$0.5–1.64/hr）已不适用——**用户确定用 Lambda
  Labs**，Lambda 没有 spot，A100/H100 on-demand 历史价位在 $1.3–3.3/hr 区间
  （以 lambda.ai/pricing 当天实际显示的为准），**按分钟计费、从 launch 到
  terminate 整段时间都算钱**（不是按 GPU 利用率），预算和时间控制要按这个
  口径重新估：4臂×5档并发的完整矩阵按 P-EAGLE 论文量级，2–3 小时内应该能
  跑完，加上搭环境/下模型/核对 CLI 语法，一个下午单卡预算仍在 $15–20 区间，
  但比 Vast.ai 那档略贵，纯粹是换了"不会被抢占、不用挑 spot"这个稳定性。
- **`src/cloud_bench/orchestrate.py` 有 `--max-runtime-min` 硬顶**（默认 180
  分钟）——超时会抛 `RuntimeBudgetExceeded` 并中止，不会讲那台机器晾在那。
  但这只挡得住"跑单个 `orchestrate.py` 进程"这段时间，**不会自动销毁云实例**。
- **跑完/出错/中途放弃，第一件事都是去 Vast.ai/RunPod 控制台销毁实例**——
  这件事没有任何脚本能替你做，必须是执行者手动确认。
- 不要在没做 §4 步骤 4 的"~95% 复现自检"之前就跑完整矩阵——配错一个 flag
  在完整矩阵上重跑是双倍花费。

## 1.5 Lambda Labs 专属风险排查（2026-08-30 新增，用户已确定用 Lambda，执行前必读）

这一节是专门针对 Lambda Labs（而不是原先设想的 Vast.ai/RunPod）做的一次全面
排查，把能在本地/查资料阶段发现的问题都列出来，代码能修的已经修了。

1. **Lambda 只有 terminate，没有 stop/pause。** 官方文档确认：instance 一
   launch 就开始按分钟计费，直到 terminate 才停，不管你有没有在跑东西；本地
   NVMe 上的所有东西 terminate 就永久消失（除非另外挂载 persistent
   filesystem——那是单独收费、按 region 锁定的东西，这次用不上）。**结论：
   这次必须一次性跑完**——环境搭建→sanity check→完整矩阵→结果 scp 回本地→
   立刻 terminate，中间不要指望"先歇一下，明天接着跑"，中途真要停就必须先把
   整个 `results/` 目录 scp 回本地再 terminate，下次是全新实例，环境要重搭。
   这也是为什么步骤5要求全程 `tmux`——一旦开始跑完整矩阵，中途唯一能保护
   进度的手段就是不让 SSH 断线杀死进程 + `orchestrate.py` 自带的续跑
   （见下方步骤5正文），terminate 本身没有任何"救回来"的余地。
2. **A100/H100 on-demand 容量在 2026 年经常紧张**，高峰期 dashboard 可能直接
   显示某型号 0 台可租——这是 Lambda 全平台的普遍情况，不是账号问题，换个
   region 或者 A100/H100 互相试一下即可。
3. **`--max-model-len` 之前漏设，已经在代码里修掉。** vLLM 不显式设这个会按
   模型原生上下文长度（Llama-3.1-8B 是 128K）去分配 KV cache/CUDA graph，
   单卡上轻则起服务巨慢，重则 OOM——GSM8K 的 prompt 和 `OUTPUT_LEN=1024`
   根本用不到那么长。`config.MAX_MODEL_LEN=4096` 现在会被
   `vllm_serve_cmd` 自动带上，不需要手动加。
4. **`VLLM_USE_V1` 环境变量已经在 vLLM 2025-11-12 那次改动里被彻底删除**
   （V1 现在是唯一引擎，没有 V0 可切换），`remote_setup.sh` 里原来
   `export VLLM_USE_V1=1` 那行是空操作，已经删掉——不影响正确性，只是避免
   照抄一个不存在的开关。`pip install "vllm>=0.9"` 在 2026-08 装到的实际
   版本会新得多，功能上没问题，但仍然要走一遍 §3 的 `--help` 核对。
5. **`ngram` / `draft_model` 这两个 method 名字这次查证仍然有效**——官方
   vLLM speculative decoding 文档现存的 method 列表（ngram / suffix / eagle /
   eagle3 / mtp / draft_model / medusa / dflash）明确包含这两个，
   `config.py` 里锁的字段基本对得上官方给的示例写法。这原本是这份计划里
   最担心的一处（之前没查证过），现在降级为低风险，§3 起服务时肉眼确认一次
   就够了。
6. **HF gated 模型批准状态是硬门槛。** 你之前截图两个模型都还是 pending；
   必须等 huggingface.co 上两个模型都显示"You have been granted access"
   （不是 pending）再租机器——pending 状态下 `huggingface-cli login` 之后
   `snapshot_download`/`vllm serve` 照样 403，等于白付了搭环境那段时间的钱。
7. **guidellm CLI 语法仍是这份计划里最大的未验证项，没有变化。** 这次查到
   vLLM 官方项目自己给的示例用的正是 `--rate-type concurrent --rate N`
   这个搭配，可信度比之前高，但 guidellm 版本迭代快，§3 仍然要求先
   `guidellm benchmark --help` 核对一遍再花钱跑矩阵。
8. **Lambda 计费按"launch 到 terminate"整段时间的分钟数算，不看 GPU 利用率**
   ——搭环境、下模型、核对 CLI 语法这些"没在真正跑基准"的时间也在计费，
   所以 §4 步骤2/3 要抓紧做，不要一边挂着机器一边慢慢查文档。
9. **Lambda 要求先在 dashboard 上传 SSH 公钥才能 launch 实例**，没有密码登录
   这条路——如果你本地还没有 SSH key，租之前先 `ssh-keygen` 生成一对。

## 2. 本次会话已经在本地准备好的东西（无需重做）

全部在没有 GPU、没有装 vllm/guidellm 的情况下写好并测过（`.venv` 里两者都
没装，`which vastai runpodctl guidellm vllm` 本机全部找不到——这是预期的，
本地只搭骨架，不跑真基准）：

- `src/cloud_bench/config.py` —— 锁定的 config 常量（模型名、
  `num_speculative_tokens=3`、`temperature=0`、`top_p=1.0`、`output_len=1024`、
  `dataset=gsm8k`、`concurrencies=(1,4,16,32,64)`）+ 四个臂的 `ArmSpec`
  （`eagle3` / `ngram` / `draft_model` / `baseline`）+ 论文复现自检阈值常量。
- `src/cloud_bench/orchestrate.py` —— 纯 subprocess 编排（不 import
  vllm/guidellm 库，所以本地能测）：
  - `vllm_serve_cmd(arm)` —— 拼 `vllm serve ... --speculative-config '{...}'`。
  - `guidellm_cmd(concurrency, output_path)` —— 拼 GuideLLM 调用（见 §3 的
    CLI 语法警告）。
  - `wait_for_health(url, ...)` —— 轮询 `/health`，`sleep`/`now`/`urlopen`
    都是可注入参数（测试用假时钟，不真的 sleep）。
  - `run_matrix(arms, concurrencies, dry_run=True, max_runtime_min=180, ...)`
    —— 遍历臂×并发，起 server → 探活 → 跑 GuideLLM → 收 JSON → 关 server。
    `dry_run=True`（默认）只打印/返回计划好的命令列表，不执行任何东西——
    **在云端先用这个模式跑一遍，肉眼核对命令再 `--execute`**。
  - `normalize_guidellm_result(raw, concurrency)` —— 从 GuideLLM 输出 JSON
    里挖 `mean_output_tokens_per_sec` / `ttft_p99_ms` / `tpot_p99_ms` /
    `mean_acceptance_rate`，容错两种可能的 schema 形状。**这是占位映射，不
    是验证过的 schema**——GuideLLM 真实输出长什么样从没在本机见过，见 §3。
  - `compute_speedup` / `to_demo_arms` / `write_results_json` / `write_demo_js`
    —— 算 speedup、写 `results/*.json`、写 `docs/site/cloud_bench.js`
    （`window.SPECTER_CLOUD_BENCH = {...}`，`status:"ready"`）。
  - CLI：`python -m cloud_bench.orchestrate --dry-run` /
    `--execute --demo-js docs/site/cloud_bench.js`。
- `src/cloud_bench/sanity_check.py` —— `~95% 复现自检`门禁：
  `python -m cloud_bench.sanity_check --eagle3-tok-per-s X --baseline-tok-per-s Y`
  ，把测到的 c=1 speedup 与论文 EAGLE3-8B 的 1.25–1.32× 区间比，低于
  `1.25 * 0.95` 就 FAIL 退出码 1，提示先别跑完整矩阵。
- `src/cloud_bench/remote_setup.sh` —— 云端环境搭建脚本（装 vllm/guidellm、
  提示先看 `guidellm --help` 确认 CLI 语法、`huggingface-cli login`
  提示、预取三个模型、结尾提醒销毁实例）。**`VLLM_USE_V1` 已经从脚本里删掉**——
  这个环境变量在 vLLM 2025-11-12 起已被移除（V1 是唯一引擎，没有 V0 可切换了），
  设了也是空操作，不影响正确性，只是删掉避免误导。
- `tests/test_cloud_bench.py` —— 24 个 hermetic 测试（mock subprocess/urlopen/
  clock），覆盖命令拼接、健康检查超时、结果归一化、speedup 数学、运行时预算
  护栏、JSON/JS 写出。`pytest -q` 全套 221→**245**。
- `docs/site/cloud_bench.js` + `docs/site/index.html` 的 "Cloud validation"
  卡片（新 `#cloud` section）——见 §5，**当前 `status:"pending"`，页面显示
  一段说明文字 + 锁定 config，不显示任何编造数字**。等真实结果出来，把
  `orchestrate.py --demo-js docs/site/cloud_bench.js` 的输出复制过来替换
  这个文件（或者直接跑那条命令生成到位），`status` 会自动变成 `"ready"`，
  页面会自动画出并发-speedup 折线图（本地已经用假数据在浏览器里验证过渲染，
  见 git 提交里的截图验证过程，无需重新验证渲染代码本身）。

## 3. 关键：不要盲信这份代码里的 GuideLLM CLI 语法

写这份计划时（2026-08-29）本机没有装 vllm/guidellm，`guidellm_cmd()` 里的语法
是从公开文档拼出来的最佳猜测（2026-08 web 搜索结果显示 GuideLLM **有两代 CLI
语法**同时在流传）：

- 旧一代（flag 式）：`guidellm benchmark --target <url> --rate-type sweep|
  synchronous|concurrent --rate <n> --data "prompt_tokens=...,output_tokens=..."`
- 新一代（registry 式）：`guidellm run --backend kind=openai_http,target=<url>
  --data kind=synthetic_text,prompt_tokens=...,output_tokens=... --constraint
  kind=max_duration,seconds=60`

本仓库代码用的是旧一代语法，且 `--data` 里塞了 `dataset=gsm8k,temperature=0,
top_p=1.0,seed=42` —— **这几个 sampling 参数是否真的被 GuideLLM 的 `--data`
接受、GSM8K 数据集加载的确切写法是什么，都没有验证过**。

**在跑任何真实请求之前**：
```bash
guidellm --help
guidellm benchmark --help
```
对着真实 `--help` 输出核对 `src/cloud_bench/orchestrate.py` 里
`guidellm_cmd()` 的每个 flag；如果版本是新一代语法，改这一个函数（其余代码
——健康检查、结果归一化、speedup 计算、demo JS 写出——都不依赖具体 CLI 语法，
不用动）。改完对着 `tests/test_cloud_bench.py` 里
`test_guidellm_cmd_uses_requested_concurrency_and_locked_config` 更新断言。

同理，`normalize_guidellm_result()` 里的字段路径（`metrics.
output_tokens_per_second.mean` 之类）是猜的两种可能形状——真正跑出第一个
GuideLLM 输出 JSON 后，`cat` 出来看一眼真实 key，照着改 `_dig()` 的路径列表。

## 4. 执行步骤

### 步骤 1 —— 租 GPU

**用户已确定用 Lambda Labs**（其余描述留作 Vast.ai/RunPod 的备选，Lambda 没
有 spot/可抢占档，不用挑"避开抢占"那一层，但 §1.5 第1条的"只有 terminate
没有 stop"仍然适用）。Lambda dashboard 选 A100（40GB 或 80GB 都够，8B fp16
weights ~16GB + KV cache + EAGLE3 draft head 留够余量）或 H100；库存紧张就
换 region 或换型号试（§1.5 第2条）。租之前确认：
- 本地已有 SSH 公钥并上传到 Lambda 账号（§1.5 第9条）；
- 两个 gated 模型在 huggingface.co 上都已经是"granted access"而不是
  "pending"（§1.5 第6条）——这一步比选机型更容易卡住,提前确认好。
记下 SSH 连接信息。

### 步骤 2 —— 环境搭建

把 `src/cloud_bench/remote_setup.sh` scp 到实例上（或者直接在实例上
`git clone` 这个仓库，脚本就在 `src/cloud_bench/remote_setup.sh`），执行：
```bash
bash remote_setup.sh
```
它会装 vllm + guidellm、提示做 §3 的 CLI 核对、提示 `huggingface-cli login`
（`meta-llama/Llama-3.1-8B-Instruct` 和 `Llama-3.2-1B-Instruct` 是 gated repo，
**必须在 HF 网页上显示"You have been granted access"而不是"pending"再租**——
pending 状态下载会 403，等于白付了搭环境那段时间的钱）、预取三个模型权重。

### 步骤 3 —— 核对 CLI 语法 + 干跑一遍命令计划

在实例上（clone 了本仓库后）：
```bash
cd Specter
python3 -m venv .venv-cloud && source .venv-cloud/bin/activate
pip install -e .  # 或者直接把 src/ 加进 PYTHONPATH，看仓库有没有 setup.py
PYTHONPATH=src python -m cloud_bench.orchestrate --dry-run
```
肉眼核对打印出来的每条 `vllm serve ...` / `guidellm benchmark ...` 命令，
对照 §3 核对完的真实 CLI flag 手动改 `src/cloud_bench/orchestrate.py` 里的
`guidellm_cmd()`（如果需要）。

### 步骤 4 —— ~95% 论文复现自检（花钱之前的门禁）

只起 `eagle3` 臂和 `baseline` 臂，只测并发=1，跑一小段拿到两个
`mean_output_tokens_per_sec`（可以手动跑一次 `vllm serve` + 单条 GuideLLM
调用，或者临时改 `orchestrate.py` 调用只测这两臂 c=1 ——两种都行，这一步允许
手动）。然后：
```bash
PYTHONPATH=src python -m cloud_bench.sanity_check \
    --eagle3-tok-per-s <测到的数> --baseline-tok-per-s <测到的数>
```
**FAIL 就停**，回去检查 `--speculative-config` 里的 `num_speculative_tokens`、
draft/target 是不是同族、tokenizer 对不对齐、`--max-model-len` 是不是被起
太大导致 KV cache 挤占异常——不要在配错的情况下跑完整矩阵。PASS 才进入步骤 5。

### 步骤 5 —— 跑完整矩阵

**先用 `tmux`（或 `screen`）起一个会话再跑**，不要在裸 SSH 前台跑——笔记本
合盖/wifi 掉线会给前台进程发 SIGHUP 直接杀掉，`tmux` 里的进程不受影响，
重新 SSH 进去 `tmux attach` 就能接着看：
```bash
tmux new -s bullet2
PYTHONPATH=src python -m cloud_bench.orchestrate \
    --execute \
    --results-json results/bullet2_vllm_eagle3.json \
    --demo-js docs/site/cloud_bench.js \
    --max-runtime-min 150
```
四个臂（`eagle3`/`ngram`/`draft_model`/`baseline`）× 五档并发
（1/4/16/32/64），全量 GSM8K（1319 题）。这一步是花钱的主体，中途盯着
`nvidia-smi`/日志，出现卡死或报错立刻 Ctrl-C 并检查——不要让一个挂死的
`guidellm benchmark` 空转吃钟点费。

**中途真断了怎么办**：`orchestrate.py` 现在是可续跑的——每个
(臂, 并发) 组合跑完立刻把 GuideLLM 原始输出写到
`results/cloud_bench_raw/guidellm_{arm}_c{c}.json`，聚合结果每跑完一个点
也会 checkpoint 到 `--results-json` 那个文件。原样重新执行上面同一条命令：
已经完成的点会被直接读缓存跳过（整臂都跑完的话连 `vllm serve` 都不会重启），
只从断点那个点继续跑，不会重复花钱。如果某个点的原始文件是半写坏的（进程
正写到一半被杀），脚本会自动判定为无效重新跑那一个点，不需要手动删文件。

### 步骤 6 —— 把结果拉回本地

```bash
scp <instance>:~/Specter/results/bullet2_vllm_eagle3.json results/
scp <instance>:~/Specter/docs/site/cloud_bench.js docs/site/
```

### 步骤 7 —— 立刻销毁云实例

回 Lambda dashboard 点 **Terminate**（Lambda 没有"stop"这个状态，只有
terminate——§1.5 第1条）。**确认步骤6的 scp 已经成功、本地能打开那两个文件
再点**，terminate 之后实例本地盘的东西没有任何恢复手段。这一步没有任何
自动化，必须手动确认——本计划里所有的时间/命令护栏都不能替代这一步。

### 步骤 8 —— 回本地写反馈

- `results/bullet2_vllm_eagle3.json` 已经是 `orchestrate.write_results_json`
  写出的格式（`{"plan": [...], "results": {arm: {"concurrency": {c: {...}}}}}`）。
- `docs/site/cloud_bench.js` 直接覆盖仓库里的占位版本——`status` 会是
  `"ready"`，demo 页的 `#cloud` section 会自动画图（渲染代码已经用假数据
  测过，见本次提交历史，不用重新验证 JS）。**如果任何一个臂在某个并发点上
  数字看着不对（比如 speedup < 0.5 或 > 3，或者某档并发完全没数据），先怀疑
  测量，不要直接把可疑数字塞进 demo 或 README**——遵循这个项目一贯的
  "诚实汇报，包括 null result"风格（参考 `docs/pitfalls.md` 坑1–26）。
- 更新 `README.md`：加第 3 个 headline 数字（`vLLM-A100 EAGLE3 <实测>×`），
  更新测试计数（本次改动后是 245，云端跑完如果没加新测试保持 245）。
- 写 `docs/engineering-notes/09-mac-vs-a100.md`（或按序号规则命名，跟着
  README 现有 8 篇的编号走）——Mac-local 从头实现 vs vLLM/A100 生产实现的
  对照，分离"根本限制"（batch=1 dead zone、α 驱动的 break-even）vs
  "硬件限制"（Apple silicon 没有 tensor core / 没有大规模并行 batch）。
  一张结果表两栏并排，标清硬件，参考 `notes/简历定稿计划-Specter_2026-08-28.md`
  第 160–161 行的要求。
- `TASKS.md` 第 112 行 Bullet 2 打勾（`[ ]`→`[x]`），第 113 行 Note #7 打勾，
  第 115 行 README 那条待办去掉"待办"部分。
- `docs/pitfalls.md` 追加坑（哪怕一个都没撞上也值得记一句"这次没撞上传闻中
  的哪几个坑"，因为§7 那张坑表本身就是这个 bullet 的价值所在，见
  `notes/简历定稿计划-Specter_2026-08-28.md` 第 67 行）。
- 更新本会话记忆（`project_specter_direction_b_deployment.md` /
  `MEMORY.md`）——Bullet 2 完成、真实数字是多少、坑表补了什么。

## 5. Demo 页集成设计（已实现，仅供理解，不需要重做）

`docs/site/index.html` 新增 `#cloud` section（在 `#numbers` 和 `#notes`
之间），JS 里的 `renderCloud()` 读 `window.SPECTER_CLOUD_BENCH`
（来自 `<script src="cloud_bench.js">`）：

- `status !== "ready"` 时渲染一段说明文字 + 锁定 config 列表（当前状态，
  没有编造任何数字）。
- `status === "ready"` 时，用 `arms: [{name, points:[{concurrency, speedup,
  ttft_p99_ms, tpot_p99_ms}]}]` 画一个 canvas 折线图（并发 x 轴，speedup y
  轴，每个臂一条线，`1.0×` 有一条参考虚线），下面配色图例。

`orchestrate.write_demo_js()` 直接产出这个 `"ready"` 形状，所以步骤 8 只要
把它生成的文件复制到 `docs/site/cloud_bench.js` 就完成集成，不需要手写 JSON。

之所以没有做成"round-by-round 回放"（像 `#lab` section 那样）：并发扫描的
自然形式是"一个 x 轴是并发、y 轴是 speedup/延迟的静态图表"，不是逐轮时间线，
两种数据形状不一样，强行塞进 `SPECTER_RUNS` 的回放机制只会更别扭。

## 6. 坑表（跑之前先读——这张表本身就是这个 bullet 的价值）

摘自 `notes/简历定稿计划-Specter_2026-08-28.md` 第 67–80 行，跑的时候留意会
不会撞上，撞上了要如实记录（不是所有坑都会复现，"没撞上"也是数据）：

1. **并发一高加速塌** —— P-EAGLE 论文：1.55×@c1 → 1.09×@c32 → 1.05×@c64；
   EAGLE 3.1：2.03×@c1 → 1.66×@c16。只测低并发会夸大收益——这正是本计划要
   扫 5 档并发的原因。
2. **真实 α ≈ 0.6–0.8，不是理论 0.95** —— 主因是通用数据集 + 现成 draft
   head 的 task-domain mismatch。GSM8K 已经是相对匹配的选择。
3. **跨框架 acceptance 指标定义不一致** —— vLLM issue #42508：vLLM 和
   SpecForge 同 config 报不同 acceptance，连方向都不一致。本计划只跑 vLLM
   一个框架，不需要跨框架对比，但如果引用别处数字做对照要小心这个坑。
4. **draft-length 饱和** —— acceptance length 和速度过了某个
   `num_speculative_tokens` 就饱和；本计划锁定 3，不扫这个维度（省预算），
   如果想额外扫需要加钱加时间，不在当前 ~$15 预算内。
5. **batch-size 依赖的最优点** —— 某 A100 研究里 EAGLE-3 在 batch 56 达峰；
   KV-cache 重的方法 batch 大了会 memory-bound。本计划的并发上限是 64，
   留意 c=64 那档是不是已经在 memory-bound 区域。
6. **固定生成长度** —— 已锁 1024（`config.OUTPUT_LEN`），避免变长输出污染
   acceptance-length 均值。
7. **vLLM 不支持 tree decoding，只支持 linear/chain** —— 写进最终 note 里
   说明这一点，不要暗示测的是 tree-based 投机解码。
8. **EAGLE3 需要每个 target 单独训的 draft head** —— 只有主流模型有现成的，
   `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` 是 Llama-3.1-8B 专用的，换目标模型
   需要换 draft head 或者退化到 ngram/draft-model 臂。

## 7. 文件清单（这次改动新增/修改的文件）

- `src/cloud_bench/__init__.py`、`config.py`、`orchestrate.py`、
  `sanity_check.py`、`remote_setup.sh` —— 新增。
- `tests/test_cloud_bench.py` —— 新增，24 个 hermetic 测试。
- `docs/site/cloud_bench.js` —— 新增，`status:"pending"` 占位数据。
- `docs/site/index.html` —— 新增 `#cloud` section + CSS + `renderCloud()`。
- 本文件 —— 新增。
