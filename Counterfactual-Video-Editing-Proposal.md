# Counterfactual Video Editing — Proposal (beating Bernini on causal/counterfactual edits)

> **⚠️ Superseded for the current build.** The architecture/novelty claims here
> (5-block model, MLLM source-inversion abduction, VACE/Wan gated renderer) are
> replaced for v0 by [[E2W-v0-Remove-Only-Spec]] — see
> [ADR-0007](e2w/docs/adr/0007-e2w-v0-remove-only-void-renderer.md). This file
> stays as the historical long-run thesis (open-domain edits, P1–P3); it is not
> what v0 builds. Read `E2W-v0-Remove-Only-Spec.md` first if you're building or
> reviewing the current system.

> 目标:在「反事实编辑」任务上超越 Bernini。统一 Pearlian 最小改动 + 物理后果级联两种语义。
> 关联笔记:[[Bernini]] · 表示选择见 [[Bernini]] 的 "The Representations"。

![[Pasted image 20260618235545.png]]
## 0. 一句话定位
反事实编辑的本质是处理**用户没说的两件事**:(1) 干预的**下游因果后果**(因果连带),(2) 必须保持的**不变量**(身份/背景/无关物体与运动)。现有视频编辑系统(Bernini、VEGGIE)只优化前者一部分,**完全不建模不变量,也不做 abduction**。

## 1. 文献地图(哪些已被占,别重做)
- **CSVC / Causally Steered Diffusion** (arXiv:2506.14404):LLM 按预设 causal graph 改写 prompt 再渲染。= prompt 层因果,无 abduction、无不变量保证、非源忠实反事实。
- **Chain of Event-Centric Causal Thought** (arXiv:2603.09094, 2026.3):因果有序事件链做**物理可信生成**。纯生成、非源编辑、无不变量。
- **Counterfactual World Models via Digital Twin (CWMDT)** (arXiv:2511.17481):数字孪生条件 + 多跳 L1/L2/L3 反事实推理评测(CLIP-Text/CLIP-F/GroundingDINO/LLM-judge)。偏 world-model/推理评测。
- **Post-Training Newton's Laws w/ Verifiable Rewards** (arXiv:2512.00425):物理可验证奖励 RL(生成)。
- **YoCausal** (arXiv:2605.30346, 2026.5):VoE 范式的视频扩散因果认知 benchmark(理解/生成)。
- **图像域 Pearlian 反事实**:Diffusion Counterfactual w/ Semantic Abduction (2506.07883)、Causal-Adapter (2509.24798)、BD-CLS (NeurIPS 2025)。仅图像、小已知属性 SCM。
- **Bernini** (arXiv:2605.22344):反事实当 self-text/self-vision-text CoT(prompt 改写 + 硬首帧中间态),最好成绩依赖外部 GPT-5.4 prompt enhancer。指标通用,不测因果正确性/过度编辑。
- **VEGGIE** (arXiv:2503.14350):learnable query tokens + 纯 diffusion loss 端到端;reasoning 隐式;无因果结构。

**白空间**:从**真实源视频**出发、做**开放域**反事实编辑,显式 (a) abduction 推断潜在场景状态,(b) 渲染因果连带,(c) 证明性保持不变量。无人做。

## 2. 统一框架:Abduction(溯因)→ Intervention(干预)→ 因果连带(Closure)→ Render(渲染)
统一视角:两种语义是同一条 do-calculus,差别只在因果连带(Closure)的深度/结构。
1. **Abduction**:源视频 → 结构化场景状态 S(实体/属性/关系/动力学) + 外生残差(保持不变项)。给出不变量:非后代变量从 abduction 重建并钉死。
   - 工程近似:source inversion 到 latent + 结构化 scene graph/caption,latent 当作"外生"重建先验。
2. **Intervention**:指令 → do(X=x)。属性 → Pearlian 分支;力/事件 → 物理分支。同一接口两种 payload。
3. **因果连带(Causal Closure)**:在 S 上传播干预 → 受影响变量集。属性=浅 DAG;物理=时序 rollout。**同一机制,不同深度 = 统一点。**
4. **Render**:condition = (abduction 钉死的非后代) + (更新后的后代)。非后代 pinned → 最小改动;后代重渲 → 因果连带。

### 接口与梯度(承接 planner+renderer 讨论)
- 沿用 **VEGGIE 式连续接口**(可学习 scene-state tokens)+ 端到端 diffusion loss,核心**不用 RL**(避开 Bernini self-vision-text 的硬中间态 / exposure bias)。
- Novelty 损失:
  - **(a) 不变量保持损失**:编辑后视频重编码,非后代 latent 必须 match abduction 的(counterfactual consistency)。← Bernini 没有。
  - **(b) 后果覆盖损失**:由 closure 监督受影响变量被正确渲染。
- RL 仅作可选末层:用人类偏好/可微 reward model 做对齐。

### 2.6 Model 具体架构(五块,核心=mask 门控源条件)
1. **Abduction 源反演**:MLLM(Qwen2.5-VL 级)读 V+I;V **inversion 进 renderer latent**=工程化 U(可从源重建的一切)→ 不变量免费拿到。← Bernini/VEGGIE/VOID 都无,是单元级真 Rung-3 的根。
2. **因果 Planner**:解析 do(X=x),输出**两类不同向量**(勿混):
   - (a) **区域 query 向量**(少量)→ 不是 mask 本身,而是"会被波及那块"的概念向量。
   - (b) **edit-plan tokens**(连续)→ 描述被改区域的目标长相,给渲染器。可微、端到端、不用 RL。
   - **逐像素 mask 不是 VLM 直接吐的**(token 稀疏、不对齐像素)。需 LISA/SAM 式 **mask 解码器**:把 query 向量 × 视觉骨干的**逐帧稠密像素特征图**做匹配(点积/注意力)→ 逐像素分数;逐帧抽特征 + 时序传播(SAM2 式)→ 三层时空 mask。
3. **门控 Renderer(强 DiT,Wan2.2 级)**:条件=(a)源 VAE latent(不变量/细节)+(b)edit-plan tokens(后代)+(c)三层 mask 作时空门。**核心机制 mask 门控(inpainting 式)**:每步去噪在`不改变`区把源 latent **贴回**(钉死,非靠自觉),`直接/间接`区在 tokens 下自由去噪 → 架构层强制最小改动。
4. **mask 来源(分阶段,关键)**:
   - **训练**:mask **精确**=仿真 factual/counterfactual(共享种子)+ 物体级因果日志。双用途:(1)门控渲染 (2)当 GT 监督 planner 学画 mask。
   - **推理(真实视频)**:无现成 mask,**靠 planner 预测**(第 2 块那套)再门控。⇒ **推理 mask 是预测的、会出错 = 本设计最大风险**:漏预测→该变的被锁(漏后果);过预测→不该解锁的解锁(过编)。`直接`区易(分割/可让用户点选,VOID 式);**`间接`区难、靠因果推理 = 真正创新点与命门,模型上限基本被它卡住。**
5. **训练损失**:主=对 sim GT 反事实视频 flow-matching;(a)**不变量保持损失**(`不改变`区生成 latent match 源 latent,sim 共享-U 监督最干净);(b)**因果 mask 损失**(预测 mask 对齐 sim 依赖图)。RL 可选末层。
6. **推理流程**:源→abduction→planner(do+预测三层 mask+tokens)→门控 Renderer 生成(接缝靠 feather + 联合去噪,不用 2nd-pass)。
- **为何赢 Bernini**:门控源条件=结构性保不变量(治过编);预测的因果连带 mask=处理没说的后果(治漏);abduction=源条件化(过 Rung-3 换源测试,Bernini prompt 主导会挂)。
- **真风险**:开放域 abduction 只能近似(inversion);**推理时三层 mask 尤其`间接`(多跳)预测是命门**;mask 门控区域接缝 artifact(feather/联合去噪)。

### 2.7 训练流程(三件套数据 + 三个误差 + 三阶段)
**一条训练数据 = 仿真免费给的三件套**(以"把桌沿杯子推下桌"为例):
- 源视频 V(不干预渲染)· 标准答案 V\*(同一随机种子下施加干预渲出)· 标准标签图 M(仿真日志逐帧给出:杯子/碎片/影子=该变,笔/桌/窗=不许动)。

**前向**:① V 编码成底稿 latent → ② MLLM 读 V+I 出三层标签图(经 mask 解码器)+ 目标长相向量 → ③ 渲染器分区生成(不许动区贴回底稿,其它区按向量画)= 预测视频 V̂。

**三个误差(相加反传,更新 MLLM + mask 解码器 + 渲染器)**:
1. **主误差**:V̂ 逼近 V\*(扩散/flow-matching)。
2. **标签误差**:预测标签图对齐 M(学对波及范围,尤其难的`间接`区)。
3. **不变量误差**:`不许动`区 V̂ 必须等于 V(直接罚乱改)。

**三阶段(不能一锅端,同时全训会崩——VEGGIE 踩过)**:
- **起点**:不从零。MLLM 用现成 Qwen2.5-VL 级,渲染器用现成 Wan2.2 级,VAE 编码器冻住;主要训中间对齐 + mask 解码 + 不变量约束等新部件。
- **① 对齐**:大量简单图像/单帧编辑例,冻 MLLM,只调对齐层 + 渲染器 → 让渲染器**听懂**标签图与向量。
- **② 端到端**:仿真反事实视频,放开 MLLM 轻量微调 → 学会预测对的标签图与后果、渲染器学时序连贯。
- **③(可选)对齐**:人类偏好轻调。

### 2.8 【暂不做 / 后续】环训练(add↔remove 自监督)
想法:add 与 remove 互为逆操作 → cycle consistency,绕一圈回原视频,**无需 GT,任何真实视频可训** → 省仿真数据、强化最小改动、校验 mask(add 区=remove 区)。

**搜后结论(方向不变,定位下调):**
- **不新**:Ouroboros(2508.14461,cycle 拉无标注真实数据进 add/remove)、Paint-by-Inpaint(2404.18212,"先删再学加"造对)、cycle 反事实生成保结构(2509.24267)已做(均图像级、无下游交互)。窄白空间 = **视频 + 因果交互 + 三层 mask** 下的环。
- **理论上会断**:Fork or Fail(PMLR)证 cycle 在**多对一/非双射**映射失败 = 我们的"交互/不可逆后果"处;随机 mask 自监督会教成"重建物体"而非移除;copy-paste 伪对有物理不一致(影子/反光)。
- **定位**:永远辅助,只喂惰性/可逆例;主力仍是仿真 GT。真做成贡献的唯一卖点 = "非可逆交互段如何拿自监督"(硬)。

**以后可直接复用的配方**:Paint-by-Inpaint 删→造加对(add 方向廉价真实数据)· SVOR/From-Ideal-to-Real(2603.09283)两阶段(无标注真实自监督预训练 + 合成对精调)· ROSE(2508.18633)side-effects = 删除任务的"间接层",推广到通用干预。

## 2.5 Pearl 地基与可识别性(设计基石)
- 反事实 = **阶梯第 3 层(Rung 3)**,严格区别于 Rung 2 干预:Rung 2 是群体级 P(Y|do(x));Rung 3 是**单元级**——"对**这一段**视频,如果当时…"。
- 形式 = **abduction → action → prediction**:用事实反推外生噪声 U 的后验 → do(X=x) → **保持同一 U** 重算。"同一 U"就是"其余一切不变"的严格定义。
- **确定性假设**:随机性全装进 U(匹配仿真:给定种子即确定)。
- **可识别性硬约束**(因果阶梯定理):Rung 3 一般无法由 Rung 1/2 唯一确定,除非加参数约束(单调性 / 双射生成机制)。⇒ **很多反事实的"正确答案"本身不唯一** → benchmark 必须区分可识别/欠定。

## 3. 数据引擎 + 真值来源:sim vs t2v
难点:真实数据无 ground-truth「哪些该变/哪些不变」。**真值来源是不可识别性下的核心抉择。**

**结论:benchmark 真值必须用 sim;t2v 生成对从 Pearl 看根本不是反事实,只能做训练增强。**
- **sim(唯一真·结构反事实)**:仿真器本身=SCM,U=随机种子+初始条件,**完全已知** → abduction **精确**。固定种子 → do(改变量) → 重跑;factual/counterfactual **共享同一 U**,非后代逐比特一致。精确 GT + 精确不变量标签。覆盖物理 + 可识别 Pearlian。
- **image→(t2v)→counter(不合法当真值)**:两 prompt 两条轨迹**不共享 U**,非后代全程漂移 = 没控噪声的 Rung 2 采样,**不是 Rung 3**。且真值=另一生成模型脑补 → 污染 + 把模型假设烤进 benchmark(违反普适性)。**仅用于训练数据增强。**
- **真实 before/after 自然实验片段**:做"真实切片"量域差/外部效度;指标放宽(只评不变量 + VLM 判后果)。
- 风险:sim-to-real 域差;sim 只覆盖其引擎建模的物理(语义/社会/生物类反事实欠覆盖,scope 里写明)。

### 因果分层 mask:三层(直接/间接/不改变),VOID quadmask 的通用化
参考 **VOID**(Video Object and Interaction Deletion,Netflix+INSAIT,arXiv:2604.02296,2026.4,**开源**)。VOID 做**反事实物体删除**,逐像素 **quadmask 4 值**:`0`=Remove(被删主物体) `63`=Overlap(主物体区∩受影响区) `127`=Causally Affected(会动/掉/轨迹变) `255`=Preserve。训练数据 = Kubric(sim)+ HUMOTO(人体)配对反事实三元组。→ **强验证了 sim 配对 + 因果区 mask + 扩散这条路;但它只做删除(remove 区是删除特化),是我们通用框架的子集。**

**我们的三层 = 按因果图距离切(比 quadmask 更普适,涵盖增/改/施力)**:
- **直接**:干预节点 + 一阶后代(do 作用点 + 1-hop)。VOID 的 remove ⊂ 此。
- **间接**:多跳传递后代(因果连带更远处)。= VOID affected(127)。**全局效应(光照/阴影/反射)归此,合法。**
- **不改变**:非后代 / 不变量。= VOID preserve(255)。

落实细则:
1. **指标严谨性锚在「不改变 vs 改变」二分**(非后代/后代,最鲁棒可识别);直接/间接边界连续,仅用于**加权与分析**(直接更可识别,间接误差累积、常欠定 → 记分更宽松)。
2. **overlap 规则**(学 VOID):标签在**物体/属性级**给,mask 是投影;像素冲突时优先级 直接 > 间接 > 不改变。
3. **时空(逐帧)mask,非静态**:间接区**晚激活**(箱子掉来那刻),天然编码"先因后果"时序。
4. **不变量在身份/物体级检验**(re-ID/属性分类器,非像素相等)→ 允许"不改变"物体的阴影像素随合法全局效应变,但几何/身份不可变。**这同时解决 diff-mask 的串味问题。**

> diff mask 仅作快速近似 + 一致性自检;真值标签由 sim 变量级依赖图给。

### 训练 / 测试 源分离(防泄漏 + 对手公平)
- **不能同源**:同引擎/资产/脚本 → 模型学 sim 专属捷径(假泛化);且我们用 sim 训、Bernini 没用,则同源 sim benchmark 既谄媚我们又坑对手(违反 §4.1)。
- **训练**:sim 引擎 A(Kubric 式)精确因果标签(稠密监督 mask/不变量损失)+ t2v 增强 + 真实数据。
- **Benchmark(报告用,均与训练源不相交)**:
  - **主集:人工标注真实视频**(无 GT 视频,详见 §4.3)——泛化 + 公平擂台,谁都没训过,比 Bernini 公平。
  - **抽查:小份异引擎仿真全 GT**——换引擎/资产/参数/种子,按场景身份切;仅校验真实集清单指标的可靠性,不作主指标。
- 训练引擎 A 的 sim 仅 dev/val,不进报告。

## 4. 评测(公平擂台,Bernini 没在上面优化)
每 case 标注:干预 + 必须出现的后果集合 E + 必须不变的不变量集合 I。
- **后果召回率**(VLM/detector 逐项检查 E 是否出现)— 欠定时改为可接受集合/分布。
- **不变量保持率**(物体/身份级一致性,检查 I 未被动)— **指标重心放这里:不变量比后果在定义上鲁棒得多。**
- **物理可信度**(物理分支)。
- **报二维分**:同时暴露 under-editing(漏后果)与 over-editing(乱改),Bernini 的通用指标做不到。

### 4.1 普适性原则(benchmark 不得偏向我们的模型)
1. **只用输出级黑盒指标**:评最终视频,不评推理/表示;不要求暴露因果图。扩散直出 / LLM-prompt / world-model / 我们的 SCM 模型同一把尺子。
2. **真值来自模型无关源**(sim 真 SCM 或自然),**永不来自生成模型**。
3. **记分不预设我们的因果图**:只对 GT 的 E/I 打分。
4. **分可识别 / 欠定**:欠定后果只评集合或只评不变量。
5. **基线覆盖多家族**并做 sanity check:复制源 → 不变量高、后果≈0;自由重生成 → 后果高、不变量崩。指标须把两者摆对位置。

### 4.2 让 benchmark "真的是 Rung 3" 的判据(模型无关,核心)
> **同一指令 + 两段不同源视频 → 正确输出必须不同**(因 abducted U 不同)。
- 若正确输出只由 prompt 决定、与源无关 → 只是 Rung 2 编辑,不是反事实,剔除。
- 这条同时定义了打 Bernini 的点:Bernini 把反事实当 prompt 改写(prompt 主导),会在"换源应换答案"上露馅。

### 4.3 Benchmark 具体规格(暂名 CF-VEdit,P0 可直接构建)
- **任务**:V+I → V̂,单元级 Rung-3(正确输出取决于 V 具体状态)。
- **关键决定:评测不需要整段 GT 视频。** 两把尺子只需**标注**:不变量集 I(和源比)+ 后果清单 E(VLM/检测器逐条核)。GT 像素对主要是**训练**用(稠密监督 mask/不变量损失),评测用不上。
- **主 benchmark = 标注法(真实视频,无 GT 视频)**:省工、真实无域差、对所有模型公平,仍严测两轴。
  - 人工标:I(哪些物体不许动)+ E(必现后果清单)+(可选)粗略受影响区;可识别性 flag。
  - 指标靠 VLM/检测器核 E + 输出 vs 源在 I 上比对。多标注者 + 抽样校验保可靠。
- **高精度抽查 = 小份仿真全 GT**:验证标注法的清单指标靠不靠谱;**不作主指标**。
- **避坑**:完全不标注、纯大模型当裁判(Bernini 式 MLLM-judge)→ 烤进裁判偏见、不可复现,**不用作主指标**。
- **一致性**:训练用仿真 GT,评测用人工标注真实视频——"真值不来自生成模型"这条仍成立(人标真实视频本就是模型无关源)。
- **干预分类**(覆盖 Pearlian + 物理),每例标签〔类型/可识别性/推理深度〕:删除(VOID 式)· 增加/插入 · 属性改变 · 物理力/事件(时序级联)。
- **指标(输出级黑盒、模型无关)**:
  1. **IP 不变量保持**(主、严谨):`不改变`集物体/身份级一致性(re-ID+属性分类+掩码重建误差)→ 罚过编。
  2. **CR 后果召回**:E 被正确渲染比例(VLM/detector)→ 罚漏后果;欠定例放宽为可接受集合。
  3. **PP 物理可信**(物理支)。
  4. **二维报告 IP×CR**(摊开过编/欠编;排名可附调和均但必展二维)。
  5. 次要:视频质量(FVD 类)+ 指令遵循 sanity。
- **Rung-3 闸门**:配对例——同 I、两源 V1/V2(abducted 态不同)→ GT 不同;源盲模型挂;报 source-sensitivity。普适性:只看输出、真值不来自生成模型。
- **基线**:Bernini · VEGGIE · VOID(删除子集)· InstructPix2Pix-video · CSVC 式 prompt-rewrite+edit · copy-source(下界)· free-regenerate(上界)。sanity:IP/CR 须把这俩摆对位置。
- **规模**:主集真实视频 ~300–500(人工标 I/E)+ 仿真高精度抽查 ~100–150,按类型×深度均衡。
- **构建步骤**:①真实视频 + 指令采集(按干预分类×推理深度均衡)→ ②人工标注 I(不许动)+ E(后果清单)+ 可识别性 flag + Rung-3 配对例,多标注者校验 → ③指标 harness(VLM judge + detector + re-ID + 源比对)→ ④跑基线 + sanity(copy-source/free-regenerate 摆对位)、验 Rung-3 配对有效 → ⑤(可选)小份仿真全 GT 抽查清单指标可靠性。
  - 训练数据另线:仿真引擎 A(Kubric 式,记录依赖图)共享种子渲 factual/counterfactual → 自动抽精确 mask+E/I 供模型监督(§3),**与评测真实集严格不相交**。

## 5. 分阶段落地
- **P0 — Benchmark + 指标**(风险最低、可独立发):构 counterfactual editing 评测集 + 二维指标;把 Bernini/VEGGIE/CSVC 跑上去,展示它们 over-edit / 漏后果。
- **P1 — 共享核心**:abduction + 不变量 pinning(连续接口 + 不变量保持损失),先在浅 DAG(属性干预)验证最小改动。
- **P2 — 因果连带深化**:接入时序 rollout,覆盖物理后果级联;接仿真数据引擎做监督。
- **P3 — 对齐**:可选 RL/可微 reward 末层做人类偏好对齐。

## 6. 与 Bernini 的差异化一句话
Bernini 把反事实当「把 prompt 写详细」;本框架把它当「abduction + do() + 闭包 + 不变量约束」,且在一个测因果正确性与过度编辑的二维指标上正面比——这是 Bernini 既没做方法、也没做评测的地方。

---
### Refs
- Bernini: arXiv:2605.22344 · VEGGIE: arXiv:2503.14350
- CSVC: arXiv:2506.14404 · Event-Centric Causal Thought: arXiv:2603.09094
- CWMDT: arXiv:2511.17481 · Newton-RL: arXiv:2512.00425 · YoCausal: arXiv:2605.30346
- Semantic Abduction: arXiv:2506.07883 · Causal-Adapter: arXiv:2509.24798 · BD-CLS: NeurIPS 2025
- **VOID**(反事实物体删除,quadmask + Kubric/HUMOTO sim 配对,开源):arXiv:2604.02296 · github.com/Netflix/void-model
- **Pearl 理论**:Book of Why(三层阶梯)· Structural Counterfactuals: A Brief Introduction (Pearl, Cognitive Science 2013) · Probabilities of Causation (Pearl, r260) · Counterfactual Identifiability of Bijective Causal Models: arXiv:2302.02228 · Potential Outcomes Perspective on Pearl's Hierarchy: arXiv:2601.20405
