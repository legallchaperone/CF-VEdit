# E2W v0(Remove-Only)—— 项目规格

> **地位**:本文件是当前 E2W 的**权威规格**,替代 [[Counterfactual-Video-Editing-Proposal]]、[[CF-VEdit-Architecture-and-Narrative (给人看的）]]、[[Sa2VA-Modification-Plan]] 三篇里与架构/训练相关的内容(abduction 源反演的 MLLM-latent 框架、Pearl 叙事包装、VACE/Wan renderer 选型)。三篇旧文档保留作历史记录,顶部已加指回本文件的说明。偏离决定见 [ADR-0007](e2w/docs/adr/0007-e2w-v0-remove-only-void-renderer.md)。
> **范围**:仅 remove 类编辑。**Renderer**:仅用 VOID 的 pass1。**目标**:AAAI 或同级别 venue,约一个月。

---

## 0. 核心主张

E2W 和 VOID 用的是**同一个 renderer 权重**(CogVideoX-Fun-V1.5-5b-InP,`void_pass1.safetensors`)、**同一套 mask 输入机制**,唯一的区别是 renderer 拿到的"编辑语义"从哪来:VOID 靠一段独立生成的文本 prompt,renderer 和上游的 mask 推理模块之间没有任何梯度联系。

E2W 的耦合是**不对称**的,需要精确地说,不能笼统带过:**edit_0..3 这条链路是真正端到端可微的**——projection 输出直接占据 renderer 的文本条件位置,这一路径全程连续,video loss 能一路传回 planner,这是相对 VOID 最核心、也是唯一站得住的"可微耦合"差异。**seg_dir/seg_ind 这条链路不是**——SAM2 的概率图要 threshold 成二值 mask、再用规则组合成 quadmask 才能喂给 renderer,threshold 和集合运算都不可导(见 1.3),video loss 传不到这条链路,seg 分支的训练信号完全来自 Stage 0 的监督 loss,和"renderer 最终生成得好不好"没有直接梯度关系。renderer 相同这件事,把"哪部分带来了提升"限定在一个变量上,这个变量精确地说是**edit token 这一条可微耦合链路**,不是笼统的"mask+edit 都耦合"——写论文/讲给别人听时,这个精度不能丢。

**创新性定位需要进一步收窄**:"query token 耦合 MLLM 和 diffusion renderer"这个机制本身,在 2025-2026 年已经是成熟设计空间,不是可以单独站住的贡献——MetaQuery(Pan et al., 2025)用可学习 query 从冻结 MLLM 里读取生成条件、只用 denoising loss 训练,InstructX(2025)专门做了 MLLM-diffusion 集成方式的系统对比,并给出和 E2W 几乎逐字重合的结论(learnable query 拼进 MLLM 序列、输出直接替换 DiT 里的文本 embedding,MLLM 上挂 LoRA、connector 用小 MLP,收敛最快),MetaCanvas 更是直接在 Wan2.2-5B 上改造输入层做同类耦合。真正立得住的贡献收窄成三点交集:(a) 物理后果感知的移除任务本身(ROSE/EffectErase 这类同期工作只处理光度学副作用,不处理这个),(b) renderer 权重和 mask 机制与 VOID 完全一致的受控对比,(c) seg/edit 双分支——一个 planner 同时驱动分割头和生成 renderer——这个具体组合。

---

## 1. 架构

### 1.1 Pipeline

```
原视频 + "remove X" 指令
        │
        ▼
┌────────────────────────────────────────────────────┐
│ SA2VA(Qwen2.5-VL 主干,LoRA)                          │
│ video token + text token +                          │
│ [seg_dir][seg_ind][edit_0][edit_1][edit_2][edit_3]  │
│  (6 个非词表、固定位置的 query token)                    │
└────────────────────────────────────────────────────┘
        │                                  │
   seg_dir / seg_ind                   edit_0..3
        │                                  │
        ▼                                  ▼
    projection                    projection(硬替换,默认)
        │                                  │
        ▼                                  │
┌─────────────────┐                        │
│ SAM2(冻结)        │                       │
│ → 两个二值 mask    │                      │
└─────────────────┘                        │
        │                                  │
        ▼                                  │
  确定性规则组合成 quadmask(不可导)              │
  (0=remove / 127=affected / 255=keep)
        │                                  │
        ▼                                  ▼
┌──────────────────────────────────────────────────┐
│ CogVideoX-Fun-V1.5-5b-InP(冻结,void_pass1 初始化,只用 pass1) │
│ mask+masked-video-latent 通道拼接 ⊕ edit token 直接占据 T5 文本位置(T5 不参与推理) │
└──────────────────────────────────────────────────┘
        │
        ▼
    编辑后视频(X 已移除)
```

### 1.2 Planner:SA2VA + 6 个 query token

6 个 slot 非词表、位置固定(紧跟 video/text token 之后:seg_dir → seg_ind → edit_0..3),不参与 next-token 预测,一次 forward 后直接按固定位置读出 hidden state。

**Attention 连通性**(自定义 4D mask):
- prompt/video 部分维持标准 causal。
- `edit_0..3` 互相双向可见——它们要顶替 renderer 里 T5 文本 token 的位置,而 T5 encoder 本身是双向的,内部理应对称,无先后依赖。
- `edit` 与 `seg_dir`/`seg_ind` 双向互相屏蔽——两组各自独立读取 prompt/video,不共享 attention 通道。mask 和 edit 之间已经有一条显式通道(SAM2 输出的 mask 参与 quadmask 构造,直接喂给 renderer),不需要再靠 attention 开一条隐式、把 mask loss 和 video loss 梯度耦合在同一 activation 上的通道。
- `seg_ind` 能看 `seg_dir`(原有顺序保留),反向不行——没有强证据支持修改,维持现状。

**RoPE**:`edit_0..3` 绑定到同一个共享 position id,消除双向 mask 生效后残留的"谁先谁后"位置偏置,让 4 个 slot 在旋转位置编码层面也真正对称。

已知限制:mask 构造假设 `batch_size=1`,尚不支持批训练。

### 1.3 Mask 分支

`seg_dir`/`seg_ind` 的 hidden state 过**同一个共享 projection**(复用 Sa2VA 自带的 `text_hidden_fcs`,Sa2VA-plan change A;不是两个独立投影),各自作为 SAM2(**冻结**)的 prompt embedding,靠两个 query token 各自的 hidden 来区分,输出两个二值 mask:direct(待移除物体)、indirect(受物理影响的区域)。

Renderer 需要的是 VOID 格式的单通道 quadmask,用确定性规则从两个二值 mask 组合出来:

| 区域 | 取值 |
|---|---|
| direct | 0(remove)|
| indirect \ direct | 127(affected)|
| 其余 | 255(keep)|

不重建 VOID 的 63(overlap)档:VOID 里 63 代表"物体和场景的物理接触面",而 `direct ∩ indirect` 这两个 mask 的交集并不真的对应这个概念——早先设想用交集近似 overlap,但那条规则和"direct → 0"是同一批像素冲突的两条规则(交集里的像素究竟算 0 还是 63,原来没写清楚),与其保留一个语义不对又有歧义的近似,不如直接砍掉,退化成三档。

**这一步(threshold + 集合运算)不可导**,是 video loss 传不回 seg_dir/seg_ind 的原因,见第 0 节。

这套映射是为了对齐 renderer 已经训练过的输入分布,不是 VOID 代码里写死的对应关系,Stage 0 训完后需要抽查重建出的 quadmask 类别占比是否合理(见第 5 节)。

Mask 进入 renderer 的方式直接复用 VOID 自己的处理逻辑:mask 和"被 mask 遮住的视频"各自过 VAE 编码,在通道维度和加噪声的 latent 拼接,一起送进 transformer——这条路径照抄 VOID 已验证的实现,不重新设计。

### 1.4 Edit 分支

CogVideoX 把 T5 文本 embedding 和视频 latent token 在序列维度拼接,一起过 full 3D self-attention(不是 cross-attention)。**默认路径是硬替换,不是混合**:`edit_0..3` 的 hidden state 过 projection 后,直接、完整地占据 renderer 文本条件应有的位置,T5 在推理时(以及 Stage 2 的 forward)完全不参与——T5 只在 Stage 1 用来离线生成回归目标,是训练阶段的"老师",不是推理时还要现场跑的组件。这样设计避免了一个之前没钉死的问题:硬替换意味着不存在"混合的另一半喂什么文本给 T5"这个悬而未决的问题,也不存在"α 最终收敛到多少"需要论证的问题——没有混合,自然没有这两个问题。InstructX 的对应机制描述用词就是"replace",不是"blend",这条路径本身是被验证过的默认选项,不是权宜之计。

**门控混合 `α · edit_embed + (1-α) · text_embed` 降级为 fallback**,只在硬替换训不稳(loss 早期发散或不降)时启用,启用时机和是否需要,视 Stage 2 早期的训练曲线决定,不预设一定会用上。原来担心的"训练早期 edit token 接近随机、硬替换会让冻结 renderer 拿到陌生分布"这个问题,现在由 Stage 1 的热身承担——Stage 1 的存在本身就是为了让硬替换从第一步开始就是安全的,α 是这个保障失效时的第二道保险,不是第一道。

### 1.5 Renderer

CogVideoX-Fun-V1.5-5b-InP,权重从 `void_pass1.safetensors` 初始化,**全程冻结**,只用 VOID 的 pass1(不引入 pass2 的变形修复)。

---

## 2. 训练:三阶段

前两阶段完全不需要编辑后的目标视频,只有 Stage 2 需要。三个训练阶段开始前,有一个不需要训练、今天就能跑的预实验,值得先做。

### Stage -1(预实验)— quadmask 分档敏感性,零训练

**数据**:复用跑 VOID baseline 时留下的 probe clip 和对应的四档 quadmask(`predictions/void/` 里已经有)。

**步骤**:对每条 quadmask 做两个修改版——版本 A 把所有 63(overlap)像素改成 0,版本 B 把 63 改成 127,其余不动。用完全相同的 `void_pass1` 权重、相同 prompt、相同 seed,原版/版本 A/版本 B 各跑一遍 renderer 推理,三组输出进 IP×CR harness 对比,顺便肉眼看几条。

**这个实验一次回答两个问题**,不是只服务于第 5 节的开放问题 2:
1. renderer 对"输入里从不出现 63"敏不敏感——如果 A、B 和原版基本没差,3.节 quadmask 三档简化是安全的;如果两个版本都明显更差,说明三档简化本身有问题,需要在 Stage 0 开始前重新设计 quadmask 构造,而不是等 Stage 2 出了烂结果再回头排查。
2. "重叠区域"这个概念行为上更接近 remove 还是 affected——如果 63→0(版本 A)明显优于 63→127(版本 B),说明 Stage 0 里"affected∪overlap → seg_ind GT"这条 GT 合并规则并的方向错了,重叠区域应该并进 direct 侧,需要在 Stage 0 训练开始前改掉,不是训完才发现。

成本是几十次 renderer 推理,不训练任何东西,一个下午能出结果,排在所有训练之前。

**结果(2026-07-02, n=6, 详见 `/data/cwx/void-runs/stage_m1/RESULTS.md`)**:
- **问题1(三档是否安全):安全。** 三组 preservation 全 1.000、edit_success 全 0.667,无崩溃;逐样本像素差(seed 固定)显示并入 remove(63→0)时输出≈四档原版(`orig-A` MAE 全部 ≤ judge 噪声底),只有并入 affected(63→127)才会在个别样本(0037,63 像素占比最大)破坏移除。→ 三档简化可进 Stage 0,**前提是 overlap 并入 remove 侧**。
- **问题2(overlap 归 remove 还是 affected):未定,暂不改默认。** 六个样本里只有 0037 一个有超噪声真信号(方向:overlap 当 affected 会害移除,偏 remove);其余样本的 judge 分差经像素差判定全落在单次 Gemini judge 的噪声内(orig 重跑 vs void 参考同内容却差 0.3 consequence,吞没 ≤0.11 的真实差,连 0053 反例也是噪声)。n=1 真信号不足以反转 GT 合并默认 → **保持并入 seg_ind 的默认**,记为 open question,Stage 0 训完后用确定性 mask IoU/Dice(而非单次 VLM judge)在更大 remove 集上复核再定。方法学教训:后续 VOID-vs-E2W 的 A/B 对照必须用确定性指标或多次 judge 平均。

### Stage 0 — seg_dir / seg_ind

| | |
|---|---|
| 数据 | (视频, "remove X" 指令) 对,视频不限 Kubric,可混入真实视频。每条跑 VOID 的 VLM-MASK-REASONER(SAM2+Gemini)得到 quadmask,拆出 direct 区域 → seg_dir GT;affected∪overlap → seg_ind GT(重叠区域并入哪一侧,以 Stage -1 的结果为准:Stage -1 n=6 未能定论,暂保持并入 seg_ind 默认,弱证据轻微倾向 direct,待 Stage 0 后用确定性 mask 指标在更大集上复核)。 |
| 冻结 | SAM2 全冻结。 |
| 训练 | SA2VA 主干(LoRA)+ seg projection。 |
| Loss | Dice + BCE。 |
| 备注 | forward 可以不拼 edit_0..3,不涉及 renderer。 |

### Stage 1 — edit_0..3 热身

| | |
|---|---|
| 数据 | 复用 Stage 0 数据,额外用 VLM 生成一句"移除后场景描述"文本,过 **CogVideoX-Fun-InP 自带的 T5**(不是 Wan 的 umT5,renderer 换了目标空间也要跟着换)离线算 mean-pooled 目标 embedding。 |
| 冻结 | renderer 不参与 forward。 |
| 训练 | edit token embedding + edit projection,SA2VA 主干继续同一个 LoRA。 |
| Loss | MSE / cosine,4 个 slot 回归到同一个目标向量。 |
| 备注 | 目的是把 edit token 从随机初始化挪到语义大致正确的区域,不追求精确。**风险与检测**:4 个 slot 回归同一个目标,除了初始 embedding 不同之外没有分化压力,存在收敛成彼此高度冗余的向量、浪费容量的风险。检测很便宜:Stage 1 训完后,直接算 4 个 edit token projection 两两 cosine similarity,几分钟出结果。数值明显偏高 → 换成按位置回归到描述文本真实 T5 token 序列(截断/补齐到 N)的目标,这样同时修掉 collapse 风险和"训练目标是 pooled、实际用法是按位置注入"这个 train/use 不一致;数值不高 → 维持现在这版更简单的设计,不做额外改动。 |

### Stage 2 — edit_0..3 精调

| | |
|---|---|
| 数据 | Kubric 生成的 remove-only 配对视频,参考 VOID 的生成方式,初始规模比照 VOID 自己的量级(约 1900 对)起步。 |
| 冻结 | renderer 全程冻结;SAM2 用 Stage 0 训好的权重(冻结)生成真实 mask。 |
| 训练 | edit token embedding + edit projection(默认硬替换,不训练 α;只有触发 fallback 时才需要训练/调度 α,见 1.4)。SA2VA 主干继续同一个共享 LoRA,不冻结——InstructX 做过的设计空间消融显示,learnable query + MLLM LoRA + 小 MLP connector 这个组合收敛最快、优于冻结 MLLM 配大 connector,现在的设计正是这个组合,不是没有依据的选择。 |
| Loss | renderer 的 flow-matching loss 训练 **edit token 这条链路**(video loss 到不了 seg_dir/seg_ind,见 1.3);继续搭配 Stage 0 的 mask loss 训练 **seg 分支**——这不是可选的保险,是 seg 分支在 Stage 2 期间唯一的直接监督来源,停掉它 seg 质量在 Stage 2 就没有任何东西继续维护。SA2VA 主干 LoRA 是两条分支共享的参数,edit 分支的梯度会更新这些共享权重,即便没有直接梯度边,持续的 mask loss 也需要在场,防止共享权重被 edit 分支单方面拉扯、间接侵蚀 seg 分支的表现。可选小权重锚定项 `‖edit_projected − stopgrad(T5_target)‖²`,防止 Stage 2 数据量小导致 edit token 过拟合、偏离 Stage 1 学到的语义方向。 |
| 增强 | 对 edit token 做条件随机丢弃(训练时按一定概率替换成学出来的"空"向量),换取推理时可用 classifier-free guidance 放大编辑信号。硬替换是默认路径,"无条件"就是单纯换成这个空向量,不存在"该丢 edit 还是丢 text 还是都丢"的歧义——这层歧义只在 α 混合真被启用时才会出现,到时候再处理。成本低、不需要新数据,**默认加入**。 |

**⚠️ 训练前置(否则无声失败)**:Stage 2 的 renderer 必须用 `model_full_load` + bf16 加载,**禁用** `model_cpu_offload_and_qfloat8`(它是 CogVideoX-Fun/`predict_v2v.py`/`VoidRendererConfig` 的默认)。float8 量化会**静默切断**反向传播——edit token 那条端到端可微链路(第 0 节的核心主张、这整个项目的立身之本)会在完全不报错的情况下断掉:loss 照常下降(seg/其他梯度还在),但 renderer 那段根本没反传,训出来的 edit token 学不到东西。与"mask 分支不可微"是同一类坑,已赶在训练开始前查明,另见 §5.8。

---

## 3. 评估计划

- 现有 CF-VEdit + IP×CR harness,VOID baseline 已跑通(`results/void/`)。因为 renderer 权重、mask 输入机制都和 VOID 一致,E2W 与 VOID 那一行的差异理论上只反映"文本 prompt 条件"vs"query token 条件"这一个变量——但严格说这不是唯一变量,mask 来源也变了(VOID 用 Gemini 生成的 oracle quadmask,E2W 用 SA2VA+SAM2 模仿出来的版本,还少了 63 档)。补一格最便宜的对照就能把这两个变量分开:**VOID 的 oracle quadmask + edit token 条件**——直接复用 VOID 已经算好的 mask,只换 conditioning,不需要新的 mask 生成,这一格单独隔离"conditioning 来源"这一个变量,是目前能拿到的、成本最低的干净对照。
- Stage 0 训完即可做的早期验证:held-out clip 上,seg query token 生成的两个 mask 相对 ground truth 的 IoU/Dice,不用等 Stage 2 出视频才知道 mask 这一步有没有问题。
- Stage 2 跑完:接入 harness,在(扩充后的)remove 子集上把 E2W 这一行和 VOID 那一行并排对比,**Kubric held-out 和真实 clip 分开报告,不合成一个数**——Stage 2 的 edit token 只在 Kubric 合成数据上训练,VOID 的文本 prompt 是 Gemini 推理时现生成的、天然不挑数据来源,E2W 有可能因为这个 domain gap 在真实 clip 上输给 VOID,跟架构设计本身对不对无关,分开报告能避免这类短板把整体对比结果搅浑。

---

## 4. 明确不做的事

- Cycle-consistency loss——完全走 supervised diffusion loss,cycle 作为 future work 一句话带过。
- 非 remove 类编辑(add/replace/attribute-change)——VOID 的 mask 语义是围绕"移除"设计的,能否迁移未验证。
- VOID pass2 的变形修复——只用 pass1。
- `seg` attend `edit`——有语义论证(seg_ind 的"受影响区域"判断和 edit 编码的"物理后果推理"高度相关),但会破坏 Stage 0 的梯度隔离(mask loss 会经这条通道漏进 edit 参数);VOID 自己完全解耦的架构在物理受影响维度也能做到 0.83,说明不加这条耦合也能达到可用水准。留作 Stage 2 跑通后的独立 ablation,不在 v0 里加。
- 门控混合 α 作为默认路径——降级为 fallback,只在硬替换训不稳时启用(见 1.4);真启用时用手动 warmup schedule,不设计成端到端可学,省时间。
- 4D attention mask 的批训练支持。
- seg 分支的 soft/可微 mask 路径(不 threshold,直接用 SAM2 概率图做 fuzzy 组合喂给 renderer,让 video loss 也能到达 seg_dir/seg_ind)——理论可行,但 renderer 冻结、只见过 VOID 训练时的离散取值,喂连续值是分布外风险,且 video loss 隔着一层冻结 renderer 传回来的信号,对空间定位这类判断大概率不如 Stage 0 的 Dice+BCE 直接监督强。收益不确定,风险是实打实的生成质量退化,v0 不做。

---

## 5. 开放问题 / 风险

1. ~~**VOID 权重 license** 尚未确认~~ — **已查(2026-07-02)**:`netflix/void-model` 权重+代码是 **Apache-2.0**(研究/衍生/商用均可);底座 CogVideoX-Fun 是 **CogVideoX License**(智谱),**学术研究明确免费允许**,商用需在 open.bigmodel.cn 注册。→ **投稿用途 blocker 解除,ADR-0007 Decision #2 成立**;唯一尾巴:日后产品化/商用需走底座 CogVideoX 的商用注册。(训练数据 Netflix 未放,HUMOTO 需向 Adobe 申请,但 Stage 2 走 Kubric,不受影响。)
2. **quadmask 简化后的分布偏移**——Stage -1 已跑(见第 2 节结果):三档简化安全,前提是 overlap 并入 remove 侧;overlap 归属本身 n=6 未定论,保持默认待 Stage 0 后用确定性 mask 指标复核。**已关闭主要风险,留一个 overlap 归属的小尾巴。**
3. Stage 1 里 VLM 生成"移除后场景描述"的 prompt 质量,需要先人工抽查几条再批量跑。
4. `seg_dir`/`seg_ind` 互相可见性维持现状(seg_ind 看得到 seg_dir),没有强证据支持修改。
5. Stage 0 伪标注的语义边界目前只在 remove 场景验证过。
6. **seg 分支在 Stage 2 期间没有直接梯度监督**,只靠持续的小权重 mask loss 维持(见第 0、2 节)——不能假设 Stage 0 训完就一劳永逸,Stage 2 跑完后需要重新测一次 mask IoU/Dice,确认共享 LoRA 权重被 edit 分支梯度更新的过程中,seg 质量没有退化。
7. **edit token 数量(现在 4 个)可能偏小**:MetaQuery 在图片任务上的 token-count 研究显示更多 token 持续提升对齐效果,InstructX 在视频任务上直接用了 512 个,比现在的数量高出两个数量级。但 Stage 2 数据只有约 1900 对,token 数量涨上去、监督数据没跟着涨,每个 token 分到的信号会更稀薄,不是越多越好。现在不改,如果 Stage 1 之后时间允许,值得做一次便宜的扫描(固定其余不变,只变 token 数量,看训练曲线或 projection 多样性有没有随数量提升),不预设答案,不直接照抄 512。
8. **Stage 2 renderer 配置会静默断梯度(已提前查明,非开放,记录在案)**:CogVideoX-Fun 的默认 `model_cpu_offload_and_qfloat8`(offload+float8 量化)在训练时会切断经 renderer 回传到 edit token 的梯度,且**完全不报错**——loss 仍下降(seg/其他分支梯度还在),但 edit 链路实际没学到东西。Stage 2 必须 `model_full_load`+bf16(见 §2 Stage 2 前置)。这跟第 0 节"seg 分支不可微"是同类的"无声失败",区别是它是配置导致、可避免;已在训练开始前记录,不等训完才排查。

---

## 6. Success Criteria

- **Stage 0**:held-out clip 上 mask IoU/Dice 达到可用水准——理论上限是复现 VOID 自己 VLM-MASK-REASONER 的输出质量,因为 GT 本来就来自这条 pipeline。
- **Stage 2**:CF-VEdit remove 子集上,IP×CR 相关指标不劣于 VOID(`results/void/`),理想情况在 quality、physical 这类 VOID 本身也不是满分的维度上有可见提升——因为 renderer 完全相同,这个提升如果出现,能比较干净地归因于 query token 耦合方式本身,而不是 renderer 差异。
