# 基于 Sa2VA 的改动 plan(搭我们系统的「定位半」)

> 结论先放:**用 Sa2VA 改,能把「因果 Planner + Mask 解码器」(也就是"决定改哪块"的那半)直接拿到手——这是工程上最琐碎的一半,基本现成。** 但「Abduction 源反演」和「门控 Renderer」(也就是"实际生成"的那半)**不在 Sa2VA 里**,要靠 VACE/Wan 另起;而真正的研究风险——三层因果 mask 里的「间接/多跳」连带——Sa2VA 只给机制、不给因果智能,要靠我们加数据+监督。
> 仓库:`github.com/bytedance/Sa2VA`(Apache-2.0)· 用 `Sa2VA-Qwen2_5-VL-7B` 这个 variant。

## 统一命名(本文从头到尾就用这套词,不用编号)
| 名称                | 干什么                                              |
| ----------------- | ------------------------------------------------ |
| **Abduction 源反演** | 读源视频,反演成 renderer 的 VAE latent(`源latent`)= 不变量先验 |
| **因果 Planner**    | 解析指令,决定改哪/改成啥;输出「区域 query 向量」+「edit-plan tokens」 |
| **Mask 解码器**      | 把 query 向量 + 视觉骨干的稠密特征 → 三层因果 mask(逐帧、跨帧一致)      |
| **门控 Renderer**   | mask 门控的 inpainting DiT:不改变区贴回源 latent,受影响区重渲    |
| **定位半**           | = 因果 Planner + Mask 解码器(决定"改哪块")                 |
| **生成半**           | = Abduction 源反演 + 门控 Renderer(实际"渲出来")           |

---

## 0. Sa2VA 现状(读完源码的事实)
核心类 `Sa2VAModel`,文件 `projects/sa2va/models/sa2va.py`。三件套:
- `self.mllm` —— Qwen2.5-VL-7B(或 InternVL),负责理解 + 生成 `[SEG]` token。
- `self.grounding_encoder` —— **SAM2**(Hiera-L 视觉骨干 + prompt encoder + mask decoder + memory 时序传播),默认全冻结,只可选解冻 `sam_mask_decoder`。
- `self.text_hidden_fcs` —— 一个 MLP,把 LLM 最后一层在 `[SEG]` 处的 hidden(`in_dim`)投影到 SAM2 的 prompt 维度(`out_dim`)。**这就是连接 MLLM 和 SAM2 的「胶水」,已经写好了。**

数据流(`Sa2VAModel.forward`):
1. MLLM 前向,拿 `output.hidden_states[-1]`。
2. `seg_token_mask = input_ids == self.seg_token_idx` 找到 `[SEG]` 位置。
3. `pred_embeddings = text_hidden_fcs(hidden)[seg_token_mask]` —— **这正是「因果 Planner」要输出的"区域 query 向量"**。
4. `grounding_encoder.inject_language_embd(...)`(在 `models/extension/sam2_base.py`)把这个 query 向量当**额外的 sparse prompt** 拼进 SAM2:`sparse_embeddings = cat([sparse_embeddings, language_embd])` → `sam_mask_decoder` → mask。SAM2 memory 跨帧传播 = **逐帧时序 mask 免费拿到**。
5. Loss:`loss_mask`(BCE)+ `loss_dice` on mask,外加 `llm_loss`(LM 自回归)。

**对号入座:**

| 我们要的部件 | Sa2VA 里对应 | 状态 |
|---|---|---|
| 因果 Planner 的 query 向量 | `[SEG]` hidden → `text_hidden_fcs` → `pred_embeddings` | ✅ 现成 |
| Mask 解码器 + 时序 | SAM2 `inject_language_embd` + `sam_mask_decoder` + memory | ✅ 现成 |
| 视觉骨干 | SAM2 Hiera-L image encoder | ✅ 现成 |
| 因果 Planner 的 edit-plan tokens | 无 | ❌ 要加 |
| 三层因果 mask | 只有单层二值 mask | ❌ 要改 |
| Abduction 源反演 | 无(属生成半) | ❌ 另起 |
| 门控 Renderer | 无 | ❌ 另起(VACE/Wan) |

---

## 1. 必改项(都在 Sa2VA 内部,工作量可控)

### 改动 A —— 单层 mask → 三层因果 mask(直接 / 间接 / 不改变)
**思路**:把单个 `[SEG]` 拆成两个特殊 token `[SEG_DIR]`、`[SEG_IND]`,各自走一遍现有 SAM2 通路得到一张 mask;`不改变` = 两者并集的补。
- `sa2va.py` `__init__`:`special_tokens=['[SEG_DIR]','[SEG_IND]']`;存两个 token idx。
- `forward`:把 `seg_token_mask` 按两个 id 拆开,分别取 `pred_embeddings`,各跑一次(或 batch 一次)`inject_language_embd` → 两套 `pred_masks`。
- Loss:GT mask 带「层标签」,`loss_mask/loss_dice` 分层各算一份(直接层、间接层)。可给间接层更低权重(它误差大、常欠定)。
- 改动局部,集中在 `forward` + 数据集,**不动 SAM2 内部**。

### 改动 B —— 新增 edit-plan tokens(喂门控 Renderer 的内容条件)
Sa2VA 只为 mask 出 `[SEG]`,**没有**描述"目标长相"的连续 token。加一路:
- 新特殊 token `[EDIT]`(可多个);新投影头 `self.edit_hidden_fcs`(仿照 `text_hidden_fcs`),把 `[EDIT]` 的 hidden 投到门控 Renderer 的条件空间。
- `forward` 里收集这些 hidden 并 return,**不进 SAM2**。
- ⚠️ 这路在 Sa2VA 内**拿不到梯度**(没有 mask 监督它)。它要等门控 Renderer 接上、用 Renderer 的去噪损失回传才学得动 → 实际上是和 Renderer **联合训练**的,不能在 Sa2VA 单独练出来。

### 改动 C —— 新数据集类:仿真配对 + 因果层标签
Sa2VA 现训练集是 referring-seg(RefCOCO/MeViS/ReVOS/Ref-SAV)。我们加 Kubric 式仿真配对数据:
- 新建 `projects/sa2va/datasets/sa2va_data_cf.py`,仿 `sa2va_data_03_refvos.py` / `sa2va_data_finetune.py`。
- 每条样本产出:`(视频帧, 指令, 直接层 mask 序列, 间接层 mask 序列)`;层标签来自仿真依赖图。
- 现有 `gt_masks` / `frames_per_batch` 管线可复用,扩成两层即可。

### 改动 D —— 训练配置
- 复制 `projects/sa2va/configs/sa2va_qwen_finetune.py` 改一份:换我们的数据集、`special_tokens`、`frozen_sam2_decoder=False`(解冻 mask decoder 学因果区)、两层 loss 权重。
- 入口照旧:`bash tools/dist.sh train <config> 8`(≥8×A100)。
- 起点权重:HF `ByteDance/Sa2VA-Qwen2_5-VL-7B` + `pretrained/sam2_hiera_large.pt`。

---

## 2. 不在 Sa2VA 里、必须另起的「生成半」
- **Abduction 源反演**:用门控 Renderer 的 VAE(Wan VAE)对源视频做 encode/inversion,得到 `源latent`。这是 Renderer 侧预处理,Sa2VA 不涉及。
- **门控 Renderer**:接 VACE(Wan2.2/Wan-14B)。把 Sa2VA 出的三层 mask + edit tokens + 源 latent 作为 VACE 的 mask/condition 输入,做 masked-V2V(不改变区贴回、受影响区重渲)。这是一条工作量更大的集成线。
- **桥**:Sa2VA 的输出(三层 mask + edit tokens)就是喂给 VACE 的接口。定位半和生成半在这里对接。

---

## 3. "是不是一半搭好了?" —— 诚实评估
**算搭好了,但要分清是哪一半。**
- ✅ **拿到手的(定位半)**:从指令到三层 mask 那套机制——query 向量、SAM2 注入、逐帧时序 mask、BCE+dice 监督、训练脚手架。这是琐碎易错的工程一半,Sa2VA 基本免费给。改动 A 是小改、C/D 是配数据+配置。
- ⚠️ **看着像有、其实没有**:三层 mask 的「**间接/多跳因果**」连带——Sa2VA 给的是"把语义 query 变 mask"的**通用机制**,但它从没学过"因果连带到哪",要靠我们的仿真依赖图监督去教。**机制现成 ≠ 能力现成,这正是命门。**
- ❌ **完全没有的(生成半)**:Abduction 源反演 + 门控 Renderer = 整个"生成"那半,要靠 VACE/Wan 另起;edit tokens 这路也得等 Renderer 才有梯度。

一句话:**Sa2VA 把"决定改哪块"的定位半骨架送你了(省一大坨工程),但"改成什么 + 怎么钉死渲染"的生成半、以及"间接因果推理"这个真命门,还得自己建。** 与其说"架构一半搭好",不如说"**最不出彩但最费体力的那 1/3 省了,最难的研究 1/3 一点没省**"。

---
### 关键文件清单(改动落点)
- `projects/sa2va/models/sa2va.py` —— `Sa2VAModel.__init__/forward`、`text_hidden_fcs`、`seg_token_idx`(改 A、B)
- `projects/sa2va/models/extension/sam2_base.py` —— `inject_language_embd`(读懂即可,基本不改)
- `projects/sa2va/models/mllm/qwenvl.py` —— MLLM 前向(加 `[EDIT]` token 时碰)
- `projects/sa2va/datasets/sa2va_data_03_refvos.py` / `sa2va_data_finetune.py` —— 仿照新建 CF 数据集(改 C)
- `projects/sa2va/configs/sa2va_qwen_finetune.py` —— 复制改训练配置(改 D)
- 权重:`ByteDance/Sa2VA-Qwen2_5-VL-7B` + `facebook/sam2-hiera-large`
