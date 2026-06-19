# CF-VEdit / E2W —— 代码仓库设计（边界 + 复用 + 防漂移）

> 目标:从零设计一个 monorepo,把 proposal 里的系统**搭得住、不跑偏**。三件事钉死:① **规定好边界**(谁能依赖谁),② **最大复用已有代码**(Sa2VA / VACE-Wan / 现有 benchmark 脚手架),③ **机制化防漂移**——让仓库长期不脱离 proposal。
> 关联:[[Counterfactual-Video-Editing-Proposal]] · [[CF-VEdit-Architecture-and-Narrative (给人看的）]] · [[CF-VEdit-Benchmark-Spec]] · [[Sa2VA-Modification-Plan]]
> 结构决定:**Monorepo,benchmark 做成可独立拆出的子包**(契合 proposal P0「benchmark 可独立发」,但与模型共享 contract/schema)。

---

## 0. 一句话设计哲学
仓库的形状 = proposal 的形状。proposal 已经替我们想好了三条最重要的边界——**「benchmark 不调模型」**(benchmark-spec §1.2)、**「数据资产 vs 运行产出」**(benchmark-spec §1.1)、**「定位半 vs 生成半」**(sa2va-plan)。**仓库要做的不是发明边界,而是把这三条边界变成目录结构 + 依赖规则 + CI 守卫,让它们物理上无法被违反。** 漂移的本质 = 边界被悄悄越过、或代码长出 proposal 里没有的东西。所以防漂移 = 把 proposal 变成仓库里**可执行的真相源**。

---

## 1. 顶层目录(monorepo)
```
e2w/
├── README.md                  # 一句话定位 + 指向 docs/proposal;新人/AI 的入口
├── AGENTS.md                  # ★ 仓库宪法:边界规则 + 改动纪律(人和 AI 协作者都读)
├── pyproject.toml             # workspace 根;声明各 package 与依赖方向
│
├── docs/
│   ├── proposal/              # ★ proposal-as-truth:四篇 note 的 canonical 版本化副本
│   │   ├── proposal.md           #   = Counterfactual-Video-Editing-Proposal
│   │   ├── architecture.md       #   = CF-VEdit-Architecture-and-Narrative
│   │   ├── benchmark-spec.md      #   = CF-VEdit-Benchmark-Spec
│   │   └── sa2va-plan.md          #   = Sa2VA-Modification-Plan
│   ├── TRACEABILITY.md        # ★ 可追溯矩阵:proposal 主张/novelty → 模块 → 测试 → 状态
│   ├── SCOPE.md               # ★ 范围/暂缓登记:cycle 训练、配对例、编辑广度…(留位不实现)
│   └── adr/                   # ★ Architecture Decision Records:任何偏离 proposal 的决定记这
│       └── 0001-monorepo-splittable-benchmark.md
│
├── packages/
│   ├── e2w_core/             # ★ 共享契约层(稳定、改动需 review)= 两半的唯一接缝
│   │   └── e2w_core/
│   │       ├── masks.py          #   三层时空 mask 定义(直接/间接/不改变)+ 优先级规则
│   │       ├── plan.py           #   edit-plan tokens / region-query 向量 的接口类型
│   │       ├── latent.py         #   abduction 源 latent 的接口(U 先验)
│   │       └── io_contract.py    #   模型 IO 落盘契约(predictions/ 目录形状,见 benchmark-spec §4)
│   │
│   ├── cf_vedit_bench/       # ★ benchmark:边界最硬,自带 pyproject,可单独 pip 安装/发布
│   │   ├── pyproject.toml        #   只依赖 e2w_core 的 schema 子集(或零内部依赖)
│   │   ├── README.md             #   benchmark card(任务/指标/如何接模型)
│   │   ├── schemas/              #   manifest / contract / judge 的 JSON schema
│   │   ├── src/cf_vedit_bench/   #   bench.py CLI、scoring、两个 judge、IO 校验
│   │   ├── data/                 #   ★ 只读资产:manifest.jsonl + contracts/ + annotations/ + videos/source/
│   │   ├── baselines/            #   copy_source.py / free_regen.py(只放脚本)
│   │   └── tests/                #   ★ spec-as-test(迁入现有 test_cf_vedit_benchmark.py)
│   │
│   ├── localization/         # 定位半:Causal Planner + Mask 解码器(基于 Sa2VA)
│   │   ├── third_party/sa2va/    #   vendored 上游(git submodule, pin commit, 不就地改)
│   │   ├── patches/              #   ★ 我们对 Sa2VA 的 deltas(overlay,与上游分离、可审计)
│   │   ├── e2w_localization/     #   新代码:三层 mask、[SEG_DIR]/[SEG_IND]/[EDIT]、CF 数据集、config
│   │   └── tests/
│   │
│   ├── generation/          # 生成半:Abduction 源反演 + 门控 Renderer(基于 VACE/Wan)
│   │   ├── third_party/vace_wan/
│   │   ├── e2w_generation/       #   源 inversion → 源latent;mask 门控 inpainting;不变量保持损失
│   │   └── tests/
│   │
│   └── data_engine/         # 仿真数据引擎(Kubric 式):共享种子 factual/CF 配对 + 依赖图 → mask/E/I
│       ├── e2w_data_engine/
│       └── tests/
│
├── integration/             # 唯一同时认识"两半"的地方:端到端流水线 + adapters
│   ├── pipelines/               #   源→abduction→planner→门控 renderer(推理/训练 三阶段编排)
│   └── adapters/                #   e2w_adapter / bernini_adapter / vace_adapter → 产出落 predictions/
│
├── configs/                 # 训练/推理配置(三阶段:对齐 / 端到端 / 可选 RL)
├── scripts/                 # dist train、数据准备、跑 benchmark 的入口脚本
├── tools/                   # 仓库级守卫:import-linter 规则、proposal-link 检查、schema lint
└── .github/workflows/       # CI:spec-test + 依赖方向检查 + schema 校验 + drift 守卫
```

---

## 2. 边界(核心):依赖方向规则
**一句话:依赖只能"向内"指向 `e2w_core`(契约层),禁止横向、禁止 benchmark→模型。** 这张图就是整个仓库的"宪法",CI 用 import-linter 强制执行。

```
                      e2w_core  (契约层,不依赖任何内部包)
                     ▲   ▲   ▲   ▲
       ┌─────────────┘   │   │   └──────────────┐
 cf_vedit_bench    localization  generation   data_engine
 (只读契约/零依赖)    │定位半│      │生成半│
       ▲                └──────┬──────┘
       │                       │ (两半唯一接缝 = e2w_core 的 mask/plan/latent 契约)
       │                  integration  ──写──► predictions/  ──被消费──►  cf_vedit_bench
       └───────────────────────────────────────────────────────────────────┘
                         benchmark 只"消费目录",永不 import 模型代码
```

五条硬边界(每条都对应 proposal 的一句原则,且都可机器校验):

| # | 边界 | 规则 | 来源 | 谁来守 |
|---|---|---|---|---|
| B1 | **benchmark ↔ 模型** | benchmark 只消费 `predictions/` 目录,**绝不 import** localization/generation/integration | benchmark-spec §1.2「生成与评测解耦」 | import-linter + spec-test |
| B2 | **数据资产 ↔ 运行产出** | `manifest/contracts/annotations/videos` 只读;模型产出全进 `predictions/`,评测结果全进 `results/` | benchmark-spec §1.1 | 目录布局 + `bench validate` |
| B3 | **定位半 ↔ 生成半** | `localization` 与 `generation` **互不 import**;唯一接缝 = `e2w_core` 的三层 mask + edit tokens + 源latent | sa2va-plan「桥」 | import-linter |
| B4 | **上游 ↔ 我们的 delta** | `third_party/` vendored 不就地改;我们的改动只在 `patches/` 或 `e2w_*` 新文件 | sa2va-plan「关键文件清单」 | CI 校验 third_party 无 diff |
| B5 | **训练源 ↔ 评测源** | sim 引擎 A 仅 dev/val,不进报告;评测 = 真实 held-out;两者严格不相交 | proposal §3「源分离」 | `provenance.jsonl` + 泄漏检查 |

> 为什么 `e2w_core` 单列一层:三层 mask、edit tokens、源latent、IO 落盘格式是**最该稳定、最不该乱改**的东西。把它们抽成一个谁都依赖、自己不依赖任何内部包的"窄腰",任何改它的 PR 自动触发 review——这就是把"接口稳定"变成结构性保证,而非口头约定。

---

## 3. 复用策略(尽量不重写)
原则:**别人写好的"机制"全拿来;我们只写 proposal 里独有的"增量"。** vendoring 用 submodule + 不就地改 + delta 外置,保证上游可升级、我们的改动可审计(对应 B4)。

| 复用对象 | 怎么接 | 我们写什么(增量) | 落点 |
|---|---|---|---|
| **Sa2VA** (Apache-2.0) | `third_party/sa2va` submodule pin commit | 单层→三层 mask(改A)、`[EDIT]` token+投影头(改B)、CF 仿真数据集(改C)、训练 config(改D) | `localization/patches` + `e2w_localization` |
| **VACE / Wan2.2** | `third_party/vace_wan` | mask 门控 inpainting(不改变区贴回源latent)、不变量保持损失、feather+联合去噪接缝 | `generation/e2w_generation` |
| **SAM2 / Qwen2.5-VL / Wan VAE** | 外部权重,configs 里 pin 版本,不进仓库 | — | `configs/` |
| **Kubric** | `data_engine` 内封装 | 共享种子渲 factual/CF 对 + 物体级因果日志 → 三层 mask + E/I 标签 | `data_engine/e2w_data_engine` |
| **现有 physics_iq benchmark 脚手架** | 直接当 `cf_vedit_bench` 的种子 | 按 benchmark-spec §9 改造:schema、外置 contract、provenance、bench.py CLI、baselines | `packages/cf_vedit_bench` |

> 现有 `physics_iq_for_simple_eval/tests/test_cf_vedit_benchmark.py` 已经是一份 **spec-as-test**(它断言 manifest 轻量化、contract 外置、二维指标不塌成单分、copy_source 落在对角下界)。**它不是普通测试,是 benchmark 的可执行规格**——迁进 `cf_vedit_bench/tests`,当作 P0 的验收闸门。

> sa2va-plan 的诚实结论同样要写进仓库认知:Sa2VA **免费送的是"定位半骨架"**(query向量→SAM2注入→逐帧 mask→BCE/dice 监督),**没送的是**「间接/多跳因果连带」的能力(机制现成 ≠ 能力现成,要靠 `data_engine` 的依赖图监督去教)和整个「生成半」。仓库结构要让这件事一眼看清:`localization` 里机制是现成的,但真正的研究风险锁在 `data_engine` 的标签质量 + 间接层 loss 上。

---

## 4. 防漂移(本仓库的命门)
漂移有两种:**(a) 边界被越过**(§2 已用 CI 堵死),**(b) 代码长出 proposal 没有的东西 / proposal 有的东西没人实现**。(b) 靠下面六件"机制",核心思想是**把 proposal 变成仓库里可执行、可追溯、改动留痕的真相源**。

**① Proposal-as-truth(版本化真相源).** 四篇 note 的 canonical 副本进 `docs/proposal/`,跟代码一起版本管理。每个模块 README **必须**反向链接到它实现的具体小节(例:`generation/README` 写「实现 architecture §A.2【4】门控 Renderer + proposal §2.6.3」)。改了设计先改 `docs/proposal/` 再改代码——**真相源永远领先实现,不许实现偷偷领先真相源**。

**② 可追溯矩阵 `TRACEABILITY.md`.** 一张表把 proposal 的每条主张/novelty 钉到 模块 + 测试 + 状态。两个用途对称:**有 novelty 无模块/测试 = 漏做**;**有模块无 proposal 锚点 = scope creep(漂移)**。重点盯三个真增量 + benchmark 的"必须满足项":

| Proposal 主张 | 模块 | 守它的测试 | 状态 |
|---|---|---|---|
| 真增量①:abduction = 源 inversion 成 latent 当 U 先验 | `generation/e2w_generation/latent` | 反演往返误差 + 不变量区 latent match | ⬜ |
| 真增量②:三层 mask 的**间接/多跳**层 | `localization`(改A)+ `data_engine`(依赖图标签) | 间接层 mask 对齐 sim 依赖图 | ⬜ |
| 真增量③:绑定 abduction 的不变量保持损失 | `generation/e2w_generation/losses` | `不改变`区 V̂ 必须 = V | ⬜ |
| benchmark:二维指标绝不塌成单分 | `cf_vedit_bench/scoring` | 已有 spec-test(`保不变量`/`命中后果` 两键必在) | ✅ |
| benchmark:真值不来自生成模型 | `cf_vedit_bench/data` | provenance 来源枚举禁 t2v 当 GT | ⬜ |
| Rung-3 闸门:同指令两源→答案须不同 | `cf_vedit_bench`(`pair_id` 位) | 暂缓,留字段(见 SCOPE) | ⏸ |

**③ Spec-as-test(规格即测试).** 把 proposal 的不变量写成 fail-on-drift 的测试,而不是注释:
- benchmark 包对模型包的 import 数 = 0(守 B1);
- 主报告对象同时含 `保不变量` 和 `命中后果` 两键,缺一即 fail(守"二维不塌");
- `copy_source` 必须落 `保不变量≈1 / 命中后果≈0`、`free_regen` 反之(守指标摆位,benchmark-spec §5.4);
- 三层 mask 枚举恰为 `{直接,间接,不改变}`,像素冲突优先级 `直接>间接>不改变`(守 architecture §A.4)。
这些测试一红,就是仓库开始漂了。

**④ ADR(偏离即留痕).** 任何偏离 proposal 的决定写一篇 `docs/adr/NNNN-*.md`(背景/决定/后果),并在相关代码 + TRACEABILITY 里反链。**没有 ADR 的偏离 = 不许 merge。** proposal 自己的"搜后定位下调"(如 cycle 训练降为辅助)正是该进 ADR 的那类决定——让偏离变成可见、可审、可回溯的工件,而不是无声的 drift。

**⑤ SCOPE 登记(暂缓项不丢不蔓).** proposal 明确"先放一放/暂不做"的(cycle/环训练、Rung-3 配对例、attribute/force_event 编辑广度)进 `docs/SCOPE.md`,每项标 状态 + 占位落点(如 manifest 已留 `pair_id`、`operation` 枚举已留 `attribute/force_event`)。作用:**暂缓项既不会半成品偷偷混进来,也不会被忘掉。**

**⑥ CI drift 守卫 + Definition-of-Done.** CI 四关:`import-linter`(依赖方向=§2 那张图)、`schema validate`(manifest/contract 卡字段)、`spec-test`(③)、`third_party 无 diff`(守 B4)。**PR 的 DoD**:过四关 + 更新 TRACEABILITY +(若偏离)附 ADR。把防漂移变成 merge 的前置条件,而非自觉。

> `AGENTS.md`(仓库宪法)把以上浓缩成给协作者(尤其 AI)的一页纸:五条边界 + "改设计先改 docs/proposal" + "偏离要 ADR" + "PR 要更新 TRACEABILITY"。**没读 proposal 的人也能被结构挡住,不会引入漂移。**

---

## 5. 搭建顺序(对齐 proposal P0–P3,先立"尺子"再造"机器")
**关键反漂移决定:先把 benchmark(尺子)做完,再做模型。** 模型一旦诞生就永远被 proposal 定义的二维指标量着,没法自己漂走。

| 阶段 | 对应 proposal | 干什么 | 产出 = 防漂移锚点 |
|---|---|---|---|
| **P0** | P0 benchmark | `cf_vedit_bench` 落地:现有脚手架按 benchmark-spec §9 改造,跑通 `validate→score→report`,copy_source/free_regen 摆对位 | 可独立发的 benchmark + 一把谁都改不动的尺子 |
| **P1** | P1 共享核心 | `e2w_core` 契约定死;`localization` 改A(三层 mask 骨架);`generation` 最小版(源 inversion + 不变量 pinning,先浅 DAG/属性) + `integration` adapter 让 E2W 能上 benchmark | 端到端能跑、两半接缝(e2w_core)冻结 |
| **P2** | P2 因果连带深化 | `data_engine`(Kubric 配对 + 依赖图标签)→ 训**间接/多跳层**(真命门);接时序 rollout 覆盖物理后果 | 真增量②落地 + 监督信号 |
| **P3** | P3 对齐 | 可选 RL/可微 reward,**单独隔离成 `alignment` 包**,核心训练不依赖它 | 末层可插拔,不污染主线 |

---

## 6. 三句话总结
1. **边界**:`e2w_core` 当窄腰,五条硬边界(B1–B5)用 import-linter + schema + 目录布局做成 CI 强制,把 proposal 的"分家原则"变成物理事实。
2. **复用**:Sa2VA 给定位半骨架、VACE/Wan 给生成半、现有 benchmark 脚手架给 P0——全部 vendored 不就地改,我们只写三个真增量。
3. **防漂移**:proposal 进 `docs/proposal` 当版本化真相源,`TRACEABILITY` 双向对账、`spec-test` fail-on-drift、`ADR` 给偏离留痕、`SCOPE` 看住暂缓项、CI+DoD 把这一切变成 merge 的前置条件。
