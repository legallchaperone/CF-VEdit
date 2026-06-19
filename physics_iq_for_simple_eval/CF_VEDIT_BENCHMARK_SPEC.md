# CF-VEdit Benchmark — 改造 Spec(把 `physics_iq_for_simple_eval` 变成合规 benchmark)

> 目标:**只写规格,不写实现**。把现有 12 条 `physics_iq_for_simple_eval` 文件夹,改造成一个"有 benchmark 形状"的可插拔评测套件——任何视频编辑模型(我们的、Bernini、VACE…)都能接进来打分。
> 术语沿用 [[CF-VEdit-Architecture-and-Narrative (给人看的）]] 的统一命名。

---

## 0. 这个 benchmark 必须满足的要求(大白话,先钉死)
**必须做到的:**
1. **真值只来自真实视频或仿真,绝不来自另一个生成模型**——否则等于把某模型的脑补当标准答案。
2. **每条只标两样东西:① 哪些东西不许动(不变量),② 编辑后必须出现的后果**;不需要整段"标准答案视频"。
3. **只看最终编辑后的视频打分**,不看模型内部、不要求它交出推理过程或因果图——这样所有模型同一把尺子。
4. **指标必须二维**:保不变量 × 命中后果;**绝不许塌成一个总分**。
5. **不变量在物体/身份级判**(那个物体还在不在、是不是它),不做逐像素相等——允许阴影随光合理变化。
6. **物理后果单独报一个分**。
7. **每条标清楚:答案唯一 还是 欠定**;欠定的题只判不变量或"落入可接受范围"。
8. **训练集和评测集不能重叠,且要查过没泄漏**。
9. **裁判要可复现**(模型版本、prompt 都记录);**不能只靠一个大模型当唯一裁判**——人评 + VLM 评双轨,互相校验。
10. **带上、下界基线**:原样复制源(下界)、无视源纯重生成(上界),确认指标摆对位置。

**这一轮先放一放(留位,以后补):**
- **配对例**:同一指令喂两段不同源、正确答案该不同——证明"真在测反事实"的关键,采集贵,先留字段位。
- **编辑类型广度**:现仅"删除/增加",属性改变、施力/事件以后再加。

---

## 1. 设计原则(先定死,后面都服从它)
1. **数据资产 与 运行产出 严格分家**:`manifest/contracts/annotations/videos` 是 benchmark 的**只读资产**;模型跑出来的东西全进 `predictions/`,评测结果全进 `results/`。三者边界不许混。
2. **生成与评测解耦**:benchmark **不调用模型**。模型在外部把编辑后视频产出来、按约定落盘,benchmark 只消费目录。任何模型零侵入接入,胶水代码归使用者。
3. **manifest 只当轻索引**:重的东西(反事实契约、mask)外置成单独文件,manifest 里只写路径——否则 manifest 会膨胀、字段越加越乱。
4. **一切可校验**:每行 manifest 用 JSON schema 卡字段;每个 run 记录版本+哈希+命令,保证可复现。

---

## 2. 目标目录结构(改造后)
```
cf_vedit_bench/
  README.md                          # benchmark card(任务/指标/如何接模型)
  schemas/
    manifest.schema.json             # 卡每行 manifest 字段(§3.1)
    contract.schema.json             # 卡每个反事实契约(§3.3)
  manifest.jsonl                     # 轻索引,每行一个样本(§3.2)
  videos/source/<sample_id>.mp4      # = 现 converted/,按 sample_id 改名
  contracts/<sample_id>.json         # ★ 反事实契约(核心资产,§3.3)
  annotations/
    masks/<sample_id>/{target,affected,preserve}_mask.npy   # 三层 mask(可选)
    provenance.jsonl                 # 每条的来源 + 防泄漏证据(§3.4)
  judges/vlm_prompts.jsonl           # = 现 vlm_judge_prompts.jsonl
  baselines/                         # 只放脚本,不放产出(§5.4)
    copy_source.py  free_regen.py  README.md
  predictions/<run_name>/            # ★ 模型产出落这里(§4)
    videos/<sample_id>.mp4
    predictions.jsonl                # 每条状态(含失败)
    run_meta.json                    # 可复现信息
  results/<run_name>/
    per_sample.jsonl  human_per_sample.jsonl
    summary.json                     # 二维聚合 + 分类拆分
    leaderboard.md
  bench.py                           # CLI(§8)
```
改造 = 重命名 + 拆出 `contracts/`、`annotations/` 资产 + 立 `predictions/`、`results/` 产出契约。**原始 12 条内容不动,只搬位置、补字段、外置契约。**

---

## 3. 数据层(轻 manifest + 外置契约 + schema)

### 3.1 `manifest.schema.json`(卡每行字段)
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "cf_vedit_bench/manifest.schema.json",
  "title": "CF-VEdit manifest row",
  "type": "object",
  "required": ["sample_id","source_video","operation","instruction","target_ref","split","category"],
  "additionalProperties": true,
  "properties": {
    "sample_id":   {"type":"string","pattern":"^[a-z0-9_]+$"},
    "source_video":{"type":"string"},
    "operation":   {"enum":["remove","add","attribute","force_event"]},
    "instruction": {"type":"string","minLength":1},
    "target_ref":  {"type":"string","minLength":1},
    "split":       {"enum":["train","val","test"]},
    "category":    {"type":"string"},
    "scene_type":  {"type":"string"},
    "difficulty":  {"enum":["easy","medium","hard"]},
    "identifiability": {"enum":["identifiable","underdetermined"]},
    "pair_id":     {"type":["string","null"]},
    "video_meta":  {"type":"object","required":["fps","num_frames","width","height"],
                    "properties":{"fps":{"type":"number"},"num_frames":{"type":"integer"},
                                  "width":{"type":"integer"},"height":{"type":"integer"}}},
    "contract":    {"type":"string"},
    "annotations": {"type":"object","properties":{
                      "target_mask":{"type":["string","null"]},
                      "affected_mask":{"type":["string","null"]},
                      "preserve_mask":{"type":["string","null"]}}}
  }
}
```

### 3.2 `manifest.jsonl`(每行一个样本,轻)
```json
{
  "sample_id": "remove_cup_0001",
  "source_video": "videos/source/remove_cup_0001.mp4",
  "operation": "remove",
  "instruction": "Remove the red cup from the table.",
  "target_ref": "the red cup on the table",
  "split": "test",
  "category": "object_removal",
  "scene_type": "tabletop",
  "difficulty": "medium",
  "identifiability": "identifiable",
  "pair_id": null,
  "video_meta": {"fps": 8, "num_frames": 49, "width": 832, "height": 480},
  "contract": "contracts/remove_cup_0001.json",
  "annotations": {
    "target_mask": "annotations/masks/remove_cup_0001/target_mask.npy",
    "affected_mask": "annotations/masks/remove_cup_0001/affected_mask.npy",
    "preserve_mask": "annotations/masks/remove_cup_0001/preserve_mask.npy"
  }
}
```
> 三层 mask `target / affected / preserve` **就是我们的"直接 / 间接 / 不改变"**;mask 是可选的(真实视频可只标契约,仿真才有精确 mask)。

### 3.3 `contracts/<sample_id>.json`(★ 反事实契约 = 核心资产)
这是和 FiVE / TGVE 拉开差距的地方:**重点不是指令,而是"这次干预后世界该变成什么样、哪些该变、哪些不许变"**。
```json
{
  "sample_id": "remove_cup_0001",
  "operation": "remove",
  "target_ref": "the red cup on the table",
  "counterfactual_state": {
    "surface": "the tabletop surface should be continuous where the cup was",
    "lighting": "lighting should match the surrounding tabletop",
    "shadow": "the cup shadow should disappear",
    "occlusion": "previously occluded tabletop should be plausibly revealed",
    "temporal": "the cup should not reappear in later frames"
  },
  "affected_regions": ["tabletop contact patch", "cup shadow", "previously occluded tabletop"],
  "preserve_regions": ["background wall", "other table objects", "camera motion"]
}
```
`contract.schema.json` 至少 require:`sample_id, operation, target_ref, counterfactual_state, affected_regions, preserve_regions`。
- **命中后果分** 打 `affected_regions` + `counterfactual_state` 各条;**保不变量分** 打 `preserve_regions`。

### 3.4 `annotations/provenance.jsonl`(来源 + 防泄漏,每条一行)
把现有 `source_full_video / source_metadata / leakage_exclusion_evidence` 挪到这里,manifest 不背这些重信息:
```json
{"sample_id":"remove_cup_0001","source_dataset":"Physics-IQ",
 "source_uri":"...","leakage_checked":true,"leaked":false,"matched_paths":[]}
```

### 3.5 现 physics_iq 字段 → 新字段映射(改造时照这个搬)
| 现字段 | 去向 |
|---|---|
| `converted_video` | `source_video` |
| `user_prompt` | `instruction` |
| `target_object` | `target_ref` |
| `operation` | `operation`(枚举不变) |
| `must_preserve[]` | `contracts/*.json` 的 `preserve_regions` |
| `expected_physical_effect` / `expected_visible_outcome` | `contracts/*.json` 的 `counterfactual_state` / `affected_regions` |
| `physics_iq_category` | `category`(也可填 `scene_type`) |
| `source_full_video`,`source_metadata`,`leakage_exclusion_evidence` | `annotations/provenance.jsonl` |
| `vlm_judge_prompt` | `judges/vlm_prompts.jsonl`(保持) |

---

## 4. 模型 IO 契约(★ 最普适的一层)
**契约 = 文件落盘,不是函数调用。** Bernini / VACE / 我们的模型 / 人肉 PS 都能接,胶水各自写。

**输入(benchmark 给模型)**:对每个 `sample_id`,给 `videos/source/<sample_id>.mp4` + manifest 行的 `instruction`、`operation`。
**输出(模型必须产出)**:
```
predictions/<run_name>/
    videos/<sample_id>.mp4     # 编辑后视频,文件名 = sample_id
    predictions.jsonl          # 每条状态,含失败
    run_meta.json              # 可复现信息(见 §4.2)
```
> mp4 收进 `videos/` 子目录,根目录留给以后的 `logs/`、`planner_outputs/` 等,不让产物在根目录铺乱。

### 4.1 `predictions.jsonl`(每条状态,失败也要记)
```json
{"sample_id":"remove_cup_0001","video":"videos/remove_cup_0001.mp4","status":"ok","runtime_sec":42.1}
{"sample_id":"remove_bottle_0002","video":null,"status":"failed","error":"CUDA OOM"}
```
→ 模型失败也能被正确统计 **failure rate**;`status!=ok` 的样本各分计 0 且标 missing。

### 4.2 `run_meta.json`(强制可复现信息)
```json
{
  "run_name": "e2w_v0_20260618",
  "model_name": "E2W",
  "model_version": "v0.1",
  "benchmark_version": "0.1.0",
  "manifest_sha256": "…",
  "command": "python run_e2w.py --manifest manifest.jsonl --out predictions/e2w_v0",
  "created_at": "2026-06-18T12:00:00Z",
  "num_samples": 100,
  "hardware": {"gpu": "A100-80GB"},
  "notes": ""
}
```
**`benchmark_version` / `manifest_sha256` / `model_version` / `command` 必填**——否则日后无法知道某个 run 是在哪个 benchmark 版本上跑的。

### 4.3 可选糖:Python `Adapter`
```python
class Adapter:                       # 使用者实现,benchmark 不依赖
    def edit(self, source_path, instruction, operation, meta) -> str: ...
```
canonical 契约永远是上面的目录落盘;接 Bernini 就是写脚本读 manifest → 调 Bernini → 按 sample_id 存 mp4 + 写 predictions.jsonl。

---

## 5. 打分层(metrics 规格)

### 5.1 裁判输出 schema(VLM 与人工**共用**)
```json
{
  "target_success": 0,          // 干预落地没(编辑成功 / 指令遵循)→ 编辑落地
  "preservation_success": 0,    // preserve_regions 有没有被乱动 → 保不变量
  "effect_hits": ["..."],       // affected_regions/counterfactual_state 命中了哪些 → 命中后果(细)
  "physical_effect_success": 0, // 物理后果整体对不对 → 物理可信
  "temporal_consistency": 0,    // 质量
  "major_artifacts": 0,         // 质量(惩罚项)
  "overall_pass": 0,            // 次要综合
  "short_reason": ""
}
```

### 5.2 字段 → 分轴
| 分轴 | 定义 | 来源 |
|---|---|---|
| **保不变量** | `preserve_regions` 未被动比例 | `preservation_success`(条件:`target_success=1` 才计) |
| **命中后果** | 命中的 `affected_regions`/`counterfactual_state` 条数 ÷ 总条数 | `effect_hits` |
| **物理可信** | 物理后果是否成立 | `physical_effect_success` |
| **编辑落地** | 干预是否成功(前置门) | `target_success` |
| **质量** | `temporal_consistency` 与 `1-major_artifacts` 均值 | 两字段 |

规则:`target_success=0` ⇒ 后果/物理分记 0 且标 `edit_failed`;欠定例后果分改判"落入可接受集合";主报告**永远二维(保不变量 × 命中后果)**,不得只给总分。

### 5.3 聚合
跨样本取均值 → 率;**按 `category`、`operation`、`difficulty` 各拆表**;产 `summary.json`:`{保不变量,命中后果,物理可信,编辑落地,质量, n, missing, failure_rate, by_*{…}}`;`leaderboard.md` 每 run 一行,**必含保不变量、命中后果两列**。

### 5.4 上下界基线(脚本在 `baselines/`,产出走 `predictions/`)
- `baselines/copy_source.py` → 产出落 `predictions/copy_source/`:把源原样输出 → 预期 **保不变量≈满分、命中后果≈0、编辑落地≈0**(下界)。
- `baselines/free_regen.py` → 产出落 `predictions/free_regen/`:无视源纯重生成 → 预期 **命中后果高、保不变量崩**(上界)。
- **`baselines/` 只放脚本,绝不放结果**,免得和 `predictions/` 边界混掉。
- 验收:两条没落在对角两端 = 指标实现有 bug。

---

## 6. 两个裁判
**VLM 裁判**:输入 `(source.mp4, predictions/<run>/videos/<id>.mp4, manifest+contract)`;prompt 用 `judges/vlm_prompts.jsonl`;输出 §5.1 schema → `results/<run>/per_sample.jsonl`;记录裁判模型名/版本/temperature/prompt hash。
**人工裁判(gradio)**:左原 / 右编辑并排,下方列 `instruction` + `preserve_regions`(逐个勾"动没动")+ `affected_regions/counterfactual_state`(逐条勾"命中没");产出**同 §5.1 schema** → `human_per_sample.jsonl`;与 VLM 算一致性(accuracy / κ)验证 VLM 裁判可信度;支持多标注者报一致性。

---

## 7. 报告产出
`results/<run>/`:`per_sample.jsonl` / `human_per_sample.jsonl`、`summary.json`(§5.3)、`leaderboard.md`(跨 run 二维表 + 上下界锚点 + 人机一致性 + failure_rate)。报告必须显式声明:放宽/暂缓项(配对例、编辑广度)、规模(n)、覆盖类型 → 诚实写清 scope。

---

## 8. CLI 接口(bench.py 规格)
```
bench validate-manifest          # 用 schemas/ 卡 manifest.jsonl + contracts/
bench list                       # 列样本/类型/可识别性/split 分布
bench validate <run>             # 检查 predictions/<run>/ 齐全、命名合规、predictions.jsonl 一致
bench score <run> --judge vlm    # VLM 裁判 → per_sample.jsonl
bench score <run> --judge human  # 起 gradio 人工裁判
bench report <run>               # 聚合 → summary.json + leaderboard.md
bench agree <run>                # 人机一致性
```

---

## 9. 改造步骤(从现文件夹到合规 benchmark)
1. **立 schema**:写 `schemas/manifest.schema.json`、`contract.schema.json`。
2. **搬家+改名**:`converted/` → `videos/source/<sample_id>.mp4`;`vlm_judge_prompts.jsonl` → `judges/vlm_prompts.jsonl`。
3. **拆资产**:按 §3.5 映射,把每条的重信息外置成 `contracts/<id>.json` + `annotations/provenance.jsonl`;manifest 只留轻字段 + 路径。
4. **补字段**:`split`(现全 test)、`category`、`identifiability`(先全 identifiable)、`pair_id=null`、`video_meta`(从 mp4 探)。
5. **立产出契约目录**:`predictions/`、`results/`,写 `README.md`(benchmark card)。
6. **写基线脚本**:`baselines/copy_source.py`、`free_regen.py`(产出落 `predictions/`)。
7. **冒烟**:`validate-manifest` 过 → 放一个 copy_source run → `validate → score(human) → report` 跑通,确认 benchmark 形状成立。

---

## 10. 改造后,对照"必须满足的要求"逐条状态
| 要求 | 状态 | 说明 |
|---|---|---|
| 真值来自真实/仿真,非生成模型 | ✅ | 真实 Physics-IQ 视频 |
| 只标不变量 + 后果 | ✅ | 契约 `preserve_regions` + `affected_regions/counterfactual_state` |
| 只看输出视频打分 | ✅ | 不看模型内部 |
| 只按标注的不变量/后果打分 | ✅ | 不预设任何因果图 |
| 上下界基线 | ✅(新增) | copy_source / free_regen |
| 指标二维不塌成单分 | ✅(强制) | 保不变量 × 命中后果 |
| 不变量物体/身份级判 | ⚠️ 暂用裁判判 | 将来换 re-ID/分类器 |
| 物理后果单独报 | ✅ | `physical_effect_success` |
| 标注答案唯一/欠定 | ✅(新增字段) | `identifiability` |
| 训练/评测不重叠、查泄漏 | ✅ | `provenance.jsonl` 记录 |
| 裁判可复现、不靠单一大模型 | ✅(放宽) | 人评 + VLM 评同 schema + run_meta 可复现 |
| 校验裁判靠不靠谱 | ✅ | 人机一致性 |
| 配对例(同指令两源不同答案) | ⏸ 暂缓 | 留 `pair_id` 位 |
| 编辑类型广度 | ⏸ 暂缓 | `operation` 枚举已留 attribute/force_event |

> 一句话:数据资产(manifest+契约+mask)与运行产出(predictions+results)分家,manifest 当轻索引、契约当核心资产,再加 schema 卡字段 + run_meta 锁版本——就从"12 条素材"升级成"有 benchmark 形状、可插拔任何模型、可复现"的评测套件。