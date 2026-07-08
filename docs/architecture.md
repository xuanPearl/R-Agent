# 项目架构与代码走读

按代码目录讲解——每个文件夹放什么、对应架构图哪一层、以及端到端的调用流。

## 顶层结构

```
R-Agent/
├── pyproject.toml           项目元数据 + 依赖 + CLI entrypoint 注册
├── README.md
├── .gitignore
├── pathology_copilot/       主包（所有代码）
├── demo/                    示例数据（case JSON）
├── docs/                    本文档
└── tests/                   单测
```

对应架构图从下往上（Base Model → Experts → Tools → Agent）的四层，主包 `pathology_copilot/` 每一层对应一个子目录，跨层的契约（schemas、state、config、llm、orchestrator、cli）放在包根目录。

---

## `pathology_copilot/` 分层详解

### 跨层契约（包根，不属于任何一层）

```
pathology_copilot/
├── __init__.py              包版本，空实现
├── __main__.py              让 `python -m pathology_copilot ...` 能跑
├── config.py                Config dataclass（seed、高不确定性阈值 0.3）
├── schemas.py    ★★★        所有跨层数据契约
├── state.py      ★★         ExternalState：可 checkpoint 的状态对象
├── llm.py        ★★★        MockLLMClient（Planner/Critic 的假 LLM）
├── orchestrator.py ★★       编排：Planner→Executor→Critic→Report
└── cli.py                   Typer CLI 命令行入口
```

**`schemas.py`** 是整个项目的"宪法"——所有跨模块传的数据都在这定义：

| 类 | 用途 | 强约束 |
|---|---|---|
| `Grounding` | 每一条证据锚定到的图像区域或知识库文档 | `region_id + source` |
| `ToolCall` | Planner 产出的一次工具调用 | 有 `call_id`、`args`、`rationale` |
| `ToolResult` | Tool 返回的结果 | **model_validator 强制：非 error 结果必须有 uncertainty + 至少 1 个 grounding** |
| `Evidence` | 报告里的一条结论 | 必须带 `call_id`（能追回是哪次工具调用产出）+ grounding |
| `DiagnosticReport` | 最终结构化报告 | 字段固定，pydantic 拒绝随意加字段 |
| `PlannerOutput` | Planner 返回的计划 | `plan: list[ToolCall]` |
| `CriticNote` / `CriticOutput` | Self-Critic 的告警 | `kind ∈ {missing_grounding, high_uncertainty, inconsistency}` |

**`state.py`** 就一个 `ExternalState`——图里"External Structured State（token 可控 / 可中断恢复）"的落点。字段：`plan / executed / pending / critic_notes / step / finished`。有 `dump()` / `load()`，纯 JSON，可断点续跑。

**`llm.py`** 是**唯一**要接真实 LLM 时改的文件。`MockLLMClient.complete(role, context, response_schema)` 签名跟 Anthropic/OpenAI SDK 对齐，内部按 `role` 分派到 `_planner_response` 或 `_critic_response`——两个假推理函数，用规则替代 LLM。

---

### Layer 1 · Base Model（基础模型层）

```
pathology_copilot/base_models/
├── __init__.py
├── pathology_vit.py         Mock DINOv3 ViT
└── pathology_vlm.py         Mock Qwen2.5-VL
```

对应架构图 **BASE MODEL** 那一栏。

- `PathologyViT.embed(patch_ref) → ViTOutput(embedding, attention)`：用 SHA256 hash 生成确定性伪 embedding，同一个 patch_ref 每次调都拿一样的结果
- `PathologyVLM.ask(image_ref, question) → VLMAnswer(text, confidence)`：if/else 匹配关键词返回预设文本

**真实化替换点**：把这两个类实现换成真实的 ViT-L 前向和 VLM 推理即可，接口不动。

---

### Layer 2 · Experts（下游专家模型层）

```
pathology_copilot/experts/
├── __init__.py              暴露 expert_registry
├── base.py                  ExpertModel 抽象基类
├── cancer_detection.py      有无癌
├── subtype_classifier.py    亚型分类（含"ambiguous_regions"逻辑，Beat 3 关键）
├── grading.py               组织学分级
├── mutation_prediction.py   突变预测
└── registry.py              ExpertRegistry + 4 个注册实例
```

对应架构图 **EXPERTS** 那一栏（"36+ 任务"位）。

- `ExpertModel.predict(vit_output, case_metadata) → dict`
- `expert_registry` 是全局单例，`expert_registry.get("cancer_detection")` 拿到具体 expert
- **加新任务的入口**：写一个 `ExpertModel` 子类 + `expert_registry.register(...)`，两行代码

**`subtype_classifier.py` 的特殊逻辑**（打回重试用到）：
```python
if region_id in hints.get("ambiguous_regions", []):
    return {"subtype": None, "probabilities": {c: 0.33, ...}}  # 三类都 ~0.33
```
让"某个 ROI 上模型给不出结论"这件事在 mock 里可控。

---

### Layer 3 · Tools（工具层）

```
pathology_copilot/tools/
├── __init__.py              触发所有 tool 模块 import → 各自 register 到 tool_registry
├── base.py                  Tool 抽象基类（run(call, *, case_metadata) → ToolResult）
├── registry.py              ToolRegistry
├── vision_primitives.py     thumbnail、region_view（vision primitives 一栏）
├── domain_models.py         把 4 个 expert 包装成 4 个 tool（domain models 一栏）
├── knowledge_tools.py       guideline_search、similar_case_retrieval（knowledge tools 一栏）
└── vlm_tool.py              vlm_ask（vlm tool 一栏）
```

对应架构图 **TOOLS**（"统一接口 + 不确定性 + grounding"）。

**Tool 契约**：所有 tool 实现 `run(call, *, case_metadata) → ToolResult`，返回的 ToolResult 必须带 `uncertainty` 和 `grounding`（`schemas.py` 里的 model_validator 强制）。

**注册流程**：
1. `tools/__init__.py` 里 `from .registry import tool_registry` 先创建全局注册表
2. 紧接着 `from . import vision_primitives`（还有另外 3 个模块）触发 import
3. 每个 tool 模块底部有 `tool_registry.register(ThumbnailTool())` 完成注册
4. 使用方 `tool_registry.call(call, case_metadata=...)` 即可

**`domain_models.py` 的关键细节**（打回重试用到）：
```python
expert_ctx = {**case_metadata, "_current_region": region_id}
output = expert.predict(vit_out, expert_ctx)
```
把当前 region_id 注入到 case_metadata 副本里，让 expert 能根据"我现在在看哪块 ROI"决定输出——比如 `SubtypeClassifier` 在 `ambiguous_regions` 里就返回模糊分布。

---

### Layer 4 · Agent（智能体层）

```
pathology_copilot/agents/
├── __init__.py              暴露 Planner / Executor / SelfCritic / SchemaReportBuilder
├── planner.py               Planner：包一层 MockLLMClient 调用
├── executor.py              Executor：按 pending 队列跑 tool，每步 checkpoint
├── self_critic.py           SelfCritic：调 MockLLMClient 做"critic"role
└── schema_report.py         SchemaReportBuilder：拼装 DiagnosticReport + grounding 审计
```

对应架构图 **AGENT** 那一栏的四个方块。

| Agent | 职责 | 关键实现 |
|---|---|---|
| `Planner` | 拆解诊断流程 → ToolCall 列表 | 把 `state` 打包成 ctx，交给 `MockLLMClient.complete(role="planner")` |
| `Executor` | 调度 tools、写 state、失败捕获 | while 循环 pop `state.pending`，用 tool_registry 调，写 `state.executed`，每步都是 checkpoint |
| `SelfCritic` | 证据一致性 & grounding 复查 | 交给 `MockLLMClient.complete(role="critic")`；**REPLACE 语义**：`state.critic_notes = list(output.notes)`（不是 append） |
| `SchemaReportBuilder` | 结构化报告 + grounding 审计 | 遍历 `state.executed` → 拼 Evidence → 组装 DiagnosticReport；`_audit()` 拒绝任何 evidence 引用了 state 里不存在的 region_id 或 call_id |

---

## `demo/` 和 `tests/`

```
demo/cases/
├── demo1.json               阳性 case（cancer=True, subtype=adenocarcinoma, grade=G2）
├── demo2_negative.json      阴性 case（cancer=False）→ Planner 只跑 2 步
└── demo3_ambiguous_roi.json 打回重试 case（ambiguous_regions=[roi_center], retry_roi=roi_periphery）

tests/
├── __init__.py
├── test_tools.py                 Tool Registry + 每个 tool 输出契约
├── test_planner.py               阳性/阴性分支、replan 时跳过已跑工具
├── test_executor.py              完整跑通、中断恢复、失败被 critic 捕获
├── test_self_critic.py           三条 critic 规则各触发一次 + 干净场景
├── test_schema_report.py         端到端报告 + grounding 审计拒绝
└── test_orchestrator_retry.py    打回重试全链路：flag → replan → clear
```

---

## 代码调用流（关键路径）

用户在终端敲：
```bash
python -m pathology_copilot run --case demo/cases/demo3_ambiguous_roi.json
```

### 1️⃣ 入口

```
__main__.py
  └→ cli.app()   (typer)
       └→ cli.run(case=Path(...))
```

`cli.py:run()` 读 JSON → 构造 `Orchestrator(step_hook=_print_step, critic_hook=_print_critic)` → 调 `orchestrator.run(case_id, case_metadata)`

### 2️⃣ Orchestrator 主循环（`orchestrator.py:run`）

```
Orchestrator.run
  ├─ [初次] state = ExternalState.from_case(...)
  │
  ├─ [初次规划] planner.plan(state) ──► MockLLMClient.complete(role="planner")
  │      │                                  └─ _planner_response(ctx)
  │      │                                       ├─ 检查 critic_notes 里有没有 "different ROI" 信号
  │      │                                       ├─ 有 + hints.retry_roi 存在 → 走 retry 分支
  │      │                                       └─ 否则 → 走初始诊断分支（thumbnail→cancer→...→vlm_ask）
  │      └─ state.register_plan(plan)
  │
  ├─ [执行] executor.run(state) ─── 循环 pop state.pending：
  │      │
  │      ├─ tool_registry.call(call, case_metadata=state.case_metadata)
  │      │      └─ tool_registry.get(name).run(call, case_metadata=...)
  │      │           │
  │      │           ├─ ThumbnailTool.run   → 返回 image_ref + grounding
  │      │           ├─ CancerDetectionTool.run（domain_models._ExpertTool）
  │      │           │    ├─ PathologyViT.embed(patch_ref)  ← Base Model 层
  │      │           │    ├─ expert_registry.get("cancer_detection")
  │      │           │    └─ CancerDetection.predict(vit_out, expert_ctx)  ← Expert 层
  │      │           ├─ VLMAskTool.run
  │      │           │    └─ PathologyVLM.ask(image_ref, question)  ← Base Model 层
  │      │           └─ ...
  │      │
  │      ├─ state.record_result(result)   ← 每步 checkpoint
  │      └─ step_hook(state, result)      ← CLI 打日志
  │
  ├─ [Critic 循环，最多 max_critic_rounds+1 轮]
  │      │
  │      ├─ 轮 1: critic.review(state)
  │      │       └─ MockLLMClient.complete(role="critic")
  │      │           └─ _critic_response(ctx)
  │      │               ├─ missing_grounding 规则
  │      │               ├─ high_uncertainty 规则
  │      │               ├─ cancer/无 grading 规则
  │      │               └─ ★ subtype max_prob<0.5 → "recommend different ROI"
  │      │       state.critic_notes = [flag]           ← REPLACE
  │      │       critic_hook(1, output)                 ← CLI 打"round 1: 1 flag"
  │      │
  │      ├─ notes 非空 → 再走一次 planner
  │      │   ├─ Planner.plan(state)   ← ctx 里带上 state.critic_notes
  │      │   └─ _planner_response 看到 "different ROI" → 走 retry 分支
  │      │       返回 [region_view(retry_roi), subtype_classifier(retry_roi)]
  │      │   state.pending.extend(new_calls) + state.plan.extend(...)
  │      │   state.finished = False
  │      │   executor.run(state)   ← 再跑 2 步
  │      │
  │      └─ 轮 2: critic.review(state)
  │              └─ 现在有 2 次 subtype 尝试，best max_prob = 0.90 → 无 flag
  │              state.critic_notes = []                ← REPLACE 清空旧 flag
  │              critic_hook(2, empty)                   ← CLI 打"round 2: 0 flags"
  │              output.notes 空 → break
  │
  └─ [报告] report_builder.build(state)
         ├─ 遍历 state.executed 拼 findings（两次 subtype 尝试都进 Evidence）
         ├─ 组装 DiagnosticReport（primary_diagnosis / subtype / grade / mutations）
         └─ _audit(report)   ← 拒绝任何 evidence 引用不存在的 region_id/call_id
```

### 3️⃣ 返回

Orchestrator 返回 `(state, report)`；CLI 用 `rich.Table` 打印 DiagnosticReport + Findings。

---

## 一句话概括每一层的"任何时候都必须成立"的不变式

| 层 | 不变式 | 强制位置 |
|---|---|---|
| Schemas | ToolResult 非 error → 必带 uncertainty + ≥1 grounding | `schemas.py:ToolResult._at_least_one_grounding_if_no_error` |
| State | 状态是纯 JSON，任意步都能 dump/load 恢复 | `state.py:ExternalState`（pydantic 全字段可序列化） |
| Tools | 所有 tool 走同一个 `run(call, *, case_metadata) → ToolResult` 接口 | `tools/base.py:Tool` 抽象方法 |
| Executor | 每一步都 checkpoint，异常也不崩溃（写入 error 结果 + critic_note） | `agents/executor.py:run` 的 try/except |
| SelfCritic | REPLACE 语义，成功的 retry 一定能"清掉"过期的 flag | `agents/self_critic.py:review` |
| Report | 报告里 grounding 引用的 region_id 必须在 state 里出现过 | `agents/schema_report.py:_audit` |

这六条不变式就是这个 skeleton 相对普通 pipeline 的核心价值——**可控、可审计、可中断恢复**。
