# Pathology AI Copilot（骨架版）

病理诊断 Agent 的四层架构骨架实现。真实的 ViT / VLM / LLM 全部 mock 掉，重点跑通 **Planner → Executor → Self-Critic → Schema Report** 的分层协作、Tool Registry 统一接口、External Structured State 中断恢复、grounding 校验。

## 架构

```
┌───────────────────────────────────────────────────────────┐
│  AGENT   Planner  Executor  Self-Critic  Schema Report    │
├───────────────────────────────────────────────────────────┤
│  TOOLS   Vision Primitives · Domain Models ·              │
│          Knowledge Tools · VLM Tool                       │
│          （统一接口 + uncertainty + grounding）            │
├───────────────────────────────────────────────────────────┤
│  EXPERTS Cancer / Subtype / Grading / Mutation ... (36+)  │
├───────────────────────────────────────────────────────────┤
│  BASE    Pathology ViT   ·   Pathology VLM                │
└───────────────────────────────────────────────────────────┘
```

四层不变式（全部由代码强制，不是口头约定）：

- 每个 `ToolResult` 必须携带 `uncertainty ∈ [0,1]` 和至少一个 `Grounding` —— 由 `schemas.py` 的 `model_validator` 拒绝
- `ExternalState` 是纯 pydantic 对象，任意步可 `dump()` / `load()` 恢复
- Self-Critic 三条规则：`missing_grounding` / `high_uncertainty` / `inconsistency`（如有癌但没分级）
- `SchemaReportBuilder._audit` 拒绝任何 evidence 引用了 state 中不存在的 `region_id` 或 `call_id`——可审计

## 安装

需要 Python 3.9+。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest
```

## 快速上手

```bash
# 一次跑完 demo case
python -m pathology_copilot run --case demo/cases/demo1.json

# 中断 + 恢复（演示 checkpoint）
python -m pathology_copilot run --case demo/cases/demo1.json \
    --stop-after 3 --dump-state state.json
python -m pathology_copilot run --case demo/cases/demo1.json \
    --resume state.json

# 单测（19 项）
pytest -v
```

## 目录

```
pathology_copilot/
├── schemas.py         # ToolCall / ToolResult / Evidence / DiagnosticReport
├── state.py           # ExternalState：可 checkpoint / resume
├── llm.py             # MockLLMClient（planner / critic 的假推理）
├── config.py
├── base_models/       # Pathology ViT / VLM（mock）
├── experts/           # 4 个下游专家 + registry（36+ 任务扩展位）
├── tools/             # Tool Registry + vision / domain / knowledge / vlm
├── agents/            # Planner / Executor / SelfCritic / SchemaReportBuilder
├── orchestrator.py    # Planner→Executor→Critic→Report 编排
└── cli.py             # typer CLI
```

## 从 mock 换成真实模型

四个替换点，每个只改一个文件：

| 层 | 文件 | 替换成什么 |
|---|---|---|
| LLM（Planner / Critic）| `pathology_copilot/llm.py` | Anthropic / OpenAI / 本地 vLLM 客户端 |
| Pathology ViT | `pathology_copilot/base_models/pathology_vit.py` | 院内自研 DINOv3 ViT-L |
| Pathology VLM | `pathology_copilot/base_models/pathology_vlm.py` | Qwen2.5-VL + Dual Encoder |
| Expert | `pathology_copilot/experts/*.py` | 训练好的 MIL / 分类头 |

`MockLLMClient.complete(role, context, response_schema)` 的签名对齐了主流 LLM SDK，换实现只要保证同名方法返回同 pydantic schema 即可，Planner / Critic 无需改。

## 添加新的下游任务

以 MSI 状态预测为例：

```python
# 1. pathology_copilot/experts/msi_prediction.py
class MSIPrediction(ExpertModel):
    name = "msi_prediction"
    def predict(self, vit_output, case_metadata):
        return {"msi_status": "MSI-H", "probability": 0.82}

# 2. pathology_copilot/experts/registry.py
expert_registry.register(MSIPrediction())

# 3. pathology_copilot/tools/domain_models.py
class MSIPredictionTool(_ExpertTool):
    name = "msi_prediction"
    expert_name = "msi_prediction"
tool_registry.register(MSIPredictionTool())

# 4. 让 Planner 在合适时机调它——在 llm.py 里加一行 add(...)
```

## 测试覆盖

- `test_tools.py` — 注册表、每个 tool 输出 schema、grounding 强制约束
- `test_planner.py` — 阳性/阴性分支、replan 时跳过已执行工具
- `test_executor.py` — 完整跑通、中断恢复、失败被 critic 捕获
- `test_self_critic.py` — 三条规则各触发一次 + 干净场景无 flag
- `test_schema_report.py` — 端到端报告构建 + grounding 审计拒绝

## 不在骨架范围内

- 真实的 WSI 读图（`openslide`）
- Web UI / FastAPI 服务
- CI（默认没配）
- 36+ 任务全部实现（示例 4 个，registry 留了 plug-in 位）
