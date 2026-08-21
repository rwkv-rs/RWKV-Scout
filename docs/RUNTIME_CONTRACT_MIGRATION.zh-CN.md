# RWKV-ECRA 正式运行态合约与全局迁移

## Task Plan 唯一运行态合约

```json
{
  "contract": "rwkv.ecra.runtime.task-plan",
  "goal": "用户原始目标",
  "records": [
    {
      "record_id": "P1",
      "question": "该记录需要回答的问题",
      "subject": "稳定主体",
      "relation": "记录身份/关系",
      "fields": [
        {"field_id": "P1:F1", "name": "version"}
      ],
      "time_scope": "current|historical|timeless|unspecified",
      "set_semantics": "single|collection|possibly_empty",
      "premise_requires_verification": false
    }
  ]
}
```

`record_id` 和 `field_id` 由 Controller 按顺序分配，是 Task Planner、Retrieval Query Plan、Evidence Ledger、Evidence Resolution、Evidence Review、Plan Revision 和 Answer Writer 之间的传输身份。任何阶段都不得重新命名、猜测或用局部别名替代它们。

所有正式对象统一使用 `contract: "rwkv.ecra.runtime.<domain>"`。名字表达领域语义，不表达实验轮次；生产代码不再生成 `schema_version` 或 `.v1/.v2/.v4`。旧名字只存在于迁移器的只读识别表与不可变审计原文中。

## 服务器迁移

先做只读审计：

```bash
.venv/bin/python scripts/migrate_runtime_contracts.py outputs data/output training/retrieval_rst
```

确认 `errors` 为空后原子写回：

```bash
.venv/bin/python scripts/migrate_runtime_contracts.py --write outputs data/output training/retrieval_rst
```

迁移器只改写 JSON/JSONL 中的结构化对象：Task Plan、已知运行态合约名、运行态 Evidence Review、离线 `oracle_evidence_assessment`，以及同一文档里的 record/field 引用。`prompt`、`raw_model_output` 等字符串是历史审计原文，不会被伪造成新协议输出。

## 上线前验证

```bash
python3 -m compileall -q agent tools utils scripts
.venv/bin/pytest -q -s
```

默认 `retrieval_query_plan_enabled=false`。完成冻结回放后，再单独决定是否启用查询扩展。
