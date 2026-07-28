# RWKV-ECRA/tools/registry.py
import inspect
import json
from typing import Callable, Dict, Any, Iterable, get_origin

from retrieval_plugins import PluginRegistry, error_result, normalize_result


def _json_type(parameter: inspect.Parameter) -> str:
    annotation = parameter.annotation
    origin = get_origin(annotation)
    if annotation is bool:
        return "boolean"
    if annotation is int:
        return "integer"
    if annotation is float:
        return "number"
    if annotation is list or origin in {list, tuple, set}:
        return "array"
    if annotation is dict or origin is dict:
        return "object"
    if parameter.default is not inspect.Parameter.empty:
        if isinstance(parameter.default, bool):
            return "boolean"
        if isinstance(parameter.default, int):
            return "integer"
        if isinstance(parameter.default, float):
            return "number"
        if isinstance(parameter.default, (list, tuple, set)):
            return "array"
        if isinstance(parameter.default, dict):
            return "object"
    return "string"

class ToolRegistry:
    _tools: Dict[str, Dict[str, Any]] = {}
    _runtime_context_keys = {
        "original_goal",
        "path_to_id",
        "id_to_path",
        "working_memory",
        "tracker",
        "agent_state",
        "task_id",
        "slm_scheduler",
        "agentic_tool_loop",
        "run_metadata",
    }

    @classmethod
    def has(cls, name: str) -> bool:
        return str(name or "") in cls._tools

    @classmethod
    def metadata(cls, name: str) -> dict[str, Any]:
        return dict(cls._tools.get(str(name or ""), {}))

    @classmethod
    def can_execute(cls, name: str, phase: str | None = None) -> bool:
        meta = cls._tools.get(str(name or ""))
        if not meta:
            return False
        return cls._phase_allows(meta, phase)

    @staticmethod
    def _phase_allows(meta: dict[str, Any], phase: str | None) -> bool:
        """Apply the declared tool phase and retrieval role contract.

        ``phase=ALL`` exposes the complete non-legacy catalog so the model can
        choose discovery, evidence, or synthesis tools itself. Other phases
        remain explicit compatibility filters for callers that need them.
        """
        if phase is None:
            return True
        if str(phase).upper() == "ALL":
            # The model-owned retrieval episode intentionally exposes the
            # complete retrieval catalog.  Discovery/evidence sequencing is
            # a model decision in this mode; the step budget remains the
            # execution boundary.
            return str(meta.get("phase") or "").upper() != "LEGACY"
        if meta.get("phase") not in {phase, "ALL"}:
            return False
        role = str(meta.get("retrieval_role") or "").strip().casefold()
        if role == "discovery":
            return phase in {"DISCOVERY", "RECOVERY"}
        if role == "evidence":
            return phase in {"EXTRACTION", "SYNTHESIS", "RECOVERY"}
        return True

    @classmethod
    def names(cls, phase: str | None = None) -> list[str]:
        return [
            name
            for name, meta in cls._tools.items()
            if cls._phase_allows(meta, phase)
        ]

    @classmethod
    def get_json_catalog(cls, phase: str | None = None) -> str:
        """Render the RWKV agent-loop tool catalog.

        rwkv-skills does not expose Python signatures as an ad-hoc prose
        prompt.  It gives the model a JSON catalog and asks for a
        ``name``/``arguments`` call.  Our registry predates full JSON schemas,
        so the registered signature remains the description while the
        arguments object is intentionally open-ended and validated by the
        concrete tool at execution time.
        """
        rows = []
        for name, meta in cls._tools.items():
            if not cls._phase_allows(meta, phase):
                continue
            rows.append(
                {
                    "name": name,
                    "description": str(meta.get("signature") or "").strip(),
                    "arguments": meta.get("argument_schema") or {"type": "object", "additionalProperties": False},
                    "plugin": meta.get("plugin", ""),
                    "capabilities": list(meta.get("capabilities") or ()),
                    "retrieval_role": meta.get("retrieval_role", ""),
                    "phase": meta.get("phase", ""),
                }
            )
        return json.dumps(rows, ensure_ascii=False, indent=2)

    @classmethod
    def register(
        cls,
        name: str,
        phase: str,
        signature: str,
        *,
        plugin: str = "",
        capabilities: Iterable[str] = (),
        retrieval_role: str = "",
        strict_args: bool = True,
    ):
        def decorator(func: Callable):
            capabilities_tuple = tuple(str(item) for item in capabilities)
            allowed = []
            argument_types = {}
            required = []
            try:
                parameters = inspect.signature(func).parameters
                for parameter_name, parameter in parameters.items():
                    if parameter_name in cls._runtime_context_keys or parameter.kind in {
                        inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD,
                    }:
                        continue
                    allowed.append(parameter_name)
                    argument_types[parameter_name] = _json_type(parameter)
                    if parameter.default is inspect.Parameter.empty:
                        required.append(parameter_name)
            except (TypeError, ValueError):
                allowed = []
                argument_types = {}
                required = []
            if plugin:
                PluginRegistry.register(
                    plugin,
                    label=plugin,
                    capabilities=capabilities_tuple,
                    tools=(name,),
                )
            cls._tools[name] = {
                "func": func,
                "signature": signature,
                "phase": phase,
                "plugin": plugin,
                "capabilities": capabilities_tuple,
                "retrieval_role": retrieval_role,
                "strict_args": strict_args,
                "allowed_args": tuple(allowed),
                "required_args": tuple(required),
                "argument_schema": {
                    "type": "object",
                    "properties": {key: {"type": argument_types.get(key, "string")} for key in allowed},
                    "required": required,
                    "additionalProperties": False,
                },
            }
            return func
        return decorator

    @classmethod
    def get_interfaces_by_phase(cls, phase: str) -> str:
        """渐进式披露：向大模型展示当前阶段可用的高度抽象的接口卡片"""
        lines = [f"### 当前可用工具接口 (Phase: {phase})"]
        for name, meta in cls._tools.items():
            if cls._phase_allows(meta, phase):
                lines.append(meta["signature"] + "\n")
        return "\n".join(lines)

    @classmethod
    def execute(
        cls,
        action: str,
        args: Dict[str, Any],
        context: Dict[str, Any],
        *,
        phase: str | None = None,
    ) -> str:
        """Execute a tool under optional phase and argument-contract checks."""
        name = str(action or "")
        meta = cls._tools.get(name)
        if not meta:
            unknown = error_result(
                provider="",
                role="discovery",
                message=f"tool is not registered: {name}",
                error_class="unknown_tool",
            )
            unknown["tool"] = name
            return json.dumps(unknown, ensure_ascii=False)
        if not cls._phase_allows(meta, phase):
            return json.dumps(
                error_result(
                    provider=meta.get("plugin", ""),
                    role=meta.get("retrieval_role") or "discovery",
                    message=f"tool '{name}' is not allowed in phase '{phase}'",
                    error_class="tool_not_allowed_in_phase",
                ),
                ensure_ascii=False,
            )

        arguments = dict(args or {})
        if meta.get("strict_args", True):
            allowed = set(meta.get("allowed_args") or ()) | cls._runtime_context_keys
            unknown = sorted(set(arguments) - allowed)
            missing = sorted(set(meta.get("required_args") or ()) - set(arguments))
            if unknown or missing:
                protocol_error = error_result(
                    provider=meta.get("plugin", ""),
                    query=str(arguments.get("query") or arguments.get("url") or ""),
                    role=meta.get("retrieval_role") or "discovery",
                    message="tool arguments do not match the registered contract",
                    error_class="tool_protocol",
                )
                protocol_error.update(
                    {
                        "unknown_arguments": unknown,
                        "missing_arguments": missing,
                        "tool": name,
                    }
                )
                return json.dumps(
                    protocol_error,
                    ensure_ascii=False,
                )

        merged_kwargs = {**(context or {}), **arguments}
        try:
            result = meta["func"](**merged_kwargs)
        except Exception as exc:
            result = json.dumps(
                error_result(
                    provider=meta.get("plugin", ""),
                    query=str(arguments.get("query") or arguments.get("url") or ""),
                    role=meta.get("retrieval_role") or "discovery",
                    message=f"{type(exc).__name__}: {exc}"[:500],
                    error_class="tool_execution",
                ),
                ensure_ascii=False,
            )

        if meta.get("plugin"):
            try:
                observed = json.loads(result) if isinstance(result, str) else result
            except (TypeError, json.JSONDecodeError):
                observed = {"status": "ok", "raw": str(result)}
            if meta.get("retrieval_role"):
                observed = normalize_result(
                    observed,
                    provider=meta.get("plugin", ""),
                    query=str(arguments.get("query") or arguments.get("url") or ""),
                    role=meta.get("retrieval_role") or "discovery",
                )
                result = json.dumps(observed, ensure_ascii=False, indent=2)
            PluginRegistry.observe(meta["plugin"], action=name, result=observed)
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)

@ToolRegistry.register(
    name="finish_task",
    phase="SYNTHESIS",
    signature="""[Tool] finish_task
- 功能: 认为用户所有的目标已经完全达成，退出系统。
- 参数: 无"""
)
def _finish_task(agent_state=None, **kwargs):
    if agent_state:
        agent_state.is_finished = True
        agent_state.final_result = "任务已达成，流程正常结束。"
    return "执行结束"
