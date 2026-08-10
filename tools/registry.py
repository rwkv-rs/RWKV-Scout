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
        "task_plan",
        "task_point_id",
    }

    @classmethod
    def has(cls, name: str) -> bool:
        return str(name or "") in cls._tools

    @classmethod
    def metadata(cls, name: str) -> dict[str, Any]:
        return dict(cls._tools.get(str(name or ""), {}))

    @classmethod
    def capability_names(
        cls,
        capability: str,
        *,
        phase: str | None = None,
        model_visible_only: bool = False,
    ) -> list[str]:
        """Return registered tools implementing one backend capability."""

        requested = str(capability or "").strip()
        if not requested:
            return []
        return [
            name
            for name, meta in cls._tools.items()
            if requested in set(meta.get("capabilities") or ())
            and cls._phase_allows(meta, phase)
            and (not model_visible_only or bool(meta.get("model_visible")))
        ]

    @classmethod
    def can_execute(cls, name: str, phase: str | None = None) -> bool:
        meta = cls._tools.get(str(name or ""))
        if not meta:
            return False
        return cls._phase_allows(meta, phase)

    @staticmethod
    def _phase_allows(meta: dict[str, Any], phase: str | None) -> bool:
        """Apply the declared tool phase and retrieval role contract.

        ``phase=ALL`` is the executor's broad compatibility gate. The model
        catalog is filtered separately by ``model_visible_only`` so backend
        adapters do not become routing choices. Other phases remain explicit
        compatibility filters for callers that need them.
        """
        if phase is None:
            return True
        if str(phase).upper() == "GENERIC_WEB":
            # The open-web episode permits one provider-agnostic retrieval
            # capability plus the terminal actions. Provider adapters remain
            # registered for the backend transaction, never as model choices.
            name = str(meta.get("name") or "")
            return name in {
                "web_search",
                "connector_lookup",
                "calculator",
                "current_time",
                "date_diff",
                "finish_task",
            } and meta.get("phase") != "LEGACY"
        # Termination is a model-visible control action throughout the
        # retrieval episode.  Keeping it SYNTHESIS-only makes a successful
        # discovery/extraction turn unable to end cleanly, so the loop keeps
        # searching until a step or timeout guard fires.
        if str(meta.get("name") or "") == "finish_task":
            return str(phase).upper() in {
                "DISCOVERY",
                "EXTRACTION",
                "RECOVERY",
                "SYNTHESIS",
                "ALL",
            }
        if str(phase).upper() == "ALL":
            # Internal callers may execute any non-legacy registered tool in
            # this phase. Planner-facing visibility is handled separately by
            # model_visible metadata.
            return str(meta.get("phase") or "").upper() != "LEGACY"
        if str(meta.get("name") or "") == "date_diff" and str(phase).upper() in {
            "DISCOVERY",
            "EXTRACTION",
        }:
            # Date arithmetic belongs after the model has selected source
            # operands; it is not a discovery or page-extraction operation.
            return False
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
    def model_visible_names(cls, phase: str | None = None) -> list[str]:
        """Return only the curated public tool surface shown to the model.

        Provider adapters and workflow helpers remain registered so the
        executor and compatibility tests can use them, but they are not
        model-facing choices. This prevents the model from treating Tavily,
        Bing, or a page fetcher as separate research strategies.
        """
        # The public model surface is phase-independent. Internal execution
        # phases are no longer routing choices; the unified research loop
        # exposes one stable catalog on every decision.
        catalog_phase = "ALL" if phase is not None else None
        names = [
            name
            for name, meta in cls._tools.items()
            if meta.get("model_visible", False) and cls._phase_allows(meta, catalog_phase)
        ]
        public_order = {
            "finish_task": 0,
            "web_search": 1,
            "connector_lookup": 2,
            "calculator": 3,
            "date_diff": 4,
            "current_time": 5,
        }
        return sorted(names, key=lambda name: (public_order.get(name, 99), name))

    @classmethod
    def get_json_catalog(
        cls,
        phase: str | None = None,
        *,
        model_visible_only: bool = False,
    ) -> str:
        """Render the RWKV agent-loop tool catalog.

        rwkv-skills does not expose Python signatures as an ad-hoc prose
        prompt.  It gives the model a JSON catalog and asks for a
        ``name``/``arguments`` call.  Our registry predates full JSON schemas,
        so the registered signature remains the description while the
        arguments object is intentionally open-ended and validated by the
        concrete tool at execution time.
        """
        rows = []
        items = list(cls._tools.items())
        if model_visible_only:
            public_order = {
                "finish_task": 0,
                "web_search": 1,
                "connector_lookup": 2,
                "calculator": 3,
                "date_diff": 4,
                "current_time": 5,
            }
            items.sort(key=lambda item: (public_order.get(item[0], 99), item[0]))
        for name, meta in items:
            catalog_phase = "ALL" if model_visible_only and phase is not None else phase
            if not cls._phase_allows(meta, catalog_phase):
                continue
            if model_visible_only and not meta.get("model_visible", False):
                continue
            explicit_description = str(meta.get("description") or "").strip()
            signature = str(meta.get("signature") or "").strip()
            if explicit_description and signature and explicit_description != signature:
                # The short description is useful for routing, but RWKV also
                # needs the legacy signature's parameter contract.  Omitting
                # it makes the model invent backend arguments such as
                # ``max_results`` for the provider-agnostic web_search tool.
                model_description = f"{explicit_description}\n{signature}"
            else:
                model_description = explicit_description or signature
            rows.append(
                {
                    "name": name,
                    # ``description`` is a first-class model contract.  Keep
                    # the signature as a compatibility fallback for legacy
                    # registrations, but every model-visible entry must have
                    # a non-empty description before it reaches RWKV.
                    "description": model_description,
                    "arguments": meta.get("argument_schema") or {"type": "object", "additionalProperties": False},
                    "plugin": meta.get("plugin", ""),
                    "capabilities": list(meta.get("capabilities") or ()),
                    "retrieval_role": meta.get("retrieval_role", ""),
                    "phase": meta.get("phase", ""),
                    "category": meta.get("category", "internal"),
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
        model_visible: bool = False,
        category: str = "internal",
        description: str = "",
    ):
        def decorator(func: Callable):
            model_description = str(description or signature or "").strip()
            if model_visible and not model_description:
                raise ValueError(
                    f"model-visible tool '{name}' must declare a non-empty description"
                )
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
                "name": name,
                "func": func,
                "signature": signature,
                "description": model_description,
                "phase": phase,
                "plugin": plugin,
                "capabilities": capabilities_tuple,
                "retrieval_role": retrieval_role,
                "strict_args": strict_args,
                "model_visible": bool(model_visible),
                "category": str(category or "internal"),
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
            # Runtime context is system-owned.  It is never part of the
            # model's argument contract and cannot be overridden by a tool
            # call emitted by RWKV.
            allowed = set(meta.get("allowed_args") or ())
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

        # Keep model arguments and system context separate.  Context wins on
        # the final call as a defensive backstop for non-strict legacy tools;
        # strict tools already reject context keys above.
        merged_kwargs = {**arguments, **(context or {})}
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
    model_visible=True,
    category="control",
    description="End the current Agent tool loop with empty arguments; this only requests synthesis and does not write the final answer.",
    signature="""[Tool] finish_task
- 功能: 认为用户所有的目标已经完全达成，退出系统。
- 参数: 无"""
)
def _finish_task(agent_state=None, **kwargs):
    if agent_state:
        agent_state.is_finished = True
        agent_state.final_result = "任务已达成，流程正常结束。"
    return "执行结束"
