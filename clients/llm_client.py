# RWKV-ECRA/clients/llm_client.py
from types import SimpleNamespace

import requests
import httpx
from openai import OpenAI
from config import get_llm_api_key, get_llm_base_url, get_llm_provider, get_llm_model, LLM_ENDPOINTS, is_local_provider
from utils.retry import retry_with_fallback
from utils.token_tracker import global_token_tracker

class LLMClient:
    def __init__(self):
        pass

    @property
    def provider(self):
        return get_llm_provider()

    @property
    def model(self):
        return get_llm_model()
        
    @property
    def client(self):
        # The local vLLM endpoint is reached through the WSL bridge.  The
        # Windows process may inherit an HTTP(S) proxy which makes httpx hang
        # even though direct requests to the bridge are healthy.
        http_client = None
        if is_local_provider(self.provider):
            http_client = httpx.Client(trust_env=False, timeout=httpx.Timeout(60.0, connect=10.0))
        return OpenAI(
            api_key=get_llm_api_key() or ("local" if self.provider == "local" else ""),
            base_url=get_llm_base_url(),
            timeout=60.0,
            http_client=http_client,
        )

    @retry_with_fallback(max_retries=3, delay=3)
    def chat_completion(
        self,
        messages: list,
        tools: list = None,
        enable_native_search: bool = False,
        max_tokens: int | None = None,
    ):
        kwargs = {
            "model": self.model,
            "messages": messages,
            "stream": False
        }
        
        provider_config = LLM_ENDPOINTS.get(self.provider, {})

        if self.provider == "baidu":
            kwargs["max_completion_tokens"] = provider_config.get("max_completion_tokens", 65536)
            
            # 🔴 全量开启文心自带的网络搜索、溯源角标及来源追踪
            if enable_native_search and provider_config.get("enable_web_search", False):
                kwargs["extra_body"] = {
                    "web_search": {
                        "enable": True,
                        "enable_citation": True,
                        "enable_trace": True
                    }
                }
                
        elif self.provider == "volcengine":
            reasoning = provider_config.get("reasoning_effort")
            if reasoning:
                kwargs["extra_body"] = {"reasoning_effort": reasoning}
        elif is_local_provider(self.provider):
            kwargs["max_tokens"] = max_tokens or provider_config.get("max_tokens", 768)
            kwargs["temperature"] = provider_config.get("temperature", 0.2)

            # Use requests for the local WSL bridge.  The OpenAI SDK's httpx
            # transport stalls in this Windows/WSL environment even though
            # direct requests to the same vLLM endpoint are healthy.
            endpoint = get_llm_base_url().rstrip("/") + "/chat/completions"
            headers = {
                "Authorization": f"Bearer {get_llm_api_key() or 'local'}",
                "Content-Type": "application/json",
            }
            session = requests.Session()
            session.trust_env = False
            response = session.post(endpoint, headers=headers, json=kwargs, timeout=(10, 120))
            response.raise_for_status()
            payload = response.json()
            choice = (payload.get("choices") or [{}])[0]
            raw_message = choice.get("message") or {}
            message = SimpleNamespace(
                role=raw_message.get("role", "assistant"),
                content=raw_message.get("content", "") or "",
                tool_calls=raw_message.get("tool_calls"),
                search_results=payload.get("search_results", []),
            )
            usage = payload.get("usage") or {}
            global_token_tracker.add_llm(
                usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0,
                usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0,
                0,
            )
            return message

        if tools: 
            kwargs["tools"] = tools
            kwargs["tool_choice"] = {"type": "function", "function": {"name": "system_router"}}
            
        resp = self.client.chat.completions.create(**kwargs)
        
        # 🌟 拦截提取 Token 消耗（兼容 OpenAI 标准格式和火山定制返回）
        if hasattr(resp, 'usage') and resp.usage:
            try:
                # 兼容不同 pydantic 版本的结构解析
                if hasattr(resp.usage, "model_dump"):
                    usage_dict = resp.usage.model_dump()
                elif hasattr(resp.usage, "__dict__"):
                    usage_dict = vars(resp.usage)
                else:
                    usage_dict = resp.usage if isinstance(resp.usage, dict) else {}
                
                in_tok = usage_dict.get("prompt_tokens") or usage_dict.get("input_tokens") or 0
                out_tok = usage_dict.get("completion_tokens") or usage_dict.get("output_tokens") or 0
                
                # 深入提取火山引擎的思考 Token (reasoning_tokens)
                details = usage_dict.get("completion_tokens_details") or usage_dict.get("output_tokens_details") or {}
                reasoning_tok = details.get("reasoning_tokens") or 0
                
                global_token_tracker.add_llm(in_tok, out_tok, reasoning_tok)
            except Exception as e:
                print(f"⚠️ [Token Tracker] 大模型用量解析失败: {e}")
        
        # 🔴 拦截解析：提取附加的溯源信息（兼容 OpenAI SDK 对 extra_body 响应的处理）
        msg = resp.choices[0].message
        raw_dict = resp.model_dump()
        msg.search_results = raw_dict.get("search_results", [])
        
        return msg

    def text_completion(self, prompt: str, max_tokens: int = 768):
        """Run a raw completion against the local vLLM endpoint.

        The RWKV checkpoint is trained around a plain ``User: ...\nAssistant:``
        transcript.  Sending that transcript through the OpenAI chat template
        makes the 1.5B model emit a visible reasoning preamble, while the raw
        completion endpoint follows the same prompt reliably.  Keep this
        method local-only so external providers retain their existing path.
        """
        if not is_local_provider(self.provider):
            raise RuntimeError("text_completion is only supported by the local provider")
        endpoint = get_llm_base_url().rstrip("/") + "/completions"
        headers = {
            "Authorization": f"Bearer {get_llm_api_key() or 'local'}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max(1, int(max_tokens)),
            "temperature": 0.2,
            "stream": False,
        }
        session = requests.Session()
        session.trust_env = False
        response = session.post(endpoint, headers=headers, json=payload, timeout=(10, 180))
        response.raise_for_status()
        data = response.json()
        choice = (data.get("choices") or [{}])[0]
        usage = data.get("usage") or {}
        global_token_tracker.add_llm(
            usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0,
            usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0,
            0,
        )
        return SimpleNamespace(content=choice.get("text", ""), usage=usage)
