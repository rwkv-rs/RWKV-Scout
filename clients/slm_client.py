# RWKV-ECRA/clients/slm_client.py
import json
import requests
import time
import concurrent.futures
import contextvars
from config import (
    get_llm_model,
    get_llm_temperature,
    get_model_backend_name,
    get_model_chunk_requests_per_task,
    get_slm_concurrency,
    get_slm_endpoint,
    get_slm_password,
    get_slm_protocol,
)
from runtime import get_model_backend
from utils.chunker import get_token_count
from utils.concurrency import shutdown_pool, submit_with_context
from utils.runtime_gate import model_request_slot
from utils.token_tracker import current_task_id, global_token_tracker, model_lane

class SLMClient:
    def __init__(self, endpoint_override=None, password_override=None):
        self.headers = {"Content-Type": "application/json"}
        self._endpoint_override = endpoint_override
        self._password_override = password_override

    @property
    def endpoint(self):
        if self._endpoint_override is not None:
            return self._endpoint_override
        return get_slm_endpoint()

    @property
    def password(self):
        if self._password_override is not None:
            return self._password_override
        return get_slm_password()

    def batch_generate(self, contents: list[str], tracker=None, task_id: str = None) -> list[str]:
        if not contents:
            return []

        token = current_task_id.set(task_id or current_task_id.get())
        try:
            return self._batch_generate_with_context(contents, tracker=tracker, task_id=task_id)
        finally:
            current_task_id.reset(token)

    def _batch_generate_with_context(self, contents: list[str], tracker=None, task_id: str = None) -> list[str]:
            
        # 1. 🚀 发送前：立即进行输入 Token 本地计算与计费拦截
        total_in_tokens = sum(get_token_count(c) for c in contents)
        global_token_tracker.add_slm(total_in_tokens, 0, task_id=task_id)

        # 🚀 发起真实的网络请求
        results = self._batch_generate_direct(contents)
        
        # 2. 📥 得到回复后：统计实际成功生成的输出 Token 并追加计费
        total_out_tokens = sum(get_token_count(r) for r in results)
        global_token_tracker.add_slm(0, total_out_tokens, task_id=task_id)

        if tracker:
            for idx, text in enumerate(results):
                tracker.track_slm(input_prompt=contents[idx], output_text=text, task_id=task_id)
        return results

    def _batch_generate_direct(self, contents: list[str]) -> list[str]:
        # SLM calls are evidence/chunk work. Keep them out of the reserved
        # control lane even when the backend implementation is shared with
        # planner and synthesis requests.
        with model_lane("chunk"):
            return self._batch_generate_direct_impl(contents)

    def _batch_generate_direct_impl(self, contents: list[str]) -> list[str]:
        selected_backend = get_model_backend_name()
        if selected_backend in {"direct_rwkv", "auto"}:
            backend = get_model_backend()
            if selected_backend == "direct_rwkv" or getattr(backend, "backend_name", "") == "direct_rwkv":
                return backend.batch_text_completion(contents, max_tokens=2400)

        if get_slm_protocol() == "openai":
            return self._openai_generate(contents)

        payload = {
            "contents": contents,
            "max_tokens": 2400,       
            "temperature": 1.0,       
            "top_k": 20,
            "top_p": 0.95,
            "alpha_presence": 2.0,    
            "alpha_frequency": 0.0,   
            "alpha_decay": 0.99,
            "stream": True,
            "password": self.password
        }

        # 1. 动态超时计算
        # 基础通信保障时间
        BASE_TIMEOUT_SECONDS = 60.0  
        # 每1000个Token额外增加的等待预留时间（覆盖缓慢的 prefill 阶段）
        PER_1K_TOKENS_TIMEOUT_SECONDS = 45.0 
        
        total_tokens = sum(get_token_count(c) for c in contents)
        dynamic_timeout = BASE_TIMEOUT_SECONDS + (total_tokens / 1000.0) * PER_1K_TOKENS_TIMEOUT_SECONDS
        
        results = {i: "" for i in range(len(contents))}
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                print(f"🚀 [SLM Client] 发射批次 (大小: {len(contents)}, 总计: {total_tokens} Tokens)。动态超时设置为 {dynamic_timeout:.1f} 秒。")
                response = requests.post(self.endpoint, json=payload, headers=self.headers, stream=True, timeout=dynamic_timeout)
                if response.status_code != 200:
                    raise RuntimeError(f"HTTP {response.status_code} - {response.content.decode('utf-8', errors='ignore')}")

                received_valid_chunk = False 

                for line in response.iter_lines():
                    if not line: continue
                    decoded = line.decode('utf-8', errors='replace').strip()
                    if "error" in decoded.lower() and not decoded.startswith("data:"):
                        raise RuntimeError(f"【后端业务致命报错】: {decoded}")

                    json_str = ""
                    if decoded.startswith("data:"):
                        json_str = decoded[5:].strip()
                    elif decoded.startswith("{"):
                        json_str = decoded
                        
                    if json_str == "[DONE]" or not json_str: continue

                    try:
                        data = json.loads(json_str)
                        if "error" in data:
                            raise RuntimeError(f"【数据流异常】: {data['error']}")
                            
                        for choice in data.get("choices", []):
                            raw_idx = choice.get("index")
                            if raw_idx is not None:
                                idx = int(raw_idx)
                                delta = choice.get("delta", {})
                                content = delta.get("content", choice.get("text", ""))
                                if idx in results and content:
                                    results[idx] += content
                                    received_valid_chunk = True
                    except json.JSONDecodeError:
                        pass
                    except Exception as e:
                        raise e 

                if not received_valid_chunk and len(contents) > 0:
                    if attempt < max_retries - 1:
                        print("⚠️ [SLM Client] 批次处理未返回任何有效数据块，可能是空内容或连接问题，将触发重试。")
                        time.sleep(2)
                        continue

                final_responses = [results[i] for i in range(len(contents))]
                return final_responses
                
            except requests.exceptions.RequestException as e:
                # 2. 优化重试日志，指示具体重试次数
                print(f"⚠️ [SLM Client] 批次 (大小: {len(contents)}) 传输超时或网络异常，触发重试 ({attempt + 1}/{max_retries})... 异常: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"网络异常，重试 {max_retries} 次后彻底失败: {str(e)}")
            except RuntimeError as e:
                raise e

    def _openai_generate(self, contents: list[str]) -> list[str]:
        """Call a local OpenAI-compatible server one prompt at a time.

        The OpenAI-compatible endpoint does not accept the legacy RWKV
        `contents` batch payload, so the local path preserves ordering while
        using a small bounded worker pool.
        """

        def generate_one(content: str) -> str:
            request_headers = dict(self.headers)
            if self.password:
                request_headers["Authorization"] = f"Bearer {self.password}"
            payload = {
                "model": get_llm_model() or "rwkv7-g1h-1.5b",
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 2400,
                "temperature": get_llm_temperature(),
                "stream": False,
            }
            with model_request_slot(current_task_id.get() or "slm-request", lane="chunk"):
                response = requests.post(self.endpoint, json=payload, headers=request_headers, timeout=180)
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code} - {response.text[:1000]}")
            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError("本地模型未返回 choices")
            message = choices[0].get("message") or {}
            return message.get("content") or choices[0].get("text", "")

        worker_count = min(
            max(1, get_slm_concurrency()),
            get_model_chunk_requests_per_task(),
            len(contents),
        )
        results = [""] * len(contents)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=worker_count)
        futures = {
            submit_with_context(executor, generate_one, content): index
            for index, content in enumerate(contents)
        }
        cancelled = False
        try:
            for future in concurrent.futures.as_completed(futures):
                results[futures[future]] = future.result()
        except concurrent.futures.TimeoutError:
            # Do not wait on active RWKV requests after the outer case has
            # expired; the runner owns the answer-shaped timeout fallback.
            cancelled = True
            raise
        finally:
            shutdown_pool(executor, list(futures), cancelled=cancelled)
        return results
