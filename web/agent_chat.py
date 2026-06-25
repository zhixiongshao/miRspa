"""miRspa Agent：多厂商 OpenAI 兼容 API + 本地 miRspa / miRspa-RNA 推理工具。

API Key 仅随请求透传至上游 LLM，服务端不持久化。
"""
from __future__ import annotations

import json
import re
import ssl
import urllib.error
import urllib.request
from typing import Any, Callable

AGENT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "predict_mirspa_dna",
            "description": (
                "Run miRspa DNA coordinate model on a genomic interval (1-based closed). "
                "Returns hairpin score/prob, seq_id, MFE, and stem sequence. "
                "Use when user asks for DNA miRspa score without RNA-seq coverage."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chrom": {"type": "string", "description": "Chromosome, e.g. chr17 or 17"},
                    "start": {"type": "integer", "description": "1-based start (inclusive)"},
                    "end": {"type": "integer", "description": "1-based end (inclusive)"},
                    "strand": {"type": "string", "enum": ["+", "-"], "description": "Strand"},
                },
                "required": ["chrom", "start", "end", "strand"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_mirspa_rna",
            "description": (
                "Run miRspa-RNA model: genomic interval + per-position coverage depths "
                "(length must equal end-start+1) + mean_cov_divisor. "
                "Returns prob, instability, 5p/3p mature regions if detected."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chrom": {"type": "string"},
                    "start": {"type": "integer", "description": "1-based start (inclusive)"},
                    "end": {"type": "integer", "description": "1-based end (inclusive)"},
                    "strand": {"type": "string", "enum": ["+", "-"]},
                    "depths": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Per-base coverage reads, one per position in the interval",
                    },
                    "mean_cov_divisor": {
                        "type": "number",
                        "description": "Mean coverage divisor (e.g. 100 for esoph, 1000 for others, 10000 default)",
                    },
                },
                "required": ["chrom", "start", "end", "strand", "depths", "mean_cov_divisor"],
            },
        },
    },
]

PROVIDER_PRESETS: dict[str, dict[str, Any]] = {
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"],
    },
    "minimax": {
        "label": "MiniMax",
        "base_url": "https://api.minimax.chat/v1",
        "default_model": "MiniMax-M2.5",
        "models": ["MiniMax-M2.5", "MiniMax-M2.5-highspeed", "abab6.5s-chat"],
    },
    "deepseek": {
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "models": ["deepseek-chat", "deepseek-reasoner"],
    },
    "moonshot": {
        "label": "Moonshot (Kimi)",
        "base_url": "https://api.moonshot.cn/v1",
        "default_model": "moonshot-v1-8k",
        "models": ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
    },
    "zhipu": {
        "label": "Zhipu (GLM)",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "default_model": "glm-4-flash",
        "models": ["glm-4-flash", "glm-4-air", "glm-4-plus"],
    },
    "openai_compatible": {
        "label": "OpenAI-compatible (custom base URL)",
        "base_url": "",
        "default_model": "",
        "models": [],
    },
}

SYSTEM_PROMPT = """You are miRspa Agent, a scientific assistant for miRNA hairpin prediction.

You may ONLY help users by calling these tools:
- predict_mirspa_dna: DNA structural model (coordinates only)
- predict_mirspa_rna: RNA model (coordinates + per-base coverage depths + mean_cov_divisor)

Rules:
1. Never invent prediction scores, coordinates, or mature miRNA sequences — always use tool results.
2. Coordinates are 1-based closed intervals on GRCh38 unless the user says otherwise.
3. For miRspa-RNA, depths array length MUST equal (end - start + 1). Ask the user if depths are missing.
4. Explain results clearly: score/prob, MFE, instability, 5p/3p regions when present.
5. Keep answers concise and scientific. Use the same language as the user (Chinese or English).
"""

_MAX_TOOL_ROUNDS = 6
_HTTP_TIMEOUT_S = 120.0


def list_providers() -> dict[str, Any]:
    return {
        "ok": True,
        "providers": [
            {
                "id": pid,
                "label": preset["label"],
                "base_url": preset["base_url"],
                "default_model": preset["default_model"],
                "models": preset["models"],
            }
            for pid, preset in PROVIDER_PRESETS.items()
        ],
    }


def _normalize_base_url(raw: str) -> str:
    base = str(raw or "").strip().rstrip("/")
    if not base:
        raise ValueError("base_url 不能为空")
    if not base.endswith("/v1"):
        if base.endswith("/v4"):
            return base
        return f"{base}/v1" if "/v1" not in base else base
    return base


def resolve_provider_config(
    *,
    provider: str,
    base_url: str | None = None,
    model: str | None = None,
) -> tuple[str, str]:
    pid = str(provider or "").strip().lower() or "openai"
    preset = PROVIDER_PRESETS.get(pid)
    if preset is None:
        raise ValueError(f"不支持的 provider: {provider}")

    if pid == "openai_compatible":
        if not str(base_url or "").strip():
            raise ValueError("OpenAI-compatible 模式需要填写 base URL")
        base = _normalize_base_url(base_url)
    else:
        base = _normalize_base_url(str(base_url or preset["base_url"]))

    mdl = str(model or preset["default_model"] or "").strip()
    if not mdl:
        raise ValueError("请选择或填写 model")
    return base, mdl


def _http_post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            **headers,
        },
    )
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S, context=ctx) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        try:
            err_json = json.loads(err_body)
            msg = err_json.get("error", err_json)
            if isinstance(msg, dict):
                msg = msg.get("message") or json.dumps(msg, ensure_ascii=False)
        except Exception:
            msg = err_body[:500] or str(e)
        raise ValueError(f"上游 API HTTP {e.code}: {msg}") from e
    except urllib.error.URLError as e:
        raise ValueError(f"无法连接上游 API: {e.reason}") from e

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"上游返回非 JSON: {raw[:300]}") from e


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1") or base.endswith("/v4"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def call_chat_completions(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = "auto",
) -> dict[str, Any]:
    key = str(api_key or "").strip()
    if not key:
        raise ValueError("api_key 不能为空")

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice

    return _http_post_json(
        _chat_completions_url(base_url),
        payload,
        headers={"Authorization": f"Bearer {key}"},
    )


def verify_api_key(
    *,
    provider: str,
    api_key: str,
    base_url: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    base, mdl = resolve_provider_config(provider=provider, base_url=base_url, model=model)
    resp = call_chat_completions(
        base_url=base,
        api_key=api_key,
        model=mdl,
        messages=[
            {"role": "system", "content": "Reply with exactly: ok"},
            {"role": "user", "content": "ping"},
        ],
        tools=None,
    )
    content = ""
    try:
        content = str(resp["choices"][0]["message"].get("content") or "")
    except (KeyError, IndexError, TypeError):
        pass
    return {
        "ok": True,
        "provider": provider,
        "base_url": base,
        "model": mdl,
        "message": "API key verified",
        "sample_reply": content[:200],
    }


def _compact_dna_result(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": data.get("ok", True),
        "model": "miRspa-DNA",
        "score": data.get("score"),
        "seq_id": data.get("seq_id"),
        "chrom": data.get("chrom"),
        "start_1b": data.get("start_1b"),
        "end_1b": data.get("end_1b"),
        "strand": data.get("strand"),
        "mfe": data.get("mfe"),
        "original_sequence_rna": data.get("original_sequence"),
        "secondary_structure": data.get("secondary_structure"),
    }


def _compact_rna_result(data: dict[str, Any]) -> dict[str, Any]:
    regions = []
    for reg in data.get("mature_regions") or []:
        if not isinstance(reg, dict):
            continue
        regions.append(
            {
                "arm": reg.get("arm"),
                "genomic": f"{reg.get('chrom')}:{reg.get('genomic_start_1b')}-{reg.get('genomic_end_1b')}:{reg.get('strand')}",
                "sequence_rna": reg.get("sequence_rna"),
                "mean_prob": reg.get("mean_prob"),
                "peak_prob": reg.get("peak_prob"),
            }
        )
    return {
        "ok": data.get("ok", True),
        "model": "miRspa-RNA",
        "prob": data.get("prob"),
        "seq_id": data.get("seq_id"),
        "chrom": data.get("chrom"),
        "start_1b": data.get("start_1b"),
        "end_1b": data.get("end_1b"),
        "strand": data.get("strand"),
        "mfe_raw": data.get("mfe_raw"),
        "instability": data.get("instability"),
        "mean_cov_divisor": data.get("mean_cov_divisor"),
        "mature_regions": regions,
    }


def _parse_tool_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return json.loads(raw)
    raise ValueError("tool arguments 无效")


def execute_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    predict_dna: Callable[..., dict[str, Any]],
    predict_rna: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    if name == "predict_mirspa_dna":
        return _compact_dna_result(
            predict_dna(
                chrom=str(arguments["chrom"]),
                start_1b=int(arguments["start"]),
                end_1b=int(arguments["end"]),
                strand=str(arguments.get("strand", "+")),
            )
        )
    if name == "predict_mirspa_rna":
        depths = arguments.get("depths")
        if isinstance(depths, str):
            depths = [float(x) for x in re.split(r"[\s,;]+", depths.strip()) if x]
        return _compact_rna_result(
            predict_rna(
                chrom=str(arguments["chrom"]),
                start_1b=int(arguments["start"]),
                end_1b=int(arguments["end"]),
                strand=str(arguments.get("strand", "+")),
                depths=depths,
                mean_cov_divisor=float(arguments.get("mean_cov_divisor", 10000.0)),
            )
        )
    raise ValueError(f"未知工具: {name}")


def run_agent_chat(
    *,
    provider: str,
    api_key: str,
    messages: list[dict[str, Any]],
    base_url: str | None = None,
    model: str | None = None,
    predict_dna: Callable[..., dict[str, Any]],
    predict_rna: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    base, mdl = resolve_provider_config(provider=provider, base_url=base_url, model=model)
    key = str(api_key or "").strip()
    if not key:
        raise ValueError("api_key 不能为空")

    convo: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "").strip()
        if role not in ("user", "assistant"):
            continue
        entry: dict[str, Any] = {"role": role}
        if msg.get("content") is not None:
            entry["content"] = str(msg["content"])
        convo.append(entry)

    tool_trace: list[dict[str, Any]] = []
    assistant_text = ""

    for _ in range(_MAX_TOOL_ROUNDS):
        resp = call_chat_completions(
            base_url=base,
            api_key=key,
            model=mdl,
            messages=convo,
            tools=AGENT_TOOLS,
            tool_choice="auto",
        )
        try:
            choice = resp["choices"][0]
            msg = choice["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"上游响应格式异常: {json.dumps(resp, ensure_ascii=False)[:400]}") from e

        tool_calls = msg.get("tool_calls") or []
        assistant_text = str(msg.get("content") or "")

        if not tool_calls:
            convo.append({"role": "assistant", "content": assistant_text})
            break

        assistant_entry: dict[str, Any] = {"role": "assistant", "content": assistant_text or None}
        assistant_entry["tool_calls"] = tool_calls
        convo.append(assistant_entry)

        for tc in tool_calls:
            fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
            tool_name = str(fn.get("name") or "")
            tool_id = str(tc.get("id") or tool_name)
            try:
                args = _parse_tool_args(fn.get("arguments"))
                result = execute_tool(
                    tool_name,
                    args,
                    predict_dna=predict_dna,
                    predict_rna=predict_rna,
                )
                tool_content = json.dumps(result, ensure_ascii=False)
                tool_trace.append({"tool": tool_name, "arguments": args, "result": result})
            except Exception as e:
                tool_content = json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)
                tool_trace.append({"tool": tool_name, "error": str(e)})

            convo.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": tool_content,
                }
            )
    else:
        raise ValueError("工具调用轮次过多，请缩小问题范围后重试")

    return {
        "ok": True,
        "provider": provider,
        "model": mdl,
        "message": assistant_text,
        "tool_trace": tool_trace,
        "messages": [m for m in convo if m.get("role") in ("user", "assistant") and m.get("content")],
    }
