import os
import json
import time
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv

from utils import empty_llm_result

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def _humanize_api_error(data: dict) -> str:
    try:
        error = data.get("error", {}) if isinstance(data, dict) else {}
        status_code = error.get("status_code")
        message = str(error.get("message", "")).strip()

        if status_code == 429 or "429" in message or "rate limit" in message.lower():
            return "LLM service is rate-limited. Free model quota is exhausted. Please try again later or switch to a paid/available model."

        if status_code and int(status_code) >= 500:
            return "LLM service is temporarily unavailable. Please try again later."

        if "not valid json" in message.lower():
            return "LLM service returned invalid response content. Please try again."

        if "request failed" in message.lower():
            return f"LLM request failed: {message}"

        if message:
            return f"LLM error: {message}"

        return "LLM service error."
    except Exception:
        return "LLM service error."


def _call_openrouter(messages: list[dict], temperature: float = 0.3) -> dict:
    last_error = None

    for _ in range(3):
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "openrouter/free",
                    "messages": messages,
                    "temperature": temperature,
                },
                timeout=60,
            )

            raw_text = response.text.strip()

            if response.status_code >= 500:
                last_error = {
                    "error": {
                        "message": "Upstream server error",
                        "status_code": response.status_code,
                        "body": raw_text[:500],
                    }
                }
                time.sleep(1)
                continue

            if response.status_code >= 400:
                return {
                    "error": {
                        "message": f"HTTP {response.status_code}",
                        "status_code": response.status_code,
                        "body": raw_text[:500],
                    }
                }

            try:
                data = response.json()
            except ValueError:
                last_error = {
                    "error": {
                        "message": "Response was not valid JSON",
                        "status_code": response.status_code,
                        "body": raw_text[:500],
                    }
                }
                time.sleep(1)
                continue

            if isinstance(data, dict) and "error" in data:
                error_message = ""
                if isinstance(data.get("error"), dict):
                    error_message = str(data["error"].get("message", ""))

                status_code = None
                if isinstance(data.get("error"), dict):
                    status_code = data["error"].get("code") or data["error"].get("status_code")

                last_error = {
                    "error": {
                        "message": error_message or "Unknown upstream error",
                        "status_code": status_code,
                        "body": json.dumps(data, ensure_ascii=False)[:500],
                    }
                }

                if str(status_code) == "429" or "rate limit" in error_message.lower():
                    return last_error

                time.sleep(1)
                continue

            return data

        except requests.RequestException as e:
            last_error = {"error": {"message": f"Request failed: {str(e)}"}}
            time.sleep(1)
            continue

        except Exception as e:
            last_error = {"error": {"message": f"Unexpected error: {str(e)}"}}
            time.sleep(1)
            continue

    return last_error or {"error": {"message": "Unknown error"}}


def _extract_content(data: dict) -> str:
    try:
        choices = data.get("choices", [])
        if not choices:
            return ""

        message = choices[0].get("message", {})
        content = message.get("content")

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    text_value = item.get("text")
                    if isinstance(text_value, str):
                        parts.append(text_value)
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts).strip()

        reasoning = message.get("reasoning")
        if isinstance(reasoning, str):
            return reasoning.strip()

        text_value = choices[0].get("text")
        if isinstance(text_value, str):
            return text_value.strip()

        return ""
    except Exception:
        return ""


def _build_context_messages(history: List[dict], limit: int = 6) -> list[dict]:
    context_messages = []

    recent = history[-limit:] if history else []
    for item in recent:
        user_text = item.get("user_code", "")
        assistant_text = item.get("result", {}).get("summary", "")

        if user_text:
            context_messages.append({"role": "user", "content": user_text})
        if assistant_text:
            context_messages.append({"role": "assistant", "content": assistant_text})

    return context_messages


def _strip_code_fences(text: str) -> str:
    text = text.strip()

    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    return text


def _extract_first_json_object(text: str) -> str:
    text = _strip_code_fences(text)

    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return text[start:]


def _repair_json_with_llm(bad_content: str) -> dict | None:
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You repair invalid JSON. "
                    "Return valid JSON only. "
                    "Do not explain. "
                    "Do not wrap in markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Fix the following content into valid JSON only. "
                    "Preserve the original meaning and keys as much as possible.\n\n"
                    f"{bad_content}"
                ),
            },
        ]

        data = _call_openrouter(messages=messages, temperature=0)
        if not isinstance(data, dict) or "choices" not in data:
            return None

        repaired = _extract_content(data)
        repaired = _extract_first_json_object(repaired)
        return json.loads(repaired)
    except Exception:
        return None


def _parse_json_safely(content: str) -> dict:
    cleaned = _extract_first_json_object(content)

    try:
        return json.loads(cleaned)
    except Exception:
        repaired = _repair_json_with_llm(cleaned)
        if repaired is not None:
            return repaired
        raise


def _normalize_text_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        cleaned = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                cleaned.append(text)
        return cleaned

    if isinstance(value, str):
        text = value.strip()

        if not text:
            return []

        lowered = text.lower()
        if lowered in {
            "none",
            "n/a",
            "no issues",
            "no major issues",
            "good",
            "correct",
            "ok",
        }:
            return []

        lines = [line.strip("-• \n\t") for line in text.splitlines()]
        lines = [line for line in lines if line]

        return lines if lines else [text]

    text = str(value).strip()
    return [text] if text else []


def _normalize_single_text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback

    text = str(value).strip()

    if not text:
        return fallback

    if text.lower() in {"good", "correct", "none", "n/a"}:
        return fallback

    return text


def _normalize_score(value: Any, default: float = 3.0) -> float:
    try:
        score = float(value)

        if 0 <= score <= 1:
            score = score * 5

        score = max(1.0, min(5.0, score))
        return round(score, 1)
    except Exception:
        return default


def _normalize_scores(scores: Dict[str, Any] | None) -> Dict[str, float]:
    scores = scores or {}
    return {
        "correctness": _normalize_score(scores.get("correctness", 3)),
        "readability": _normalize_score(scores.get("readability", 3)),
        "maintainability": _normalize_score(scores.get("maintainability", 3)),
        "documentation_quality": _normalize_score(scores.get("documentation_quality", 3)),
    }


def _failure_result(result: Dict[str, Any], message: str) -> Dict[str, Any]:
    result["summary"] = message
    result["bugs"] = []
    result["style_issues"] = []
    result["maintainability_issues"] = []
    result["documentation_issues"] = []
    result["suggestions"] = []
    result["improved_code"] = ""
    result["generated_doc"] = ""
    result["scores"] = {
        "correctness": "-",
        "readability": "-",
        "maintainability": "-",
        "documentation_quality": "-",
    }
    result["overall_score"] = "-"
    return result


def review_code_with_llm(
    code: str,
    mode: str,
    rule_summary: Dict[str, Any],
    history: List[dict] | None = None,
) -> Dict[str, Any]:
    result = empty_llm_result()

    try:
        prompt = f'''
You are a senior software engineer and Python code reviewer.

IMPORTANT LANGUAGE RULE:
- Detect the user's language and reply in the SAME language.
- If the user writes in Chinese, reply in Chinese.
- If the user writes in English, reply in English.

Return VALID JSON only.

Required JSON format:
{{
  "summary": "string",
  "bugs": ["string"],
  "style_issues": ["string"],
  "maintainability_issues": ["string"],
  "documentation_issues": ["string"],
  "suggestions": ["string"],
  "improved_code": "string",
  "scores": {{
    "correctness": 5,
    "readability": 5,
    "maintainability": 5,
    "documentation_quality": 5
  }},
  "overall_score": 5
}}

Rules:
- Return JSON only
- No markdown
- No code fences
- Scores must be numbers from 1 to 5
- overall_score must be a number from 1 to 5
- Decimals are allowed, for example 4.5
- The code is Python
- Review mode: {mode}

Additional strict rules:
- All list fields MUST be valid JSON arrays, never strings
- If no items, return []
- Do NOT return "None", "Good", "Correct", or similar placeholders
- summary must be a full sentence, not a single word
- improved_code must be valid code or an empty string

Rule-based analysis:
{rule_summary}

Code:
{code}
'''

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert Python code reviewer. "
                    "Use recent conversation context when helpful, but focus on the current code. "
                    "Always match the user's language."
                ),
            }
        ]
        messages.extend(_build_context_messages(history or []))
        messages.append({"role": "user", "content": prompt})

        data = _call_openrouter(messages=messages, temperature=0.2)

        if "choices" not in data:
            return _failure_result(result, _humanize_api_error(data))

        content = _extract_content(data)
        if not content:
            return _failure_result(result, "LLM returned empty content.")

        parsed = _parse_json_safely(content)

        result["summary"] = _normalize_single_text(
            parsed.get("summary", ""),
            fallback="Review completed."
        )
        result["bugs"] = _normalize_text_list(parsed.get("bugs", []))
        result["style_issues"] = _normalize_text_list(parsed.get("style_issues", []))
        result["maintainability_issues"] = _normalize_text_list(parsed.get("maintainability_issues", []))
        result["documentation_issues"] = _normalize_text_list(parsed.get("documentation_issues", []))
        result["suggestions"] = _normalize_text_list(parsed.get("suggestions", []))

        improved_code = _normalize_single_text(parsed.get("improved_code", ""), fallback="")
        result["improved_code"] = "" if improved_code.lower() in {"good", "none"} else improved_code
        result["generated_doc"] = _normalize_single_text(parsed.get("generated_doc", ""), fallback="")

        normalized_scores = _normalize_scores(
            parsed.get(
                "scores",
                {
                    "correctness": 3,
                    "readability": 3,
                    "maintainability": 3,
                    "documentation_quality": 3,
                },
            )
        )
        result["scores"] = normalized_scores

        overall_from_model = parsed.get("overall_score", None)
        if overall_from_model is None:
            result["overall_score"] = round(sum(normalized_scores.values()) / 4, 1)
        else:
            result["overall_score"] = _normalize_score(overall_from_model)

        return result

    except Exception as e:
        return _failure_result(result, f"LLM call failed or JSON parsing failed: {str(e)}")


def review_non_python_code_with_llm(
    code: str,
    language: str,
    history: List[dict] | None = None,
) -> Dict[str, Any]:
    result = empty_llm_result()

    try:
        prompt = f'''
You are a senior multi-language code reviewer.

IMPORTANT LANGUAGE RULE:
- Detect the user's language and reply in the SAME language.
- If the user writes in Chinese, reply in Chinese.
- If the user writes in English, reply in English.

Return VALID JSON only.

Required JSON format:
{{
  "summary": "string",
  "bugs": ["string"],
  "style_issues": ["string"],
  "maintainability_issues": ["string"],
  "documentation_issues": ["string"],
  "suggestions": ["string"],
  "improved_code": "string",
  "scores": {{
    "correctness": 5,
    "readability": 5,
    "maintainability": 5,
    "documentation_quality": 5
  }},
  "overall_score": 5
}}

Rules:
- Return JSON only
- No markdown
- No code fences
- Scores must be numbers from 1 to 5
- overall_score must be a number from 1 to 5
- Decimals are allowed, for example 4.5
- The code language is: {language}
- Review syntax, readability, maintainability, documentation, and common bugs for that language.

Additional strict rules:
- All list fields MUST be valid JSON arrays, never strings
- If no items, return []
- Do NOT return "None", "Good", "Correct", or similar placeholders
- summary must be a full sentence, not a single word
- improved_code must be valid code or an empty string

Code:
{code}
'''

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert programming reviewer for Python, JavaScript, Java, SQL, C/C++, HTML, CSS, and other common languages. "
                    "Always match the user's language."
                ),
            }
        ]
        messages.extend(_build_context_messages(history or []))
        messages.append({"role": "user", "content": prompt})

        data = _call_openrouter(messages=messages, temperature=0.2)

        if "choices" not in data:
            return _failure_result(result, _humanize_api_error(data))

        content = _extract_content(data)
        if not content:
            return _failure_result(result, "LLM returned empty content.")

        parsed = _parse_json_safely(content)

        result["summary"] = _normalize_single_text(
            parsed.get("summary", ""),
            fallback="Review completed."
        )
        result["bugs"] = _normalize_text_list(parsed.get("bugs", []))
        result["style_issues"] = _normalize_text_list(parsed.get("style_issues", []))
        result["maintainability_issues"] = _normalize_text_list(parsed.get("maintainability_issues", []))
        result["documentation_issues"] = _normalize_text_list(parsed.get("documentation_issues", []))
        result["suggestions"] = _normalize_text_list(parsed.get("suggestions", []))

        improved_code = _normalize_single_text(parsed.get("improved_code", ""), fallback="")
        result["improved_code"] = "" if improved_code.lower() in {"good", "none"} else improved_code
        result["generated_doc"] = _normalize_single_text(parsed.get("generated_doc", ""), fallback="")

        normalized_scores = _normalize_scores(
            parsed.get(
                "scores",
                {
                    "correctness": 3,
                    "readability": 3,
                    "maintainability": 3,
                    "documentation_quality": 3,
                },
            )
        )
        result["scores"] = normalized_scores

        overall_from_model = parsed.get("overall_score", None)
        if overall_from_model is None:
            result["overall_score"] = round(sum(normalized_scores.values()) / 4, 1)
        else:
            result["overall_score"] = _normalize_score(overall_from_model)

        return result

    except Exception as e:
        return _failure_result(result, f"LLM call failed or JSON parsing failed: {str(e)}")


def chat_with_llm(user_input: str, history: List[dict] | None = None) -> Dict[str, Any]:
    result = empty_llm_result()

    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful multi-language coding assistant. "
                    "You can answer natural language questions, explain programming concepts, "
                    "and generate code in Python, JavaScript, Java, SQL, C/C++, HTML, CSS, and more. "
                    "If the user asks for code, provide runnable code and a short explanation. "
                    "Always answer in the same language as the user's latest message."
                ),
            }
        ]
        messages.extend(_build_context_messages(history or []))
        messages.append({"role": "user", "content": user_input})

        data = _call_openrouter(messages=messages, temperature=0.4)

        if "choices" not in data:
            result["summary"] = _humanize_api_error(data)
            return result

        content = _extract_content(data)

        if not content:
            result["summary"] = "LLM returned empty content."
            return result

        result["summary"] = content
        result["reasoning"] = content
        result["bugs"] = []
        result["style_issues"] = []
        result["maintainability_issues"] = []
        result["documentation_issues"] = []
        result["suggestions"] = []
        result["improved_code"] = ""
        result["generated_doc"] = ""
        result["scores"] = {
            "correctness": "-",
            "readability": "-",
            "maintainability": "-",
            "documentation_quality": "-",
        }
        result["overall_score"] = "-"

        return result

    except Exception as e:
        result["summary"] = f"LLM call failed: {str(e)}"
        return result