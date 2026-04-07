import os
import json
import time
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv

from utils import empty_llm_result

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

FREE_MODELS = [
    "openai/gpt-oss-20b",
    "meta-llama/llama-3.1-8b-instruct",
    "google/gemma-4-26b-a4b-it",
    "openai/sora-2-pro",
]


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


def _call_openrouter(messages: list[dict], temperature: float = 0.2, model: str = "openrouter/free") -> dict:
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
                    "model": model,
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


def _call_openrouter_multi(messages: list[dict], temperature: float = 0.2) -> list[dict]:
    results = []

    for model in FREE_MODELS:
        data = _call_openrouter(messages=messages, temperature=temperature, model=model)
        content = _extract_content(data) if isinstance(data, dict) and "choices" in data else ""

        results.append(
            {
                "model": model,
                "raw": data,
                "content": content,
            }
        )

    return results


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
        assistant_text = ""

        result = item.get("result", {})
        if isinstance(result, dict):
            assistant_text = result.get("summary", "")
        elif isinstance(result, list) and result:
            first_item = result[0]
            if isinstance(first_item, dict):
                inner_result = first_item.get("result", {})
                if isinstance(inner_result, dict):
                    assistant_text = inner_result.get("summary", "")

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


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _enforce_score_consistency(result: Dict[str, Any]) -> Dict[str, Any]:
    scores = result.get("scores", {})
    if not isinstance(scores, dict):
        return result

    correctness = _normalize_score(scores.get("correctness", 3))
    readability = _normalize_score(scores.get("readability", 3))
    maintainability = _normalize_score(scores.get("maintainability", 3))
    documentation_quality = _normalize_score(scores.get("documentation_quality", 3))

    bugs = result.get("bugs", []) or []
    style_issues = result.get("style_issues", []) or []
    maintainability_issues = result.get("maintainability_issues", []) or []
    documentation_issues = result.get("documentation_issues", []) or []
    summary = str(result.get("summary", "")).lower()

    bug_text = " ".join(str(x) for x in bugs).lower()
    style_text = " ".join(str(x) for x in style_issues).lower()
    maintain_text = " ".join(str(x) for x in maintainability_issues).lower()
    doc_text = " ".join(str(x) for x in documentation_issues).lower()
    all_issue_text = " ".join([bug_text, style_text, maintain_text, doc_text, summary])

    has_syntax_error = _contains_any(
        all_issue_text,
        [
            "syntaxerror",
            "syntax error",
            "invalid syntax",
            "indentationerror",
            "unindent does not match",
            "expected an indented block",
            "unexpected indent",
            "parse error",
            "compilation error",
        ],
    )

    has_confirmed_runtime_failure = _contains_any(
        bug_text,
        [
            "division by zero",
            "zerodivisionerror",
            "indexerror",
            "typeerror",
            "valueerror",
            "arithmeticexception",
            "nullpointerexception",
            "will raise",
            "raises ",
            "guaranteed runtime",
            "guaranteed failure",
        ],
    )

    has_edge_case_failure = _contains_any(
        all_issue_text,
        [
            "empty list",
            "empty input",
            "null input",
            "common edge case",
            "may fail",
            "can fail",
            "potential runtime error",
        ],
    )

    has_readability_issue = (
        bool(style_issues)
        or _contains_any(
            style_text,
            [
                "indent",
                "format",
                "formatting",
                "naming",
                "readability",
                "mixed spaces",
                "mixed tabs",
                "unclear structure",
            ],
        )
    )

    has_maintainability_issue = (
        bool(maintainability_issues)
        or _contains_any(
            maintain_text,
            [
                "no error handling",
                "missing error handling",
                "poor structure",
                "hardcoded",
                "duplicated",
                "tight coupling",
                "difficult to extend",
                "no input validation",
            ],
        )
    )

    has_documentation_issue = (
        bool(documentation_issues)
        or _contains_any(
            doc_text + " " + all_issue_text,
            [
                "docstring",
                "documentation",
                "missing comments",
                "no comments",
                "no documentation",
                "missing jsdoc",
                "missing javadoc",
            ],
        )
    )

    if has_syntax_error:
        correctness = min(correctness, 2.0)

    if has_confirmed_runtime_failure:
        correctness = min(correctness, 2.5)
    elif has_edge_case_failure:
        correctness = min(correctness, 3.5)

    if has_readability_issue:
        readability = min(readability, 3.5)
        if _contains_any(style_text, ["indent", "formatting", "mixed spaces", "mixed tabs"]):
            readability = min(readability, 3.0)

    if has_maintainability_issue:
        maintainability = min(maintainability, 3.5)
        if _contains_any(maintain_text, ["no error handling", "missing error handling", "no input validation"]):
            maintainability = min(maintainability, 3.0)

    if has_documentation_issue:
        documentation_quality = min(documentation_quality, 2.5)
        if _contains_any(doc_text + " " + all_issue_text, ["docstring", "no documentation", "missing jsdoc", "missing javadoc"]):
            documentation_quality = min(documentation_quality, 2.0)

    scores = {
        "correctness": round(correctness, 1),
        "readability": round(readability, 1),
        "maintainability": round(maintainability, 1),
        "documentation_quality": round(documentation_quality, 1),
    }
    result["scores"] = scores

    computed_overall = round(sum(scores.values()) / 4, 1)
    if scores["correctness"] <= 2.0:
        computed_overall = min(computed_overall, 2.5)

    result["overall_score"] = computed_overall
    return result


def _is_placeholder_result(parsed: dict) -> bool:
    if not isinstance(parsed, dict):
        return True

    summary = str(parsed.get("summary", "")).strip().lower()
    bugs = parsed.get("bugs", [])
    style_issues = parsed.get("style_issues", [])
    maintainability_issues = parsed.get("maintainability_issues", [])
    documentation_issues = parsed.get("documentation_issues", [])
    suggestions = parsed.get("suggestions", [])
    improved_code = str(parsed.get("improved_code", "")).strip()
    scores = parsed.get("scores", {})

    has_any_content = any([
        bugs,
        style_issues,
        maintainability_issues,
        documentation_issues,
        suggestions,
        improved_code,
    ])

    if not isinstance(scores, dict) or not scores:
        return True

    default_like_scores = (
        str(scores.get("correctness")) in {"3", "3.0"} and
        str(scores.get("readability")) in {"3", "3.0"} and
        str(scores.get("maintainability")) in {"3", "3.0"} and
        str(scores.get("documentation_quality")) in {"3", "3.0"}
    )

    if summary in {"", "review completed.", "review completed"} and not has_any_content:
        return True

    if default_like_scores and not has_any_content:
        return True

    return False


def _call_review_with_retry(messages: list[dict], max_attempts: int = 3) -> dict:
    last_error = None

    for _ in range(max_attempts):
        data = _call_openrouter(messages=messages, temperature=0)

        if "choices" not in data:
            last_error = data
            continue

        content = _extract_content(data)
        if not content:
            last_error = {"error": {"message": "LLM returned empty content."}}
            continue

        try:
            parsed = _parse_json_safely(content)
        except Exception as e:
            last_error = {"error": {"message": f"JSON parsing failed: {str(e)}"}}
            continue

        if _is_placeholder_result(parsed):
            last_error = {"error": {"message": "LLM returned placeholder/incomplete review content."}}
            continue

        return {"parsed": parsed}

    return last_error or {"error": {"message": "LLM failed after retries."}}


def _build_multi_model_review_results(messages: list[dict]) -> list[dict]:
    multi_results = _call_openrouter_multi(messages=messages, temperature=0)
    all_results = []

    for item in multi_results:
        model_name = item.get("model", "unknown-model")
        raw_data = item.get("raw", {})
        content = item.get("content", "")

        if not content:
            error_message = _humanize_api_error(raw_data) if isinstance(raw_data, dict) else "No response"
            all_results.append(
                {
                    "model": model_name,
                    "result": _failure_result(empty_llm_result(), error_message),
                }
            )
            continue

        try:
            parsed = _parse_json_safely(content)
        except Exception:
            all_results.append(
                {
                    "model": model_name,
                    "result": _failure_result(empty_llm_result(), "JSON parse failed"),
                }
            )
            continue

        if _is_placeholder_result(parsed):
            all_results.append(
                {
                    "model": model_name,
                    "result": _failure_result(empty_llm_result(), "LLM returned placeholder/incomplete review content."),
                }
            )
            continue

        temp_result = empty_llm_result()

        temp_result["summary"] = _normalize_single_text(
            parsed.get("summary", ""),
            fallback="LLM returned incomplete review content."
        )
        temp_result["bugs"] = _normalize_text_list(parsed.get("bugs", []))
        temp_result["style_issues"] = _normalize_text_list(parsed.get("style_issues", []))
        temp_result["maintainability_issues"] = _normalize_text_list(parsed.get("maintainability_issues", []))
        temp_result["documentation_issues"] = _normalize_text_list(parsed.get("documentation_issues", []))
        temp_result["suggestions"] = _normalize_text_list(parsed.get("suggestions", []))

        improved_code = _normalize_single_text(parsed.get("improved_code", ""), fallback="")
        temp_result["improved_code"] = "" if improved_code.lower() in {"good", "none"} else improved_code
        temp_result["generated_doc"] = _normalize_single_text(parsed.get("generated_doc", ""), fallback="")

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
        temp_result["scores"] = normalized_scores

        overall_from_model = parsed.get("overall_score", None)
        if overall_from_model is None:
            temp_result["overall_score"] = round(sum(normalized_scores.values()) / 4, 1)
        else:
            temp_result["overall_score"] = _normalize_score(overall_from_model)

        temp_result = _enforce_score_consistency(temp_result)

        all_results.append(
            {
                "model": model_name,
                "result": temp_result,
            }
        )

    return all_results


def review_code_with_llm(
    code: str,
    mode: str,
    rule_summary: Dict[str, Any],
    history: List[dict] | None = None,
) -> Dict[str, Any]:
    try:
        prompt = f'''
You are a senior software engineer and expert Python code reviewer.

OUTPUT LANGUAGE RULE:
- Always respond in English only.
- Never reply in Chinese or any other language.

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

VERY IMPORTANT REVIEW RULES:
- Report a syntax error ONLY if the code is actually syntactically invalid.
- Do NOT call non-standard indentation a syntax error unless it truly makes the code invalid.
- In Python, indentation that is non-PEP8 but internally consistent is a style/readability issue, not a syntax error.
- Report a bug only for confirmed or highly likely functional problems.
- Do NOT report missing type hints, missing docstrings, naming style, or non-pythonic style as bugs.
- Put style issues in style_issues.
- Put structure, validation, and extensibility concerns in maintainability_issues.
- Put missing docstrings/comments in documentation_issues.
- Be careful not to misclassify best-practice suggestions as correctness bugs.

STRICT SCORING POLICY:
- Start each dimension from 5 and deduct only for real issues.
- Never give all 5s if any issue is reported.
- Do not give 5 unless the code is close to production quality.
- Scores must match the problems listed in bugs, style_issues, maintainability_issues, and documentation_issues.

Correctness scoring:
- 5: Code is syntactically valid and works for normal expected inputs.
- 4: Mostly correct, but has minor risks or misses some edge-case handling.
- 3: Works for common cases but has a meaningful edge-case bug.
- 2: Has a syntax error, guaranteed runtime failure, or serious logic problem.
- 1: Fundamentally broken or non-executable.

Readability scoring:
- Deduct for inconsistent indentation, poor naming, unclear structure, or formatting problems.
- Non-standard but valid indentation should reduce readability, not correctness.

Maintainability scoring:
- Deduct for missing validation, missing error handling for risky operations, duplicated logic, hardcoded values, poor structure, or difficult extension.

Documentation quality scoring:
- Deduct for missing docstrings, missing comments where explanation is needed, or unclear parameter/return behavior.
- If there is no docstring and no useful comments, documentation_quality should usually be <= 2.

Overall scoring:
- overall_score must reflect the real quality level across dimensions.
- If correctness <= 2, overall_score MUST be <= 2.5.
- Do not make overall_score higher than the average quality impression.

Before finalizing the JSON, perform a consistency check:
- If you reported a real syntax error or guaranteed runtime failure, lower correctness.
- If you reported formatting or indentation issues, lower readability.
- If you reported missing validation or poor structure, lower maintainability.
- If you reported missing docstrings or comments, lower documentation_quality.
- Ensure the listed issues and scores do not contradict each other.
- Be conservative, precise, and realistic.

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
                    "Focus only on the current code and the provided rule-based analysis. "
                    "Do not use prior review context for scoring. "
                    "Always respond in English. "
                    "Be strict about accuracy and avoid false syntax-error claims."
                ),
            }
        ]
        messages.append({"role": "user", "content": prompt})

        multi_review_results = _build_multi_model_review_results(messages)
        return {"multi": multi_review_results}

    except Exception as e:
        fallback_results = []
        for model in FREE_MODELS:
            fallback_results.append(
                {
                    "model": model,
                    "result": _failure_result(empty_llm_result(), f"LLM call failed or JSON parsing failed: {str(e)}"),
                }
            )
        return {"multi": fallback_results}


def review_non_python_code_with_llm(
    code: str,
    language: str,
    history: List[dict] | None = None,
) -> Dict[str, Any]:
    try:
        prompt = f'''
You are a senior multi-language code reviewer.

OUTPUT LANGUAGE RULE:
- Always respond in English only.
- Never reply in Chinese or any other language.

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

VERY IMPORTANT REVIEW RULES:
- Report a syntax/parse/compile error ONLY if the code is actually invalid in that language.
- Do NOT treat mere style-guide violations as syntax errors.
- Report a bug only for confirmed or highly likely functional problems.
- Do NOT report naming, missing type hints, missing comments, or formatting alone as bugs.
- Put formatting and naming in style_issues.
- Put validation, structure, robustness, and extensibility concerns in maintainability_issues.
- Put missing docs/comments/JSDoc/Javadoc in documentation_issues.
- Be careful not to misclassify best-practice suggestions as correctness bugs.

STRICT SCORING POLICY:
- Start each dimension from 5 and deduct only for real issues.
- Never give all 5s if any issue is reported.
- Do not give 5 unless the code is close to production quality.
- Scores must match the problems listed in bugs, style_issues, maintainability_issues, and documentation_issues.

Correctness scoring:
- 5: Code is syntactically valid and works for normal expected use.
- 4: Mostly correct, but has minor risks or misses some edge-case handling.
- 3: Works for common cases but has a meaningful edge-case bug.
- 2: Has a syntax/parse error, guaranteed runtime failure, invalid query, or serious logic problem.
- 1: Fundamentally broken.

Readability scoring:
- Deduct for inconsistent formatting, poor naming, unclear structure, or confusing layout.

Maintainability scoring:
- Deduct for lack of validation/error handling for risky operations, duplicated logic, hardcoded values, poor modularity, or difficult extension.

Documentation quality scoring:
- Deduct for missing comments, missing explanation, missing JSDoc/Javadoc/docstrings, or unclear intent where documentation is needed.

Overall scoring:
- overall_score must reflect the actual quality level across dimensions.
- If correctness <= 2, overall_score MUST be <= 2.5.
- Do not make overall_score higher than the average quality impression.

Before finalizing the JSON, perform a consistency check:
- If you reported a real syntax/parse/compile error or guaranteed runtime failure, lower correctness.
- If you reported formatting or naming issues, lower readability.
- If you reported missing validation or poor structure, lower maintainability.
- If you reported missing comments or explanation, lower documentation_quality.
- Ensure the listed issues and scores do not contradict each other.
- Be conservative, precise, and realistic.

Code:
{code}
'''

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert programming reviewer for Python, JavaScript, Java, SQL, C/C++, HTML, CSS, and other common languages. "
                    "Focus only on the current code. "
                    "Do not use prior review context for scoring. "
                    "Always respond in English. "
                    "Be strict about accuracy and avoid false syntax-error claims."
                ),
            }
        ]
        messages.append({"role": "user", "content": prompt})

        multi_review_results = _build_multi_model_review_results(messages)
        return {"multi": multi_review_results}

    except Exception as e:
        fallback_results = []
        for model in FREE_MODELS:
            fallback_results.append(
                {
                    "model": model,
                    "result": _failure_result(empty_llm_result(), f"LLM call failed or JSON parsing failed: {str(e)}"),
                }
            )
        return {"multi": fallback_results}


def chat_with_llm(user_input: str, history: List[dict] | None = None) -> Dict[str, Any]:
    result = empty_llm_result()

    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful coding assistant. "
                    "You can answer natural language questions, explain programming concepts, "
                    "and generate code in Python, JavaScript, Java, SQL, C/C++, HTML, CSS, and more. "
                    "If the user asks for code, provide runnable code and a short explanation. "
                    "Always answer in English only."
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