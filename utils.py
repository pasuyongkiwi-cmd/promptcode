import json
from typing import Any, Dict, List


def normalize_mode(mode: str) -> str:
    if not mode:
        return "full"

    mode = mode.strip().lower()

    if mode in {"bug", "bug_review"}:
        return "bug"
    if mode in {"style_docs", "style", "style_review", "style_docs_review"}:
        return "style_docs"
    return "full"


def safe_json_loads(text: str) -> Dict[str, Any]:
    if not text:
        return {}

    cleaned = text.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```"):].strip()

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


def _ensure_text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback

    text = str(value).strip()
    if not text:
        return fallback

    return text


def _ensure_list(value: Any) -> List[str]:
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
        if lowered in {"none", "n/a", "good", "correct", "ok", "no issues", "no major issues"}:
            return []

        lines = [line.strip("-• \n\t") for line in text.splitlines()]
        lines = [line for line in lines if line]
        return lines if lines else [text]

    text = str(value).strip()
    return [text] if text else []


def _ensure_scores(value: Any) -> Dict[str, Any]:
    default_scores = {
        "correctness": "-",
        "readability": "-",
        "maintainability": "-",
        "documentation_quality": "-",
    }

    if not isinstance(value, dict):
        return default_scores

    return {
        "correctness": value.get("correctness", "-"),
        "readability": value.get("readability", "-"),
        "maintainability": value.get("maintainability", "-"),
        "documentation_quality": value.get("documentation_quality", "-"),
    }


def empty_llm_result() -> Dict[str, Any]:
    return {
        "summary": "",
        "reasoning": "",
        "bugs": [],
        "style_issues": [],
        "maintainability_issues": [],
        "documentation_issues": [],
        "suggestions": [],
        "improved_code": "",
        "generated_doc": "",
        "scores": {
            "correctness": "-",
            "readability": "-",
            "maintainability": "-",
            "documentation_quality": "-",
        },
        "overall_score": "-",
        "rule_results": {},
    }


def merge_review_results(llm_result: Dict[str, Any], rule_summary: Dict[str, Any]) -> Dict[str, Any]:
    llm_result = llm_result or {}
    rule_summary = rule_summary or {}

    merged = empty_llm_result()

    merged["summary"] = _ensure_text(llm_result.get("summary", ""), "")
    merged["reasoning"] = _ensure_text(llm_result.get("reasoning", ""), "")
    merged["bugs"] = _ensure_list(llm_result.get("bugs", []))
    merged["style_issues"] = _ensure_list(llm_result.get("style_issues", []))
    merged["maintainability_issues"] = _ensure_list(llm_result.get("maintainability_issues", []))
    merged["documentation_issues"] = _ensure_list(llm_result.get("documentation_issues", []))
    merged["suggestions"] = _ensure_list(llm_result.get("suggestions", []))
    merged["improved_code"] = _ensure_text(llm_result.get("improved_code", ""), "")
    merged["generated_doc"] = _ensure_text(llm_result.get("generated_doc", ""), "")
    merged["scores"] = _ensure_scores(llm_result.get("scores", {}))
    merged["overall_score"] = llm_result.get("overall_score", "-")

    merged["rule_results"] = {
        "syntax_check": rule_summary.get("syntax_check", {}),
        "docstring_check": rule_summary.get("docstring_check", {}),
        "naming_check": rule_summary.get("naming_check", {}),
        "flake8_check": rule_summary.get("flake8_check", {}),
    }

    return merged