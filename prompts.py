from typing import Any, Dict


def build_messages(code: str, mode: str, rule_summary: Dict[str, Any]) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are an expert senior Python code reviewer. "
                "Review the user's code based on correctness, style, readability, "
                "maintainability, and documentation. Be specific and concise."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Review mode: {mode}\n\n"
                f"Rule-based analysis:\n{rule_summary}\n\n"
                f"Python code:\n{code}"
            ),
        },
    ]