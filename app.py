import os
import uuid
from flask import Flask, render_template, request, session
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

from rules import (
    check_syntax,
    run_flake8,
    check_docstrings,
    check_naming,
    build_rule_summary,
)
from reviewer import (
    review_code_with_llm,
    review_non_python_code_with_llm,
    chat_with_llm,
)
from utils import normalize_mode, empty_llm_result, merge_review_results

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-key")

CHAT_STORE = {}


def get_session_id() -> str:
    if "chat_session_id" not in session:
        session["chat_session_id"] = str(uuid.uuid4())
    return session["chat_session_id"]


def get_history() -> list:
    sid = get_session_id()
    if sid not in CHAT_STORE:
        CHAT_STORE[sid] = []
    return CHAT_STORE[sid]


def is_likely_code(text: str) -> bool:
    if not text:
        return False

    code_signals = [
        "def ", "class ", "import ", "from ", "return ", "print(",
        "if ", "elif ", "else:", "for ", "while ", "try:", "except",
        "function ", "const ", "let ", "var ", "=>", "{", "}", ";",
        "public class ", "System.out.println", "#include", "SELECT ",
        "INSERT ", "UPDATE ", "DELETE ", "CREATE TABLE", "<html", "</",
        "body {", "color:", "margin:", "="
    ]
    lowered = text.lower()
    return any(signal.lower() in lowered for signal in code_signals)


def detect_language(text: str) -> str:
    lowered = text.lower().strip()

    if "system.out.println" in lowered or "public class" in lowered:
        return "java"

    if "#include" in lowered or "int main(" in lowered or "std::" in lowered:
        return "c/c++"

    if "console.log(" in lowered or "function " in lowered or "const " in lowered or "let " in lowered:
        return "javascript"

    if lowered.startswith("select ") or " insert into " in lowered or "create table " in lowered or (" from " in lowered and " where " in lowered):
        return "sql"

    if "<html" in lowered or "<div" in lowered or "<body" in lowered or "</html>" in lowered:
        return "html"

    if "body {" in lowered or "color:" in lowered or "font-size:" in lowered or "display:" in lowered:
        return "css"

    if "def " in lowered or "import " in lowered or "print(" in lowered or "elif " in lowered or "__name__" in lowered:
        return "python"

    return "text"


def build_multi_model_results(llm_result, rule_summary=None):
    """
    Normalize reviewer output into a list:
    [
        {"model": "...", "result": {...}},
        ...
    ]
    """
    if isinstance(llm_result, dict) and "multi" in llm_result:
        items = llm_result["multi"]
        normalized = []

        for item in items:
            model_name = item.get("model", "unknown-model")
            model_result = item.get("result", empty_llm_result())

            if rule_summary and isinstance(model_result, dict):
                merged = merge_review_results(model_result, rule_summary)
            else:
                merged = model_result

            normalized.append({
                "model": model_name,
                "result": merged
            })

        return normalized

    if isinstance(llm_result, dict):
        single_result = merge_review_results(llm_result, rule_summary) if rule_summary else llm_result
        return [
            {
                "model": "single-model",
                "result": single_result
            }
        ]

    return [
        {
            "model": "single-model",
            "result": empty_llm_result()
        }
    ]


@app.route("/", methods=["GET", "POST"])
def index():
    history = get_history()

    if request.method == "POST":
        action = request.form.get("action", "send")

        if action == "clear":
            if "chat_session_id" in session:
                CHAT_STORE[session["chat_session_id"]] = []
            return render_template(
                "index.html",
                history=[],
                code="",
                mode="auto",
                error=None,
            )

        user_input = request.form.get("code", "").strip()
        mode = request.form.get("mode", "auto").strip().lower()

        if not user_input:
            return render_template(
                "index.html",
                history=history,
                code="",
                mode=mode,
                error="Input cannot be empty.",
            )

        # --- Chat mode ---
        if mode == "chat":
            final_result = chat_with_llm(user_input, history)

            history.append(
                {
                    "input_kind": "chat",
                    "user_code": user_input,
                    "mode": "chat",
                    "language": "natural-language",
                    "result": final_result,
                    "syntax_result": {},
                    "docstring_result": {},
                    "naming_result": {},
                    "flake8_result": {},
                }
            )

            return render_template(
                "index.html",
                history=history,
                code="",
                mode=mode,
                error=None,
            )

        # --- Auto mode: natural language or code ---
        if mode == "auto":
            if not is_likely_code(user_input):
                final_result = chat_with_llm(user_input, history)

                history.append(
                    {
                        "input_kind": "chat",
                        "user_code": user_input,
                        "mode": "chat",
                        "language": "natural-language",
                        "result": final_result,
                        "syntax_result": {},
                        "docstring_result": {},
                        "naming_result": {},
                        "flake8_result": {},
                    }
                )

                return render_template(
                    "index.html",
                    history=history,
                    code="",
                    mode=mode,
                    error=None,
                )

            detected_language = detect_language(user_input)

            # Python: rule-based + 4 LLMs
            if detected_language == "python":
                actual_mode = "full"

                syntax_result = check_syntax(user_input)
                docstring_result = check_docstrings(user_input)
                naming_result = check_naming(user_input)
                flake8_result = run_flake8(user_input)

                rule_summary = build_rule_summary(
                    syntax_result=syntax_result,
                    docstring_result=docstring_result,
                    naming_result=naming_result,
                    flake8_result=flake8_result,
                )

                llm_result = review_code_with_llm(
                    code=user_input,
                    mode=actual_mode,
                    rule_summary=rule_summary,
                    history=history,
                )

                final_result = build_multi_model_results(llm_result, rule_summary)

                history.append(
                    {
                        "input_kind": "code_review",
                        "user_code": user_input,
                        "mode": actual_mode,
                        "language": "python",
                        "result": final_result,
                        "syntax_result": syntax_result,
                        "docstring_result": docstring_result,
                        "naming_result": naming_result,
                        "flake8_result": flake8_result,
                    }
                )
            else:
                # Other languages: 4 LLMs
                llm_result = review_non_python_code_with_llm(
                    code=user_input,
                    language=detected_language,
                    history=history,
                )

                final_result = build_multi_model_results(llm_result, rule_summary=None)

                history.append(
                    {
                        "input_kind": "code_review",
                        "user_code": user_input,
                        "mode": "generic-review",
                        "language": detected_language,
                        "result": final_result,
                        "syntax_result": {},
                        "docstring_result": {},
                        "naming_result": {},
                        "flake8_result": {},
                    }
                )

            return render_template(
                "index.html",
                history=history,
                code="",
                mode=mode,
                error=None,
            )

        # --- Explicit review mode ---
        actual_mode = normalize_mode(mode)
        detected_language = detect_language(user_input)

        if detected_language == "python":
            syntax_result = check_syntax(user_input)
            docstring_result = check_docstrings(user_input)
            naming_result = check_naming(user_input)
            flake8_result = run_flake8(user_input)

            rule_summary = build_rule_summary(
                syntax_result=syntax_result,
                docstring_result=docstring_result,
                naming_result=naming_result,
                flake8_result=flake8_result,
            )

            llm_result = review_code_with_llm(
                code=user_input,
                mode=actual_mode,
                rule_summary=rule_summary,
                history=history,
            )

            final_result = build_multi_model_results(llm_result, rule_summary)

            history.append(
                {
                    "input_kind": "code_review",
                    "user_code": user_input,
                    "mode": actual_mode,
                    "language": "python",
                    "result": final_result,
                    "syntax_result": syntax_result,
                    "docstring_result": docstring_result,
                    "naming_result": naming_result,
                    "flake8_result": flake8_result,
                }
            )
        else:
            llm_result = review_non_python_code_with_llm(
                code=user_input,
                language=detected_language,
                history=history,
            )

            final_result = build_multi_model_results(llm_result, rule_summary=None)

            history.append(
                {
                    "input_kind": "code_review",
                    "user_code": user_input,
                    "mode": "generic-review",
                    "language": detected_language,
                    "result": final_result,
                    "syntax_result": {},
                    "docstring_result": {},
                    "naming_result": {},
                    "flake8_result": {},
                }
            )

        return render_template(
            "index.html",
            history=history,
            code="",
            mode=mode,
            error=None,
        )

    return render_template(
        "index.html",
        history=history,
        code="",
        mode="auto",
        error=None,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))