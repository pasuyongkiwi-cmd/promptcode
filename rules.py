import ast
import os
import subprocess
import tempfile
from typing import Any, Dict


WEAK_NAMES = {"x", "y", "z", "tmp", "foo", "bar", "test", "data", "var", "obj"}


def check_syntax(code: str) -> Dict[str, Any]:
    try:
        compile(code, "<string>", "exec")
        return {"passed": True, "error": ""}
    except SyntaxError as exc:
        return {
            "passed": False,
            "error": f"SyntaxError: {exc.msg} (line {exc.lineno})",
        }
    except Exception as exc:
        return {
            "passed": False,
            "error": str(exc),
        }


def check_docstrings(code: str) -> Dict[str, Any]:
    result = {
        "passed": True,
        "missing_docstrings": [],
        "error": "",
    }

    try:
        tree = ast.parse(code)
        missing = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if ast.get_docstring(node) is None:
                    missing.append(
                        {
                            "type": type(node).__name__,
                            "name": node.name,
                            "line": node.lineno,
                        }
                    )

        result["missing_docstrings"] = missing
        result["passed"] = len(missing) == 0
        return result

    except Exception as exc:
        result["passed"] = False
        result["error"] = str(exc)
        return result


def check_naming(code: str) -> Dict[str, Any]:
    result = {
        "passed": True,
        "issues": [],
        "error": "",
    }

    try:
        tree = ast.parse(code)
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if node.id.lower() in WEAK_NAMES:
                    issues.append(
                        {
                            "name": node.id,
                            "line": node.lineno,
                            "reason": "Name too generic",
                        }
                    )

        result["issues"] = issues
        result["passed"] = len(issues) == 0
        return result

    except Exception as exc:
        result["passed"] = False
        result["error"] = str(exc)
        return result


def run_flake8(code: str) -> Dict[str, Any]:
    result = {
        "passed": True,
        "issues": [],
        "error": "",
        "available": True,
    }

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        process = subprocess.run(
            ["flake8", tmp_path],
            capture_output=True,
            text=True,
        )

        if process.returncode != 0:
            stdout = process.stdout.strip()
            if stdout:
                result["issues"] = stdout.split("\n")
            result["passed"] = False

        return result

    except FileNotFoundError:
        result["available"] = False
        result["error"] = "flake8 not installed"
        return result

    finally:
        if "tmp_path" in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)


def build_rule_summary(
    syntax_result,
    docstring_result,
    naming_result,
    flake8_result,
):
    return {
        "syntax_check": syntax_result,
        "docstring_check": docstring_result,
        "naming_check": naming_result,
        "flake8_check": flake8_result,
    }