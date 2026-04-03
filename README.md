# PromptCode – LLM-Powered Code Reviewer

PromptCode is an AI-powered code review tool that combines rule-based analysis with Large Language Models (LLMs) to evaluate code quality, detect bugs, and provide structured feedback.

## Features

- AI-assisted code review using LLM (OpenRouter API)
- Rule-based analysis for Python (syntax, naming, docstrings, flake8)
- Multi-language support:
  - Python
  - JavaScript
  - Java
  - SQL
  - HTML / CSS
- Structured review output:
  - Summary
  - Bugs
  - Style Issues
  - Maintainability Issues
  - Documentation Issues
  - Suggestions
  - Improved Code
- Scoring system (1–5 scale):
  - Correctness
  - Readability
  - Maintainability
  - Documentation Quality
  - Overall Score
- Natural language chat mode
- Markdown rendering + syntax highlighting
- Copy-to-clipboard for code blocks
- Session-based interaction history

---

## Tech Stack

**Backend**
- Python
- Flask
- OpenRouter API (LLM integration)
- Rule-based static analysis (custom + flake8)

**Frontend**
- HTML / CSS
- Jinja2 templates
- Prism.js (syntax highlighting)
- Marked.js (markdown rendering)

---

## System Architecture

1. User submits code or natural language input
2. System detects input type (code vs text) :contentReference[oaicite:3]{index=3}
3. For Python:
   - Run rule-based checks (syntax, naming, docstring, flake8)
   - Combine with LLM analysis
4. For other languages:
   - Use LLM-based review only
5. LLM returns structured JSON review :contentReference[oaicite:4]{index=4}
6. Backend normalizes and merges results
7. Frontend displays:
   - Scores
   - Issues
   - Suggestions
   - Improved code

---

## Prompt Engineering Highlights

- Enforced structured JSON output
- Strict classification:
  - Bugs vs Style vs Maintainability vs Documentation
- Prevented false positives (e.g. indentation ≠ syntax error)
- Designed scoring consistency rules
- Added retry + JSON repair mechanism for robustness :contentReference[oaicite:5]{index=5}

---

## UI Features

- Chat + Code Review unified interface :contentReference[oaicite:6]{index=6}  
- Auto mode (detects code vs natural language)
- Multiple review modes:
  - Bug Review
  - Style & Docs Review
  - Full Review
- Real-time markdown rendering
- Syntax highlighting for multiple languages
- Copy buttons for all code blocks

---

## How to Run

```bash
pip install -r requirements.txt
python app.py
