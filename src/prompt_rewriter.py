# src/prompt_rewriter.py

from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()
import os
from dataclasses import dataclass

try:
    # OpenAI client (>= 1.44.0) as recommended in the original project
    from openai import OpenAI

    _OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None
    _OPENAI_AVAILABLE = False


# --------- Configuration for the rewriting behaviour ---------

@dataclass
class PromptRewriteConfig:
    """
    Controls how user queries are rewritten into high-quality prompts.

    Key dimensions:
    - target_role: which expert role the model should adopt
    - domain: legal / compliance domain the system is working in
    - output_language: language of the rewritten prompt
    """

    target_role: str = "senior AML compliance auditor"
    domain: str = (
        "the Hong Kong Anti-Money Laundering and Counter-Terrorist Financing "
        "Ordinance (Cap. 615) and ERC-3643 concepts"
    )
    output_language: str = "English"
    model: str = "gpt-4.1-mini"
    temperature: float = 0.3


# --------- System prompt for the AI-based rewriter ---------

_SYSTEM_PROMPT_TEMPLATE = """You are an expert prompt engineer specialized in legal and compliance RAG systems.

Your job is to take a SHORT, possibly ambiguous user query and rewrite it into a HIGH-QUALITY, NATURAL, and DETAILED instruction prompt for a large language model.

The downstream model will:
- Work over {domain};
- Receive several excerpts of legal text (sections of the ordinance);
- Be asked to extract ENFORCEABLE, OPERATIONAL obligations.

When you rewrite the prompt, you MUST:
- Start with a clear description of the overall task;
- Explicitly describe the role the model should adopt (e.g., {target_role});
- Explain what kind of legal text the model will see;
- Clarify the goal: identify enforceable, operational obligations relevant to the user query;
- Describe how the model should think (step-by-step analysis, focus on triggers, actions, responsible parties, and references);
- Specify the desired output structure (for example a JSON array with fields such as
  "obligation_id", "trigger", "action", "subject", "references");
- Add constraints to avoid hallucinations (e.g., base all statements on the provided text, say "not found" if unsure).

VERY IMPORTANT:
- Do NOT simply restate a generic template.
- Do NOT copy any previous structure verbatim.
- Rewrite the prompt from scratch in your own words, as if you were writing a task specification for a colleague.
- The rewritten prompt must be written in {output_language}.
- Output ONLY the rewritten prompt, with no explanation and no surrounding quotes.
"""


def _build_system_prompt(cfg: PromptRewriteConfig) -> str:
    """Build the system prompt for the AI-based rewriter."""
    return _SYSTEM_PROMPT_TEMPLATE.format(
        target_role=cfg.target_role,
        domain=cfg.domain,
        output_language=cfg.output_language,
    )


# --------- LLM-based rewriting ---------


def _rewrite_with_llm(user_query: str, cfg: PromptRewriteConfig) -> str:
    """
    Use an LLM (e.g., gpt-4.1-mini) to rewrite the raw user query
    into a natural, detailed, and domain-aware extraction prompt.
    """

    if not _OPENAI_AVAILABLE:
        raise RuntimeError(
            "openai package is not available. "
            "Install openai>=1.44.0 or use dry_run=True to fall back to heuristic rewriting."
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it in your environment "
            "or use dry_run=True to rely on heuristic rewriting."
        )

    client = OpenAI(api_key=api_key)

    system_prompt = _build_system_prompt(cfg)

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": (
                "Here is the raw user query:\n"
                f"{user_query}\n\n"
                "Rewrite this into ONE single, high-quality task instruction for the extraction model. "
                "Do not explain what you are doing. Just output the final prompt."
            ),
        },
    ]

    completion = client.chat.completions.create(
        model=cfg.model,
        messages=messages,
        temperature=cfg.temperature,
        max_tokens=700,
    )

    rewritten = completion.choices[0].message.content.strip()
    return rewritten


# --------- Heuristic (rule-based) rewriting ---------


def _rewrite_heuristic(user_query: str, cfg: PromptRewriteConfig) -> str:
    """
    A simple, deterministic rewriting function without LLM calls.

    This version is intentionally more "template-like" and mechanical,
    so that its style is clearly different from the AI-based rewrite.
    """

    template = f"""
Task:
You work as a {cfg.target_role} analysing legal obligations in {cfg.domain}.
Based on the user's query, you must extract relevant, enforceable obligations
from the legal text provided.

User query:
\"\"\"{user_query.strip()}\"\"\"

Instructions:
- Focus only on obligations that are operational and enforceable.
- Ignore general background or high-level policy statements unless needed.
- Base all findings strictly on the provided text.

Required output format (JSON):
[
  {{
    "obligation_id": "CDD-001",
    "trigger": "<when the obligation is activated>",
    "action": "<what must be done>",
    "subject": "<who must act>",
    "references": ["<section or paragraph number>"]
  }},
  ...
]

Return ONLY the JSON array, with no extra explanation.
""".strip()

    return template


# --------- Public entry point ---------


def rewrite_prompt(
    user_query: str,
    cfg: PromptRewriteConfig | None = None,
    dry_run: bool = False,
) -> str:
    """
    Public entry point for the prompt rewriter.

    Parameters
    ----------
    user_query : str
        Raw user input / query string.
    cfg : PromptRewriteConfig | None
        Optional configuration. If None, defaults are used.
    dry_run : bool
        If True, skip LLM calls and use heuristic rewriting only.

    Returns
    -------
    str
        The rewritten, high-quality prompt for the downstream extraction model.
    """
    if cfg is None:
        cfg = PromptRewriteConfig()

    if dry_run:
        # Heuristic / offline mode
        return _rewrite_heuristic(user_query, cfg)

    try:
        # Primary path: AI-based rewrite
        return _rewrite_with_llm(user_query, cfg)
    except Exception:
        # Safe fallback: if the LLM or network fails, use heuristic rewrite
        return _rewrite_heuristic(user_query, cfg)


# --------- Simple CLI for local testing ---------


if __name__ == "__main__":
    import argparse
    import textwrap

    parser = argparse.ArgumentParser(
        description="Rewrite a raw user query into a high-quality extraction prompt."
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Raw user query, e.g. 'customer due diligence obligations'.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Do not call the LLM; use heuristic rewriting only.",
    )
    args = parser.parse_args()

    rewritten = rewrite_prompt(args.query, dry_run=args.dry_run)
    print("\n=== REWRITTEN PROMPT ===\n")
    print(textwrap.dedent(rewritten))