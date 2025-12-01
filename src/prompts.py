ROLE_PROMPT = """Act as a lead auditor and virtual compliance controller with deep knowledge \
of the HKMA Anti-Money Laundering and Counter-Financing of Terrorism Ordinance (Cap. 615) \
and smart contract frameworks such as ERC-3643. Your goal is to extract legitimate and \
enforceable obligations from the provided Cap. 615 sections that could be audited in smart \
contracts. Work ONLY from the supplied context."""

TASK_INSTRUCTION = """For each major section provided, analyze carefully, classify, and explain \
each clause using this JSON shape:
[
  {{
    "provision": "<section or schedule citation>",
    "severity": "Very High|High|Medium",
    "reinforcement_type": "Must-have|Must-not-have",
    "compliance_objective": "<brief objective>",
    "feasibility": "On-chain|Off-chain|Hybrid",
    "explanation": "<concise but precise mapping to the law>",
    "erc_3643": "<suggested modules/functions>"
  }}
]
Rules:
- Answer only from the provided context; do not invent text outside it.
- Avoid hypothetical clauses or those requiring subjective judgment.
- Keep language aligned with Cap. 615 wording; make it readable for non-technical reviewers.
- If no enforceable obligation is present in the context, return an empty list.
"""


def build_prompt(context: str, query: str, erc_whitelist: str = "") -> str:
    """Create the full prompt string for the LLM."""
    whitelist_block = ""
    if erc_whitelist:
        whitelist_block = f"""ERC-3643 allowed modules/functions:
{erc_whitelist}

Rules for erc_3643 field:
- Choose ONLY from the allowed list above; do not invent new functions.
- If no suitable ERC-3643 hook applies, set "erc_3643": "N/A (off-chain)".\n"""

    return f"""{ROLE_PROMPT}

Task: Extract enforceable obligations relevant to: "{query}"

Context:
{context}

{whitelist_block}
{TASK_INSTRUCTION}
Return ONLY valid JSON."""
