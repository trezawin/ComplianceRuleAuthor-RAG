import argparse
import json
import re
from pathlib import Path

from src.config import paths
from src.ingest import clean


def extract_functions(text: str):
    # Capture camelCase or snake_case identifiers followed by '('
    fn_pattern = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)
    matches = set(fn_pattern.findall(text))
    # Filter to likely ERC-3643 terms
    allowed_prefixes = {"can", "is", "force", "freeze", "unfreeze", "recover", "register", "update", "delete", "pause", "unpause", "batch"}
    functions = []
    for m in matches:
        lower = m.lower()
        if any(lower.startswith(p) for p in allowed_prefixes):
            functions.append(m)
    return sorted(set(functions))


def extract_modules(text: str):
    # Common module/class names in ERC-3643 spec
    module_pattern = re.compile(r"\b(IdentityRegistry|Compliance|ComplianceModule|TrustedIssuersRegistry|ClaimTopicsRegistry|Token|ModularCompliance|IdentityRegistryStorage|Recovery)\b")
    return sorted(set(module_pattern.findall(text)))


def main():
    ap = argparse.ArgumentParser(description="Build ERC-3643 whitelist from spec text.")
    ap.add_argument("--source", type=Path, default=paths.erc_raw, help="Path to ERC-3643 spec text")
    ap.add_argument("--out", type=Path, default=paths.erc_whitelist, help="Output whitelist JSON")
    args = ap.parse_args()

    text = clean(args.source.read_text(encoding="utf-8", errors="ignore"))
    functions = extract_functions(text)
    modules = extract_modules(text)

    # Add a core fallback list to ensure coverage even if regex misses items
    core = [
        "canTransfer",
        "isVerified",
        "forceTransfer",
        "freeze",
        "unfreeze",
        "pause",
        "unpause",
        "recoverAddress",
        "registerIdentity",
        "updateIdentity",
        "deleteIdentity",
        "batchTransfer",
    ]

    whitelist = sorted(set(functions + modules + core))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"modules_functions": whitelist}, indent=2))
    print(f"Wrote whitelist with {len(whitelist)} entries to {args.out}")


if __name__ == "__main__":
    main()
