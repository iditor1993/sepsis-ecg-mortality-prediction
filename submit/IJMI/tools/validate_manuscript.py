"""Validate citation consistency and approximate word count."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "manuscript_IJMI.md"
REFS = json.loads((ROOT / "references" / "final_references.json").read_text(encoding="utf-8"))


def parse_citations(text: str) -> set[int]:
    """Parse numbered citations, including comma-separated groups and ranges."""
    found: set[int] = set()
    for match in re.finditer(r"\[([0-9,\s\u2013-]+)\]", text):
        for part in match.group(1).split(","):
            part = part.strip()
            if not part:
                continue
            if re.fullmatch(r"\d+", part):
                found.add(int(part))
                continue
            range_match = re.fullmatch(r"(\d+)\s*[\u2013-]\s*(\d+)", part)
            if range_match:
                start, end = int(range_match.group(1)), int(range_match.group(2))
                if end < start:
                    # A descending range is malformed; skip so the validator fails loudly.
                    found.add(-1)
                else:
                    found.update(range(start, end + 1))
    return found


def parse_citation_order(text: str) -> list[int]:
    """Return citation numbers in first-appearance order, expanding ranges."""
    order: list[int] = []
    for match in re.finditer(r"\[([0-9,\s\u2013-]+)\]", text):
        for part in match.group(1).split(","):
            part = part.strip()
            if not part:
                continue
            if re.fullmatch(r"\d+", part):
                order.append(int(part))
                continue
            range_match = re.fullmatch(r"(\d+)\s*[\u2013-]\s*(\d+)", part)
            if range_match:
                start, end = int(range_match.group(1)), int(range_match.group(2))
                order.extend(range(start, end + 1))
    return order


def main() -> None:
    text = MD.read_text(encoding="utf-8")

    # Reference list starts after the marker.
    list_part = text.split("## References")[1]
    listed = set(int(n) for n in re.findall(r"^\[(\d+)\]", list_part, flags=re.MULTILINE))

    body = text.split("## 1. Introduction")[1].split("## References")[0]
    # Remove figure legends and table bodies for the approximate main-text count.
    body_no_resources = body.split("## Figure legends")[0].split("## Tables")[0]

    citation_order = parse_citation_order(body)
    cited = set(citation_order)
    first_appearance = list(dict.fromkeys(citation_order))
    sequential = sorted(cited)
    max_citation = max(cited) if cited else 0
    all_expected = set(range(1, len(REFS) + 1))
    figures_cited = set(int(n) for n in re.findall(r"Figure\s+(\d+)", body))
    tables_cited = set(int(n) for n in re.findall(r"Table\s+(\d+)", body))

    words = len(re.findall(r"\b[\w\u00b4\u2019'-]+\b", body_no_resources))

    print(f"References in list: {len(listed)} ({min(listed) if listed else None}\u2013{max(listed) if listed else None})")
    print(f"Distinct in-text citations: {len(cited)}")
    print(f"Distinct citation numbers (sorted): {sequential}")
    print(f"First-appearance order: {first_appearance}")
    print(f"Approximate main-text words (excluding abstract/tables/figure legends/refs): {words}")
    print(f"Missing from text: {sorted(all_expected - cited)}")
    print(f"Listed but not in JSON: {sorted(listed - all_expected)}")
    print(f"JSON refs not listed: {sorted(all_expected - listed)}")
    print(f"Max citation: {max_citation}")
    print(f"Figures cited in main text: {sorted(figures_cited)}")
    print(f"Tables cited in main text: {sorted(tables_cited)}")

    problems = []
    if list_part.count("[") != sum(1 for _ in re.finditer(r"^\[\d+\]", list_part, flags=re.MULTILINE)):
        pass
    if cited != all_expected:
        problems.append("citation/reference mismatch")
    if first_appearance != sequential:
        problems.append("citations not numbered in order of first appearance")
    if listed != all_expected:
        problems.append("reference-list mismatch")
    if figures_cited != set(range(1, 9)):
        problems.append("figure reference mismatch")
    if tables_cited != set(range(1, 7)):
        problems.append("table reference mismatch")
    if words > 3300:
        problems.append("main text may exceed IJMI guidance")

    if problems:
        print("PROBLEMS:", "; ".join(problems))
        raise SystemExit(1)
    print("Validation passed.")


if __name__ == "__main__":
    main()
