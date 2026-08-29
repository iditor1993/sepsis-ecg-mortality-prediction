"""Renumber manuscript and reference JSON so citations follow first-appearance order."""

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "manuscript_IJMI.md"
REFS = ROOT / "references" / "final_references.json"

OLD_TO_NEW = {
    43: 38,
    44: 39,
    40: 40,
    41: 41,
    38: 42,
    39: 43,
    42: 44,
}
NEW_TO_OLD = {new: old for old, new in OLD_TO_NEW.items()}


def map_token(token: str) -> str:
    token = token.strip()
    if re.fullmatch(r"\d+", token):
        return str(OLD_TO_NEW.get(int(token), int(token)))
    range_match = re.fullmatch(r"(\d+)\s*([\u2013-])\s*(\d+)", token)
    if range_match:
        a = OLD_TO_NEW.get(int(range_match.group(1)), int(range_match.group(1)))
        b = OLD_TO_NEW.get(int(range_match.group(3)), int(range_match.group(3)))
        return f"{a}{range_match.group(2)}{b}"
    return token


def replace_group(match: re.Match) -> str:
    content = match.group(1)
    parts = content.split(",")
    return "[" + ",".join(map_token(part) for part in parts) + "]"


def main() -> None:
    text = MD.read_text(encoding="utf-8")
    head, tail = text.split("## References", 1)
    head = re.sub(r"\[([0-9,\s\u2013-]+)\]", replace_group, head)

    reference_block = tail
    reference_end = reference_block.find("\n---\n")
    list_text = reference_block[:reference_end] if reference_end >= 0 else reference_block
    rest = reference_block[reference_end:] if reference_end >= 0 else ""

    lines = list_text.splitlines(keepends=True)
    references = {}
    for line in lines:
        match = re.match(r"^\[(\d+)\]\s*(.*)$", line, flags=re.S)
        if match:
            references[int(match.group(1))] = match.group(2).rstrip("\r\n")
    new_lines = []
    for new_id in range(1, 47):
        old_id = NEW_TO_OLD.get(new_id, new_id)
        content = references[old_id]
        new_lines.append(f"[{new_id}] {content}\n")
    new_text = head + "## References\n\n" + "".join(new_lines) + rest
    MD.write_text(new_text, encoding="utf-8")

    refs = json.loads(REFS.read_text(encoding="utf-8"))
    by_old = {ref["id"]: ref for ref in refs}
    reordered = []
    for new_id in range(1, 47):
        old_id = NEW_TO_OLD.get(new_id, new_id)
        ref = dict(by_old[old_id])
        ref["id"] = new_id
        reordered.append(ref)
    REFS.write_text(json.dumps(reordered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("References renumbered; reference list and JSON updated.")


if __name__ == "__main__":
    main()
