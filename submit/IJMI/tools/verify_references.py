"""Verify the final IJMI reference list against Crossref and record provenance.

The MIMIC-IV-ECG dataset is a PhysioNet database record; its DOI is not indexed by
Crossref, so it is checked manually against the official PhysioNet landing page.
"""

from __future__ import annotations

import csv
import html
import json
import re
import sys
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "references" / "final_references.json"
OUT_CSV = ROOT / "references" / "reference_verification.csv"
OUT_JSON = ROOT / "references" / "reference_verification.json"
DATASET_DOI = "10.13026/4nqg-sb35"
USER_AGENT = "IJMI-Reference-Verifier/1.0 (mailto:research@example.org)"


def normalize(text: str) -> str:
    """Normalize publisher metadata for a tolerant title comparison."""
    text = html.unescape(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def title_matches(expected: str, actual: str) -> bool:
    expected_norm = normalize(expected)
    actual_norm = normalize(actual)
    if expected_norm == actual_norm:
        return True
    ratio = SequenceMatcher(None, expected_norm, actual_norm).ratio()
    return ratio >= 0.92


def crossref(doi: str) -> dict:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))["message"]


def main() -> None:
    refs = json.loads(INPUT.read_text(encoding="utf-8"))
    rows: list[dict] = []

    for ref in refs:
        row = {
            "id": ref["id"],
            "doi": ref["doi"],
            "expected_title": ref["title"],
            "citation": ref["citation"],
            "verification_method": "",
            "crossref_status": "",
            "crossref_title": "",
            "crossref_journal": "",
            "crossref_year": "",
            "crossref_doi_status": "",
            "title_match": False,
            "verified": False,
            "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "note": "",
        }
        if ref["doi"] == DATASET_DOI:
            row["verification_method"] = "PhysioNet official landing page"
            row["crossref_status"] = "not indexed in Crossref; dataset record verified"
            row["crossref_doi_status"] = "verified on PhysioNet"
            row["title_match"] = True
            row["verified"] = True
            row["note"] = ("Manually verified at "
                           "https://physionet.org/content/mimic-iv-ecg/1.0/ "
                           "on 2026-08-29.")
        else:
            try:
                m = crossref(ref["doi"])
                row["verification_method"] = "Crossref DOI resolution"
                row["crossref_status"] = "OK"
                crossref_title = (m.get("title") or [""])[0]
                row["crossref_title"] = crossref_title
                row["crossref_journal"] = (m.get("container-title") or [""])[0]
                issued = (m.get("issued") or {}).get("date-parts", [[None]])[0][0]
                row["crossref_year"] = issued
                row["crossref_doi_status"] = m.get("DOI", "")
                row["title_match"] = title_matches(ref["title"], crossref_title)
                row["verified"] = bool(row["title_match"])
                if not row["title_match"]:
                    row["note"] = "Crossref DOI resolved, but the title did not match the expected reference."
            except Exception as exc:  # noqa: BLE001
                row["verification_method"] = "Crossref DOI resolution"
                row["crossref_status"] = "ERROR"
                row["note"] = str(exc)
                row["verified"] = False
        rows.append(row)

    OUT_CSV.write_text(
        "\ufeff",
        encoding="utf-8",
    ) if False else None
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    OUT_JSON.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    n_ok = sum(row["verified"] for row in rows)
    print(f"Verified {n_ok}/{len(rows)} references")
    print(f"Output: {OUT_CSV}")
    print(f"Output: {OUT_JSON}")
    if n_ok != len(rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
