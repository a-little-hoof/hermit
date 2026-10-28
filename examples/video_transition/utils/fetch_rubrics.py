"""Extract SEGUE Tables 19-26 into a local evaluator cache with provenance."""
from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from examples.video_transition.utils.protocol import METRICS

PAPER_URL = "https://arxiv.org/html/2609.38691v2"
ROOT = Path(__file__).resolve().parents[3]


class TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables = []
        self.figure = None
        self.in_caption = False
        self.in_table = False
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "figure" and "ltx_table" in attrs.get("class", "").split():
            self.figure = {"caption": "", "rows": []}
        if self.figure is None:
            return
        if tag == "figcaption":
            self.in_caption = True
        elif tag == "table":
            self.in_table = True
        elif tag == "tr" and self.in_table:
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = ""

    def handle_endtag(self, tag):
        if self.figure is None:
            return
        if tag == "figcaption":
            self.in_caption = False
        elif tag in ("td", "th") and self.cell is not None:
            self.row.append(" ".join(self.cell.split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.figure["rows"].append(self.row)
            self.row = None
        elif tag == "table":
            self.in_table = False
        elif tag == "figure":
            self.tables.append(self.figure)
            self.figure = None

    def handle_data(self, data):
        if self.figure is not None and self.in_caption:
            self.figure["caption"] += data
        if self.cell is not None:
            self.cell += data


def extract_rubrics(html: str) -> dict[str, list[str]]:
    parser = TableParser()
    parser.feed(html)
    found = {}
    for table in parser.tables:
        metric = next((name for name in METRICS if re.search(r"\(" + re.escape(name) + r"\)", table["caption"])), None)
        if metric is None:
            continue
        rows = {int(row[0]): row[1] for row in table["rows"]
                if len(row) >= 2 and re.fullmatch(r"\d+", row[0])}
        if set(rows) != set(range(1, 11)):
            raise ValueError(f"{metric}: expected ten numbered criteria, found {sorted(rows)}")
        found[metric] = [rows[number] for number in range(1, 11)]
    if set(found) != set(METRICS):
        raise ValueError(f"missing metric tables: {sorted(set(METRICS) - found.keys())}")
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html-file", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / ".demo/video-transition/evaluator/rubrics.json")
    args = parser.parse_args()
    if args.html_file:
        raw = args.html_file.read_bytes()
    else:
        request = urllib.request.Request(PAPER_URL, headers={"User-Agent": "hermit-video-transition/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read()
    rubrics = extract_rubrics(raw.decode("utf-8"))
    digest = hashlib.sha256(json.dumps(rubrics, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    payload = {"source_url": PAPER_URL, "paper_html_sha256": hashlib.sha256(raw).hexdigest(),
               "rubric_sha256": digest, "metrics": rubrics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "criteria_counts": {k: len(v) for k,v in rubrics.items()},
                      "rubric_sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
