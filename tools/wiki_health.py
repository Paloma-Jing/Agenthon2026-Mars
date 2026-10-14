"""Check project wiki structure and regenerate the extracted link graph."""

import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
WIKI = ROOT / "wiki"
REQUIRED = ("title", "type", "tags", "sources", "last_updated")


def main():
    errors = []
    for folder in ("raw", "wiki", "wiki/sources", "wiki/entities", "wiki/concepts",
                   "wiki/syntheses", "graph", "tools", "inbox", "archive"):
        if not (ROOT / folder).is_dir():
            errors.append(f"Missing directory: {folder}")
    pages = sorted(WIKI.rglob("*.md"))
    names = {}
    edges = []
    for page in pages:
        if page.stem in names:
            errors.append(f"Ambiguous page name: {page.stem}")
        names[page.stem] = page
    for page in pages:
        text = page.read_text()
        parts = text.split("---", 2)
        if not text.startswith("---\n") or len(parts) < 3:
            errors.append(f"Missing frontmatter: {page.name}")
        else:
            for key in REQUIRED:
                if not re.search(rf"^{key}:", parts[1], re.M):
                    errors.append(f"Missing {key}: {page.name}")
            if len(parts[2].strip()) < 40:
                errors.append(f"Empty or stub body: {page.name}")
        for target in re.findall(r"\[\[([^\]|#]+)(?:[^\]]*)\]\]", text):
            name = Path(target).stem
            if name not in names:
                errors.append(f"Broken link: {page.stem} -> {target}")
            else:
                edges.append({"source": page.stem, "target": name, "type": "EXTRACTED"})
    index_targets = {e["target"] for e in edges if e["source"] == "index"}
    for name in names:
        if name != "index" and name not in index_targets:
            errors.append(f"Not indexed: {name}")
    inbound = {e["target"] for e in edges}
    orphans = sorted(set(names) - inbound)
    log = names.get("log")
    if not log or not re.search(r"^## \[\d{4}-\d{2}-\d{2}\]", log.read_text(), re.M):
        errors.append("Operation log has no dated entry")
    graph = {"nodes": [{"id": p.stem, "path": str(p.relative_to(ROOT))} for p in pages],
             "edges": edges}
    report = {"pages": len(pages), "links": len(edges), "errors": errors, "orphans": orphans,
              "scope": "Structural checks only; factual accuracy and stale rules require review."}
    (ROOT / "graph").mkdir(exist_ok=True)
    (ROOT / "graph/graph.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "graph/health.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(bool(errors or orphans))


if __name__ == "__main__":
    main()
