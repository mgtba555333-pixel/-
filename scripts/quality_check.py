from __future__ import annotations

import re
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    "app/main.py",
    "app/model_service.py",
    "web/index.html",
    "web/app.css",
    "web/app.js",
    "requirements.txt",
    "Dockerfile",
    "docker-compose.yml",
)


class IdCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name == "id" and value:
                self.ids.append(value)


def main() -> int:
    errors: list[str] = []
    for relative in REQUIRED_FILES:
        if not (ROOT / relative).is_file():
            errors.append(f"Missing required file: {relative}")

    html_path = ROOT / "web/index.html"
    js_path = ROOT / "web/app.js"
    css_path = ROOT / "web/app.css"
    if errors:
        print("\n".join(f"ERROR: {item}" for item in errors))
        return 1

    html = html_path.read_text(encoding="utf-8")
    js = js_path.read_text(encoding="utf-8")
    css = css_path.read_text(encoding="utf-8")

    parser = IdCollector()
    parser.feed(html)
    seen: set[str] = set()
    duplicates: set[str] = set()
    for element_id in parser.ids:
        if element_id in seen:
            duplicates.add(element_id)
        seen.add(element_id)
    if duplicates:
        errors.append("Duplicate HTML IDs: " + ", ".join(sorted(duplicates)))

    js_ids = set(re.findall(r'\$\(["\']([^"\']+)["\']\)', js))
    missing = js_ids - seen
    if missing:
        errors.append("JavaScript references missing HTML IDs: " + ", ".join(sorted(missing)))

    if 'href="/assets/app.css"' not in html or 'src="/assets/app.js"' not in html:
        errors.append("HTML must load the stylesheet and script from /assets.")
    if "Math.random" in js or "mockPrediction" in js:
        errors.append("Fake/random prediction code is not allowed.")
    if "model_not_ready" not in (ROOT / "app/main.py").read_text(encoding="utf-8"):
        errors.append("Backend must explicitly handle missing model weights.")
    if "not a medical" not in (ROOT / "README.md").read_text(encoding="utf-8").lower():
        errors.append("README must include the medical-use disclaimer.")
    if ":root" not in css or "@media" not in css:
        errors.append("Responsive CSS design tokens/media queries are missing.")

    if errors:
        print("\n".join(f"ERROR: {item}" for item in errors))
        return 1

    print(f"Quality checks passed: {len(REQUIRED_FILES)} required files, {len(seen)} unique HTML IDs, {len(js_ids)} JS ID references.")
    print("No dummy prediction marker found; model availability is handled explicitly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
