"""Small dependency-free HTML to Markdown converter for retrieval evidence.

The retrieval pipeline needs readable text, but flattening HTML into one line
destroys the row/column relationship that makes lists and tables verifiable.
This converter keeps headings, links, lists and HTML tables as Markdown while
remaining deterministic and bounded for model evidence prompts.
"""

from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser
from typing import Iterable


# Page chrome is not factual page evidence.  The original response remains in
# the retrieval trace; this only removes menus and controls from the bounded
# Markdown projection used by ranking, chunking, and synthesis.
_SKIP_TAGS = {
    "head",
    "script",
    "style",
    "noscript",
    "svg",
    "template",
    "nav",
    "footer",
    "form",
}
_BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "div",
    "dl",
    "dt",
    "dd",
    "figure",
    "figcaption",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "main",
    "nav",
    "p",
    "pre",
    "section",
}
_VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
_CHROME_TOKENS = {
    "breadcrumb",
    "breadcrumbs",
    "headerlink",
    "navigation",
    "navbar",
    "related",
    "skip-link",
    "sphinxsidebar",
    "theme-switcher",
    "theme-toggle",
    "wy-nav",
}


def _is_chrome_container(tag: str, attributes: dict[str, str | None]) -> bool:
    if tag not in {"aside", "div", "section", "ul"}:
        return False
    role = str(attributes.get("role") or "").casefold()
    if role in {"navigation", "search"}:
        return True
    identity = " ".join(
        str(attributes.get(key) or "").casefold()
        for key in ("id", "class")
    )
    tokens = set(re.findall(r"[a-z0-9_-]+", identity))
    return any(
        token in _CHROME_TOKENS or any(token.startswith(prefix + "-") for prefix in _CHROME_TOKENS)
        for token in tokens
    )


def _clean_cell(value: object) -> str:
    text = unescape(str(value or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace("|", r"\|")


def markdown_table(rows: Iterable[Iterable[object]]) -> list[str]:
    """Render rows as a valid Markdown table, retaining every cell."""

    normalized = [[_clean_cell(cell) for cell in row] for row in rows]
    normalized = [row for row in normalized if any(row)]
    if not normalized:
        return []
    width = max(len(row) for row in normalized)
    normalized = [row + [""] * (width - len(row)) for row in normalized]
    output = [
        "| " + " | ".join(normalized[0]) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    output.extend("| " + " | ".join(row) + " |" for row in normalized[1:])
    return output


def _remove_anchor_only_toc(value: str) -> str:
    """Remove one-row in-page TOC tables while retaining factual tables."""

    lines = value.splitlines()
    output: list[str] = []
    index = 0
    anchor_link = re.compile(r"\[[^\]]+\]\(#[^)]+\)")
    while index < len(lines):
        current = lines[index].strip()
        next_line = lines[index + 1].strip() if index + 1 < len(lines) else ""
        cells = [cell.strip() for cell in current.strip("|").split("|")]
        is_separator = bool(next_line) and all(
            re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in next_line.strip("|").split("|")
        )
        is_anchor_only = bool(cells) and all(
            cell and not anchor_link.sub("", cell).strip() for cell in cells
        )
        if current.startswith("|") and is_separator and is_anchor_only:
            index += 2
            continue
        output.append(lines[index])
        index += 1
    return "\n".join(output)


class _MarkdownParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lines: list[str] = []
        self.line_parts: list[str] = []
        self.line_prefix = ""
        self.skip_depth = 0
        self.list_stack: list[tuple[str, int]] = []
        self.table_rows: list[list[str]] = []
        self.table_row: list[str] | None = None
        self.table_cell: list[str] | None = None
        self.table_depth = 0
        self.link_stack: list[str] = []
        self.pre_depth = 0
        self.pre_parts: list[str] = []

    @property
    def active_parts(self) -> list[str]:
        return self.table_cell if self.table_cell is not None else self.line_parts

    def _append(self, value: str) -> None:
        raw = unescape(value or "")
        if self.pre_depth:
            self.pre_parts.append(raw.replace("\r\n", "\n").replace("\r", "\n"))
            return
        value = re.sub(r"\s+", " ", raw)
        if value.strip():
            self.active_parts.append(value)
            return
        # HTML commonly separates adjacent inline elements with a text node
        # containing only whitespace. Dropping it corrupts source facts such
        # as ``-X gil`` and ``python -VV`` before they reach the model.
        parts = self.active_parts
        if raw and any(char.isspace() for char in raw) and parts and not parts[-1].endswith((" ", "\n")):
            parts.append(" ")

    def _flush_line(self) -> None:
        value = re.sub(r"\s+", " ", "".join(self.line_parts)).strip()
        if value:
            self.lines.append(f"{self.line_prefix}{value}".rstrip())
        self.line_parts = []
        self.line_prefix = ""

    def _flush_table_row(self) -> None:
        # HTML permits closing ``th``, ``td`` and ``tr`` tags to be omitted.
        # Documentation generators commonly emit compact tables such as
        # ``<tr><th>Version<th>Changes<tbody><tr><td>v21<td>Stable``.
        # ``HTMLParser`` does not synthesize the omitted end tags, so flush the
        # active cell whenever a row boundary is observed. Without this, the
        # last cell of one row is shifted into the next row and version/status
        # relations are corrupted before they reach the evidence extractor.
        if self.table_cell is not None:
            if self.table_row is None:
                self.table_row = []
            self.table_row.append(re.sub(r"\s+", " ", "".join(self.table_cell)).strip())
            self.table_cell = None
        if self.table_row is not None:
            self.table_rows.append(self.table_row)
            self.table_row = None

    def _emit_table(self) -> None:
        self._flush_table_row()
        if self.table_rows:
            self.lines.extend(markdown_table(self.table_rows))
            self.lines.append("")
        self.table_rows = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.casefold()
        attributes = dict(attrs)
        if self.skip_depth:
            if tag not in _VOID_TAGS:
                self.skip_depth += 1
            return
        if tag in _SKIP_TAGS or _is_chrome_container(tag, attributes):
            self.skip_depth = 1
            return
        if tag == "pre" and self.table_cell is None:
            self._flush_line()
            self.pre_depth += 1
            if self.pre_depth == 1:
                self.pre_parts = []
            return
        if self.pre_depth:
            if tag == "br":
                self.pre_parts.append("\n")
            return
        if tag == "table":
            self._flush_line()
            self.table_depth += 1
            return
        if tag == "tr" and self.table_depth:
            self._flush_table_row()
            self.table_row = []
            return
        if tag in {"td", "th"} and self.table_depth:
            if self.table_row is None:
                self.table_row = []
            if self.table_cell is not None:
                self.table_row.append("".join(self.table_cell))
            self.table_cell = []
            return
        if tag == "a":
            href = str(attributes.get("href") or "").strip()
            self._append("[")
            self.link_stack.append(href)
            return
        if tag == "br":
            if self.table_cell is not None:
                self._append(" ")
            else:
                self._flush_line()
            return
        if tag in _BLOCK_TAGS and self.table_cell is None:
            self._flush_line()
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"} and self.table_cell is None:
            self.line_prefix = "#" * int(tag[1]) + " "
        elif tag == "li" and self.table_cell is None:
            self._flush_line()
            if self.list_stack:
                kind, number = self.list_stack[-1]
                self.line_prefix = f"{number}. " if kind == "ol" else "- "
                self.list_stack[-1] = (kind, number + 1)
            else:
                self.line_prefix = "- "
        elif tag in {"ul", "ol"} and self.table_cell is None:
            self.list_stack.append((tag, 1))
        elif tag in {"strong", "b"}:
            self._append("**")
        elif tag in {"em", "i"}:
            self._append("*")
        elif tag == "code":
            self._append("`")
        elif tag == "sup":
            # Preserve scientific notation such as 10<sup>22</sup> as 10^22
            # instead of flattening it into the ambiguous integer 1022.
            self._append("^")
        elif tag == "sub":
            self._append("_{")
        elif tag == "img":
            alt = str(attributes.get("alt") or "").strip()
            src = str(attributes.get("src") or "").strip()
            if alt or src:
                self._append(f"![{alt}]({src})")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if self.skip_depth:
            self.skip_depth -= 1
            return
        if tag == "pre" and self.pre_depth:
            self.pre_depth -= 1
            if not self.pre_depth:
                body = "".join(self.pre_parts).strip("\n")
                if body.strip():
                    fence = "````" if "```" in body else "```"
                    self.lines.extend([fence, body.rstrip(), fence, ""])
                self.pre_parts = []
            return
        if self.pre_depth:
            return
        if tag in {"td", "th"} and self.table_cell is not None:
            if self.table_row is None:
                self.table_row = []
            self.table_row.append(re.sub(r"\s+", " ", "".join(self.table_cell)).strip())
            self.table_cell = None
            return
        if tag == "tr" and self.table_depth:
            self._flush_table_row()
            return
        if tag == "table" and self.table_depth:
            self.table_depth -= 1
            if not self.table_depth:
                self._emit_table()
            return
        if tag == "a":
            href = self.link_stack.pop() if self.link_stack else ""
            self._append(f"]({href})" if href else "]")
            return
        if tag in {"strong", "b"}:
            self._append("**")
        elif tag in {"em", "i"}:
            self._append("*")
        elif tag == "code":
            self._append("`")
        elif tag == "sub":
            self._append("}")
        elif tag == "li" and self.table_cell is None:
            self._flush_line()
        elif tag in {"ul", "ol"} and self.table_cell is None:
            self._flush_line()
            if self.list_stack:
                self.list_stack.pop()
        elif tag in _BLOCK_TAGS and self.table_cell is None:
            self._flush_line()

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            self._append(data)

    def finish(self) -> str:
        if self.pre_depth and self.pre_parts:
            body = "".join(self.pre_parts).strip("\n")
            if body.strip():
                fence = "````" if "```" in body else "```"
                self.lines.extend([fence, body.rstrip(), fence, ""])
            self.pre_depth = 0
            self.pre_parts = []
        if self.table_cell is not None and self.table_row is not None:
            self.table_row.append("".join(self.table_cell))
            self.table_cell = None
        self._flush_table_row()
        if self.table_rows:
            self._emit_table()
        self._flush_line()
        output: list[str] = []
        previous_blank = False
        for line in self.lines:
            line = line.rstrip()
            blank = not line
            if blank and previous_blank:
                continue
            output.append(line)
            previous_blank = blank
        return "\n".join(output).strip()


def html_to_markdown(value: object, max_chars: int | None = None) -> str:
    """Convert HTML into bounded Markdown without flattening tables."""

    source = str(value or "")
    if not source.strip():
        return ""
    parser = _MarkdownParser()
    parser.feed(source)
    parser.close()
    result = parser.finish()
    if not result and "<" not in source:
        result = "\n".join(" ".join(line.split()) for line in source.splitlines() if line.strip())
    result = _remove_anchor_only_toc(result)
    return result[:max_chars] if max_chars else result


__all__ = ["html_to_markdown", "markdown_table"]
