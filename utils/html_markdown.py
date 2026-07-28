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


_SKIP_TAGS = {"script", "style", "noscript", "svg", "template"}
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

    @property
    def active_parts(self) -> list[str]:
        return self.table_cell if self.table_cell is not None else self.line_parts

    def _append(self, value: str) -> None:
        value = re.sub(r"\s+", " ", unescape(value or ""))
        if value.strip():
            self.active_parts.append(value)

    def _flush_line(self) -> None:
        value = re.sub(r"\s+", " ", "".join(self.line_parts)).strip()
        if value:
            self.lines.append(f"{self.line_prefix}{value}".rstrip())
        self.line_parts = []
        self.line_prefix = ""

    def _flush_table_row(self) -> None:
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
            if tag in _SKIP_TAGS:
                self.skip_depth += 1
            return
        if tag in _SKIP_TAGS:
            self.skip_depth = 1
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
        elif tag == "img":
            alt = str(attributes.get("alt") or "").strip()
            src = str(attributes.get("src") or "").strip()
            if alt or src:
                self._append(f"![{alt}]({src})")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if self.skip_depth:
            if tag in _SKIP_TAGS:
                self.skip_depth -= 1
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
    return result[:max_chars] if max_chars else result


__all__ = ["html_to_markdown", "markdown_table"]
