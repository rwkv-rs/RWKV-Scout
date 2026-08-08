import unittest
from unittest.mock import patch

from tools.web_search_keyless import _page_excerpt
from utils.html_markdown import html_to_markdown, markdown_table


class HtmlMarkdownTests(unittest.TestCase):
    def test_html_table_preserves_rows_and_columns(self):
        value = html_to_markdown(
            "<h2>Stations</h2><table><thead><tr><th>Name</th><th>Line</th></tr></thead>"
            "<tbody><tr><td>罗湖</td><td>1号线</td></tr><tr><td>国贸</td><td>1号线</td></tr></tbody></table>"
        )
        self.assertIn("## Stations", value)
        self.assertIn("| Name | Line |", value)
        self.assertIn("| 罗湖 | 1号线 |", value)
        self.assertIn("| 国贸 | 1号线 |", value)

    def test_html_table_with_omitted_cell_and_row_end_tags_preserves_relations(self):
        value = html_to_markdown(
            "<details><summary>History</summary><table><thead><tr>"
            "<th>Version<th>Changes<tbody><tr><td>v21.0.0"
            "<td><p>No longer experimental.<tr><td>v18.0.0"
            "<td>No longer behind the flag.</table></details>"
        )

        self.assertIn("| Version | Changes |", value)
        self.assertIn("| v21.0.0 | No longer experimental. |", value)
        self.assertIn("| v18.0.0 | No longer behind the flag. |", value)

    def test_links_and_lists_remain_markdown(self):
        value = html_to_markdown(
            '<p>Source <a href="https://example.com">link</a></p><ul><li>first</li><li>second</li></ul>'
        )
        self.assertIn("[link](https://example.com)", value)
        self.assertIn("- first", value)
        self.assertIn("- second", value)

    def test_page_chrome_is_not_projected_as_evidence(self):
        value = html_to_markdown(
            "<header><p>Published: 2026-07-30</p><nav><a href='/'>Home</a><a href='/docs'>Docs</a></nav></header>"
            "<main><h1>Release date</h1><p>The release date is 2026-07-30.</p></main>"
            "<footer>Cookie settings</footer>"
        )
        self.assertIn("# Release date", value)
        self.assertIn("2026-07-30", value)
        self.assertIn("Published: 2026-07-30", value)
        self.assertNotIn("Home", value)
        self.assertNotIn("Cookie settings", value)

    def test_role_navigation_and_sphinx_related_chrome_are_removed(self):
        value = html_to_markdown(
            '<div class="related" role="navigation">index modules next previous</div>'
            '<main><h1>Release notes</h1><p>The release date is 2026-07-29.</p></main>'
        )
        self.assertNotIn("index modules", value)
        self.assertIn("# Release notes", value)

    def test_markdown_table_pads_short_rows(self):
        value = "\n".join(markdown_table([["Name", "Line"], ["罗湖"]]))
        self.assertIn("| 罗湖 |  |", value)


    def test_inline_code_whitespace_is_not_destroyed(self):
        value = html_to_markdown(
            "<p>Use <code>-X</code> <code>gil=0</code> and run "
            "<code>python</code> <code>-VV</code>.</p>"
        )
        self.assertIn("`-X` `gil=0`", value)
        self.assertIn("`python` `-VV`", value)

    def test_superscript_and_subscript_preserve_scientific_meaning(self):
        value = html_to_markdown(
            "<p>Oceans contain 5 x 10<sup>22</sup> g and H<sub>2</sub>O.</p>"
        )
        self.assertIn("5 x 10^22 g", value)
        self.assertIn("H_{2}O", value)

    def test_anchor_only_toc_table_is_removed_but_data_table_remains(self):
        value = html_to_markdown(
            "<table><tr><td><a href='#a'>Section A</a><a href='#b'>Section B</a></td></tr></table>"
            "<table><tr><th>Name</th><th>Value</th></tr><tr><td>A</td><td>42</td></tr></table>"
        )
        self.assertNotIn("Section A", value)
        self.assertIn("| Name | Value |", value)
        self.assertIn("| A | 42 |", value)

    def test_page_excerpt_selects_main_before_raw_html_bounding(self):
        html = (
            "<html><body>"
            + "<nav>menu entry</nav>" * 20000
            + "<main><h1>Kubernetes v1.30</h1>"
            "<p>Released Wednesday, April 17, 2024.</p></main></body></html>"
        )
        with patch("tools.web_search_keyless.fetch_text", return_value=html):
            value = _page_excerpt("https://kubernetes.io/release", limit=2000)
        self.assertIn("Kubernetes v1.30", value)
        self.assertIn("April 17, 2024", value)
        self.assertNotIn("menu entry", value)


if __name__ == "__main__":
    unittest.main()
