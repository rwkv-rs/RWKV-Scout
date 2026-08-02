import unittest

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

    def test_markdown_table_pads_short_rows(self):
        value = "\n".join(markdown_table([["Name", "Line"], ["罗湖"]]))
        self.assertIn("| 罗湖 |  |", value)


    def test_anchor_only_toc_table_is_removed_but_data_table_remains(self):
        value = html_to_markdown(
            "<table><tr><td><a href='#a'>Section A</a><a href='#b'>Section B</a></td></tr></table>"
            "<table><tr><th>Name</th><th>Value</th></tr><tr><td>A</td><td>42</td></tr></table>"
        )
        self.assertNotIn("Section A", value)
        self.assertIn("| Name | Value |", value)
        self.assertIn("| A | 42 |", value)


if __name__ == "__main__":
    unittest.main()
