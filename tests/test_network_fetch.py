from __future__ import annotations

import unittest

from utils.network_fetch import decode_http_body


class NetworkEncodingTests(unittest.TestCase):
    def test_utf8_body_wins_over_broken_iso88591_header(self):
        body = "深圳地铁一号线站点".encode("utf-8")
        self.assertEqual(
            decode_http_body(body, declared_encoding="text/html; charset=ISO-8859-1"),
            "深圳地铁一号线站点",
        )

    def test_gb18030_body_is_decoded_when_utf8_is_invalid(self):
        body = "深圳地铁一号线站点".encode("gb18030")
        self.assertEqual(decode_http_body(body), "深圳地铁一号线站点")

    def test_html_meta_charset_is_considered(self):
        body = '<meta charset="gb18030"><p>深圳地铁</p>'.encode("gb18030")
        self.assertIn("深圳地铁", decode_http_body(body))


if __name__ == "__main__":
    unittest.main()
