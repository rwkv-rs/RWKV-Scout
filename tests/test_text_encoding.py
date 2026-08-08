import unittest

from utils.text_encoding import repair_mojibake


class TextEncodingTests(unittest.TestCase):
    def test_repairs_cp1252_utf8_mojibake(self):
        original = "深圳站点"
        broken = original.encode("utf-8").decode("cp1252")
        self.assertEqual(repair_mojibake(broken), original)

    def test_keeps_normal_ascii(self):
        value = '{"name":"web_search","arguments":{}}'
        self.assertEqual(repair_mojibake(value), value)

    def test_repairs_utf8_smart_punctuation_decoded_as_cp1252(self):
        original = "you\u2019re ready \u2014 run rustup\u2011init"
        broken = original.encode("utf-8").decode("cp1252")
        self.assertEqual(repair_mojibake(broken), original)
