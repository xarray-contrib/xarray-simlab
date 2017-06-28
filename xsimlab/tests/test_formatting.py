import unittest

from xsimlab.formatting import pretty_print, maybe_truncate, wrap_indent


class TestFormatting(unittest.TestCase):

    def test_maybe_truncate(self):
        self.assertEqual(maybe_truncate('test', 10), 'test')
        self.assertEqual(maybe_truncate('longteststring', 10), 'longtes...')

    def test_pretty_print(self):
        self.assertEqual(pretty_print('test', 10), 'test' + ' ' * 6)

    def test_wrap_indent(self):
        text = "line1\nline2"

        expected = '-line1\n line2'
        self.assertEqual(wrap_indent(text, start='-'), expected)

        expected = 'line1\n line2'
        self.assertEqual(wrap_indent(text, length=1), expected)
