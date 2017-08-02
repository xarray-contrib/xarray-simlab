from xsimlab.formatting import pretty_print, maybe_truncate, wrap_indent


def test_maybe_truncate():
    assert maybe_truncate('test', 10) == 'test'
    assert maybe_truncate('longteststring', 10) == 'longtes...'


def test_pretty_print():
    assert pretty_print('test', 10) == 'test' + ' ' * 6


def test_wrap_indent():
    text = "line1\nline2"

    expected = '-line1\n line2'
    assert wrap_indent(text, start='-') == expected

    expected = 'line1\n line2'
    assert wrap_indent(text, length=1) == expected
