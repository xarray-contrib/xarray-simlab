from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys

PY2 = sys.version_info[0] < 3
PY3 = sys.version_info[0] >= 3

if PY3:  # pragma: no cover
    basestring = str
    unicode_type = str
    bytes_type = bytes

    def iteritems(d):
        return iter(d.items())

    def itervalues(d):
        return iter(d.values())

    range = range
    zip = zip
    from functools import reduce
    import builtins
else:  # pragma: no cover
    # Python 2
    basestring = basestring
    unicode_type = unicode
    bytes_type = str

    def iteritems(d):
        return d.iteritems()

    def itervalues(d):
        return d.itervalues()

    range = xrange
    from itertools import izip as zip, imap as map
    reduce = reduce
    import __builtin__ as builtins
