
from pystig import stig

# Other modules rely on non-standard libraries (e.g. numpy).
try:
    from pystig import prob
except ImportError as e:
    import sys
    print >>sys.stderr, 'Failed import:', e
