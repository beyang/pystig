
import stig

# Other modules rely on non-standard libraries (e.g. numpy).
try:
    import prob
except ImportError as e:
    import sys
    print >>sys.stderr, 'Failed import:', e
