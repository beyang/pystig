#!/usr/bin/env python

'''
A set of reinvented wheels, "useful utilities", and so forth.

Ought to work with both Python 3 and recent Python 2s (tested with
2.7.5).

A few are derivatives of utilities from
http://aima.cs.berkeley.edu/python/utils.html.
'''

from __future__ import print_function

import collections
import datetime
import getopt
import json
import math
import operator
import pickle
import pprint
import random
import subprocess
import sys

########################################################################
# Python 2/3 compatibility/interop

def print_(*args):
    '''
    `print` is not a function in Python 2 but `print_` will be a
    function in any Python.
    '''
    print(*args)

def printerr(x):
    sys.stderr.write(x)
    sys.stderr.write('\n')

try:
    xrange
except NameError:
    xrange = range

# This might be a terrible idea.
try:
    unicode
except NameError:
    unicode = str

########################################################################
# Unix/OS

def sh(arg, shell=True, bash=False):
    '''Issue a shell command as subprocess.'''
    kwargs = {}
    if bash:
        kwargs['executable'] = '/bin/bash'
    return subprocess.check_call(arg, shell=shell, **kwargs)

def sh_slurp(arg, shell=True, bash=False):
    '''Issue a shell command as subprocess and return its output'''
    kwargs = {}
    if bash:
        kwargs['executable'] = '/bin/bash'
    return subprocess.check_output(arg, shell=shell, universal_newlines=True, **kwargs)

def system(arg):
    return sh(arg, shell=False)

def system_slurp(arg):
    return sh_slurp(arg, shell=False)

########################################################################
# Iteration

def chunked(n, it):
    assert n > 0
    buf = []
    for i, x in enumerate(it):
        buf.append(x)
        if (i+1) % n == 0:
            yield buf
            buf = []
    if buf:
        yield buf

def take(n, it):
    buf = []
    for i, x in enumerate(it):
        if i == n:
            break
        buf.append(x)
    return buf

def xtake(n, it):
    for i, x in enumerate(it):
        if i == n:
            return
        yield x

def sliding(n, it):
    buf = take(n, it)
    yield tuple(buf)
    for x in it:
        buf = buf[1:] + [x]
        yield tuple(buf)

def repeat(it):
    '''
    Like itertools.chain but friendly to one-pass iterators at the cost
    of memory.
    '''
    buf = list(it)
    while True:
        for x in buf:
            yield x

def growing(n, it):
    buf = []
    for i, x in enumerate(it):
        if i % n == 0:
            yield buf
        buf.append(x)
    yield buf

def skip(n, it):
    for i, x in enumerate(it):
        if i >= n:
            yield x

def ticked(it, step, logger=print_):
    assert step > 0
    for i, x in enumerate(it):
        if i % step == 0:
            logger(i)
            yield x

def time_ticked(it, step, logger=print_):
    assert step > 0
    last = datetime.datetime.now()
    for i, x in enumerate(it):
        if i % step == 0:
            now = datetime.datetime.now()
            logger(i, str(now - last))
            last = now
        yield x

def subsample(it, rate, rng=random):
    for x in it:
        if rng.random() <= rate:
            yield x

def round_robin(n, it):
    xs = [[] for _ in xrange(n)]
    for i, x in enumerate(it):
        xs[i % n].append(x)
    return xs

########################################################################
# Collections

def concat(lists):
    '''Return the concatenation of lists.'''
    return sum(lists, [])

def make_grid(*args, **kwargs):
    '''
    Return a `len(args)`-dimensional grid.

    >>> g = make_grid(5, 7, 11)
    >>> len(g)
    5
    >>> len(g[0])
    7
    >>> len(g[0][0])
    11
    >>> g[4][6][10] is None
    True
    >>> make_grid(5, 7, 11, val=42)[0][0][0]
    42
    '''
    cur = kwargs.get('val')
    for n in reversed(args):
        cur = [cur] * n
    return cur

def getattr_trail(trail, d):
    '''
    Iterated getattr rooted at `d` for every name in `trail`.
    '''
    for t in trail:
        d = d[t]
    return d

########################################################################
# Sequences / iteration

def require_first(seq):
    try:
        return next(iter(seq))
    except StopIteration:
        raise ValueError('argument is an empty sequence')

def lfold(op, seq, init=None):
    acc = init
    for i, x in enumerate(seq):
        if i == 0 and init is None:
            acc = x
        else:
            acc = op(acc, x)
    return acc

def rfold(op, seq, init=None):
    return lfold(flipped(op), reversed(seq), init)

def cumsum(xs):
    '''Cumulative sum.'''
    return [sum(xs[:i+1]) for i in xrange(len(xs))]

def diff(xs):
    return [xs[i] - xs[i-1] for i in xrange(1, len(xs))]

# Useful for abbreviated tuple manipulation (e.g. sorted(pairs, key=snd))

def first(x):
    return x[0]

def second(x):
    return x[1]

def third(x):
    return x[2]

def fourth(x):
    return x[3]

def fifth(x):
    return x[4]

fst = first
snd = second

def enumeration(xs):
    '''
    Give each unique element of xs an integer in 0..len(xs) in order of
    first appearance.
    '''
    there = {}
    back = []
    for i, x in enumerate(xs):
        if x not in there:
            there[x] = len(there)
            back.append(x)
    return there, back

def arguniq(xs):
    seen = set()
    args = []
    uniqs = []
    for i, x in enumerate(xs):
        if x not in seen:
            seen.add(x)
            args.append(i)
            uniqs.append(x)
    return uniqs, args

def argsorted(xs):
    return sorted(xrange(len(xs)), key=xs.__getitem__)

########################################################################
# Strings

def deepsplit(s, seps):
    assert type(s) is str
    if len(seps) == 0:
        return s
    else:
        return [deepsplit(x, seps[1:]) for x in s.split(seps[0])]

class StrDB(collections.Mapping):
    '''
    Maps strings to integers and vice versa.
    '''
    def __init__(self, capacity=0):
        self.stoi = {}
        self.itos = list(xrange(capacity))

    def index_of(self, s):
        return self.stoi.get(s, -1)

    def lookup(self, i):
        return self.itos[i]

    def insert(self, s):
        '''Insert a string forever into this db.'''
        if not isinstance(s, str):
            raise TypeError('StrDB can only insert strs.')
        if s in self.stoi:
            i = self.stoi[s]
        else:
            i = len(self.itos)
            self.itos.append(s)
            self.stoi[s] = i
        return i

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, x):
        if isinstance(x, str):
            return self.index_of(x)
        if isinstance(x, int):
            return self.lookup(x)
        raise TypeError('StrDB subscript is neither int not str.')

    def __iter__(self):
        return iter(self.stoi)

    def __contains__(self, x):
        if isinstance(x, str):
            return x in self.stoi
        if isinstance(x, int):
            return 0 <= x < len(self.itos)
        raise TypeError('StrDB __contains__ given neither int nor str.')

########################################################################
# Functions

def curry(f, *args, **kwargs):
    def g(*args_, **kwargs_):
        kwargs.update(kwargs_)
        return f(*(args + args_), **kwargs)
    g.__name__ = f.__name__
    g.__doc__ = f.__doc__
    return g

def compose(f, g):
    def fg(*args, **kwargs):
        return f(g(*args, **kwargs))
    fg.__name__ = '(' + f.__name__ + ' . ' + g.__name__ + ')'
    return fg

def flipped(f):
    def g(*args, **kwargs):
        return f(*reversed(args), **kwargs)
    g.__name__ = 'flipped(' + f.__name__ + ')'
    return g

def logged(f, logger=print_):
    def g(*args, **kwargs):
        y = f(*args, **kwargs)
        logger(f.__name__ + '(' + ', '.join(str(a) for a in args) + ') -> ' + str(y))
        return y
    g.__name__ = 'logged(' + f.__name__ + ')'
    return g

def timed(f, logger=print_):
    '''
    If `logger` is None then the returned function returns the
    execution time of `f` instead of its return value.
    '''
    def g(*args, **kwargs):
        start, y = datetime.datetime.now(), None
        try:
            y = f(*args, **kwargs)
        finally:
            pass
        end = datetime.datetime.now()
        if logger is not None:
            logger(end - start)
        else:
            y = end - start
        return y
    g.__name__ = 'timed(' + f.__name__ + ')'
    return g

def hashable_args(*args, **kwargs):
    return (args, tuple(kwargs.items()))

def memoized(f, key=hashable_args):
    def g(*args, **kwargs):
        k = key(*args, **kwargs)
        if k not in g.memo_cache:
            g.memo_cache[k] = f(*args, **kwargs)
        return g.memo_cache[k]
    g.memo_cache = {}
    g.__name__ = 'memoized( ' + f.__name__ + ')'
    return g

########################################################################
# Math

inf = float('inf')

def ident(x):
    return x

def log2(x):
    return math.log(x, 2)

def log10(x):
    return math.log(x, 10)

def log_sum_exp(xs, log=math.log, exp=math.exp):
    if len(xs) == 0:
        return -inf
    def lse(x, y):
        lo = min(x, y)
        hi = max(x, y)
        return log(1.0 + exp(lo - hi)) + hi;
    return lfold(lse, xs)

def sigmoid(x, exp=math.exp):
    return 1.0 / (1.0 + exp(-x))

def mean(ls):
    assert len(ls) > 0
    return sum(ls) / len(ls)

def median(ls, rng=random):
    assert len(ls) > 0
    n = len(ls)
    ls = sorted(ls)
    if n % 2 == 1:
        return ls[n // 2]
    else:
        mids = (ls[n // 2 - 1], ls[n // 2])
        try:
            return mean(mids)
        except TypeError:
            return rng.choice(mids)

def mode(ls):
    assert len(ls) > 0
    return histogram(ls, by_count=True)[0][0]

def variance(ls, meanval=None):
    '''Provide mean if already known.'''
    assert len(ls) > 0
    if meanval is None:
        meanval = mean(ls)
    return mean([(x - meanval)**2 for x in ls])

def stddev(ls, meanval=None):
    '''Provide the mean if already known.'''
    return math.sqrt(variance(ls, meanval))

def sample_variance(ls, meanval=None):
    '''
    Provide the mean if already known.

    Sample variance is the unbiased estimator of the variance of the
    sampled distribution using Bessel's correction.
    '''
    assert len(ls) > 0
    if meanval is None:
        meanval = mean(ls)
    return sum([(x - meanval)**2 for x in ls]) / (len(ls) - 1)

def sample_stddev(ls, meanval=None):
    '''Provide the mean if already known.'''
    return math.sqrt(sample_variance(ls, meanval))

def argmins(seq, key=ident):
    '''Key must output a number given a sequence element.'''
    xs, min_ = [], key(require_first(seq))
    for x, y in enumerate(seq):
        y = key(y)
        if y < min_:
            xs, min_ = [x], y
        elif y == min_:
            xs.append(x)
    return xs

def argmin(seq, key=ident):
    '''
    Key must output a number given a sequence element.
    Ties favor elements earlier in the sequence
    '''
    return argmins(seq, key)[0]

def argmaxs(seq, key=ident):
    '''Key must output a number given a sequence element.'''
    return argmins(seq, key=lambda x: -key(x))

def argmax(seq, key=ident):
    '''
    Key must output a number given a sequence element.
    Ties favor elements earlier in the sequence
    '''
    return argmaxs(seq, key)[0]

def histogram(seq, key=ident, by_count=False):
    '''Key must output a number given a sequence element.'''
    h = {}
    for x in seq:
        x = key(x)
        h[x] = h.get(x, 0) + 1
    if by_count:
        return sorted(h.items(), key=snd, reverse=True)
    else:
        return sorted(h.items())

def coinflip(p, rng=random):
    '''True with probability p.'''
    return rng.uniform(0, 1) < p

def cdf_to_pdf(cdf):
    assert len(cdf) > 0
    return [cdf[0]] + diff(cdf)

def diceroll(cdf, rng=random):
    assert len(cdf) > 0
    # assert 0.0 <= sum(cdf_to_pdf(cdf)) <= 1.0
    x = rng.uniform(0, 1)
    return bsearch(x, cdf)

def prod(seq, start=None):
    return lfold(operator.mul, seq, start)

def dotprod(x, y):
    return sum(a * b for a, b in zip(x, y))

def vec_add(x, y):
    return [a + b for a, b in zip(x, y)]

def scale(s, seq):
    return [s * x for x in seq]

def transpose(mat):
    return zip(*mat)

def transform(mat, vec):
    '''`mat` is a matrix (list of rows)'''
    mat = transpose(mat)
    assert len(mat) > 0
    assert len(mat) == len(vec)
    return lfold(vec_add, (scale(x, col) for x, col in zip(vec, mat)))

def norm(v, l=2.0):
    '''l-norm of vector v.'''
    if l == inf:
        return max(abs(x) for x in v)
    else:
        return sum(abs(x)**l for x in v)**(1/float(l))

def dist(x, y, l=2.0):
    try:
        return norm((a - b for a, b in zip(x, y)), l)
    except TypeError:
        return norm([x - y], l)

def hamming_weight(seq):
    return sum(x != 0 for x in seq)

def normalize(seq, total=1.0):
    '''Scale numbers in `seq` so that they sum to `total`.'''
    seq = list(seq)
    s = total / float(sum(seq))
    return scale(s, seq)

########################################################################
# Misc

def bsearch(x, xs):
    lo, hi = 0, len(xs)
    while lo < hi:
        mid = (lo + hi) // 2
        if x < xs[mid]:
            hi = mid
        elif x > xs[mid]:
            lo = mid + 1
        else:
            return mid
    return lo

def abs_impl():
    raise NotImplementedError('Must override')

class Tree(object):
    @staticmethod
    def from_list(ls):
        return Tree(ls[0], [Tree.from_list(x) for x in ls[1:]])

    def __init__(self, label, children=[]):
        self.label = label
        self.children = children

    def to_list(self):
        return [self.label] + [to_list(kid) for kid in self.children]

    def __repr__(self):
        return repr(self.to_list())

    def leaves(self):
        if len(self.children) == 0:
            return [self.label]
        else:
            return concat([x.leaves() for x in self.children])

    def preorder(self, f=None):
        if f is not None:
            f(self.label)
            for kid in self.children:
                kid.preorder(f)
        else:
            return [self.label] + concat([kid.preorder() for kid in self.children])

    def postorder(self, f=None):
        if f is not None:
            for kid in self.children:
                kid.postorder(f)
            f(self.label)
        else:
            return concat([kid.postorder() for kid in self.children]) + [self.label]

    def map(self, f):
        return Tree(f(self.label), [kid.map(f) for kid in self.children])

class Struct(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __cmp__(self, other):
        if isinstance(other, Struct):
            return cmp(self.__dict__, other.__dict__)
        else:
            return cmp(self.__dict__, other)

    def __repr__(self):
        return 'Struct(%s)' % ', '.join('%s=%s' % (k, repr(v)) for k, v in self.__dict__.items())

    def items(self):
        return self.__dict__.items()

def str_kv_pairs(pairs, outer_sep=', ', inner_sep='='):
    return outer_sep.join([inner_sep.join([str(x) for x in p]) for p in pairs])

def dict_collect(seq):
    '''
    From a sequence of (k, v) pairs, produces a dict mapping every
    key to a list of corresponding values.
    '''
    d = collections.defaultdict(list)
    for k, v in seq:
        d[k].append(v)
    return dict(d)

def parse_rows(rows, fields, delim=None):
    return [
        tuple([f(s) for f, s in zip(fields, r.split(delim))])
        for r in rows
        ]

########################################################################
# Scripting

def run_job_queue(queue, conc,
                  run=lambda job: job.run(),
                  done=lambda job: job.poll() is not None,
                  tick=0.1):
    working = []
    while len(queue) > 0 or len(working) > 0:
        for job in working:
            if done(job):
                working.remove(job)
        while len(queue) > 0 and len(working) < conc:
            cur, queue = queue[0], queue[1:]
            run(job)
            working.append(cur)
        time.sleep(tick)

class DummyAttrs(object):
    '''
    Useful, for instance, for silent-failing on import and logging
    usage.  Example:

    >>> try: import foo           # Suppose we don't have this
    >>> except ImportError: foo = DummyAttrs(verbose=1)
    >>> foo.bar()                 # prints 'Dummy call: bar' to stderr
    '''
    def __init__(self, verbose=0):
        self.verbose = verbose
    def __getattr__(self, x):
        def dummy(*args, **kwargs):
            if self.verbose > 1:
                print >>sys.stderr, 'Dummy call: {}(*{}, **{})'.format(x, args, kwargs)
            elif self.verbose > 0:
                print >>sys.stderr, 'Dummy call:', x
        dummy.__name__ = 'dummy(' + x + ')'
        return dummy

class ScriptOptions(object):
    '''
    Users might be interested in `argparse` from the Python standard
    library.  This remains here because `argparse` only lives in
    Python 2.7 and higher, and because it is arguably more lightweight
    (at the cost of being far less featureful).
    '''
    def __init__(self, opts):
        '''
        `opts` is a dict mapping option names to (arg, short, desc)
        triples whose:
        - first element is an argument-parsing function if the option
          takes a parameter and None if the option takes no parameter,
        - second element is a one-letter option abbreviation or '' if
          no abbreviation is desired,
        - third element is a string description of the option, and
        e.g. `{ 'my-option': (int, 'm', 'A most excellent parameter') }`
        '''
        assert isinstance(opts, dict)
        assert all(isinstance(k, str) for k in opts.keys()), 'Keys are str.'
        assert all(len(v) == 3 for v in opts.values()), 'Values are triples.'
        assert all(len(v[1]) == 0 or \
                       len(v[1]) == 1 and v[1].isalpha() \
                       for v in opts.values()), \
            'Shorts are single alphabet letters or empty strings'
        self.opts = opts
        self.short_to_opt = dict((v[1], k) for k, v in opts.items())

    def getopt(self, args):
        shorts = ''
        for k, v in self.short_to_opt.items():
            shorts += k
            if self.opts[v][0] is not None:
                shorts += ':'
        longs = []
        for k, v in self.opts.items():
            opt = k
            if v[0] is not None:
                opt += '='
            longs.append(opt)
        optvals, rest = getopt.gnu_getopt(args, shorts, longs)

        resolved = {}
        for k, v in optvals:
            if len(k) == 2:
                opt = self.short_to_opt[k[1]] # strip leading '-' and resolve
            else:
                opt = k[2:]                   # strip leading '--'
            parser = self.opts[opt][0]
            if parser is not None:
                v = parser(v)
            resolved[opt] = v

        return resolved, rest

    def print_help(self, out=print_):
        out('Options:')
        for opt, (arg, short, desc) in self.opts.items():
            if arg is not None:
                out('  -' + short + ', --' + opt + '=' + opt.upper())
                out('    ' + desc)
            else:
                out('  -' + short + ', --' + opt)
                out('    ' + desc)

########################################################################
# Web / text / encoding / data formats

# Python 2/3 interop
try:
    import HTMLParser
except ImportError:
    import html.parser
    HTMLParser = html.parser

def html_unescape(s):
    return HTMLParser.HTMLParser().unescape(s)

def build_jsonlike_query(query):
    '''
    Returns a function that takes a dict as obtained by `json.loads()`
    and executes the jsonlike query on it.

    Example:
    The query (string)

       'foo.bar.#2.baz'

    return `47` when applied to

       {
         'foo': {
           'bar': [{}, { 'zap': 62 }, { 'baz': 47 }, { 'bzzp': 5 }],
           'baf': [5]
         }
       }

    '''
    trail = query.split('.')
    trail = [
        int(x[1:]) if (len(x) >= 2 and x[0] == '#') else unicode(x)
        for x in trail
    ]
    return curry(getattr_trail, trail)

########################################################################
# Main

def _run_tests():
    # Unit tests!

    print_('Testing print_ function.')
    assert list(xrange(3)) == [0, 1, 2]
    assert list(chunked(3, [1,2,3,4,5,6,7,8])) == [[1,2,3], [4,5,6], [7,8]]

    ticks = iter(xrange(0, 42, 5))
    def f(ticks, i):
        assert i == next(ticks)
    ticked(xrange(42), 5, curry(f, ticks))
    del f, ticks

    assert concat([[1,2,3], [4], [5,6]]) == [1,2,3,4,5,6]

    m, n, o = 2, 3, 4
    g = make_grid(m, n, o)
    for i, j, k in [(i, j, k) for i in xrange(m) for j in xrange(n) for k in xrange(o)]:
        assert g[i][j][k] is None
    g = make_grid(m, n, o, val=5)
    for i, j, k in [(i, j, k) for i in xrange(m) for j in xrange(n) for k in xrange(o)]:
        assert g[i][j][k] == 5
    del g, m, n, o

    assert require_first([42]) == 42
    try:
        require_first([])
        assert False, 'Expected exception'
    except ValueError:
        pass

    assert lfold(lambda a,b: a+b, [1,2,3,4]) == sum([1,2,3,4])
    assert rfold(lambda a,b: a+b, [1,2,3,4]) == sum([1,2,3,4])
    assert lfold(lambda a,b: a-b, [1,2,3,4]) == 1-2-3-4
    assert rfold(lambda a,b: a-b, [1,2,3,4]) == 4-3-2-1
    assert cumsum([1,2,3,4]) == [1,3,6,10]
    assert diff([1,3,6,10]) == [2,3,4]
    assert fst((1,2)) == 1
    assert snd((1,2)) == 2

    assert deepsplit('a.b,c.d', ',.') == [['a', 'b'], ['c', 'd']]
    assert deepsplit('a.b,c.d', '.,') == [['a'], ['b', 'c'] ,['d']]

    strs = [str(x) for x in xrange(123123123, 123123123 + 10000)]
    d = StrDB()
    for s in strs:
        d.insert(s)
    assert len(d) == len(strs)
    assert len(d) == len(set(d.values()))
    a, b, c = strs[:3]
    i, j, k = d[a], d[b], d[c]
    assert i != j
    assert i != k
    assert j != k
    assert i == d[a]
    assert j == d[b]
    assert k == d[c]
    del strs, d, a, b, c, i, j, k

    assert compose(lambda x: x+1, lambda x: x+2)(39) == 42
    assert flipped(lambda x,y: x-y)(1, 2) == 1
    assert flipped(flipped(lambda x,y: x-y))(1, 2) == -1

    def f(x): assert x == '<lambda>(1, 2) -> 3'
    logged(lambda x,y: 3, f)(1,2)
    del f

    hash(hashable_args(1, 2, 3, a=4, b=5)) # shouldn't raise
    try:
        hash(hashable_args({}, a={})) # should raise
        assert False
    except TypeError:
        pass

    x = {'a': 0}
    def f(x, y):
        x['a'] += 1
        return x['a']
    f = memoized(curry(f, x))
    assert f(1) == 1
    assert f(1) == 1
    assert f(1) == 1
    assert f(2) == 2
    assert f(2) == 2
    del x, f

    # TODO unit test math, misc, scripting

    print('Passed all tests!')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        cmd, args = sys.argv[1], sys.argv[2:]

        if cmd == 'test':
            _run_tests()
