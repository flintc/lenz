from lenz.helpers import is_dict_like, is_list_like, always, arityn, freeze, assign, protoless0
from functools import reduce, partial
import lenz.algebras as A
from lenz.algebras import Select
import lenz.infestines as I
from copy import deepcopy, copy
from lenz.contract import nth_arg
import logging
import sys


logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL+1)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.CRITICAL)
formatter = logging.Formatter('%(name)-12s: %(levelname)-10s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def lens_from(get, set):
    def lens_from_inner(i):
        def lens_from_wrapper(x, _i, F, xi2yF):
            return F.map(
                lambda v: set(i, v, x),
                xi2yF(get(i, x), i)
            )
        lens_from_wrapper.length = 4
        return lens_from_wrapper
    return lens_from_inner


def get_prop(k, o):
    if is_dict_like(o):
        return o[k]
    return None


def safe_prop(k, o):
    if is_dict_like(o):
        return o[k] if k in o.keys() else None


def set_index(k, v, o):
    on = deepcopy(o)
    if v is not None:
        #on[k] = v
        #print('here...', k, v)
        on = [v if ix == k else i for ix, i in enumerate(on)]
    else:
        on = [i for ix, i in enumerate(on) if ix != k]
    return on


def set_prop(k, v, o):
    r = type(o)()
    for key in o:
        if (key != k):
            r[key] = o[key]
        else:
            if v is not None:
                r[key] = v
            k = None
    return r


fun_prop = lens_from(get_prop, set_prop)


def get_index(i, xs):
    logger.critical('[get_index] - {}, {}'.format(i, xs))
    if is_list_like(xs):
        return xs[i]
    return None


fun_index = lens_from(get_index, set_index)


def composed_middle(o, r):
    def outer(F, xi2yF):
        xi2yF = r(F, xi2yF)
        return lambda x, i: o(x, i, F, xi2yF)
    return outer


def identity(x, i, _F, xi2yF):
   #print('identity', x, i, _F, xi2yF)
    return xi2yF(x, i)


identity.length = 4


def from_reader(wi2x):
    def from_reader_wrapper(w, i, F, xi2yF):
        return F.map(
            always(w),
            xi2yF(wi2x(w, i), i)
        )
    from_reader_wrapper.length = 4
    return from_reader_wrapper


def to_function(o):
    if isinstance(o, str):
        return fun_prop(o)
    if isinstance(o, int):
        return fun_index(o)
    if is_list_like(o):
        return composed(0, o)
    # return o  # if (hasattr(o, '__len__') and len(o) == 4) else from_reader(o)
    logger.critical('[to_function] - {}, {}'.format(o,
                                                    hasattr(o, 'length') and o.length == 4))
    return o if (hasattr(o, 'length') and o.length == 4) else from_reader(o)


def composed(oi0, os):
    n = len(os) - oi0
    logger.debug(
        '[{}] - len(os): {} - oi0: {}'.format('composed', len(os), oi0))
   #print('[composed] - ', oi0, os, n)
    if n < 2:
        # if n==1 and oi0==1:
        #    return identity
        if n != 0:
            return to_function(os[oi0])
        return identity
    else:
        n -= 1
        last = to_function(os[oi0+n])
       #print('here', oi0, n, os[oi0+n], last)

        def r(F, xi2yF): return lambda x, i: last(x, i, F, xi2yF)
        n -= 1
        while n > 0:
            r = composed_middle(to_function(os[oi0+n]), r)
            n -= 1
        first = to_function(os[oi0])

        def composed_wrapper(x, i, F, xi2yF):
            return first(x, i, F, r(F, xi2yF))
        composed_wrapper.length = 4
        return composed_wrapper


def modify_composed(os, xi2y, x, y=None):
    n = len(os)
    logger.debug('[{}] - len(os): {}'.format('modify_composed', len(os)))
    xs = []
    for i in range(len(os)):
        xs.append(x)
        if isinstance(os[i], str):
            x = get_prop(os[i], x)
        elif isinstance(os[i], int):
            x = get_index(os[i], x)
        else:
           #print('[modify_composed] - else - xi2y, y', xi2y, y)
            x = composed(i, os)(
                x, os[i-1] if len(os) >= (i-1) else None, I.Identity, xi2y or always(y))
            #n = n-3 if n-i == 2 else i
            n = i
    if (n == len(os)):
       #print('[modify_composed] - n==len(os) -', xi2y, x, os[n-1], y)
        x = xi2y(x, os[n-1]) if xi2y else y
    n -= 1
    while 0 <= n:
       #print('[modify_comnposed] - while -', n, x, xs[n], os[n])
        if callable(os[n]):  # and os[n].__name__ in ['elems', 'subseq_u']:
           #print('[TMP FIX] - continue if callable(optic)')
            n -= 1
            continue
        x = set_prop(os[n], x, xs[n]) if isinstance(
            os[n], str) else set_index(os[n], x, xs[n])
        n -= 1
    return x


def modify_u(o, xi2x, s):
    logger.debug('Test message')
    xi2x = nth_arg(0, xi2x)
    if isinstance(o, str):
        return set_prop(o, xi2x(get_prop(o, s), o), s)
    if isinstance(o, int):
        return set_index(o, xi2x(get_index(o, s), o), s)
    if is_list_like(o):
        return modify_composed(o, xi2x, s)
    # if (hasattr(o, '__len__') and len(o) == 4) else (xi2x(o(s, None), None), s)
    return o(s, None, I.Identity, xi2x)
    # return (xi2x(o(s, None), None), s)


def set_u(o, x, s):
    if isinstance(o, str):
        return set_prop(o, x, s)
    if isinstance(o, int):
        return set_index(o, x, s)
    if is_list_like(o):
        return modify_composed(o, 0, s, x)
    # if (hasattr(o, '__len__') and len(o) == 4) else (xi2x(o(s, None), None), s)
    return o(s, None, I.Identity, always(x)) if hasattr(o, 'length') and o.length == 4 else s


def id(x, *algebras):
   #print('id', x, algebras)
    return x


@arityn(2)
def transform(o, s):
    return modify_u(o, id, s)


def modify_op(xi2y):
    def modify_op_wrapper(x, i, C, _xi2yC=None):
        logger.debug(
            '[modify_op({}) - x: {}, i: {}, C: {}'.format(xi2y.__name__, x, i, C))
        # TODO: Figure out why the if else is needed and if another solution exists
        result = C.of(xi2y(x)) if isinstance(x, int) else x
        return result
    modify_op_wrapper.length = 4
    modify_op_wrapper.__name__ = 'modify_op({})'.format(xi2y.__name__)
    return modify_op_wrapper


def set_pick(template, value, x):
    for k in template:
        v = value[k] if k in value.keys() else None
        t = template[k]
        x = set_pick(t, v, x) if is_dict_like(t) else set_u(t, v, x)
    return x


def get_as_u(xi2y, l, s):
    if isinstance(l, str):
        return xi2y(get_prop(l, s), l)
    if isinstance(l, int):
        return xi2y(get_index(l, s), l)
    if is_list_like(l):
        n = len(l)
        logger.debug('[{}] - l: {} - s: {}'.format('get_as_us', l, s))
        for i in range(0, n):
            if isinstance(l[i], str):
                s = get_prop(l[i], s)
            elif isinstance(l[i], int):
                s = get_index(l[i], s)
            else:
                return composed(i, l)(s, l[i-1] if len(l) >= (i-1) else None, A.Select, xi2y)
        return xi2y(s, l[n-1] if len(l) >= (n-1) and n > 0 else None)
    if xi2y is not id and (l.length != 4 if hasattr(l, 'length') else False):
       #print('get_as_u if', l)
        logger.critical('[get_as_u] if - {}'.format(l))
        return xi2y(l(s, None), None)
    else:
       #print('get_as_u else', l)
        logger.critical('[get_as_u] else - {}'.format(l))
        return l(s, None, A.Select, xi2y)


def get_u(l, s): return get_as_u(id, l, s)


def get_pick(template, x):
    r = None
    for k in template.keys():
        t = template[k]
        v = get_pick(t, x) if is_dict_like(t) else get_as_u(id, t, x)
        if v is not None:
            if not r:
                r = {}
            r[k] = v
    return r


def pick(template):
    def pick_wrapper(x, i, F, xi2yF):
        return F.map(
            lambda v: set_pick(template, v, x),
            xi2yF(get_pick(template, x), i)
        )
    pick_wrapper.length = 4
    return pick_wrapper


def subseq_u(begin, end, t):
    t = to_function(t)

    def subseq_u_wrapper(x, i, F, xi2yF):
        n = -1

        def inner(x, i):
            nonlocal n
            n += 1
            if begin <= n and not (end <= n):
                return xi2yF(x, i)
            else:
                return F.of(x)
        return t(x, i, F, inner)
    subseq_u_wrapper.length = 4
    subseq_u_wrapper.__name__ = 'subseq_u<wrapper>'
    return subseq_u_wrapper


def elems(xs, _i, A, xi2yA):
    result = reduce(
        lambda ysF, x: A.ap(
            A.map(
                lambda ys: lambda y: [*ys, y], ysF),
            xi2yA(x, xs)),
        xs,
        A.of([]))
    return result


elems.length = 4


@arityn(3)
def collect_as(xi2y, t, s):
    results = []

    def as_fn(x, i):
        y = nth_arg(0, xi2y)(x, i)
       #print(x, i, y, t)
        if y is not None:
            results.append(y)
    get_as_u(
        as_fn,
        t, s)
   #print('results', results)
    return results


collect = collect_as(id)
modify = arityn(3)(modify_u)
get = arityn(2)(get_u)
set = arityn(3)(set_u)


def find_index_hint(hint, xi2b, xs):
    u = safe_prop('hint', hint)
    n = len(xs)
    d = u-1
    if n <= u:
        u = n-1
    if d < 0:
        u = 0
    while 0 <= d and u < n:
        if xi2b(xs[u], u, hint):
            return u
        if xi2b(xs[d], d, hint):
            return d
        u += 1
        d -= 1
    while u < n:
        if xi2b(xs[u], u, hint):
            return u
        u += 1
    while 0 <= d:
        if xi2b(xs[d], d, hint):
            return d
        d -= 1
    return n


def find(xih2b, hint={'hint': 0}):
    xih2b = nth_arg(0, xih2b)

    def find_wrapper(xs, _i, F, xi2yF):
        logger.critical('[find_wrapper] - {}, {}'.format(xs, is_list_like(xs)))
        ys = xs if is_list_like(xs) else ''
        hint['hint'] = find_index_hint(hint, xih2b, ys)
        i = hint['hint']
        return F.map(lambda v: set_index(i, v, ys), xi2yF(ys[i], i))
    find_wrapper.length = 4
    return find_wrapper


def satisfying(p):
    def satisfying_wrapper(x, i, C, xi2yC):
        def rec(x, i):
            if nth_arg(0, p)(x, i):
                return xi2yC(x, i)
            return children(x, i, C, rec)
        return rec(x, i)
    satisfying_wrapper.length = 4
    return satisfying_wrapper


leafs = satisfying(lambda x, *args: x is not None and not is_list_like(
    x) and not is_dict_like(x))


@arityn(3)
def all(xi2b, t, s):
    return not get_as_u(lambda x, i: True if not nth_arg(0, xi2b)(x, i) else None, t, s)


def iso_u(bwd, fwd):
    # @arityn(4)
    def iso_u_wrapper(x, i, F, xi2yF):
        return F.map(fwd, xi2yF(bwd(x), i))
    iso_u_wrapper.length = 4
    return iso_u_wrapper


def _compare(x, y):
    logger.critical('[_compare] - {} == {} ? {}'.format(x, y, x == y))
    return x == y


def is_(v):
    logger.critical('[is_] - {}'.format(v))
    return iso_u(lambda x: _compare(x, v), lambda b: v if b is True else None)


and_ = all(id)


def where_eq(template): return satisfying(
    and_(branch(modify(leafs, is_, template))))


def to_object(x):
    return assign({}, x)


def cpair(xs):
    return lambda x: [x, xs]


def branch_assemble(ks):
    def branch_assemble_inner(xs):
        r = dict()
        i = len(ks)
        while i:
            v = xs[0]
            if v is not None:
                r[ks[i]] = v
            xs = xs[1]
            i -= 1
        return r
    return branch_assemble_inner


def is_same(x, y):
    # return (x == y & (x != 0 | 1/x == 1/y)) | (x != x and y != y)
    return (x == y) or (x != x and y != y)


def _branch_or_1_level_identity(otherwise, k2o, xO, x, A, xi2yA):
    written = None
    same = True
    r = {}
    for k in k2o.keys():
        written = 1
        x = xO[k]
        y = k2o[k](x, k, A, xi2yA)
        if y is not None:
            r[k] = y
            if (same):
                same = is_same(x, y)
            else:
                same = False
    t = written
    for k in xO.keys():
        if t is None or k2o[k] is None:
            written = 1
            x = xO[k]
            y = otherwise(x, k, A, xi2yA)
            if y is not None:
                r[k] = y
                if (same):
                    same = is_same(x, y)
                else:
                    same = False
    return (x if (same and (xO == x)) else r) if written else x


def __branch_or_1_level_identity(fn):
    def __branch_or_1_level_identity(otherwise, k2o, x0, x, A, xi2yA):
        y = fn(otherwise, k2o, x0, x, A, xi2yA)
        if x != y:
            #y = freeze(y)
            y = y
        return y
    #__branch_or_1_level_identity.length = 4
    return __branch_or_1_level_identity


branch_or_1_level_identity = __branch_or_1_level_identity(
    _branch_or_1_level_identity)


def branch_or_1_level(otherwise, k2o):
    def branch_or_1_level_wrapper(x, _i, A, xi2yA):
        x0 = to_object(x) if is_dict_like(x) else dict()  # freeze(dict())
        if I.Identity == A:
            return branch_or_1_level_identity(otherwise, k2o, x0, x, A, xi2yA)
        elif (Select == A):
            for k in k2o.keys():
                logger.critical(
                    '[branch_or_1_level_wrapper] - {}, {}, {}, {}, {}'.format(k2o, x0, x, A, xi2yA))
                tmp = k2o[k]
                tmp.length = 4
                y = tmp(safe_prop(k, x0), k, A, xi2yA)
                if y is not None:
                    return y
            for k in x0.keys():
                if k not in k2o.keys():
                    y = otherwise(x0[k], k, A, xi2yA)
                    if y is not None:
                        return y
        else:
            map = A.map
            ap = A.ap
            of = A.of
            xsA = of(cpair)
            ks = []
            for k in k2o.keys():
                ks.append(k)
                xsA = ap(map(cpair, xsA), k2o[k](x0[k], k, A, xi2yA))
            t = True if len(ks) else None
            for k in x0.keys():
                if (t is None or k2o[k] is None):
                    ks.append(k)
                    xsA = ap(map(cpair, xsA), otherwise(x0[k], k, A, xi2yA))
            return map(branch_assemble(ks), xsA) if len(ks) else of(x)
    branch_or_1_level_wrapper.length = 4
    return branch_or_1_level_wrapper


def branch_or_u(otherwise, template):
    k2o = dict()
    for k in template.keys():
        v = template[k]
        k2o[k] = branch_or_u(otherwise, v) if is_dict_like(
            v) else to_function(v)
    return branch_or_1_level(otherwise, k2o)


def branch_or(otherwise):
    otherwise = to_function(otherwise)

    def branch_or_inner(template):
        return branch_or_u(otherwise, template)
    return branch_or_inner


def zero(x, _i, C, _xi2yC):
    return C.of(x)


zero.length = 4

branch = branch_or(zero)

values = branch_or_1_level(identity, protoless0)
values.length = 4


def children(x, i, C, xi2yC):
    if is_list_like(x):
        return elems(x, i, C, xi2yC)
    if is_dict_like(x):
        return values(x, i, C, xi2yC)
        #raise NotImplementedError('Not ready for dict like')
    return C.of(x)


children.length = 4


def rewrite(yi2y):
    def rewrite_wrapper(x, i, F, xi2yF):
        return F.map(lambda y: nth_arg(0, yi2y)(y, i) if y is not None else y, xi2yF(x, i))
    rewrite_wrapper.length = 4
    return rewrite_wrapper


def reread(xi2x):
    def reread_wrapper(x, i, _F, xi2yF):
        return xi2yF(nth_arg(0, xi2x)(x, i) if x is not None else x, i)
    reread_wrapper.length = 4
    return reread_wrapper
