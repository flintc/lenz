from helpers import is_dict_like, is_list_like, always, arityn
from functools import reduce, partial
import algebras as A
from copy import deepcopy
from contract import nth_arg


def lens_from(get, set):
    def wrapper(i):
        def inner(x, _i, F, xi2yF):
            return F.map(
                lambda v: set(i, v, x),
                xi2yF(get(i, x), i)
            )
        return inner
    return wrapper


def get_prop(k, o):
    if is_dict_like(o):
        return o[k]
    return None


def set_prop(k, v, o):
    #print('set_prop', k, v, o)
    on = deepcopy(o)
    on[k] = v
    return on


fun_prop = lens_from(get_prop, set_prop)


def get_index(i, xs):
    if is_list_like(xs):
        #print(i, xs)
        return xs[i]
    return None


set_index = set_prop
fun_index = lens_from(get_index, set_index)


def composed_middle(o, r):
    def outer(F, xi2yF):
        xi2yF = r(F, xi2yF)
        return lambda x, i: o(x, i, F, xi2yF)
    return outer


def identity(x, i, _F, xi2yF):
    return xi2yF(x, i)


def from_reader(wi2x):
    def wrapper(w, i, F, xi2yF):
        #print('from_reader', wi2x, w, i, F, xi2yF)
        return F.map(
            always(w),
            xi2yF(wi2x(w, i), i)
        )
    return wrapper


def to_function(o):
    if isinstance(o, str):
        return fun_prop(o)
    if isinstance(o, int):
        return fun_index(o)
    if is_list_like(o):
        return composed(0, o)
    #print('to_function', o, o.length if hasattr(o, 'length') else 'no length')
    # return o  # if (hasattr(o, '__len__') and len(o) == 4) else from_reader(o)
    return o if (hasattr(o, 'length') and o.length == 4) else from_reader(o)


def composed(oi0, os):
    n = len(os) - oi0
    if n < 2:
        if n != 0:
            return to_function(os[oi0])
        return identity
    else:
        n -= 1
        last = to_function(os[oi0+n])

        def r(F, xi2yF): return lambda x, i: last(x, i, F, xi2yF)
        n -= 1
        while n > 0:
            r = composed_middle(to_function(os[oi0+n]), r)
            n -= 1
        first = to_function(os[oi0])
        return lambda x, i, F, xi2yF: first(x, i, F, r(F, xi2yF))


def modify_composed(os, xi2y, x, y=None):
    #print('enter modify composed', os, xi2y, x, y)
    n = len(os)
    xs = []
    for i in range(len(os)):
        xs.append(x)
        if isinstance(os[i], str):
            x = get_prop(os[i], x)
        elif isinstance(os[i], int):
            x = get_index(os[i], x)
        else:
            #print('modify composed, else', xi2y, y)
            x = composed(i, os)(x, os[i-1], A.Identity, xi2y or always(y))
            #print('n before, i', n, i)
            #n = n-3 if n-i == 2 else i
            n = i
            #print('n after, i', n, i)
    if (n == len(os)):
        #print('modify composed n==len(os)', xi2y, x, os[n-1], y)
        x = xi2y(x, os[n-1]) if xi2y else y
    # print('modify composed xs', xs)
    # print('modify composed x', x)
    # print('modify composed os', os)
    # print('modify composed n', n)
    n -= 1
    while 0 <= n:
        #print('modify comnposed while', n, x, xs[n], os[n])
        x = set_prop(os[n], x, xs[n]) if isinstance(
            os[n], str) else set_index(os[n], x, xs[n])
        n -= 1
    return x


def modify_u(o, xi2x, s):
    #print('modify_u', o, xi2x, s)
    xi2x = nth_arg(0, xi2x)
    if isinstance(o, str):
        #print(o, xi2x, s)
        return set_prop(o, xi2x(get_prop(o, s), o), s)
    if isinstance(o, int):
        return set_index(o, xi2x(get_index(o, s), o), s)
    if is_list_like(o):
        return modify_composed(o, xi2x, s)
    # if (hasattr(o, '__len__') and len(o) == 4) else (xi2x(o(s, None), None), s)
    return o(s, None, A.Identity, xi2x)
    # return (xi2x(o(s, None), None), s)


def set_u(o, x, s):
    if isinstance(o, str):
        return set_prop(o, x, s)
    if isinstance(o, int):
        return set_index(o, x, s)
    if is_list_like(o):
        return modify_composed(o, 0, s, x)
    # if (hasattr(o, '__len__') and len(o) == 4) else (xi2x(o(s, None), None), s)
    return o(s, None, A.Identity, x)


def id(x, y):
    #print('id', x, y)
    return x


@arityn(2)
def transform(o, s):
    return modify_u(o, id, s)


def modify_op(xi2y):
    def wrapper(x, i, C=A.Identity, _xi2yC=None):
        #print('modify_op wrapper', x, i, C, _xi2yC)
        #print('modify_op', x, i, C)
        if i == elems:
            print('elems?', A.Constant.map(xi2y, x), _xi2yC)
            # return C.of(_xi2yC(x, i))
        return C.of(nth_arg(0, xi2y)(x, i))
    wrapper.length = 4
    return wrapper


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
        for i in range(0, n):
            if isinstance(l[i], str):
                s = get_prop(l[i], s)
            elif isinstance(l[i], int):
                s = get_index(l[i], s)
            else:
                return composed(i, l)(s, l[i-1], A.Select, xi2y)
        return xi2y(s, l[n-1])
    #print(xi2y, xi2y == id)
    if xi2y is not id and (l.length != 4 if hasattr(l, 'length') else False):
        #print('get_as_u if', l)
        return xi2y(l(s, None), None)
    else:
        #print('get_as_u else', l)
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
    def wrapper(x, i, F, xi2yF):
        return F.map(
            lambda v: set_pick(template, v, x),
            xi2yF(get_pick(template, x), i)
        )
    return wrapper


def subseq_u(begin, end, t):
    t = to_function(t)

    def wrapper(x, i, F, xi2yF):
        n = -1
        #print('subseq_u', x, i, F, xi2yF)

        def inner(x, i):
            nonlocal n
            n += 1
            if begin <= n and not (end <= n):
                #print('subseq u wrapper.inner if')
                return xi2yF(x, i)
            else:
                #print('subseq u wrapper.inner else')
                return F.of(x)
        return t(x, i, F, inner)
    wrapper.length = 4
    wrapper.__name__ = 'subseq_u<wrapper>'
    return wrapper


def elems(xs, _i, A, xi2yA):
    #print('elems', _i, A, xs, xi2yA)
    result = reduce(
        lambda ysF, x: A.ap(
            A.map(
                lambda ys: lambda y: [*ys, y], ysF),
            xi2yA(x, xs)),
        xs,
        A.of([]))
    #print('elems result', result)
    return result


def get_values(ys):
    print('get_values', ys)
    return ys.values()


def values_helper(ys, y, ysF, x):
    print(ys, y, ysF, x)


def values(xs, _i, A, xi2yA):
    print('values', _i, A, xs, xi2yA)
    result = reduce(
        lambda ysF, x: A.ap(
            A.map(
                lambda ys: lambda y: values_helper(ys, y, ysF, x), ysF),
            xi2yA(x, xs)),
        xs.items(),
        A.of(dict()))
    #print('elems result', result)
    return result


elems.length = 4
values.length = 4


@arityn(3)
def collect_as(xi2y, t, s):
    results = []

    def as_fn(x, i):
        y = xi2y(x, i)
        print(x, i, y, t)
        if y is not None:
            results.append(y)
    get_as_u(
        as_fn,
        t, s)
    print('results', results)
    return results


collect = collect_as(id)
modify = arityn(3)(modify_u)
get = arityn(2)(get_u)
