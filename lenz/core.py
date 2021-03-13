from lenz.helpers import is_dict_like, is_list_like, safe_ix, always, arityn, tap
from functools import reduce, partial
import lenz.algebras as A
import lenz.infestines as I
from copy import deepcopy
from lenz.contract import nth_arg
import logging
import sys
import numpy as np
import pandas as pd
from typing import Mapping, Iterable

# from types import NoneType
MUTABLE = False


def enable_mutability():
    global MUTABLE
    MUTABLE = True


def disable_mutability():
    global MUTABLE
    MUTABLE = False


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.setLevel(100)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)-12s: %(levelname)-10s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# LENSES = [
#   [is_dict_like, lens_from(get_prop, set_prop)],
#   [is_list_like, lens_from(get_index, set_index)],
# ]

def lens_from(get, set):
    def wrapper(i):
        def inner(x, _i, F, xi2yF):
            return F.map(
                lambda v: set(i, v, x),
                xi2yF(get(i, x), i)
            )
        inner.length = 4
        return inner
    return wrapper


def mapping_get(key, data):
    return data.get(key, Undefined())


def mapping_set(key, value, data):
    if value is None:
        data.pop(key)
    elif not isinstance(value, Undefined):
        data[key] = value
    return data


def object_get(key, data):
    return getattr(data, key)


def object_set(key, value, data):
    if value is None:
        delattr(key, data)
    elif not isinstance(value, Undefined):
        setattr(data, key, value)
    return data


DICT_LIKE_LENSES = [
    [lambda x: isinstance(x, Mapping), [mapping_get, mapping_set]],
    [lambda x: True, [object_get, object_set]]
]


def get_prop(k, o):
    if is_dict_like(o):
        for cond, lens in DICT_LIKE_LENSES:
            if cond(o):
                return lens[0](k, o)
    return None


def set_prop(k, v, o):
    if MUTABLE:
        on = o
    else:
        on = deepcopy(o)
    if isinstance(on, Undefined) and isinstance(k, int):
        return [v]
    if is_dict_like(o):
        for cond, lens in DICT_LIKE_LENSES:
            if cond(o):
                return lens[1](k, v, on)
    return None


# def set_prop(k, v, o):
#     if MUTABLE:
#         on = o
#     else:
#         on = deepcopy(o)
#     if isinstance(on, Undefined) and isinstance(k, int):
#         return [v]
#     if v is None:
#         try:
#             on.pop(k)
#         except AttributeError:
#             delattr(on, k)
#     elif not isinstance(v, Undefined):
#         try:

#             on[k] = v
#         except TypeError:
#             setattr(on, k, v)
#     return on


def _iter(v):
    try:
        return iter(v)
    except TypeError:
        return iter(v.__dict__)


fun_prop = lens_from(get_prop, set_prop)


def iterable_get(i, xs):
    # if is_list_like(xs):
    return xs[i]
    # return None


def iterable_set(k, v, o):
    on = mapping_set(k, v, list(o))
    return type(o)(on)


def ndarray_set(k, v, o):
    on = mapping_set(k, v, list(o))
    return np.array(on)


def df_get(k,  o):
    return o.iloc[k]


def df_set(k, v, o):
    o.iloc[k] = v
    return o


# def df_is_same


LIST_LIKE_LENSES = [
    [lambda x: isinstance(x, pd.DataFrame), [df_get, df_set]],
    [lambda x: isinstance(x, np.ndarray), [iterable_get, ndarray_set]],
    [lambda x: isinstance(x, Iterable), [iterable_get, iterable_set]],
    [lambda x: True, [iterable_get, iterable_set]]
]


def get_index(i, xs):
    for cond, lens in LIST_LIKE_LENSES:
        if cond(xs):
            return lens[0](i, xs)


def set_index(k, v, o):
    for cond, lens in LIST_LIKE_LENSES:
        if cond(o):
            return lens[1](k, v, o)


def is_same(a, b):
    pass


fun_index = lens_from(get_index, set_index)


def composed_middle(o, r):
    def outer(F, xi2yF):
        xi2yF = r(F, xi2yF)
        return lambda x, i: o(x, i, F, xi2yF)
    return outer


def identity(x, i, _F, xi2yF):
    return xi2yF(x, i)


identity.length = 4


def from_reader(wi2x):
    wi2x = maybe_reader(wi2x)

    def wrapper(w, i, F, xi2yF):
        return F.map(
            always(w),
            xi2yF(wi2x(w, i), i)
        )
    wrapper.length = 4
    return wrapper


def to_function(o):
    if isinstance(o, str):
        return fun_prop(o)
    if isinstance(o, int):
        return fun_index(o)
    if is_list_like(o):
        return composed(0, o)
    # return o  # if (hasattr(o, '__len__') and len(o) == 4) else from_reader(o)
    return o if (hasattr(o, 'length') and o.length == 4) else from_reader(o)


def composed(oi0, os):
    n = len(os) - oi0
    logger.debug(
        '[{}] - len(os): {} - oi0: {}'.format('composed', len(os), oi0))
    if n < 2:
        # if n==1 and oi0==1:
        #    return identity

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
    n = len(os)
    logger.debug('[{}] - len(os): {}'.format('modify_composed', len(os)))
    xs = []
    for i in range(len(os)):
        xs.append(x)
        if isinstance(safe_ix(i, os), str):
            x = get_prop(safe_ix(i, os), x)
        elif isinstance(safe_ix(i, os), int):
            x = get_index(safe_ix(i, os), x)
        else:
            x = composed(i, os)(x, safe_ix(i-1, os),
                                A.Identity, xi2y or always(y))
            n = i

            break  # addresses test L.modify(['xs', L.elems, 'x'], func, data)
    if (n == len(os)):
        x = xi2y(x, safe_ix(n-1, os)) if xi2y else y
    n -= 1
    while 0 <= n:
        if callable(safe_ix(n, os)):
            n -= 1
            continue
            # pass
        # if isinstance(safe_ix(n,os), str):
        #     if is_dict_like(xs[n]):
        #          x = set_prop(os[n], x, xs[n])
        #     else:
        #         x = set_index(os[n], x, xs[n])
        # else:
        #     if is_dict_like(xs[n]):
        #         x =
        x = set_prop(os[n], x, xs[n]) if isinstance(
            safe_ix(n, os), str) else set_index(safe_ix(n, os), x, xs[n])
        n -= 1
    return x


def modify_u(o, xi2x, s):
    logger.debug('Test message')
    xi2x = maybe_reader(xi2x)
    if isinstance(o, str):
        return set_prop(o, xi2x(get_prop(o, s), o), s)
    if isinstance(o, int):
        return set_index(o, xi2x(get_index(o, s), o), s)
    if is_list_like(o):
        return modify_composed(o, xi2x, s)
    if (hasattr(o, 'length') and o.length == 4):
        return o(s, None, A.Identity, xi2x)
    else:
        return (xi2x(o(s, None), None), s)


def set_u(o, x, s):
    if isinstance(o, str):
        return set_prop(o, x, s)
    if isinstance(o, int):
        return set_index(o, x, s)
    if is_list_like(o):
        return modify_composed(o, 0, s, x)
    if (hasattr(o, 'length') and o.length == 4):
        return o(s, None, A.Identity, I.always(x))
    else:
        return s


def id_(x, *algebras):
    return x


@arityn(2)
def transform(o, s):
    return modify_u(o, id_, s)


def modify_op(xi2y):
    def wrapper(x, i, C, _xi2yC=None):
        result = C.of(xi2y(x)) if isinstance(x, int) else x
        return result  # if isinstance(x, int) else 'b'

    wrapper.length = 4
    wrapper.__name__ = 'modify_op({})'.format(xi2y.__name__)
    return wrapper


def set_pick(template, value, x):
    for k in template:
        v = value[k] if k in value.keys() else None
        t = template[k]
        x = set_pick(t, v, x) if is_dict_like(t) else set_u(t, v, x)
    return x


def maybe_reader(fn):
    def wrapper(l, *args):
        try:
            result = fn(l, args[0])
        except TypeError as e:
            try:
                result = fn(l)
            except TypeError as e:
                result = fn(l, *args)
        return result
    return wrapper


def get_as_u(xi2y, l, s):
    # xi2y = maybe_reader(xi2y)
    if isinstance(l, str):
        return xi2y(get_prop(l, s), l)
    if isinstance(l, int):
        return xi2y(get_index(l, s), l)
    if is_list_like(l):
        n = len(l)
        logger.debug('[{}] - l: {} - s: {}'.format('get_as_us', l, s))
        # TODO: find permanent solution for when optic is empty list: []
        if n == 0:

            return s
        for i in range(0, n):
            if isinstance(l[i], str):
                s = get_prop(l[i], s)
            elif isinstance(l[i], int):
                s = get_index(l[i], s)
            else:
                return composed(i, l)(s, l[i-1], A.Select, xi2y)
        return xi2y(s, l[n-1])
    if xi2y is not id_ and (l.length != 4 if hasattr(l, 'length') else False):
        return xi2y(l(s, None), None)
    else:
        return maybe_reader(l)(s, None, A.Select, xi2y)


def get_u(l, s): return get_as_u(id_, l, s)


class Undefined(dict):
    pass


def get_pick(template, x):
    r = {}
    for k in template.keys():
        t = template[k]
        # v = get_pick(t, x) if is_dict_like(t) else get_as_u(id_, t, x)
        if is_dict_like(t):
            v = get_pick(t, x)
        else:
            try:
                v = get_as_u(id_, t, x)
            except KeyError:
                v = Undefined()
        if v is not None:
            if r is None:
                r = type(x)()
            r[k] = v

    return r if len(r) > 0 else None


def _filter_undefined(x):
    return type(x)((key, value) for key, value in x.items() if not isinstance(value, Undefined))


def pick(template):
    """L.pick creates a lens out of the given possibly nested object template of
    lenses and allows one to pick apart a data structure and then put it back
    together. When viewed, undefined properties are not added to the result and
    if the result would be an empty object, the result will be undefined. This
    allows L.pick to be used with e.g. L.choices. Otherwise an object is created,
    whose properties are obtained by viewing through the lenses of the template.
    When set with an object, the properties of the object are set to the context
    via the lenses of the template.

    For example, let's say we need to deal with data and schema in need of some
    semantic restructuring:

    sample_flat = {'px': 1, 'py': 2, 'vx': 1, 'vy': 0}

    We can use L.pick to create a lens to pick apart the data and put it back
    together into a more meaningful structure:

    sanitize = L.pick({'pos': {'x': 'px', 'y': 'py'},
                      vel: {'x': 'vx', 'y': 'vy'}})

    In the template object the lenses are relative to the root focus of L.pick.

    We now have a better structured view of the data:

    L.get(sanitize, sampleFlat)
    # { 'pos': { 'x': 1, 'y': 2 }, vel: { 'x': 1, 'y': 0 } }

    Arguments:
        template {dict} -- mapping of keys to lenses relative to the root focus

    Returns:
        dict -- sanitized result
    """
    def wrapper(x, i, F, xi2yF):
        print("aaaa", F, F.map)
        out = F.map(
            lambda v: set_pick(template, v, x),
            xi2yF(get_pick(template, x), i)
        )
        # handle case where all keys in pick don't exist in data
        if (is_dict_like(out)):
            if all(isinstance(x, Undefined) for x in out.values()):
                return None
            else:
                return _filter_undefined(out)
        else:
            return out
    wrapper.length = 4
    return wrapper


def subseq_u(begin, end, t):
    t = to_function(t)

    def wrapper(x, i, F, xi2yF):
        n = -1

        def inner(x, i):
            nonlocal n
            n += 1
            if begin <= n and not (end <= n):
                return xi2yF(x, i)
            else:
                return F.of(x)
        return t(x, i, F, inner)
    wrapper.length = 4
    wrapper.__name__ = 'subseq_u<wrapper>'
    return wrapper


# def elems(xs, _i, A, xi2yA):
#     result = reduce(
#         lambda ysF, x: A.ap(
#             A.map(
#                 lambda ys: lambda y: [*ys, y], ysF),
#             xi2yA(x, xs)),
#         xs,
#         A.of([]))
#     return result

def tmp(*args):
    return args


def boo(A, xi2yA):
    def _boo(ysF, x):
        try:
            foo = xi2yA(*reversed(x))
        except Exception as e:
            return [x[1]]
        out = A.ap(
            A.map(
                lambda ys: lambda y: [*ys, y], ysF),
            foo
        )
        return out
    return _boo


def elems(xs, _i, A, xi2yA):
    n = len(xs) if is_list_like(xs) else 0
    ys = []
    j = 0
    same = True
    for i in range(n):
        # x = xs[i]
        x = get_index(i, xs)
        try:
            y = xi2yA(x, i)
        except KeyError:
            y = x
        j += 1
        if y is not None and not isinstance(y, Undefined):
            ys.append(y)
        if (same):
            print("aldf", x, y)
            try:
                same = (x == y and (x != 0 or (x == 0 and y == 0))) or (
                    x != x and y != y)
            except ValueError:
                same = x.equals(y)
    if (j != n):
        for _ in range(n-j):
            ys.append(None)
    elif same:
        return xs
    else:
        return type(xs)(ys)


def elems_(xs, _i, A, xi2yA):
    result = reduce(
        boo(A, xi2yA),
        enumerate(xs),
        A.of([]))
    return result


def get_values(ys):
    return ys.values()


def values_helper(ys, y, ysF, x):
    pass


# def values(xs, _i, A, xi2yA):
#    # print('values', _i, A, xs, xi2yA)
#     result = reduce(
#         lambda ysF, x: A.ap(
#             A.map(
#                 lambda ys: lambda y: values_helper(ys, y, ysF, x), ysF),
#             xi2yA(x, xs)),
#         xs.items(),
#         A.of(dict()))
#     return result


elems.length = 4


@arityn(3)
def collect_as(xi2y, t, s):
    results = []

    def as_fn(x, i):
        y = xi2y(x, i)
        if y is not None and not isinstance(y, Undefined):
            results.append(y)
    get_as_u(
        as_fn,
        t, s)
    return results


@arityn(2)
def collect(t, s):
    return collect_as(id_, t, s)


@arityn(3)
def modify(o, xi2x, s):
    return modify_u(o, xi2x, s)


@arityn(3)
def set(o, x, s):
    return set_u(o, x, s)


@arityn(3)
def setf(o, s, x):
    return set_u(o, x, s)


@arityn(3)
def subseq(begin, end, t):
    return subseq_u(begin, end, t)


@arityn(2)
def get(l, s):
    return get_u(l, s)


def cpair(xs):
    def _cpair(x):
        return [x, xs]
    return _cpair


def branch_assemble(ks):
    def _branch_assemble(xs):
        r = {}
        i = len(ks)
        while (i > 0):
            try:
                v = xs[0]
            except TypeError as e:
                return xs
            if v is not None:
                try:
                    r[ks[i]] = v
                except IndexError as e:
                    return v
            try:
                xs = xs[1]
            except TypeError as e:
                return xs
        return r
    return _branch_assemble


def branch_or_1_level_identity(otherwise, k2o, x0, x, A, xi2yA):
    written = None
    same = True
    r = {}
    _k2o = I.to_dict_like(k2o) if is_list_like(k2o) else k2o

    if is_list_like(k2o):
        for k, bar in enumerate(k2o):
            written = 1
            # x = x0.get(k, Undefined())
            x = x0[k]
            y = k2o[k](x, k, A, xi2yA)

            # if is_list_like(k2o):
            # y = type(k2o)(y.values())
            if (y is not None):
                r[k] = y
                if same:
                    same = (x == y and (x != 0 or (x == 0 and y == 0))) or (
                        x != x and y != y)
            else:
                same = False
    else:
        for k in _k2o:
            written = 1
            x = get_prop(k, x0)  # x0.get(k, Undefined())
            y = k2o[k](x, k, A, xi2yA)
            # if is_list_like(k2o):
            # y = type(k2o)(y.values())
            if (y is not None):
                r[k] = y
                if same:
                    same = (x == y and (x != 0 or (x == 0 and y == 0))) or (
                        x != x and y != y)
            else:
                same = False
    # for k in _k2o:
    #     written = 1
    #     x = x0.get(k, Undefined())
    #     print("aaaa", x, k, k2o, k2o.__class__)
    #     y = k2o[k](x, k, A, xi2yA)
    #     print("bbbbb", k2o.__class__, y)
    #     if is_list_like(k2o):
    #         y = type(k2o)(y.values())
    #     if (y is not None):
    #         r[k] = y
    #         if same:
    #             same = (x == y and (x != 0 or (x == 0 and y == 0))) or (
    #                 x != x and y != y)
    #     else:
    #         same = False
    t = written
    for k in _iter(x0):
        if (t is None or _k2o.get(k, None) is None):
            written = 1
            x = get_prop(k, x0)
            y = otherwise(x, k, A, xi2yA)
            if (y is not None):
                r[k] = y
                if same:
                    same = (x == y and (x != 0 or (x == 0 and y == 0))) or (
                        x != x and y != y)
            else:
                same = False
    return (x if (same and x0 == x) else r) if written else x


def branch_or_1_level(otherwise, k2o):
    def _branch_or_1_level(x, _i, A_, xi2yA):
        x0 = I.to_dict_like(x) if (is_list_like(
            x) or is_dict_like(x)) else I.create(None)
        # TODO: move all algebras to one file
        if I.Identity == A_ or A.Identity == A_:
            out = branch_or_1_level_identity(otherwise, k2o, x0, x, A_, xi2yA)
            return out
            # if is_list_like(x):
            #     return type(x)(out)
            # else:
            #     return out
        elif A.Select == A_:
            for k in k2o:
                y = k2o[k](x0[k], k, A_, xi2yA)
                if y is None:
                    return y
            for k in x0:
                if k2o.get(k, None) is None:
                    y = otherwise(x0[k], k, A_, xi2yA)
                    # if y is None:
                    #     return y
        else:
            xsA = A_.of(cpair)
            ks = []
            for ks in k2o:
                ks.append(k)
                xsA = A_.ap(A_.map(cpair, xsA), k2o[k](x0[k], k, A_, xi2yA))
            t = None if len(ks) == 0 else True
            foo = range(len(x0)) if is_list_like(x0) else x0
            for k in foo:
                if (t is None or k2o[k] is None):
                    ks.append(k)
                    xsA = A_.ap(
                        A_.map(cpair, xsA),
                        otherwise(x0[k], k, A_, xi2yA))
            return A_.map(branch_assemble(ks), xsA) if len(ks) != 0 else A_.of(x)
    _branch_or_1_level.length = 4
    return _branch_or_1_level


def branch_or_u(otherwise, template):
    k2o = I.create(template)
    _template = range(len(template)) if is_list_like(template) else template
    for k in _template:
        v = template[k]
        # len(v) !=0 needed for case when optic is an empty list
        v2 = branch_or_u(otherwise, v) if (
            is_dict_like(v)) else to_function(v)
        try:
            k2o[k] = v2
        except IndexError:
            k2o.append(v2)
    out = branch_or_1_level(otherwise, k2o)
    return out


@arityn(2)
def branch_or(otherwise, template):
    otherwise = to_function(otherwise)
    out = branch_or_u(otherwise, template)
    a = 1
    return out


def zero(x, _i, C, _xi2yC):
    return C.of(x)


zero.length = 4

branch = branch_or(zero)

values = branch_or_1_level(identity, I.protoless0)
values.length = 4


@arityn(2)
def remove(o, s):
    return set_u(o, None, s)
# remove = arityn(2)(function remove(o, s) {
#     return setU(o, void 0, s)
# })


def disperse_u(traversal, values, data):
    if not is_list_like(values):
        values = ''
    i = 0

    def do(*args, **kwargs):
        nonlocal i
        # i += 1
        v = values[i]
        i += 1
        return v
    return modify_u(traversal, do, data)


@arityn(3)
def disperse(traversal, values, data):
    return disperse_u(traversal, values, data)


class Lens:
    def __init__(self, obj):
        self._obj = obj
        self.set = partial(set, s=self._obj)
        self.get = partial(get, s=self._obj)
        self.remove = partial(remove, s=self._obj)
        self.modify = partial(modify, s=self._obj)
        self.collect = partial(collect, s=self._obj)
        self.disperse = partial(disperse, data=self._obj)
