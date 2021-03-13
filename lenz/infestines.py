from copy import deepcopy
from .helpers import is_list_like, is_dict_like

ENV = 'dev'

LENGTH = 'length'


def always(x):
    return lambda *args, **kwargs: x


def _define_name_u(fn, value):
    fn.__name__ = value
    fn.name = value
    fn.configurable = True
    return fn


def define_name_u(*args, **kwargs):
    try:
        return _define_name_u(_define_name_u, 'defineName')
    except Exception as e:
        return lambda fn, _: fn


def id(x):
    return x


def dev_set_name(to, name):
    return define_name_u(to, name)


def dev_copy_name(to, from_):
    return define_name_u(to, from_.name)


def dev_with_name(ary):
    return lambda fn: copy_name(ary(fn), fn)


set_name = id if ENV == 'production' else dev_set_name

copy_name = id if ENV == 'production' else dev_copy_name

with_name = id if ENV == 'production' else dev_with_name


class Functor:
    def __init__(self, map):
        self.map = map


class Applicative(Functor):
    def __init__(self, map, of, ap):
        super().__init__(map)
        self.of = of
        self. ap = ap


class Monad(Applicative):
    def __init__(self, map, of, ap, chain):
        super().__init__(map, of, ap)
        self.chain = chain


def apply_u(x2y, x):
    return x2y(x)


Identity = Monad(apply_u, id, apply_u, apply_u)


def assign(a, b):
    return dict(**a, **b)


def create(x):
    if is_list_like(x):
        return []
    else:
        return {}


def protoless(o):
    return assign(create(None), o)


protoless0 = protoless({})


def to_dict_like(x):
    if is_dict_like(x):
        return deepcopy(x)
    elif is_list_like(x):
        return dict(enumerate(x))
