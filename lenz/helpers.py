from functools import partial, reduce, wraps
from itertools import chain, islice
from collections import namedtuple
from types import new_class
from abc import ABCMeta
from copy import deepcopy, copy
import inspect


class ListLike(metaclass=ABCMeta):
    pass


class DictLike(metaclass=ABCMeta):
    pass


DictLike.register(dict)

immutable_types = set((int, str))


class Frozen(object):
    def __init__(self, value):
        self._value = value

    def __getattribute__(self, name):
        if name == '_value':
            return super(Frozen, self).__getattribute__(name)
        v = getattr(self._value, name)
        return v if v.__class__ in immutable_types else freeze(v)

    def __setattr__(self, name, value):
        if name == '_value':
            super(Frozen, self).__setattr__(name, value)
        else:
            raise Exception(
                "Can't modify frozen object {0}".format(self._value))


def freeze(value):
    return Frozen(value)


def arityn(n):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, args_state=(), kwargs_state=dict(), **kwargs):
            args_state = args_state + args
            kwargs_state = dict(**kwargs_state, **kwargs)
            if len(args_state)+len(kwargs_state.keys()) == n:
                result = fn(*args_state, **kwargs_state)
                return result
            elif len(args_state)+len(kwargs_state.keys()) > n:
                result = fn(*args_state[0:n])
                return result
            else:
                return partial(wrapper, args_state=args_state, kwargs_state=kwargs_state)
        return wrapper
    return decorator


def n_args(fn):
    argspec = inspect.getfullargspec(fn)
    return len(argspec.args)


def drop_extra_args(fn, *args):
    error = None
    for i in reversed(range(len(args))):
        try:
            return fn(args[0:i])
        except TypeError as e:
            error = e
            continue
    if error is not None:
        raise error


def snd_u(x, y):
    return y


def always(x):
    def no_op(*args, **kwargs):
        return x
    return no_op


def apply(value, operation):
    return operation(value)


def pipe(operations):
    def pipe_op(start_value):
        return reduce(apply, operations, start_value)
    return pipe_op


def register_list_like(typ):
    ListLike.register(typ)


def is_list_like(x):
    return isinstance(x, (list, tuple, map, chain, ListLike))


def is_dict_like(x):
    return isinstance(x, DictLike) or (hasattr(x, 'keys') and hasattr(x, 'values') and hasattr(x, 'items') and callable(x.values))


def to_list_like(x):
    if not is_list_like(x):
        return (x,)
    return x


def tap(fn):
    @wraps(fn)
    def wrapper(x):
        fn(x)
        return x
    return wrapper


def log_inputs(fn):
    print('here')
    return pipe([tap(print), fn])


def fmap(f, data):
    if isinstance(data, list):
        return [f(x) for x in data]
    if is_dict_like(data):
        print('fmap dict like', data)

        result = {k: f(v) for (k, v) in data.items()}
        print('fmap dict result', result)
        return result
    print('fmap default', data)
    return data


def cata(f, data):
    def cata_on_f(x): return cata(f, x)
    recursed = fmap(cata_on_f, data)
    print('cata recursed', recursed)
    return f(recursed)


def assign(obj, src):
    for k, v in src.items():
        v2 = copy(v)
        obj[k] = v2
    return obj


def protoless(o): return assign(dict(), o)


protoless0 = protoless({})
