from functools import partial, reduce, wraps
from itertools import chain, islice
from collections import namedtuple
from types import new_class
from abc import ABCMeta


class ListLike(metaclass=ABCMeta):
    pass


class DictLike(metaclass=ABCMeta):
    pass


DictLike.register(dict)


def arityn(n):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, args_state=(), kwargs_state=dict(), **kwargs):
            args_state = args_state + args
            kwargs_state = dict(**kwargs_state, **kwargs)
            #print(args_state, kwargs_state)
            if len(args_state)+len(kwargs_state.keys()) >= n:
                result = fn(
                    *args_state[0:(n-len(kwargs_state.keys()))], **kwargs_state)
                return result
            else:
                return partial(wrapper, args_state=args_state, kwargs_state=kwargs_state)
        return wrapper
    return decorator


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
