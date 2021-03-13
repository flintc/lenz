import operator as op
import lenz.helpers as H
from functools import reduce, wraps, partial

equals = H.arityn(2)(op.eq)


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


def pipe(*fns):
    return lambda x: reduce(lambda acc, y: y(acc), fns, x)


@H.arityn(1)
def negate(x):
    # print("????", x)
    return -1*x


# @H.arityn(1)
def inc(x, *args):
    # print("????", x)
    return x+1


def dec(x, *args):
    # print("????", x)
    return x-1


@H.arityn(2)
def add(x, y):
    return x+y


@H.arityn(1)
def identity(x):
    return x
