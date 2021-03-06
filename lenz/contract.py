
from copy import deepcopy
from functools import wraps


def nth(n, fn):
    @wraps(fn)
    def wrapper(xs):
        result = deepcopy(xs)
        result[n] = fn(result[n])
        return result
    return wrapper


def nth_arg(n, fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(args[n], **kwargs)
    return wrapper
