
from copy import deepcopy


def nth(n, fn):
    def wrapper(xs):
        result = deepcopy(xs)
        result[n] = fn(result[n])
        return result
    return wrapper


def nth_arg(n, fn):
    def wrapper(*args, **kwargs):
        return fn(args[n], **kwargs)
    return wrapper
