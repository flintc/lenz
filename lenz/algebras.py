
from lenz.helpers import always, snd_u


def logger(fn):
    def wrapper(*args):
        out = fn(*args)
        print(f"LOGGER: {fn.__name__} out", out)
        return out
    return wrapper


class Applicative:
    def __init__(self, map, of, ap):
        # self.call = logger(map)
        self.call = map
        # self.of = logger(of)
        self.of = of
        self.ap = ap
        self.map = map


class Constant:
    def map(x2y, c):
        return c


def ConstantWith(ap, empty=None): return Applicative(snd_u, always(empty), ap)


def foo(l, r):
    if l is not None:
        return l
    return r


Select = ConstantWith(foo)
Select.__name__ = "Select"


class Pair:
    def of(x):
        return lambda y: (x, y)

    def ap(x2yS, xS):
        def wrapper(x0):
            x1, x2y = x2yS(x0)
            x, y1 = xS(x1)
            return (x, x2y(y1))
        return wrapper

    def map(x2y, xS):
        return Pair.ap(Pair.of(x2y), xS)

    def run(s, xS):
        return xS(s)[0]


class State:
    def of(result):
        return lambda state: {'state': state, 'result': result}

    def ap(x2yS, xS):
        def wrapper(state0):
            tmp = x2yS(state0)
            tmp1 = xS(tmp['state'])
            return {
                'state': tmp1['state'],
                'result': tmp['result'](tmp1['result'])
            }
        return wrapper

    def map(x2y, xS):
        return State.ap(State.of(x2y), xS)

    def run(s, xS):
        return xS(s)['result']


class TupleM:
    def of(x):
        return (x,)

    def concat(x, y):
        return x+y

    def empty():
        return ()


class Identity:
    def of(x):
        return x

    def ap(x2y, x):
        out = x2y(x)
        return type(x)(out)

    def map(x2y, x):
        out = x2y(x)
        return out


class First:
    def of(x):
        return x[0]


class Function:
    def map(x2y, c):
        return x2y(c)


def elems(algebra):
    def transformer(x2yF):
        def exec_transform(xs):
            return reduce(lambda ysF, x: algebra.ap(algebra.map(lambda ys: lambda y: [*ys, y], ysF), x2yF(x)), xs, algebra.of([]))
        return exec_transform
    return transformer
