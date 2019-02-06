

import lenz as L
from lenz.log import stream_handler
import lenz.operations as R
from functools import wraps
import inspect
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(stream_handler)


def spread(fn):
    @wraps(fn)
    def wrapper(xs):
        return fn(*xs)
    return wrapper


def assert_eq(test, expected):
    logger.info('TEST: {}'.format(inspect.getsource(test)))
    assert(test() == expected)


find_tests = [
    (lambda: L.set(L.find(R.equals(2)), None, [2]), []),
    (lambda: L.set(L.find(R.equals(2)))(None, [1, 2, 3]), [1, 3]),
    (lambda: L.set(L.find(R.equals(2)))(4)([1, 2, 3]), [1, 4, 3]),
    #(lambda: L.set(L.find(R.equals(2)), 2)([1, 4, 3]), [1, 4, 3, 2]),
]


def run_tests():
    list(map(spread(assert_eq), find_tests))


if __name__ == '__main__':
    run_tests()
