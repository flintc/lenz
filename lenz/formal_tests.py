

import lenz as L
from lenz.log import stream_handler
import lenz.operations as R
import lenz.infestines as I
from functools import wraps
import inspect
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)
logger.addHandler(stream_handler)


def spread(fn):
    @wraps(fn)
    def wrapper(xs):
        return fn(*xs)
    return wrapper


def describe(desc, run):
    logger.info('Running tests for: {}'.format(desc))
    run()


def assert_eq(test, expected):
    logger.info('TEST: {}'.format(inspect.getsource(test)))
    try:
        result = test()
        try:
            assert(result == expected)
        except AssertionError as e:
            logger.critical('TEST: {}'.format(inspect.getsource(test)))
            logger.critical(
                '    ASSERTION ERROR: {} {}'.format(result, expected))
    except Exception as e:
        logger.critical('TEST: {}'.format(inspect.getsource(test)))
        logger.critical('    FAILED - error: {}, {}'.format(e, type(e)))


find_tests = [
    (lambda: L.set(L.find(R.equals(2)), None, [2]), []),
    (lambda: L.set(L.find(R.equals(2)))(None, [1, 2, 3]), [1, 3]),
    (lambda: L.set(L.find(R.equals(2)))(4)([1, 2, 3]), [1, 4, 3]),
    # (lambda: L.set(L.find(R.equals(2)), 2)([1, 4, 3]), [1, 4, 3, 2]),
    # (lambda: L.set(L.find(R.equals(2)), 2, None), [2]),
    # (lambda: L.set(L.find(R.equals(2)), 2, []), [2]),
    (lambda: L.get(L.find(R.equals(2)), None), None),
    (lambda: L.get(L.find(R.equals(2)), [3]), None),
    (lambda: L.get(L.find(R.equals(1), {'hint': 2}), [2, 2, 2, 1, 2]), 1),
    (lambda: L.set(L.find(R.equals(2), {
     'hint': 2}), 55, [2, 1, 2]), [2, 1, 55]),
    (lambda: L.set(L.find(R.equals(2), {
     'hint': 0}), 55, [2, 1, 2]), [55, 1, 2]),
    (lambda: L.set(L.find(R.equals(2), {
     'hint': 1}), 55, [2, 1, 2]), [55, 1, 2]),
    (
        lambda: \
        L.get(
            L.find(
                R.pipe(
                    abs,
                    R.equals(2)
                ),
                {'hint': 2}
            ),
            [-1, -2, 3, 1, 2, 1]
        ),
        -2
    ),
    (lambda:  L.get([], [[{'x': {'y': 101}}]]), [[{'x': {'y': 101}}]]),
    (lambda: L.set([0], None, [None]), []),
    (lambda:  L.set(1, '2', ['1', '2', '3']), ['1', '2', '3']),
    (lambda: L.modify('x', lambda x: x + 1, {'x': 1}), {'x': 2}),
]


def rewrite_tests():
    assert_eq(lambda: L.get(L.rewrite(lambda x: x - 1), 1), 1)
    assert_eq(lambda: L.get(L.rewrite(lambda x: x - 1), None), None)
    assert_eq(lambda: L.set(L.rewrite(lambda x: x - 1), None, 1), None)
    assert_eq(lambda: L.set(L.rewrite(lambda x: x - 1), 3, 1), 2)


def reread_tests():
    assert_eq(lambda: L.get(L.reread(lambda x: x - 1), 1), 0)
    assert_eq(lambda: L.get(L.reread(lambda x: x - 1), None), None)
    assert_eq(lambda: L.set(L.reread(lambda x: x - 1), None, 1), None)
    assert_eq(lambda: L.set(L.reread(lambda x: x - 1), 3, 1), 3)

# describe('composing with plain functions', lambda:


def composing_plain_functions_tests():
    assert_eq(lambda: L.get(lambda x: x + 1, 2), 3)
    assert_eq(lambda: L.modify(R.inc, R.negate, 1), 1)
    assert_eq(lambda: L.get(['x', lambda x, i: [x, i]], {'x': -1}), [-1, 'x'])
    assert_eq(lambda: L.collect([L.elems, lambda x, i: [x, i]], ['x', 'y']), [
        ['x', 0],
        ['y', 1]
    ])
    assert_eq(lambda: L.collect([L.values, lambda x, i: [x, i]], {'x': 1, 'y': -1}), [
        [1, 'x'],
        [-1, 'y']
    ])
    assert_eq(lambda: L.get([0, lambda x, i: [x, i]], [-1]), [-1, 0])
    assert_eq(lambda: L.get([0, 'x', R.negate], [{'x': -1}]), 1)
    assert_eq(lambda: L.set([0, 'x', R.negate], 2, [{'x': -1}]), [{'x': -1}])
    assert_eq(lambda: L.get(I.always('always'), 'anything'), 'always')
    assert_eq(lambda: L.set(I.always('always'),
                            'anything', 'original'), 'original')


def run_test(test, *args):
    return test()


def elems_tests():
    run_test(lambda: L.modify(L.elems, R.identity, [0, -1]), [0, -1])
    assert_eq(lambda: L.modify(L.elems, R.identity,
                               {'x': 1, 'y': 2}), {'x': 1, 'y': 2})
    assert_eq(lambda: L.modify(L.elems, R.inc, {
              'x': 1, 'y': 2}), {'x': 1, 'y': 2})
    assert_eq(lambda: L.modify(L.elems, R.negate, []), [])
    #assert_eq(lambda: L.remove(L.elems, [1]), [])
    assert_eq(
        lambda:
        L.modify(['xs', L.elems, 'x', L.elems], R.add(1), {
            'xs': [{'x': [1]}, {'x': [2, 3, 4]}]
        }),
        {'xs': [{'x': [2]}, {'x': [3, 4, 5]}]}
    )
    assert_eq(
        lambda:
        L.set(['xs', L.elems, 'x', L.elems], 101, {
            'xs': [{'x': [1]}, {'x': [2, 3, 4]}]
        }),
        {'xs': [{'x': [101]}, {'x': [101, 101, 101]}]}
    )
    # assert_eq(
    #     lambda:
    #     L.remove(['xs', L.elems, 'x', L.elems], {
    #         'ys': 'hip',
    #         'xs': [{'x': [1]}, {'x': [2, 3, 4]}]
    #     }),
    #     {'ys': 'hip', 'xs': [{'x': []}, {'x': []}]}
    # )
    assert_eq(
        lambda:
        L.modify(['xs', L.elems, 'x'], lambda x: (None if x < 2 else x), {
            'xs': [{'x': 3}, {'x': 1}, {'x': 4}, {'x': 1, 'y': 0}, {'x': 5}, {'x': 9}, {'x': 2}]
        }),
        {'xs': [{'x': 3}, {}, {'x': 4}, {'y': 0}, {'x': 5}, {'x': 9}, {'x': 2}]}
    )
    assert_eq(
        lambda:
        L.modify([L.elems, ['x', L.elems]], R.add(1), [
            {'x': [1]},
            {},
            {'x': []},
            {'x': [2, 3]}
        ]),
        [{'x': [2]}, {}, {'x': []}, {'x': [3, 4]}]
    )
    assert_eq(
        lambda:
        L.modify([[L.elems, 'x'], L.elems], R.add(1), [
            {'x': [1]},
            {'y': 'keep'},
            {'x': [], 'z': 'these'},
            {'x': [2, 3]}
        ]),
        [{'x': [2]}, {'y': 'keep'}, {'x': [], 'z': 'these'}, {'x': [3, 4]}]
    )


L.get([0, 'x', R.negate], [{'x': -1}])
describe('L.rewrite', rewrite_tests)
describe('L.reread', reread_tests)
describe('composing with plain functions', composing_plain_functions_tests)
describe('L.elems', elems_tests)


def run_tests():
    list(map(spread(assert_eq), find_tests))


if __name__ == '__main__':
    # run_tests()
    pass
