import os
import pytest
import inspect
import sys
import lenz as L
from lenz.log import stream_handler
import lenz.operations as R
import lenz.infestines as I
import lenz.helpers as H
from functools import wraps
import inspect
import logging
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)
logger.setLevel(logging.INFO)
logger.addHandler(stream_handler)


@dataclass
class XYZ:
    x: int
    y: int
    z: int


def add10(x):
    return x+10


@H.register_dict_like
class XYZ:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"{self.x}, {self.y}, {self.z}"


def assert_eq(test, expected):
    lines, lineno = inspect.getsourcelines(test)

    def test_assert_eq():
        assert test() == expected
    test_assert_eq.__name__ = f"test_assert_eq:{lineno}"
    return test_assert_eq


testEq = assert_eq


compose_test = [
    assert_eq(lambda: L.get(L.compose(), 'any'), 'any'),
    assert_eq(lambda: L.compose('x'), 'x'),
    assert_eq(lambda: L.compose(101), 101),
    assert_eq(
        lambda:
        L.compose(
            101,
            'x'
        ),
        [101, 'x']
    )
]
# @pytest.mark.parametrize("run_test", compose_test)
# def test_compose(run_test):
#     run_test()

identity_tests = [
    assert_eq(lambda: L.get(L.identity, 'any'), 'any'),
    assert_eq(lambda: L.modify(L.identity, R.add(1), 2), 3),
    assert_eq(lambda: L.modify([], R.add(1), 2), 3),
    assert_eq(lambda: L.remove(['x', L.identity], {'x': 1, 'y': 2}), {'y': 2})
]


@pytest.mark.parametrize("run_test", identity_tests)
def test_identity(run_test):
    run_test()


composing_plain_functions = [
    assert_eq(lambda: L.get(lambda x: x + 1, 2), 3),
    assert_eq(lambda: L.modify(R.inc, R.negate, 1), 1),
    assert_eq(lambda: L.get(['x', lambda x, i: [x, i]], {'x': -1}), [-1, 'x']),
    assert_eq(lambda: L.collect([L.elems, lambda x, i: [x, i]], ['x', 'y']), [
        ['x', 0],
        ['y', 1]
    ]),
    assert_eq(lambda: L.collect([L.values, lambda x, i: [x, i]], {'x': 1, 'y': -1}), [
        [1, 'x'],
        [-1, 'y']
    ]),
    assert_eq(lambda: L.get([0, lambda x, i: [x, i]], [-1]), [-1, 0]),
    assert_eq(lambda: L.get([0, 'x', R.negate], [{'x': -1}]), 1),
    assert_eq(lambda: L.set([0, 'x', R.negate], 2, [{'x': -1}]), [{'x': -1}]),
    assert_eq(lambda: L.get(I.always('always'), 'anything'), 'always'),
    assert_eq(lambda: L.set(I.always('always'),
                            'anything', 'original'), 'original')
]


@pytest.mark.parametrize("run_test", composing_plain_functions)
def test_composing_plain_functions(run_test):
    run_test()


find_tests = [
    (lambda: L.set(L.find(R.equals(2)), None, [2]), []),
    (lambda: L.set(L.find(R.equals(2)))(None, [1, 2, 3]), [1, 3]),
    (lambda: L.set(L.find(R.equals(2)))(4)([1, 2, 3]), [1, 4, 3]),
    (lambda: L.set(L.find(R.equals(2)), 2)([1, 4, 3]), [1, 4, 3, 2]),
    (lambda: L.set(L.find(R.equals(2)), 2, None), [2]),
    (lambda: L.set(L.find(R.equals(2)), 2, []), [2]),
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
        lambda:
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
    # fails b/c unary modifier
    (lambda: L.modify('x', lambda x: x + 1, {'x': 1}), {'x': 2}),
]


@pytest.mark.parametrize("run_test", find_tests)
def test_find(run_test):
    assert_eq(*run_test)


rewrite_tests = [
    assert_eq(lambda: L.get(L.rewrite(lambda x: x - 1), 1), 1),
    assert_eq(lambda: L.get(L.rewrite(lambda x: x - 1), None), None),
    assert_eq(lambda: L.set(L.rewrite(lambda x: x - 1), None, 1), None),
    assert_eq(lambda: L.set(L.rewrite(lambda x: x - 1), 3, 1), 2)
]


@pytest.mark.parametrize("run_test", rewrite_tests)
def test_rewrite(run_test):
    run_test()


reread_tests = [
    testEq(
        lambda:
        L.get(
            L.reread(lambda x: x - 1),
            1
        ),
        0
    ),
    testEq(
        lambda:
        L.get(
            L.reread(lambda x: x - 1),
            None
        ),
        None
    ),
    testEq(
        lambda:
        L.set(
            L.reread(lambda x: x - 1),
            None,
            1
        ),
        None
    ),
    testEq(
        lambda:
        L.set(
            L.reread(lambda x: x - 1),
            3,
            1
        ),
        3
    )
]


@pytest.mark.parametrize("run_test", reread_tests)
def test_reread(run_test):
    run_test()


elems_tests = [
    assert_eq(lambda: L.modify(L.elems, R.identity, [0, -1]), [0, -1]),
    assert_eq(lambda: L.modify(L.elems, R.identity,
                               {'x': 1, 'y': 2}), {'x': 1, 'y': 2}),
    assert_eq(lambda: L.modify(L.elems, R.inc, {
        'x': 1, 'y': 2}), {'x': 1, 'y': 2}),
    assert_eq(lambda: L.modify(L.elems, R.negate, []), []),
    assert_eq(lambda: L.remove(L.elems, [1]), []),
    assert_eq(
        lambda:
        L.modify(['xs', L.elems, 'x', L.elems], R.add(1), {
            'xs': [{'x': [1]}, {'x': [2, 3, 4]}]
        }),
        {'xs': [{'x': [2]}, {'x': [3, 4, 5]}]}
    ),
    assert_eq(
        lambda:
        L.set(['xs', L.elems, 'x', L.elems], 101, {
            'xs': [{'x': [1]}, {'x': [2, 3, 4]}]
        }),
        {'xs': [{'x': [101]}, {'x': [101, 101, 101]}]}
    ),
    assert_eq(
        lambda:
        L.remove(['xs', L.elems, 'x', L.elems], {
            'ys': 'hip',
            'xs': [{'x': [1]}, {'x': [2, 3, 4]}]
        }),
        {'ys': 'hip', 'xs': [{'x': []}, {'x': []}]}
    ),
    assert_eq(
        lambda:
        L.modify(['xs', L.elems, 'x'], lambda x: (None if x < 2 else x), {
            'xs': [{'x': 3}, {'x': 1}, {'x': 4}, {'x': 1, 'y': 0}, {'x': 5}, {'x': 9}, {'x': 2}]
        }),
        {'xs': [{'x': 3}, {}, {'x': 4}, {'y': 0}, {'x': 5}, {'x': 9}, {'x': 2}]}
    ),
    assert_eq(
        lambda:
        L.modify([L.elems, ['x', L.elems]], R.add(1), [
            {'x': [1]},
            {},
            {'x': []},
            {'x': [2, 3]}
        ]),
        [{'x': [2]}, {}, {'x': []}, {'x': [3, 4]}]
    ),
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
]


@pytest.mark.parametrize("run_test", elems_tests)
def test_elems(run_test):
    run_test()


values_tests = [
    assert_eq(lambda: L.modify(
        # keys maybe should be strings? probably not though
        L.values, R.identity, [1, 2]), {0: 1, 1: 2}),
    assert_eq(lambda: L.modify(L.values, R.inc, [1, 2]), {0: 2, 1: 3}),
    assert_eq(lambda: L.modify(L.values, R.negate, {
              'x': 11, 'y': 22}), {'x': -11, 'y': -22}),
    assert_eq(
        lambda:
        L.remove([L.values, L.when(lambda x: 11 < x & x < 33)], {
            'x': 11,
            'y': 22,
            'z': 33
        }),
        {'x': 11, 'z': 33}
    ),
    assert_eq(lambda: L.remove(L.values, {'x': 11, 'y': 22, 'z': 33}), {}),
    assert_eq(lambda: L.modify(L.values, R.inc, {}), {}),
    assert_eq(lambda: L.remove(L.values, {'x': 1}), {}),
    assert_eq(lambda: L.remove(L.values, None), None),
    assert_eq(lambda: L.modify(L.values, R.inc, None), None),
    assert_eq(lambda: L.modify(L.values, R.inc, 'anything'), 'anything'),
    assert_eq(lambda: L.modify(L.values, R.inc, XYZ(3, 1, 4)),
              XYZ(**{'x': 4, 'y': 2, 'z': 5}))
]


@pytest.mark.parametrize("run_test", values_tests)
def test_values(run_test):
    run_test()


branch_tests = [
    assert_eq(
        lambda: L.modify(L.branch_or([], {'x': []}), R.identity, {
                         'x': 0, 'y': np.nan, 'z': 0}),
        {
            'x': 0,
            'y': np.nan,
            'z': 0
        }
    ),
    assert_eq(lambda: L.modify(L.branch({}), lambda x: x + 1, None), None),
    assert_eq(lambda: L.modify(L.branch({}),
                               lambda x: x + 1, 'anything'), 'anything'),
    assert_eq(lambda: L.modify(L.branch({}), lambda x: + 1, {}), {}),
    assert_eq(lambda: L.set(L.branch({'x': []}), 1, 9), {'x': 1}),
    assert_eq(lambda: L.remove(L.branch({'x': []}), 1), {}),
    assert_eq(lambda: L.remove(L.branch({}), {}), {}),
    assert_eq(lambda: L.modify(L.branch({}),
                               lambda x: x + 1, {'x': 1}), {'x': 1}),
    assert_eq(
        lambda:
        L.modify(L.branch({'a': 'x', 'b': [], 'c': 0, 'd': L.identity}), lambda x: x + 1, {
            'a': {'x': 1},
            'b': 2,
            'c': [3],
            'd': 4,
            'extra': 'one'
        }),
        {'a': {'x': 2}, 'b': 3, 'c': [4], 'd': 5, 'extra': 'one'}
    ),
    assert_eq(lambda: L.set(L.branch({'a': ['x', 0], 'b': []}), 0, None), {
        'a': {'x': [0]},
        'b': 0
    }),
    assert_eq(lambda: L.modify(L.branch({'y': L.identity}), R.inc, XYZ(3, 1, 4)), {
        'x': 3,
        'y': 2,
        'z': 4
    }),
    assert_eq(lambda: L.or_(L.branch({'x': [], 'y': []}), {
              'x': False, 'y': False}), False),
    assert_eq(
        lambda: L.modify(L.branch({'x': {'a': []}, 'y': []}), R.negate, {
                         'x': {'a': 1}, 'y': 2}),
        {'x': {'a': -1}, 'y': -2}
    )
]


@pytest.mark.parametrize("run_test", branch_tests)
def test_branch(run_test):
    run_test()


branch_or_tests = [
    assert_eq(
        lambda:
        L.transform(L.branch_or(L.modify_op(R.inc), {'x': L.modify_op(R.dec)}), {
            'x': 1,
            'y': 1
        }),
        {'x': 0, 'y': 2}
    )
]


@pytest.mark.parametrize("run_test", branch_or_tests)
def test_branch_or(run_test):
    run_test()


collect_tests = [
    assert_eq(
        lambda:
        L.collect(['xs', L.elems, 'x', L.elems], {
            'xs': [{'x': [3, 1]}, {'x': [4, 1]}, {'x': [5, 9, 2]}]
        }),
        [3, 1, 4, 1, 5, 9, 2]
    ),
    assert_eq(
        lambda:
        L.collect([L.elems, 'x', L.elems], [
                  {'x': [1]}, {}, {'x': []}, {'x': [2, 3]}]),
        [1, 2, 3]
    ),
    assert_eq(lambda: L.collect(L.elems, []), []),
    assert_eq(lambda: L.collect('x', {'x': 101}), [101]),
    assert_eq(lambda: L.collect('y', {'x': 101}), []),
    assert_eq(
        lambda:
        L.collect(['a', L.elems, 'b', L.elems, 'c', L.elems], {
            'a': [{'b': []}, {'b': [{'c': [1]}]}, {'b': []}, {'b': [{'c': [2]}]}]
        }),
        [1, 2]
    ),
]


@pytest.mark.parametrize("run_test", collect_tests)
def test_collect(run_test):
    run_test()


collect_as_tests = [
    assert_eq(lambda: L.collect_as(
        R.negate, L.elems, [1, 2, 3]), [-1, -2, -3]),
    assert_eq(
        lambda:
        L.collect_as(lambda x, ix: None if x < 0 else x +
                     1, L.elems, [0, -1, 2, -3]),
        [1, 3]
    )
]


@pytest.mark.parametrize("run_test", collect_as_tests)
def test_collect_as(run_test):
    run_test()


pick_tests = [
    assert_eq(lambda: L.get(L.pick({'x': 'c'}),
                            {'a': [2], 'b': 1}), None),
    assert_eq(lambda: L.get(L.pick({'x': 'c', 'y': 'b'}),
                            {'a': [2], 'b': 1}), {'y': 1}),
    assert_eq(lambda: L.get(L.pick({'x': {'y': 'z'}}), None), None),
    assert_eq(lambda: L.set([L.pick({'x': 'c'}), 'x'], 4, {'a': [2], 'b': 1}), {
        'a': [2],
        'b': 1,
        'c': 4
    }),
    assert_eq(lambda: L.get(L.pick({'x': 'b', 'y': 'a'}), {
              'a': [2], 'b': 1}), {'x': 1, 'y': [2]}),
    assert_eq(lambda: L.set([L.pick({'x': 'b', 'y': 'a'}), 'x'], 3, {'a': [2], 'b': 1}), {
        'a': [2],
        'b': 3
    }),
    assert_eq(lambda: L.remove([L.pick({'x': 'b', 'y': 'a'}), 'y'], {'a': [2], 'b': 1}), {
        'b': 1
    }),
    assert_eq(lambda: L.remove([L.pick({'x': 'b'}), 'x'], {
        'a': [2], 'b': 1}), {'a': [2]}),
    assert_eq(lambda: L.get(L.pick({'x': 0, 'y': 1}), [
              'a', 'b']), {'x': 'a', 'y': 'b'}),
    assert_eq(
        lambda:
        L.get(L.pick({'x': {'y': 'a', 'z': 'b'}, 'b': ['c', 0]}), {
              'a': 1, 'b': 2, 'c': [3]}),
        {'x': {'y': 1, 'z': 2}, 'b': 3}
    ),
    assert_eq(
        lambda:
        L.set(
            L.pick({'x': {'y': 'a', 'z': 'b'}, 'b': ['c', 0]}),
            {'x': {'y': 4}, 'b': 5, 'z': 2},
            {'a': 1, 'b': 2, 'c': [3]}
        ),
        {'a': 4, 'c': [5]}
    )
]


@pytest.mark.parametrize("run_test", pick_tests)
def test_pick(run_test):
    run_test()


def merge_with(x):
    return dict(**x, **dict(merge_data=10))


custom_tests = [
    assert_eq(lambda: L.collect(L.subseq_u(
        0, 1, ['b', L.elems]), {'a': 1, 'b': [1, 2, 3]}), [1]),
    assert_eq(lambda: L.transform([L.subseq_u(0, 1, ['b', L.elems]), L.modify_op(add10)], {'a': 1, 'b': [1, 2, 3]}), {
              'a': 1, 'b': [11, 2, 3]}),
    assert_eq(lambda: L.transform(
        ['b', 'b', 'b', 'b', 'b', 'b', L.elems, L.modify_op(add10)], dict(a=1, b=dict(b=dict(b=dict(b=dict(b=dict(b=[1, 2, 3]))))))), {'a': 1, 'b': {
            'b': {'b': {'b': {'b': {'b': [11, 12, 13]}}}}}}),
    assert_eq(lambda: L.modify(
        ['b', 'b', 'b', 'b', 'b', 'b', L.elems], add10, dict(a=1, b=dict(b=dict(b=dict(b=dict(b=dict(b=[1, 2, 3]))))))), {'a': 1, 'b': {
            'b': {'b': {'b': {'b': {'b': [11, 12, 13]}}}}}}),
    assert_eq(lambda: L.transform(
        ['b', L.elems, L.modify_op(add10)], {'a': 1, 'b': [1, 2, 3]}), {'a': 1, 'b': [11, 12, 13]}),
    assert_eq(lambda: H.pipe(
        [L.modify(0, R.add(1)), L.modify(1, R.add(-10))])([10, 10]), [11, 0]),
    assert_eq(lambda: L.modify(
        ['a', 'b', 'c', 1, L.subseq_u(2, 5, L.elems)], add10, {'a': {'b': {'c': [0, [90, 1, 2, 3, 4, 5, 6]]}}}), {'a': {'b': {'c': [0, [90, 1, 12, 13, 14, 5, 6]]}}}),
    assert_eq(lambda: L.get(
        L.pick({'z': ['b', 1], 'q': 'a', 'data': ['b', 2]}), {'a': 1, 'b': [1, 2, 3]}), {'z': 2, 'q': 1, 'data': 3}),
    assert_eq(lambda: L.transform(L.pick({'z': ['b', 1, L.modify_op(add10)], 'q': 'a'}), {'a': 1, 'b': [1, 2, 3]}), {
              'a': 1, 'b': [1, 12, 3]}),
    assert_eq(lambda: L.modify(L.pick({'z': ['b', 1], 'q': 'a'}), merge_with, {'a': 1, 'b': [1, 2, 3]}), {
              'a': 1, 'b': [1, 2, 3]}),
    assert_eq(lambda: L.collect([
        L.elems,
        L.get(L.pick({
            'description': 'Description',
            'amount': 'Amount',
            'date': 'Date',
        })),
        L.modify('date', lambda x: x+'!'),
        #     L.set('account',
        #           'checking',
        #     })),
    ], [{'Description': 'A', 'Amount': 10, 'Date': '2018/01/01'},
        {'Description': 'B', 'Amount': 20, 'Date': '2018/01/02'}]), [{'description': 'A', 'amount': 10, 'date': '2018/01/01!'},
                                                                     {'description': 'B', 'amount': 20, 'date': '2018/01/02!'}]),
    assert_eq(lambda: L.get([], {'a': 1, 'b': [1, 2, 3]}), {
              'a': 1, 'b': [1, 2, 3]}),
    assert_eq(lambda: L.set_u([], {}, {'a': 1, 'b': [1, 2, 3]}), {})
]


@pytest.mark.parametrize("run_test", custom_tests)
def test_custom(run_test):
    run_test()


when_tests = [
    testEq(
        lambda:
        L.get(
            L.when(lambda x: x > 2),
            1
        ),
        None
    ),
    testEq(lambda: L.get([L.when(lambda x: x > 2), I.always(2)], 1), None),
    testEq(
        lambda:
        L.get(
            L.when(lambda x: x > 2),
            3
        ),
        3
    ),
    testEq(lambda: L.collect(
        [L.elems, L.when(lambda x: x > 2)], [1, 3, 2, 4]), [3, 4]),
    testEq(
        lambda: L.modify([L.elems, L.when(lambda x: x > 2)],
                         R.negate, [1, 3, 2, 4]),
        [1, -3, 2, -4]
    )
]


@pytest.mark.parametrize("run_test", when_tests)
def test_when(run_test):
    run_test()


satisfying_tests = [
    testEq(lambda: L.collect(L.satisfying(R.is_(int)), [3, '1', 4, {'x': 1}]), [
        3,
        4,
        1,
    ])
]


@pytest.mark.parametrize("run_test", satisfying_tests)
def test_satisfying(run_test):
    run_test()


where_eq_tests = [
    testEq(
        lambda:
        L.collect(L.where_eq({'type': 'foo'}), [
            {'type': 'foo', 'children': [{'type': 'foo'}]},
            {'type': 'bar', 'children': [{'type': 'foo', 'value': 'bar'}]},
        ]),
        [
            {'type': 'foo', 'children': [{'type': 'foo'}]},
            {'type': 'foo', 'value': 'bar'},
        ]
    )
]


@pytest.mark.parametrize("run_test", where_eq_tests)
def test_where_eq(run_test):
    run_test()


leafs_tests = [
    testEq(lambda: L.collect(L.leafs, 101), [101]),
    testEq(lambda: L.collect(L.leafs, XYZ(1, 2, 3)), [XYZ(1, 2, 3)]),
    testEq(lambda: L.collect(L.leafs, [['x'], [1, [], {'y': 2}], [[False]]]), [
        'x',
        1,
        2,
        False,
    ]),
    testEq(lambda: L.set(L.leafs, 1, None), None),
    testEq(lambda: L.set(L.leafs, 1, 'defined'), 1),
    testEq(lambda: L.collect(L.leafs, [{
        'key': 3,
        'value': 'a',
        'lhs': {'key': 1, 'value': 'r'},
        'rhs': {'key': 2, 'value': 'd'}
    }, {'key': 4}]), [3, 'a', 1, 'r', 2, 'd', 4]),
    testEq(lambda: L.get(L.leafs, [{
        'key': 3,
        'value': 'a',
        'lhs': {'key': 1, 'value': 'r'},
        'rhs': {'key': 2, 'value': 'd'}
    }, {'key': 4}]), 3),
]


@pytest.mark.parametrize("run_test", leafs_tests)
def test_leafs(run_test):
    run_test()


# const flatten = [
#     L.optional,
#     L.lazy((rec)=>
#            L.cond(
#         [R.is(Array), [L.elems, rec]],
#         [R.is(Object), [L.values, rec]],
#         [L.identity]
#     )
#     ),
# ]

lazy_folds_tests = [
    testEq(lambda: L.get([L.elems, 'y'], [{'x': 1}, {'y': 2}, {'z': 3}]), 2),
    # testEq(lambda: L.get(flatten, [[[[[[[[[[101]]]]]]]]]]), 101),
    testEq(lambda: L.get(L.elems, []), None),
    testEq(lambda: L.get(L.values, {}), None),
    # testEq(
    #     lambda:
    #     L.getAs((x, i)=> (x > 3 ? [x + 2, i]: None), L.elems, [
    #         3,
    #         1,
    #         4,
    #         1,
    #         5,
    #     ]),
    #     [6, 2]
    # ),
    # testEq(
    #     lambda:
    #     L.getAs((x, i)=> (x > 3 ? [x + 2, i]: None), L.values, {
    #         a: 3,
    #         b: 1,
    #         c: 4,
    #         d: 1,
    #         e: 5,
    #     }),
    #     [6, 'c']
    # ),
    # testEq(lambda: L.getAs((_)= > {}, L.values, {x: 1}), None),
    # testEq(
    #     lambda:
    #     L.getAs(lambda x: (x < 9 ? None: [x]), flatten, [
    #         [[1], 2],
    #         {y: 3},
    #         [{l: 41, r: [5]}, {x: 6}],
    #     ]),
    #     [41]
    # ),
    testEq(lambda: L.any_(lambda x, i: x > i, L.elems, [0, 1, 3]), True),
    testEq(lambda: L.any_(lambda x, i: x > i, L.elems, [0, 1, 2]), False),
    testEq(lambda: L.all_(lambda x, i: x > i, L.elems, [1, 2, 3]), True),
    testEq(lambda: L.all_(lambda x, i: x > i, L.elems, [1, 2, 2]), False),
    # testEq(lambda: L.all1(lambda x, i: x > i, L.elems, [1, 2, 3]), True),
    # testEq(lambda: L.all1(lambda x, i: x > i, L.elems, []), False),
    # testEq(lambda: L.none(lambda x, i: x > i, L.elems, [0, 1, 3]), False),
    # testEq(lambda: L.none(lambda x, i: x > i, L.elems, [0, 1, 2]), True),
    testEq(lambda: L.and_(L.elems, []), True),
    # testEq(lambda: L.and1(L.elems, [1]), True),
    # testEq(lambda: L.and1(L.elems, [1, 0]), False),
    # testEq(lambda: L.and1(L.elems, []), False),
    testEq(lambda: L.or_(L.elems, []), False),
]


@pytest.mark.parametrize("run_test", lazy_folds_tests)
def test_lazy_folds(run_test):
    run_test()


props_tests = [
    testEq(lambda: L.get(L.props('x', 'y'), {
           'x': 1, 'y': 2, 'z': 3}), {'x': 1, 'y': 2}),
    testEq(lambda: L.get(L.props('x', 'y'), {'z': 3}), None),
    testEq(lambda: L.get(L.props('x', 'y'), {'x': 2, 'z': 3}), {'x': 2}),
    testEq(lambda: L.remove(L.props('x', 'y'), {
           'x': 1, 'y': 2, 'z': 3}), {'z': 3}),
    testEq(lambda: L.set(L.props('x', 'y'), {},
                         {'x': 1, 'y': 2, 'z': 3}), {'z': 3}),
    testEq(lambda: L.set(L.props('x', 'y'), {'y': 4}, {'x': 1, 'y': 2, 'z': 3}), {
        'y': 4,
        'z': 3,
    }),
    testEq(lambda: L.remove(L.props('x', 'y'), {'x': 1, 'y': 2}), {}),
    testEq(lambda: L.set(L.props('a', 'b'), {
           'a': 2}, {'a': 1, 'b': 3}), {'a': 2}),
    # testEq(lambda: I.keys(L.get(L.props('x', 'b', 'y'), {'b': 1, 'y': 1, 'x': 1})), [
    #     'x',
    #     'b',
    #     'y',
    # ]),
]


@pytest.mark.parametrize("run_test", props_tests)
def test_props(run_test):
    run_test()
