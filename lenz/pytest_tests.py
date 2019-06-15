import pytest
import lenz as L
import lenz.operations as R
import lenz.infestines as I

data = {'a': [1, 2, 3], 'b': {'c': [{'d': 1, 'e': 20}, {'d': 10}]}}

tests = [
    (lambda: L.get(['a', 0], data), 1),
    (lambda: L.get(['a', 0, lambda x: x+1], data), 2),
    (lambda: L.get([], data), data),
    (lambda: L.get(['a', 1], data), -1)
]


def test_get1():
    assert L.get(['a', 0], data) == 1


def test_get2():
    assert L.get(['a', 0])(data) == 1


def test_get3():
    assert L.get([], data) == data


def test_get_pick1():
    assert L.get(L.pick({
        'aa': ['a', 0],
        'bb': ['b', 'c', 0, 'e']
    }))(data) == {
        'aa': 1,
        'bb': 20,
    }


def test_get_pick2():
    assert L.get([L.pick({
        'aa': ['a', 0],
        'bb': ['b', 'c', 0, 'e']
    }), L.get([])])(data) == {
        'aa': 1,
        'bb': 20,
    }


def test_get_pick_modify1():
    assert L.get([L.pick({
        'aa': ['a', 0],
        'bb': ['b', 'c', 0, 'e']
    }), L.modify([], lambda x: x['aa'])])(data) == 1


def test_get_pick_modify1a():
    assert L.get([L.pick({
        'aa': ['a', 0],
        'bb': ['b', 'c', 0, 'e']
    }), L.modify([], lambda x, i: x['aa'])])(data) == 1


def test_get_pick_modify1b():
    assert L.get([L.pick({
        'aa': ['a', 0],
        'bb': ['b', 'c', 0, 'e']
    }), L.modify(['aa', []], lambda x, i: x+1)])(data) == {
        'aa': 2,
        'bb': 20,
    }


# def test_get():
#     assert L.get(['a', 0], data) == 1


# def test_b():
#     result = L.get(['a', 0, lambda x: x+1], data)
#     print(result)
#     assert result == 2


# def test_c(): assert L.get(['a', 0], data) == 1
def test_plain_funcs1():
    assert L.get(R.inc, 2) == 3


def test_plain_funcs1b():
    # want to ensure it works with arity 1
    assert L.get(lambda x, *args: x+1, 2) == 3


def test_plain_funcs1c():
    # want to ensure it works with arity 1
    # fails
    def foo(x, y):
        return x+1
    assert L.get(foo, 2) == 3


def test_plain_funcs2():
    assert L.modify(R.inc, R.negate, 1) == 1


def test_plain_funcs3():
    assert L.get(['x', lambda x, i: [x, i]], {'x': -1}) == [-1, 'x']


def test_plain_funcs4():
    assert L.collect([L.elems, lambda x, i: [x, i]], ['x', 'y']) == [
        ['x', 0],
        ['y', 1]
    ]


def test_plain_funcs5():
    assert L.collect([L.values, lambda x, i: [x, i]], {'x': 1, 'y': -1}) == [
        [1, 'x'],
        [-1, 'y']
    ]


def test_plain_funcs6():
    assert L.get([0, lambda x, i: [x, i]], [-1]) == [-1, 0]


def test_plain_funcs7():
    assert L.get([0, 'x', R.negate], [{'x': -1}]) == 1


def test_plain_funcs8():
    assert L.set([0, 'x', R.negate], 2, [{'x': -1}]) == [{'x': -1}]


def test_plain_funcs9():
    assert L.get(I.always('always'), 'anything') == 'always'


# def test_plain_funcs10():
#     assert L.set(I.always('always'), 'anything', 'original') == 'original'

def test_elems1():
    assert L.modify(L.elems, R.identity, [0, -1]) == [0, -1]


def test_elems2():
    # known to fail b/c elems and dict-like is broken
    assert L.modify(L.elems, R.identity, {'x': 1, 'y': 2}) == {'x': 1, 'y': 2}


def test_elems3():
    assert L.modify(L.elems, R.negate, []) == []


def test_values1():
    assert L.modify(L.values, R.identity, [1, 2]) == {'0': 1, '1': 2}
