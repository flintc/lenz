import logging
import sys
from lenz import handler
import lenz as L
import lenz.helpers as H

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def log_test(fn):
    def wrapper():
        try:
            result = fn()
            logger.info('PASSED: {}'.format(fn.__name__))
            return result
        except Exception as e:
            logger.critical(
                'FAILED: {}\n\t\t\t\tEXCEPTION: {}'.format(fn.__name__, e))
    return wrapper


def should_work():
    testA()
    testB()
    testC()
    testD()
    testE()
    testF()
    testG()
    testH()
    testI()
    testJ()
    testK()
    testM()

    @log_test
    def testN():
        data = [{'Description': 'A', 'Amount': 10, 'Date': '2018/01/01'},
                {'Description': 'B', 'Amount': 20, 'Date': '2018/01/02'}]
        expected = [{'description': 'A', 'amount': 10, 'date': '2018/01/01!'},
                    {'description': 'B', 'amount': 20, 'date': '2018/01/02!'}]
        operation = L.collect([
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
        ])
        result = operation(data)
        #print('result', result)
        assert(result == expected)
        return result
    testN()


@log_test
def testB():
    expected = [1]
    data = {'a': 1, 'b': [1, 2, 3]}
    optic = L.subseq_u(0, 1, ['b', L.elems])
    result = L.collect(optic, data)
    assert(result == expected)
    return result

# should work but doesn't technically cause of the combination of elems with modify_op, a hack is in place temporarily


@log_test
def testC():
    def add10(x):
        return x+10
    expected = {'a': 1, 'b': [11, 2, 3]}
    data = {'a': 1, 'b': [1, 2, 3]}
    optic = [L.subseq_u(0, 1, ['b', L.elems]), L.modify_op(add10)]
    result = L.transform(optic, data)
    assert(result == expected)
    return result


@log_test
def testD():
    def add10(x):
        return x+10
    expected = {'a': 1, 'b': {
        'b': {'b': {'b': {'b': {'b': [11, 12, 13]}}}}}}
    data = dict(a=1, b=dict(b=dict(b=dict(b=dict(b=dict(b=[1, 2, 3]))))))
    optic = ['b', 'b', 'b', 'b', 'b', 'b', L.elems, L.modify_op(add10)]
    result = L.transform(optic, data)
    logger.debug(result)
    assert(result == expected)
    return result


@log_test
def testE():
    def add10(x):
        return x+10
    expected = {'a': 1, 'b': {
        'b': {'b': {'b': {'b': {'b': [11, 12, 13]}}}}}}
    data = dict(a=1, b=dict(b=dict(b=dict(b=dict(b=dict(b=[1, 2, 3]))))))
    optic = ['b', 'b', 'b', 'b', 'b', 'b', L.elems]
    result = L.modify(optic, add10, data)
    assert(result == expected)
    return result
# should work but doesn't cause of the combination of elems with modify_op
#L.transform(['b', 2], data)


@log_test
def testF():
    def add10(x):
        return x+10
    data = {'a': 1, 'b': [1, 2, 3]}
    optic = ['b', L.elems, L.modify_op(add10)]
    result = L.transform(optic, data)
    assert(result == {'a': 1, 'b': [11, 12, 13]})
    return result


@log_test
def testG():
    def add(x):
        return lambda y: y+x
    data = ([10, 10])
    op = H.pipe([L.modify(0, add(1)), L.modify(1, add(-10))])
    result = op(data)
    assert(result == [11, 0])
    return result


@log_test
def testH():
    def modifier(x):
        return x+10
    inputs = {'a': {'b': {'c': [0, [90, 1, 2, 3, 4, 5, 6]]}}}
    expected = {'a': {'b': {'c': [0, [90, 1, 12, 13, 14, 5, 6]]}}}
    optic = ['a', 'b', 'c', 1, L.subseq_u(2, 5, L.elems)]
    result = L.modify(optic, modifier, inputs)
    assert(result == expected)
    return result


@log_test
def testI():
    expected = {'z': 2, 'q': 1, 'data': 3}
    data = {'a': 1, 'b': [1, 2, 3]}
    optic = L.pick({'z': ['b', 1], 'q': 'a', 'data': ['b', 2]})
    result = L.get(optic, data)
    assert(result == expected)
    return result
#L.transform(L.pick({'z': ['b', 1, L.modify_op(do_something)], 'q': 'a'}), data)


@log_test
def testJ():
    # should work b/c transform instead of modify
    def add10(x):
        return x+10
    data = {'a': 1, 'b': [1, 2, 3]}
    expected = {'a': 1, 'b': [1, 12, 3]}
    optic = L.pick({'z': ['b', 1, L.modify_op(add10)], 'q': 'a'})
    result = L.transform(optic, data)
    assert(result == expected)
    return result


@log_test
def testK():
    # point is, using L.modify with L.pick shouldn't have any effect
    def merge_with(x):
        return dict(**x, **dict(merge_data=10))
    data = {'a': 1, 'b': [1, 2, 3]}
    expected = {'a': 1, 'b': [1, 2, 3]}
    optic = L.pick({'z': ['b', 1], 'q': 'a'})
    result = L.modify(optic, merge_with, data)
    assert(result == expected)
    return result


def known_exceptions():
    testL()


@log_test
def testA():
    expected = {'a': 1, 'b': [1, 2, 3]}
    data = {'a': 1, 'b': [1, 2, 3]}
    result = L.get([], data)
    assert(result == expected)
    return result


@log_test
def testL():
    # shouldn't do anything, but should fail silently?
    def add10(x):
        return x+10
    data = {'a': 1, 'b': [1, 2, 3]}
    expected = {'a': 1, 'b': [1, 2, 3]}
    optic = L.pick({'z': ['b', 1, L.modify_op(add10)], 'q': 'a'})
    result = L.modify(optic, add10, data)
    assert(result == expected)
    return result


@log_test
def testM():
    data = {'a': 1, 'b': [1, 2, 3]}
    result = L.set_u([], {}, data)
    return result


if __name__ == '__main__':
    logger.info('THE FOLLOWING SHOULD ALL PASS')
    should_work()
    logger.info('THE FOLLING ARE KNOWN TO FAIL')
    known_exceptions()
