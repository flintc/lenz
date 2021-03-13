import lenz as L
import operations as R


def testN():
    data = [{'Description': 'A', 'Amount': 10, 'Date': '2018/01/01'},
            {'Description': 'B', 'Amount': 20, 'Date': '2018/01/02'}]
    expected = [{'description': 'A', 'amount': 10, 'date': '2018/01/01!'},
                {'description': 'B', 'amount': 20, 'date': '2018/01/02!'}]
    operation = L.collect([
        L.elems,
        L.get([L.pick({
            'description': 'Description',
            'amount': 'Amount',
            'date': 'Date',
        })]),
        L.modify('date', lambda x: x+'!'),
        #     L.set('account',
        #           'checking',
        #     })),
    ])
    result = operation(data)
    # print('result', result)
    assert(result == expected)
    return result


# testN()

# result = L.modify(['xs', L.elems, 'x'], lambda x: (None if x < 2 else x), {
#     'xs': [{'x': 3}, {'x': 1}, {'x': 4}, {'x': 1, 'y': 0}, {'x': 5}, {'x': 9}, {'x': 2}]
# })
# print(result)
# result = L.modify(L.elems, R.inc, {
#     'x': 1, 'y': 2})
# print(result)

#print(L.modify('x', lambda x: x + 1, {'x': 1}))

#print(L.get([L.elems, lambda x, i: x], [1, 2, 3]))
import pandas as pd
df = pd.DataFrame(dict(a=[1, 2, 3], b=[4, 5, 6]))
L.helpers.DictLike.register(pd.DataFrame)
# print(L.get(L.pick({'aa': 'a', 'bb': 'b'}), df))
