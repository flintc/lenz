
import lenz.infestines as I
from copy import deepcopy

from lenz.helpers import is_dict_like, is_list_like, always, arityn

ENV = I.ENV


def id(x):
    return x


def get_prop(k, o):
    if is_dict_like(o):
        return o[k]
    return None


def set_prop(k, v, o):
    on = deepcopy(o)
    on[k] = v
    return on


def get_index(i, xs):
    if is_list_like(xs):
        return xs[i]
    return None


def dev_copy_name(to, from_):
    return I.define_name_u(to, from_.name)


set_name = id if ENV == 'production' else I._define_name_u
copy_name = id if ENV == 'production' else dev_copy_name


def from_reader(wi2x):
    def wrapper(w, i, F, xi2yF):
        return F.map(I.always(w), xi2yF(wi2x(w, i), i))
    return copy_name(wrapper, wi2x)


def get_as_u(xi2y, l, s):
    if isinstance(l, str):
        return xi2y(get_prop(l, s), l)
    if isinstance(l, int):
        return xi2y(get_index(l, s), l)
    if is_list_like(l):
        n = getattr(l, I.LENGTH)
