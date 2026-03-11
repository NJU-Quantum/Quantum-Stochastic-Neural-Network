from functools import reduce
from operator import mul
from collections.abc import Iterable
from time import time

import numpy as np
from scipy.interpolate import interp1d


class NoValue(object):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance


NO_VALUE = NoValue()


class TextedTimer(object):
    def __init__(self):
        self._most_recent_called = time()

    def __call__(self, *text, **_kwargs):
        called_at = time()
        print(*text, ':\n\t',
              called_at - self._most_recent_called
              )
        self._most_recent_called = called_at


TEXT_TIME = TextedTimer()


def prod(iterable, initial_value=1, reversed_order=False):
    """
    Product of elements from an iterable object. 

    Args:
        reversed_order: a bool
            If it is False, which is the default case, the order of the product is that early elements are to the left of late elements. If it is True, the order is then reversed.

    Notes:
        There is now a build-in function of this in Python 3.8 when reversed_order == False.
    """
    return prod_reversed_order(iterable, initial_value) if reversed_order else prod_normal_order(iterable, initial_value)


def prod_normal_order(iterable, initial_value=1):
    return reduce(mul, iterable, initial_value)


def prod_reversed_order(iterable, initial_value=1):
    return reduce(
        lambda value, element: mul(element, value),
        iterable,
        initial_value
    )


def binary_repr(int_to_repr, length_of_bits):
    """
    Convert an integer to its binary representation.

    Return:
        s: a string
    """
    return format(
        int_to_repr,
        '0'+str(length_of_bits)+'b'
    )


def complex_interp1d(dom_known, cod, **kwarg_for_interpolator):
    """
    Interpolate complex data on a 1-D domain.
    """
    cod_real, cod_imag = np.real(cod), np.imag(cod)
    return lambda dom: (
        interp1d(dom_known, cod_real, **kwarg_for_interpolator)(dom)
        + 1j * interp1d(dom_known, cod_imag, **kwarg_for_interpolator)(dom)
    )


def flatten(iterable, ignore_types=(str, bytes)):
    """
    Flatten an iterable.
    """
    for x in iterable:
        if isinstance(x, Iterable) and not isinstance(x, ignore_types):  # 将字符串和字节排除在可迭代对象之外
            yield from flatten(x)
        else:
            yield x
'''
# 应用实例
items = [1, 2, [3, 4, [5, 6], 7], 8]
for x in flatten(items):
    print(x)
# output：
 1 2 3 4 5 6 7 8
'''

def getvalue(nested_list, indices_tuple):
    """
    Get the value of an element of a nested list by specifying the index of that element with a tuple.

    Args:
        nested_list: a list
        indices_tuple: a tuple of ints

    Return:
        value: (to check whether a view or an original object is returned)
    """
    return reduce(
        lambda result, index: result[index],
        indices_tuple,
        nested_list
    )


def setitem(nested_list, indices_tuple, new_value):
    """
    Assign a new value to an element of a nested list by specifying the index of that element with a tuple.

    Args:
        nested_list: a list
        indices_tuple: a tuple of ints
        new_value: any thing

    Notes:
        The original list nested_list will be modified.
    """
    if len(indices_tuple) == 1:
        nested_list[indices_tuple[-1]] = new_value
    else:
        setitem(nested_list[indices_tuple[0]], indices_tuple[1:], new_value)


def import_file(module_name, path):
    """
    Import a module by specifying a path to a script file.

    Args:
        module_name: a string
        path: a string

    Return:
        mod: a module
    """
    from importlib import util
    spec = util.spec_from_file_location(module_name, path)
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def perm_struct(array, structure, new_structured_order):
    """
    Change the structure of an array without modifying its data.

    Args:
        array: array_like
        structure: a list of lists of ints
            prod(structure[i]) should == array.shape[i]
        new_structured_order: a list of lists of ints
            Every int from 0 to len(sum(structure,[]))-1 should appear exactly once.
            len(sum(new_structured_order,[])) should == len(sum(structure,[]))

    Return:
        new_array: array_like
            This will be a new view object if possible; otherwise, it will be a copy, which depends on the new_structured_order.

    Notes:
        None of the above conditions will be checked in this function.

    Examples:
    >>> a = np.arange(48).reshape((3,4,4))
    >>> struct = [[3],[4,1],[2,2]]
    >>> new_order = [[2,3,4],[1,0]]
    >>> perm_struct(a, struct, new_order)
    array([[ 0, 16, 32,  4, 20, 36,  8, 24, 40, 12, 28, 44],
           [ 1, 17, 33,  5, 21, 37,  9, 25, 41, 13, 29, 45],
           [ 2, 18, 34,  6, 22, 38, 10, 26, 42, 14, 30, 46],
           [ 3, 19, 35,  7, 23, 39, 11, 27, 43, 15, 31, 47]])
    """
    flattened_structure = sum(structure, [])
    new_struct = [
        [flattened_structure[index] for index in piece]
        for piece in new_structured_order
    ]
    return np.transpose(
        array.reshape(flattened_structure),
        axes=sum(new_structured_order, [])
    ).reshape(tuple(map(prod, new_struct)))


######################################################
# Test
if __name__ == '__main__':
    '''
    getvalue, setitem
    '''
    # shape = (2,3,6,4,2)
    # indices = (0,0,0,0,0)
    # nested_list = np.array(range(prod(shape))).reshape(shape).tolist()
    # print('value before: ', getvalue(nested_list, indices))
    # setitem(nested_list, indices, 1)
    # print('value after: ', getvalue(nested_list, indices))
    '''
    complex_interp1d
    '''
    # rl = np.array(range(12)).reshape((3, 2, 2))
    # im = np.array(list(reversed(list(range(12))))).reshape((3, 2, 2))
    # z = rl + 1j * im
    # t = np.array(range(2))
    # print('z:\n', z)
    # print('t:\n', t)
    # z_t = complex_interp1d(t, z, axis=1)
    # print('z(0.2):\n', z_t(0.2))
    '''
    perm_struct
    '''
    # structure = [[2, 1], [3, 4], [2, 2]]
    # new_structured_order = [[3, 5, 2], [0, 1, 4]]
    # array = np.arange(prod(sum(structure, []))).reshape(sum(structure, []))
    # new_array = perm_struct(array, structure, new_structured_order)
    # print(new_array)

    # structure = [[2, 1, 3, 4, 2, 2], ]
    # new_structured_order = [[3, 5, 2], [0, 1, 4]]
    # array = np.arange(prod(sum(structure, []))).reshape(sum(structure, []))
    # new_array = perm_struct(array, structure, new_structured_order)
    # print(new_array)
    pass
