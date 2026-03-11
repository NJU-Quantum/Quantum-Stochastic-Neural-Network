import itertools as it

import qutip as qt
import numpy as np

import miscellany as mcl

######################################################
# Constants
SIGMA_OPERATORS_LIST = [
    qt.sigmax(),
    qt.sigmay(),
    qt.sigmaz()
]

EXTENDED_SIGMA_OPERATORS_LIST = [
    qt.identity(2),
    qt.sigmax(),
    qt.sigmay(),
    qt.sigmaz()
]

######################################################
# States


def qubits_base(num_qubits, base_index=0):
    dim = 2**num_qubits
    data = np.array([1], dtype=complex)
    ind = np.array([0], dtype=np.int32)
    ptr = np.array([0]*(base_index+1)+[1]*(dim - base_index), dtype=np.int32)
    return qt.Qobj(
        qt.fastsparse.fast_csr_matrix(
            (data, ind, ptr), shape=(dim, 1)
        ),
        dims=[[2]*num_qubits, [1]*num_qubits],
        isherm=False
    )

######################################################
# Operators


def operator_base(dims, row_index=0, col_index=0):
    dim_row, dim_col = map(mcl.prod, dims)
    data = np.array([1], dtype=complex)
    ind = np.array([col_index], dtype=np.int32)
    ptr = np.array([0]*(1 + row_index)+[1] *
                   (dim_row - row_index), dtype=np.int32)
    return qt.Qobj(
        qt.fastsparse.fast_csr_matrix(
            (data, ind, ptr),
            shape=(dim_row, dim_col)
        ),
        isherm=(dim_row == dim_col) and (row_index == col_index),
        dims=dims
    )


def qft(dim=1):
    phase = 2.0j * np.pi / dim
    arr = np.arange(dim)
    L, M = np.meshgrid(arr, arr)
    L = phase * (L * M)
    L = np.exp(L)
    qft_op_dims = [[dim]]*2
    return qt.Qobj(1.0 / np.sqrt(dim) * L, dims=qft_op_dims)


def qft_on_qubits(num_qubits=1):
    dim = 2 ** num_qubits
    phase = 2.0j * np.pi / dim
    arr = np.arange(dim)
    L, M = np.meshgrid(arr, arr)
    L = phase * (L * M)
    L = np.exp(L)
    qft_op_dims = [[2] * num_qubits, [2] * num_qubits]
    return qt.Qobj(1.0 / np.sqrt(dim) * L, dims=qft_op_dims)


def circular_permute_qubits_basis(num_qubits, offset=1):
    dim = 2**num_qubits
    return sum(
        qubits_base(num_qubits, (base_index+offset) % dim)
        * qubits_base(num_qubits, base_index).dag()
        for base_index in range(dim)
    )


def circular_permute_basis(dim, offset=1):
    return sum(
        qt.basis(dim, (base_index + offset) % dim)
        * qt.basis(dim, base_index).dag()
        for base_index in range(dim)
    )


def random_permute_qubits_basis(num_qubits):
    def shuffled(lst):
        from random import shuffle
        shuffle(lst)
        return lst
    dim = 2**num_qubits
    return sum(
        qubits_base(num_qubits, index_base_ket)
        * qubits_base(num_qubits, index_base_to_bra).dag()
        for index_base_ket, index_base_to_bra in zip(
            shuffled(list(range(dim))),
            range(dim)
        )
    )


def permute_qubits_basis(num_qubits, perm_method='circular', offset=1):
    if perm_method == 'circular':
        return circular_permute_qubits_basis(num_qubits, offset)
    elif perm_method == 'random':
        return random_permute_qubits_basis(num_qubits)
    else:
        raise ValueError('method for permute not recognized')


def flip_all_qubits(num_qubits=1):
    return qt.tensor([qt.sigmax()]*num_qubits)


def qubit_hermitian(weights):
    return sum(
        weight*sigma_op
        for weight, sigma_op in zip(weights, EXTENDED_SIGMA_OPERATORS_LIST)
    )

######################################################
# Functions


def distance(super_op_1, super_op_2):
    return (super_op_1 - super_op_2).norm()


def inverse_Qobj(Q_obj):
    '''there is now a same method in QuTiP'''
    return qt.Qobj(
        Q_obj.data.todense().I,
        dims=Q_obj.dims
    )


def sup_on_op(super_op, op):
    return qt.vector_to_operator(
        super_op * qt.operator_to_vector(op)
    )

######################################################
# Iterators


def iter_qubits_base(num_qubits):
    return map(
        qubits_base,
        it.repeat(num_qubits),
        range(2**num_qubits)
    )


# TO DO: faster implement without using iter_operator_base
def iter_qubits_operator_base(num_qubits):
    return iter_operator_base([[2]*num_qubits]*2)


def iter_operator_base(dims):
    return map(
        lambda indices: operator_base(dims, *indices),
        it.product(*map(
            lambda dims_hilbert: range(mcl.prod(dims_hilbert)),
            dims
        ))
    )

######################################################
# Lists


def qubits_basis(num_qubits):
    return [qubits_base(num_qubits, base_index) for base_index in range(2**num_qubits)]


# TO DO: faster implement without using operator_basis
def qubits_operator_basis(num_qubits):
    return operator_basis([[2]*num_qubits]*2)


def operator_basis(dims):
    return [
        operator_base(dims, *indices)
        for indices in it.product(*map(
            lambda dh: range(mcl.prod(dh)),
            dims
        ))
    ]

######################################################
# Others


######################################################
# Tests
if __name__ == "__main__":
    '''
    qubits_base
    '''
    # print(qubits_base(3, 2))

    '''
    operator_base,
    qubit_hermitian
    '''
    # print(operator_base([[2,2],[3,3]],3,7))
    # print(qubit_hermitian(np.array([1,2,0,0])))

    '''
    distance,
    inverse_Qobj
    '''
    # print(distance(
    #     qt.to_super(qt.sigmax()),
    #     qt.to_super(qt.sigmax())
    # ))
    # print(distance(
    #     qt.to_super(qt.sigmay()),
    #     qt.to_super(qt.sigmaz())
    # ))
    # U = qt.rand_unitary(2)
    # print(inverse_Qobj(
    #     U
    # ) * U)
    # print(inverse_Qobj(
    #     qt.to_super(U)
    # ) * qt.to_super(U))

    '''
    iter_qubits_base,
    iter_qubits_operator_base,
    iter_operator_base
    '''
    # for base in iter_qubits_base(2): print(base)
    # for base in iter_qubits_operator_base(2): print(base)
    # for base in iter_operator_base(((1,2),(3,1))): print(base)

    '''
    qubits_basis, qubits_operator_basis, operator_basis
    '''
    # print(qubits_basis(2))
    # print(qubits_operator_basis(2))
    # print(operator_basis(((1,2),(3,1))))
    pass
