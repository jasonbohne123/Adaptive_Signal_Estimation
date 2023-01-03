import time

import numpy as np
from difference_matrix import Difference_Matrix


def sherman_morrison_recursion(a_ij, DDT_inv):
    """Compute the inverse of a matrix using the Sherman-Morrison formula

    Utilizes forward recrusion for computational efficiency
    """
    start = time.time()
    A_inv = DDT_inv
    k = 1

    # Loop over the columns of the matrix
    while k <= len(a_ij):

        # Create the vectors u and v which are scaled unit vectors
        e_n = np.zeros(len(a_ij))
        e_n[k - 1] = 1
        u = e_n.reshape(-1, 1)
        v = e_n.reshape(1, -1)

        num = A_inv.dot(a_ij[k - 1] * u.dot(v)).dot(A_inv)
        den = 1 + a_ij[k - 1] * v.dot(A_inv).dot(u)

        A_inv = A_inv - num / den

        k = k + 1
    end = time.time()
    total_time = end - start
    return A_inv, total_time


def test_sherman_morrison(n):
    diff = Difference_Matrix(n, 2)

    DDT = diff.DDT
    DDT_inv = diff.DDT_inv

    a_ij = np.random.rand(DDT.shape[0])

    # compute the inverse of DDT+a_ij*I using the Sherman-Morrison formula
    A_inv, total_time = sherman_morrison_recursion(a_ij, DDT_inv)

    # check that the inverse is correct
    computed_inv = A_inv.dot(DDT + np.diag(a_ij))

    print(f"Max error: {np.max(np.abs(computed_inv-np.eye(DDT.shape[0])))}")
    print(f"Total time: {total_time}")

    return
