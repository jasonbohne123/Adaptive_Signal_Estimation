import sys
import time

import numpy as np

path = "/home/jbohne/sbu/research/"
sys.path.append(f"{path}Adaptive_Signal_Estimation_Private/matrix_algorithms/")

from difference_matrix import Difference_Matrix
from sherman_morrison import sherman_morrison_recursion


def woodbury_matrix_inversion(a_ij, DDT_inv, step=20, tests=False):
    """Compute the inverse of a matrix using the Woodbury formula

    k-blocks of the matrix are inverted at a time
    """
    start = time.time()
    A_inv = DDT_inv
    k = 1

    # Loop over the columns of the matrix
    while k <= len(a_ij):

        len_block = min(len(a_ij[k - 1 : k - 1 + step]), step)

        # Create the vectors u and v which are scaled unit vectors
        u = np.zeros((len(a_ij), len_block))
        v = np.zeros((len_block, len(a_ij)))
        c = np.zeros((len_block, len_block))

        for i in range(0, len_block):

            u[k + i - 1, i] = 1
            v[i, k + i - 1] = 1
            c[i, i] = a_ij[k + i - 1]

        #  extract kth block of A_inv
        truncated_mat = v.dot(A_inv).dot(u)

        # compute the inverse of the kth block of A_inv
        inv_truncated_mat = np.linalg.inv(truncated_mat)

        if tests:
            # check that the inverse using numpy is correct
            assert np.max(np.abs(truncated_mat.dot(inv_truncated_mat) - np.eye(len_block))) < 1e-10

        c_a_inv, tot_time = sherman_morrison_recursion(1 / a_ij[k - 1 : k - 1 + step], inv_truncated_mat)

        A_inv = A_inv - A_inv.dot(u).dot(c_a_inv.dot(v).dot(A_inv))

        k = k + step
    end = time.time()
    total_time = end - start
    return A_inv, total_time


def test_woodbury_inversion(n, step=None):

    diff = Difference_Matrix(n, 2)

    DDT = diff.DDT
    DDT_inv = diff.DDT_inv

    a_ij = np.random.rand(DDT.shape[0])

    if step is None:
        # compute the inverse of DDT+a_ij*I using the Woodbury formula
        A_inv, total_time = woodbury_matrix_inversion(a_ij, DDT_inv)
    else:
        # compute the inverse of DDT+a_ij*I using the Woodbury formula
        A_inv, total_time = woodbury_matrix_inversion(a_ij, DDT_inv, step=step)

    # check that the inverse is correct
    computed_inv = A_inv.dot(DDT + np.diag(a_ij))

    print(f"Max error: {np.max(np.abs(computed_inv-np.eye(DDT.shape[0])))}")
    print(f"Total time: {total_time}")

    return
