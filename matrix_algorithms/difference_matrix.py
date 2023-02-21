import sys

sys.path.append("../")
import numpy as np
from scipy.linalg import get_lapack_funcs
from scipy.sparse import dia_matrix

from matrix_algorithms.k_differences import differences
from matrix_algorithms.matrix_sequence import Matrix_Sequence


class Difference_Matrix:
    """General class for creating difference matrices of unequally spaced and equally spaced data"""

    def __init__(self, n, k, t: np.ndarray = None, prior: np.ndarray = None) -> None:

        self.n = n
        self.k = k
        self.l, self.u = k + 1, k + 1

        self.t = t
        self.prior = prior

        self.time_enabled = False
        self.prior_enabled = False

        # sequence of matrices used to create D
        self.sequence = Matrix_Sequence()

        # time not provided construct difference matrix
        if t is None:

            D = self.compose_difference_matrix(n, k + 1).toarray()

            self.sequence.add_matrix(np.eye(n - k - 1))
            self.sequence.add_matrix(D)

        # if time increments are  provided, recurse through the difference matrix
        else:

            # returns the time matrix T
            D, mat_to_append = self.construct_time_matrix(t)

            for mat in mat_to_append:

                self.sequence.add_matrix(mat)

            self.time_enabled = True

        # account for conditional prior
        if prior is not None:

            D, mat_to_append = self.construct_prior_matrix(D, prior)

            for mat in mat_to_append:
                self.sequence.add_matrix_left(mat)

            self.prior_enabled = True

        self.D = D

        # create DDT
        DDT = D.dot(D.T)

        # save the DDT matrix
        self.DDT = DDT

        # save the transpose sequence of D
        self.sequence_transpose = self.sequence.compute_transpose()

        # save the full sequence of D
        self.DDT_sequence = self.sequence_transpose.get_sequence() + self.sequence_transpose.get_sequence()

        # save the composite sequence of D
        self.composite_sequence = self.sequence.compute_matrix().dot(self.sequence_transpose.compute_matrix())

        assert np.allclose(self.composite_sequence, DDT)

        condition_number = np.linalg.cond(D)

        if condition_number > 1e8:
            print(" WARNING Condition number is large: {}".format(condition_number))

    def compose_difference_matrix(self, n, k):
        """Extracts the kth difference matrix for any n-size array using pascal's triangle"""

        def pascals(k):
            pas = [0, 1, 0]
            counter = k
            while counter > 0:
                pas.insert(0, 0)
                pas = [np.sum(pas[i : i + 2]) for i in range(0, len(pas))]
                counter -= 1
            return pas

        coeff = pascals(k)
        coeff = [i for i in coeff if i != 0]
        coeff = [coeff[i] if i % 2 == 0 else -coeff[i] for i in range(0, len(coeff))]

        if k == 0:
            D = dia_matrix((np.ones(n), 0), shape=(n - k, n))
        elif k == 1:
            D = dia_matrix((np.vstack([i * np.ones(n) for i in [-1, 1]]), range(0, k + 1)), shape=(n - k, n))
        else:
            D = dia_matrix((np.vstack([i * np.ones(n) for i in coeff]), range(0, k + 1)), shape=(n - k, n))

        return D

    def LU_decomposition(self, diag, b=None):
        """
        LU decomposition specifically for banded matrices using LAPACK routine gbsv

        Inspiration taken from scipy solve_banded function

        Parameters
        ----------
        D : Array
            Difference matrix

        Returns
        -------
        L : Array
            Lower triangular matrix
        U : Array
            Upper triangular matrix
        """

        # setup identity matrix
        if b is None:
            b = np.eye(self.n - self.k - 1)
        else:
            b = np.asarray(b)

        # number of lower and upper diagonals
        (nlower, nupper) = self.l, self.u

        # get lapack function
        (gbsv,) = get_lapack_funcs(("gbsv",), (diag, b))

        # setup problem
        a2 = np.zeros((2 * nlower + nupper + 1, self.n - self.k - 1), dtype=gbsv.dtype)
        a2[nlower:, :] = diag
        lu, piv, x, info = gbsv(nlower, nupper, a2, b, overwrite_ab=True, overwrite_b=True)

        return x

    def compute_k_difference(self, k: int):
        """
        Computes the kth order difference matrix

        Parameters
        ----------
        k : int
            Order of difference

        Returns
        -------
        D : Array
            Difference matrix
        """

        # if the order is the same as the original, return the original saving computation time
        if k == self.k:
            return self.D

        # else compute the difference matrix
        else:
            D = self.compose_difference_matrix(self.n, k + 1)
            return D.toarray()

    def construct_time_matrix(self, t):
        """Accounts for unequal time increments via recursion by Tibshirani"""

        # initialize D_k to be D_1
        D_k = Difference_Matrix(self.n, 0, t=None).D

        mat_to_append = []

        mat_to_append.append(np.eye(self.n))
        mat_to_append.append(D_k)

        # loop through up to kth order tf
        for k in range(0, self.k):

            # for kth order system
            D_1 = Difference_Matrix(self.n - k - 1, 0, t=None).D

            diff = np.array(differences(t, k=k + 1))
            scale = np.diag((k + 1) / diff)
            # recursively account for time increments
            D_k = D_1.dot(scale.dot(D_k))

            mat_to_append.append(scale)
            mat_to_append.append(D_1)

        return D_k, reversed(mat_to_append)

    def construct_prior_matrix(self, D, prior):
        """Constructs prior matrix assuming univariate influences"""

        mat_to_append = []

        # construct the prior matrix from the prior vector
        prior = np.diag(prior[: self.n - self.k - 1])

        # construct the prior matrix
        D_prior = prior.dot(D)

        mat_to_append.append(prior)

        return D_prior, mat_to_append
