import sys

sys.path.append("../")
import numpy as np
import scipy
from scipy.linalg import get_lapack_funcs
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import spsolve

from matrix_algorithms.k_differences import differences


class Difference_Matrix:
    """General class for creating difference matrices of unequally spaced and equally spaced data"""

    def __init__(self, n, k, t=None, prior: np.ndarray = None, style="lapack") -> None:

        self.n = n
        self.k = k
        self.l, self.u = k + 1, k + 1

        self.t = t
        self.prior = prior

        self.time_enabled = False
        self.prior_enabled = False

        if prior is not None or t is not None:
            self.style = "sparse"
        else:
            self.style = style

        # create the kth order difference matrix (sparse)
        D = self.compose_difference_matrix(n, k + 1).toarray()

        # if time increments are not provided, recurse through the difference matrix
        if t is not None:

            # returns the time matrix T
            D = self.construct_time_matrix(t)
            self.time_enabled = True

        if prior is not None:

            D = self.construct_prior_matrix(D, prior)
            self.prior_enabled = True

        self.D = D

        # create DDT
        DDT = D.dot(D.T)

        # save the DDT matrix
        self.DDT = DDT

        if self.style == "lapack":
            # determine the projected coefficients across diagonals
            DDT_diag_coeff = [DDT.diagonal(i)[0] for i in range(-k - 1, k + 2)]

            self.DDT_diag = np.array([i * np.ones(n - k - 1) for i in DDT_diag_coeff])

            self.DDT_to_invert = self.DDT_diag

        elif self.style == "sparse":

            self.DDT_to_invert = DDT

        # save the inverse of the DDT matrix as C Contigous array
        self.DDT_inv = np.asarray(self.invert(self.DDT_to_invert, style=self.style), order="C")

        # confirm this is in fact the inverse
        assert self.DDT.dot(self.DDT_inv).all() == np.eye(n - k - 1).all()

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

    def invert(self, diag, style):
        """
        Inverts the banded difference matrix

        Parameters
        ----------

        diag: Array
            Diagonals of the difference matrix

        style : str
            "lapack" or "sparse"

        Returns
        -------
        DDT_inv : Array
            Inverse of the difference matrix
        """
        if style == "lapack":
            DDT_inv = self.LU_decomposition(diag)

            return DDT_inv
        elif style == "sparse":
            DDT_inv = self.sparse_banded(diag)
            return DDT_inv
        else:

            return None

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

    def sparse_banded(self, diag):
        """
        Solves the system using scipy sparse banded matrix solver

        * Much better performance for irregularly spaced data
        """

        # convert to sparse matrix if not already
        if not isinstance(diag, scipy.sparse.csc.csc_matrix):
            diag = scipy.sparse.csc.csc_matrix(diag)

        inv = spsolve(diag, np.eye(self.n - self.k - 1))
        return inv

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

        # loop through up to kth order tf
        for k in range(0, self.k):

            # for kth order system
            D_1 = Difference_Matrix(self.n - k - 1, 0, t=None).D

            diff = np.array(differences(t, k=k + 1))
            scale = np.diag((k + 1) / diff)

            # recursively account for time increments
            D_k = D_1.dot(scale.dot(D_k))

        return D_k

    def construct_prior_matrix(self, D, prior):
        """Constructs prior matrix assuming univariate influences"""

        assert (
            len(prior) - 2 == D.shape[0]
        ), "Prior must be the same length as the number of rows in the difference matrix"

        # construct the prior matrix
        prior = np.diag(prior[1:-1])

        # construct the prior matrix
        D_prior = prior.dot(D)

        return D_prior
