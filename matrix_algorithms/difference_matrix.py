import numpy as np
import pentapy as pp
from scipy.linalg import get_lapack_funcs
from scipy.sparse import dia_matrix, spdiags
from scipy.sparse.linalg import spsolve


class Second_Order_Difference_Matrix:
    def __init__(self, n, style=None) -> None:


        self.n = n
        self.k=2


        self.style = style if style is not None else "lapack"

        # number of upper and lower diagonals of DDT
        self.l = 2
        self.u = 2

        # create the naive second order difference matrix (sparse)
        D = self.extract_second_order_difference_matrix(n)

        # save the difference matrix
        self.D = D.toarray()

        # create DDT
        DDT = D.dot(D.T)

        # determine the projected coefficients across diagonals
        DDT_diag_coeff = [DDT.diagonal(i)[0] for i in range(-2, 2 + 1)]

        self.DDT_diag = np.array([i * np.ones(n - 2) for i in DDT_diag_coeff])

        # save the DDT matrix
        self.DDT = DDT.toarray()

        # save the inverse of the DDT matrix as C Contigous array
        self.DDT_inv = np.asarray(self.invert(style=self.style), order="C")

    def invert(self, style):
        """
        Inverts the difference matrix

        Parameters
        ----------
        style : str
            "lapack" or "pentapy" or "sparse"

        Returns
        -------
        DDT_inv : Array
            Inverse of the difference matrix
        """
        if style == "lapack":
            DDT_inv = self.LU_decomposition()

            return DDT_inv
        elif style == "pentapy":
            DDT_inv = self.ptrans_algorithm()
            return DDT_inv
        elif style == "sparse":
            DDT_inv = self.sparse_banded()
            return DDT_inv
        else:

            return None

    def LU_decomposition(self, b=None):
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
            b = np.eye(self.n - 2)
        else:
            b = np.asarray(b)

        # number of lower and upper diagonals
        (nlower, nupper) = self.l, self.u

        # get lapack function
        (gbsv,) = get_lapack_funcs(("gbsv",), (self.DDT_diag, b))

        # setup problem
        a2 = np.zeros((2 * nlower + nupper + 1, self.n - 2), dtype=gbsv.dtype)
        a2[nlower:, :] = self.DDT_diag
        lu, piv, x, info = gbsv(nlower, nupper, a2, b, overwrite_ab=True, overwrite_b=True)

        return x

    def sparse_banded(self):
        """
        Solves the system using scipy sparse banded matrix solver

        Returns
        -------
        inv : Array
            Inverse of the difference matrix
        """

        # confirm this works
        sparse_mat = dia_matrix((self.DDT_diag, range(-self.k, self.k + 1)), shape=(self.n - 2, self.n - 2))
        inv = spsolve(sparse_mat, np.eye(self.n - 2))
        return inv

    def ptrans_algorithm(self):
        """Solves pentadiagonal system using pentapy package"""

        inv = np.zeros((self.n - 2, self.n - 2))
        for i in range(0, self.n - 2):
            unit = np.zeros(self.n - 2)
            unit[i] = 1
            inv[i] = pp.solve(self.DDT, unit, is_flat=False)
        return inv

    def extract_second_order_difference_matrix(self, n):
        """ Extracts the first difference matrix for any n-size array"""

        # create the diagonals of the difference matrix
        diagonals = np.array([np.ones(n ), -np.ones(n )])

        # create the difference matrix
        D = spdiags(diagonals, [0, -1], n , n)

        D=D.dot(D).T
        D=D[:-2,:]

        return D

def test_lapack(n=100):
    """Test the LU decomposition method"""
    k = 2
    print("Testing LU decomposition method")
    D = Second_Order_Difference_Matrix(n, k, style="lapack")

    DDT = D.DDT
    DDT_inv = D.DDT_inv

    DDT_rank = np.linalg.matrix_rank(DDT)

    print(f"Rank of DDT matrix is {DDT_rank} out of {n-2}.")

    # check that the inverse is correct
    assert np.allclose(DDT.dot(DDT_inv), np.eye(n - 2), rtol=1e-8, atol=1e-8)

    return


def test_sparse(n=100):
    """Test the sparse method"""
    k = 2
    print("Testing sparse method")
    D = Second_Order_Difference_Matrix(n, k, style="sparse")

    DDT = D.DDT
    DDT_inv = D.DDT_inv

    DDT_rank = np.linalg.matrix_rank(DDT)

    print(f"Rank of DDT matrix is {DDT_rank} out of {n-2}.")

    # check that the inverse is correct
    assert np.allclose(DDT.dot(DDT_inv), np.eye(n - 2), rtol=1e-8, atol=1e-8)

    return


def test_pentapy(n=100):
    """Test the pentapy method"""
    k = 2
    print("Testing pentapy method")
    D = Second_Order_Difference_Matrix(n, k, style="pentapy")

    DDT = D.DDT
    DDT_inv = D.DDT_inv

    DDT_rank = np.linalg.matrix_rank(DDT)

    print(f"Rank of DDT matrix is {DDT_rank} out of {n-2}.")

    # check that the inverse is correct
    assert np.allclose(DDT.dot(DDT_inv), np.eye(n - 2), rtol=1e-8, atol=1e-8)

    return
