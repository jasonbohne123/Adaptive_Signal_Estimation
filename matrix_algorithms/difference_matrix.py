

import pentapy as pp
import numpy as np
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import get_lapack_funcs  



class Difference_Matrix():
    def __init__(self,n,k,style=None) -> None:
        

        self.n=n
        self.k=k
        self.style=style if style is not None else "lapack"

        # number of upper and lower diagonals of DDT
        self.l=k
        self.u=k

        # create the difference matrix
        diff_diag=self.extract_matrix_diagonals(self.n,self.k)
        self.D=dia_matrix((diff_diag, np.arange(-(k-1), self.k+1)), shape=(self.n, self.n-2)).toarray()

        # extract only the diagonals of the DDT matrix (which is pascals(k+2))
        self.diag=self.extract_matrix_diagonals(self.n,self.k+2)

        # save the DDT matrix 
        self.DDT=dia_matrix((self.diag, np.arange(-self.l, self.u + 1)), shape=(self.n, self.n)).toarray()
        
        # save the inverse of the DDT matrix
        self.DDT_inv=self.invert(style=self.style)
      
    
    
    def invert(self,style):
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
        if style=="lapack":
            DDT_inv=self.LU_decomposition()
            return DDT_inv
        elif style=="pentapy":
            DDT_inv=self.ptrans_algorithm()
            return DDT_inv
        elif style=="sparse":
            DDT_inv=self.sparse_banded()
            return DDT_inv
        else:
            
            return None

    def LU_decomposition(self,b=None):
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
            b = np.eye(self.n)
        else:
            b = np.asarray(b)

        (nlower, nupper) = self.l,self.u # number of lower and upper diagonals

        # get lapack function
        gbsv, = get_lapack_funcs(('gbsv',), (self.diag, b))

        # setup problem
        a2 = np.zeros((2*nlower + nupper + 1, self.n), dtype=gbsv.dtype)
        a2[nlower:, :] = self.diag
        lu, piv, x, info = gbsv(nlower, nupper, a2, b, overwrite_ab=True,
                                overwrite_b=True)

        return x

    def sparse_banded(self):
        """
        Solves the system using scipy sparse banded matrix solver

        Returns
        -------
        inv : Array
            Inverse of the difference matrix
        """
        inv=spsolve(self.DDT,np.eye(self.n))
        return inv


    def ptrans_algorithm(self):
        """Solves pentadiagonal system using pentapy package"""

        inv=np.zeros((self.n,self.n))
        for i in range(0,self.n):
            unit=np.zeros(self.n)
            unit[i]=1
            inv[i] =pp.solve(self.DDT, unit, is_flat=False)
        return inv


    def extract_matrix_diagonals(self,n,k):
        """
        Extracts only the diagonals of the difference matrix
        """
        def pascals(k):
            pas = [0, 1, 0]
            counter = k
            while counter > 0:
                pas.insert(0, 0)
                pas = [np.sum(pas[i: i + 2]) for i in range(0, len(pas))]
                counter -= 1
            return pas

        coeff = pascals(k)
        coeff = [i for i in coeff if i != 0]
        coeff = [coeff[i] if i % 2 == 0 else -coeff[i]
                for i in range(0, len(coeff))]

        if k == 0:
            diag=np.ones(n)
        else:
            diag=np.vstack([i * np.ones(n) for i in coeff])
 
        return diag


def test_lapack(n=100,sig_figs=10):
    """Test the LU decomposition method"""
    k=2
    print("Testing LU decomposition method")
    D=Difference_Matrix(n,k,style="lapack")
    
    return

def test_sparse(n=100,sig_figs=10):
    """Test the sparse method"""
    k=2
    print("Testing sparse method")
    D=Difference_Matrix(n,k,style="sparse")
    
    return

def test_pentapy(n=100,sig_figs=10):
    """Test the pentapy method"""
    k=2
    print("Testing pentapy method")
    D=Difference_Matrix(n,k,style="pentapy")
 
    return 



