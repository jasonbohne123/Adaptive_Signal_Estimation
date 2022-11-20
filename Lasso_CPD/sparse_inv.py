from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np
from scipy.sparse import csr_matrix


def sparse_inversion(A, algorithm="lanczos"):
    """
    Inverts sparse matrices via SVD through SLEPc and PETSc
    Default Algorithm is Lanczos Algorithm
    """

    r, c = A.shape

    A = csr_matrix(A)

    p1 = A.indptr
    p2 = A.indices
    p3 = A.data
    # creates matrix witin PETSC data type
    A_p = PETSc.Mat().createAIJ(size=A.shape, csr=(p1, p2, p3))

    # constructs eigenvalue problem within SLEPC 
    E = SLEPc.SVD()
    E.create()
    E.setOperators(A_p)
    E.setDimensions(r)
    E.setProblemType(SLEPc.SVD.ProblemType.STANDARD)  
    E.setType(algorithm)
    E.setFromOptions()

    E.solve()

    # gets eigenvalues and eigenvectors
    eigs = np.array([E.getValue(i) for i in range(0, r)])

    u = []
    v = []

    for i in range(0, r):

        # allocates memory for eigenvectors
        u_ = PETSc.Vec().createSeq(r)
        v_ = PETSc.Vec().createSeq(c)

        E.getVectors(i, u_, v_)
        u.append(u_.getArray())
        v.append(v_.getArray())

    u = np.array(u).T
    v = np.array(v).T

    s = np.diagflat((1 / eigs) ** 2)
    l = np.matmul(np.matmul(u, s), u.T)

    return l
