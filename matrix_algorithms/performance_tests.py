def test_lapack(n=100):
    """Test the LU decomposition method"""
    k = 2
    print("Testing LU decomposition method")
    D = Difference_Matrix(n, k, style="lapack")

    DDT = D.DDT
    DDT_inv = D.DDT_inv

    DDT_rank = np.linalg.matrix_rank(DDT)

    print(f"Rank of DDT matrix is {DDT_rank} out of {n-k}.")

    # check that the inverse is correct
    assert np.allclose(DDT.dot(DDT_inv), np.eye(n - k), rtol=1e-8, atol=1e-8)

    return


def test_sparse(n=100):
    """Test the sparse method"""
    k = 2
    print("Testing sparse method")
    D = Difference_Matrix(n, k, style="sparse")

    DDT = D.DDT
    DDT_inv = D.DDT_inv

    DDT_rank = np.linalg.matrix_rank(DDT)

    print(f"Rank of DDT matrix is {DDT_rank} out of {n-k}.")

    # check that the inverse is correct
    assert np.allclose(DDT.dot(DDT_inv), np.eye(n - k), rtol=1e-8, atol=1e-8)

    return
