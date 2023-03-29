def compute_lambda_max(D: Difference_Matrix, x: np.ndarray):
    """Computes the maximum lambda value for the adaptive trend filtering algorithm"""

    DDT_inv = np.linalg.solve(D.DDT, np.eye(D.DDT.shape[0]))

    max_error = np.max(abs(D.DDT.dot(DDT_inv) - np.eye(D.DDT.shape[0])))

    if max_error > 1e-6:
        print(f"Warning: DDT is not invertible. Max error: {max_error}")

    D_D = D.D

    # lambda value which gives best affine fit
    lambda_max = np.max(abs(DDT_inv.dot(D_D).dot(x)))

    return lambda_max, D
