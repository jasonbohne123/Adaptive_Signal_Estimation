### DEPRECATED Functions ###

# Idea was to generate the difference matrix in one shot for all orders using pascals triangle recursion
# With the alterative method, for L1-Trend Filtering seems many inconsistencies 

import numpy as np
from scipy.sparse import spdiags


def extract_matrix_diagonals( n, k):
    """
    Extracts only the diagonals of the difference matrix
    """

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
    coeff = [coeff[i] if i % 2 == 1 else -coeff[i] for i in range(0, len(coeff))]

    if k == 0:
        diag = spdiags([np.ones(n)], [0], n, n)
    else:

        diag = spdiags([i * np.ones(n) for i in coeff], np.arange(-k, k ), n-2 , n)

    return diag

### PTrans Algorithm ###

## Efficient solver for pentadiagonal systems 

 def ptrans_algorithm(self):
        """Solves pentadiagonal system using pentapy package"""

        inv = np.zeros((self.n - 2, self.n - 2))
        for i in range(0, self.n - 2):
            unit = np.zeros(self.n - 2)
            unit[i] = 1
            inv[i] = pp.solve(self.DDT, unit, is_flat=False)
        return inv
