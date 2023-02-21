from collections import deque

import numpy as np


class Matrix_Sequence:
    def __init__(self):

        self.sequence = deque()

    def add_matrix(self, mat):

        self.sequence.append(mat)

    def compute_matrix(self):
        return np.linalg.multi_dot(self.sequence)

    def compute_transpose(self):

        sequence_transpose = Matrix_Sequence()

        for mat in reversed(self.sequence):
            sequence_transpose.add_matrix(mat.T)

        return sequence_transpose
