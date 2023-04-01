from collections import deque

import numpy as np


class Matrix_Sequence:
    def __init__(self):

        self.sequence = deque()

    def get_sequence(self):
        return self.sequence

    def add_matrix(self, mat):

        self.sequence.append(mat)

    def add_matrix_left(self, mat):

        self.sequence.appendleft(mat)

    def compute_matrix(self):
        return np.linalg.multi_dot(self.sequence)

    def compute_transpose(self):

        sequence_transpose = Matrix_Sequence()

        for mat in reversed(self.sequence):
            sequence_transpose.add_matrix(mat.T)

        return sequence_transpose
