from network.Activation import Activation
from network.Activation import ARRAY

import numpy as np

class ReLU(Activation):
    """
    Функция активации ReLU

    { a, x > 0,
    { 0, x <= 0

    Производная

    { 1, x > 0,
    { 0, x <= 0
    """
    def activate(self, Z: ARRAY, A: ARRAY):
        A = np.where(Z > 0, Z, 0)
        return A

    def apply_jacobian(self, Z: ARRAY, A: ARRAY, F: ARRAY, G: ARRAY):
        G = np.where(A > 0, 1, 0)
        return G