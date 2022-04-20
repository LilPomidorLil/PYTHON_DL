from network.Activation import Activation
from network.Activation import ARRAY

import numpy as np

class ReLU(Activation):
    def activate(self, Z: ARRAY, A: ARRAY):
        A = np.where(Z >= 0, Z, 0)
        return A