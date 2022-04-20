import numpy as np

from ..OUTPUT import Output

from numpy import ndarray as ARRAY

class RegressionMSE(Output):
    """
    Подсчет ошибки на последнем слое как задача регрессии - ошибка MSE
    """
    def evaluate(self, prev_layer_data: ARRAY, target: ARRAY):
        nrows, ncols = prev_layer_data.shape

        if (target.shape[0] != nrows) or (target.shape[1] != ncols):
            raise IndexError(f"Target dimension is not last hidden layer dimension")

        self.__m_din = prev_layer_data - target


    def backprop_data(self):
        return self.__m_din

    def loss(self):
        return 0.5 * np.square(self.__m_din).sum()

