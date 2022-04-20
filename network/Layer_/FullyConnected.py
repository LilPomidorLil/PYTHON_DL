from ..LAYER import Layer
from ..ACTIVATION import Activation
from ..OPTIMIZER import Optimizer

from numpy import ndarray as ARRAY
import numpy as np

from copy import deepcopy

class FullyConnected(Layer):
    """
    Реализация полносвязного слоя
    """


    def __init__(self, in_size: int, out_size: int, Activation: Activation):
        super(FullyConnected, self).__init__(in_size=in_size, out_size=out_size)
        self.__Activation = Activation

    def init(self, mu: float, sigma: float):
        self.__m_weight = np.ndarray(shape=(self._m_in_size, self._m_out_size),
                                    buffer=np.random.normal(size = (self._m_in_size, self._m_out_size), scale=sigma, loc=mu))

        self.__m_bias = np.ndarray(shape=(self._m_out_size, 1),
                                    buffer=np.random.normal(size = (self._m_out_size, ), scale=sigma, loc=mu))
        self.__m_dw = np.empty(shape = (self._m_in_size, self._m_out_size))
        self.__m_db = np.empty(shape = (self._m_out_size, 1))



    def forward(self, prev_layer_data: ARRAY):
        ncols = prev_layer_data.shape[1]

        self.__m_z = np.dot(self.__m_weight.T,  prev_layer_data)
        self.__m_z += self.__m_bias


        self.__m_a = np.empty(shape = (self._m_in_size, ncols))
        self.__m_a = self.__Activation.activate(self.__m_z, self.__m_a)

    def backprop(self, prev_layer_data: ARRAY, next_layer_data: ARRAY):
        ncols = prev_layer_data.shape[1]

        dLz = self.__m_z

        # вычислили производную слоя по d_a
        dLz = self.__Activation.apply_jacobian(self.__m_z, self.__m_a, next_layer_data, dLz)

        # показывает насколько сильно нужно изменить каждый вес
        self.__m_dw = np.dot(prev_layer_data, dLz.T) / ncols
        self.__m_db = dLz.mean(axis = 1).reshape(self._m_out_size, 1)

        self.__m_din = np.dot(self.__m_weight, dLz)

    def backprop_data(self):
        return self.__m_din

    def output(self):
        return self.__m_a

    def update(self, opt: Optimizer):
        self.__m_weight = opt.update(self.__m_dw, self.__m_weight)
        self.__m_bias = opt.update(self.__m_db, self.__m_bias)

