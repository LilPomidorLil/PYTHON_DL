from numpy import ndarray as ARRAY
from network.OptimiZer import Optimizer
from network.Activation import Activation

class Layer:
    """
    Базовый класс для всех слоев
    """

    def __init__(self, in_size: int, out_size: int):
        self.__m_in_size = in_size
        self.__m_out_size = out_size

    def in_size(self):
        return self.__m_in_size

    def out_size(self):
        return self.__m_out_size

    def init(self, mu: float, sigma: float, Activation: Activation):
        """
        Инициализация слоя

        :param mu: мат.ожидание
        :param sigma: дисперсия
        :param Activation: функция активации, которая будет применена к этому слою
        :return: None
        """
        pass

    def forward(self, prev_layer_data: ARRAY):
        """
        Проход вперед внутри одного слоя
        :param prev_layer_data: - данные нейронов предыдущего слоя
        :return: None
        """
        pass

    def output(self):
        """
        :return: Значения нейронов в слою после функции активации
        """
        pass

    def backprop(self, prev_layer_data: ARRAY, next_layer_data: ARRAY):
        """
        Реализация метода обратного распространения ошибки.

        :param prev_layer_data: значения нейронов предыдущего слоя,
	    которые также являются входными значениями этого слоя.

        :param next_layer_data: значения нейронов следующего слоя,
        которые также являются выходными значениями этого слоя

        :return:
        """
        pass

    def backprop_data(self):
        """
        :return: next_layer_data в Layer.backprop()
        """
        pass

    def update(self, opt: Optimizer):
        """
        Обновление весов сетки
        :param opt: - объект класса Optimizer
        :return:
        """
        pass


