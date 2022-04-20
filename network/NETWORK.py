from .Callback_ import VerboseCallback
from .LAYER import Layer
from .OUTPUT import Output
from .CALLBACK import Callback

import numpy as np

class NeuralNetwork:
    """Нейронная сеть - основной класс управления сеткой"""

    def __init__(self):
        self.m_output = False
        self.m_callback = VerboseCallback()
        self.m_layers = []

    def count_layers(self):
        """
        :return: кол-во слоев в сетке
        """
        return self.m_layers.__len__()

    def add_layer(self, layer: Layer):
        """
        Добавить слой в сетку
        :param layer: слой
        :return:
        """
        self.m_layers.append(layer)

    def set_output(self, out: Output):
        """
        Установить задачу, на которую будет учиться сетка
        :param out: объект установленного класса
        :return:
        """
        if self.m_output:
            del self.m_output

        self.m_output = out

    def get_output(self):
        return self.m_output

    def set_callback(self, call: Callback):
        """
        Установить пользовательский вывод информации об обучении
        :param call: объект установленного класса
        :return:
        """
        self.m_callback = call

    def set_default_callback(self):
        self.m_callback = VerboseCallback()

    def init(self, mu: float = 0, sigma: float = 0.01):
        """
        Инициализация слоев сетки.

        Начальные значения весов и смещений генерируются из нормального распределения,
        при желании их можно изменить
        :param mu: - мат.ожидание
        :param sigma: - дисперсия
        :return:
        """

        nlayer = self.count_layers()

        for i in range(nlayer):
            self.m_layers[i].init(mu, sigma)




