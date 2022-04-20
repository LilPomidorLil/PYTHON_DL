from ..OPTIMIZER import Optimizer

from numpy import ndarray as ARRAY

class SGD(Optimizer):
    """
    Оптимайзер

    Стохастический градиентный спуск
    """
    def __init__(self, lrate: float = 0.01, decay: float = 0):
        """
        Инициализация класса
        :param lrate: длина шага
        :param decay: поправочный коэффицент
        """
        self.lrate = lrate
        self.decay = decay

    def update(self, dvec: ARRAY, vec: ARRAY):
        vec -= self.lrate * (dvec + self.decay * vec)
        return vec