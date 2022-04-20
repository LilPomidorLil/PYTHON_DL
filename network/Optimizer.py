from numpy import ndarray as ARRAY

class Optimizer:
    """
    Базовый класс, реализация оптимайзеров сетки
    """

    def reset(self):
        """
        :return: Сброс параметров модели
        """
        pass

    def update(self, dvec: ARRAY, vec: ARRAY):
        """
        Обновление весов сетки
        :param dvec: производная весов
        :param vec: веса
        :return: обновленные значения весов (смещений)
        """
        pass