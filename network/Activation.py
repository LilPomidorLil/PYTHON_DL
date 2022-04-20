from numpy import ndarray as ARRAY

class Activation:
    """
    Базовый класс для функций активаций
    """

    def activate(self, Z: ARRAY, A: ARRAY):
        """
        Активация нейронов, используется при прямом распространении
        :param Z: значения нейронов до активации
        :param A: значения нейронов после активации
        :return:
        """
        pass

    def apply_jacobian(self, Z: ARRAY, A: ARRAY, F: ARRAY, G: ARRAY):
        """
        Расчет Якобиана, используется при обратном распространении
        :return:
        """
        pass