from numpy import ndarray as ARRAY

class Output:
    """
    Базовый класс, отвечающий за выходной слой сетки
    """

    def check_target_data(self, target: ARRAY):
        pass

    def evaluate(self, prev_layer_data: ARRAY, target: ARRAY):
        """
        Подсчет прямого и обратного распространения для последнего скрытого слоя и для таргета.
        :param prev_layer_data: - последний скрытый слой
        :param target: - таргет
        :return:
        """
        pass

    def backprop_data(self):
        pass

    def loss(self):
        """
        Подсчет лосса
        :return:
        """
        pass