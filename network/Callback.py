from NeuralNetwork import NeuralNetwork

from numpy import ndarray as ARRAY

class Callback:
    """

    Базовый класс для отслеживания обучения.

    """

    def __init__(self, m_nbatch: int, m_nepoch: int, m_batch_id: int, m_epoch_id: int):
        self._m_nbatch = m_nbatch
        self._m_nepoch = m_nepoch
        self._m_batch_id = m_batch_id
        self._m_epoch_id = m_epoch_id

    def pre_trained_batch(self, net: NeuralNetwork, sub_X: ARRAY, y: ARRAY):
        """
        Срабатывает до прохода батча по сетке

        :param net: - объект класса нейросеть
        :param sub_X: - обучающие данные
        :param y: - таргет
        :return: сообщение
        """
        pass

    def post_trained_batch(self, net: NeuralNetwork, sub_X: ARRAY, y: ARRAY):
        """
        Срабатывает после прохода батча по сетке

        :param net: - объект класса нейросеть
        :param sub_X: - обучающие данные
        :param y: - таргет
        :return: сообщение
        """
        pass