from .Callback_ import VerboseCallback
from .LAYER import Layer
from .OUTPUT import Output
from .CALLBACK import Callback

from numpy import ndarray as ARRAY

class NeuralNetwork:
    """Нейронная сеть - основной класс управления сеткой"""

    def __init__(self):
        self.m_output = False
        self.m_callback = VerboseCallback()
        self.m_layers = []

    def __check_unit_sizes(self):
        nlayer = self.count_layers()

        if nlayer <= 1: return

        for i in range(1, nlayer):
            if self.m_layers[i].in_size() != self.m_layers[i].out_size():
                raise ValueError(f"Check input data layers!")


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

    def forward(self, input: ARRAY):
        """
        Проход вперед по всей сетке
        :param input: входные данные
        :return:
        """

        nlayer = self.count_layers()

        if nlayer <= 0: return

        if input.shape[0] != self.m_layers[0].in_size():
            raise IndexError(f"Check input and first layer dimension")

        self.m_layers[0].forward(input)

        for i in range(1, nlayer):
            self.m_layers[i].forward(self.m_layers[i - 1].output())

    def backprop(self, input: ARRAY, target: ARRAY):
        """
        Проход назад по всей сетке
        :param input: входные данные
        :param target: таргет
        :return:
        """
        nlayer = self.count_layers()

        if nlayer <= 0: return

        self.m_output.evaluate(self.m_layers[-1].output(), target)

        if nlayer == 1:
            self.m_layers[0].backprop(input, self.m_output.backprop_data())
            return

        self.m_layers[-1].backprop(self.m_layers[-2].output(), self.m_output.backprop_data())

        for i in range(nlayer - 2, 0, -1):
            self.m_layers[i].backprop(
                self.m_layers[i - 1].output(),
                self.m_layers[i + 1].backprop_data()
            )

        self.m_layers[0].backprop(input, self.m_layers[1].backprop_data())

    def update(self, opt):
        """
        Обновление весов модели
        :param opt: объект класса оптимайзер
        :return:
        """
        nlayer = self.count_layers()

        if nlayer <= 0: return

        for i in range(0, nlayer):
            self.m_layers[i].update(opt)

    def fit(self, x: ARRAY, y: ARRAY, batch_size: int, nepoch: int, opt):
        # TODO: доделать батчи
        self.m_callback._m_nbatch = batch_size
        self.m_callback._m_nepoch = nepoch

        for ep in range(nepoch):
            self.m_callback._m_epoch_id = ep

            for batch in range(20):
                self.m_callback._m_batch_id = batch
                self.forward(x)
                self.backprop(x, y)
                self.update(opt)
                self.m_callback.post_trained_batch(self, x, y)

        return True

    def predict(self, x: ARRAY):
        nlayer = self.count_layers()

        if nlayer <= 0: return ARRAY

        self.forward(x)
        
        return self.m_layers[-1].output()
