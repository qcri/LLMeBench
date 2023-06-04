import random
from abc import ABC, abstractmethod


class TaskBase(ABC):
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset

    def load_data(self, data_path):
        return self.dataset.load_data(data_path)

    def get_random_prediction(self, label_set):
        return random.choice(list(label_set))

    @abstractmethod
    def evaluate(self, true_labels, predicted_labels):
        pass
