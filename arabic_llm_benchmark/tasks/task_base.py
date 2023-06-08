import random
from abc import ABC, abstractmethod


class TaskBase(ABC):
    def __init__(self, dataset, seed=2023, **kwargs):
        self.dataset = dataset

        random.seed(seed)

    def load_data(self, data_path):
        return self.dataset.load_data(data_path)

    def get_random_prediction(self, label_set):
        return random.choice(list(label_set))

    def get_random_continuous_prediction(self, range_set):
        return random.uniform(range_set[0], range_set[1])

    @abstractmethod
    def evaluate(self, true_labels, predicted_labels):
        pass
