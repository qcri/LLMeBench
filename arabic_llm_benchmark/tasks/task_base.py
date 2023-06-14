import random
from abc import ABC, abstractmethod


class TaskBase(ABC):
    def __init__(self, dataset, seed=2023, **kwargs):
        self.dataset = dataset

        random.seed(seed)

    def get_random_prediction(self, label_set):
        return random.choice(list(label_set))

    @abstractmethod
    def evaluate(self, true_labels, predicted_labels):
        pass
