import random
from abc import ABC, abstractmethod

import numpy as np


class TaskBase(ABC):
    def __init__(self, dataset, seed=2023, **kwargs):
        self.dataset = dataset

        random.seed(seed)

    def get_random_prediction(self, label_set):
        return random.choice(list(label_set))

    def get_random_continuous_prediction(self, score_range):
        return random.uniform(score_range[0], score_range[1])

    def create_random_binary_array(self, length):
        return np.random.randint(low=0, high=2, size=(length,))

    @abstractmethod
    def evaluate(self, true_labels, predicted_labels):
        pass
