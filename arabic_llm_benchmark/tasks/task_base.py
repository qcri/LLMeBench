import random
from abc import ABC, abstractmethod


class TaskBase(ABC):
    def __init__(self, dataset, seed=2023, **kwargs):
        self.dataset = dataset

        random.seed(seed)

    # TODO: Remove the dataset calling from the task base
    def load_data(self, data_path):
        return self.dataset.load_data(data_path)

    def load_train_data(self, train_data_path):
        return self.dataset.load_train_data(train_data_path)

    def prepare_fewshots(self, target_data, train_data, n_shots):
        return self.dataset.prepare_fewshots(target_data, train_data, n_shots)

    def get_random_prediction(self, label_set):
        return random.choice(list(label_set))


    @abstractmethod
    def evaluate(self, true_labels, predicted_labels):
        pass
