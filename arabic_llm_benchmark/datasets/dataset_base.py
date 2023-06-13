from abc import ABC, abstractmethod

class DatasetBase(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def citation(self):
        pass

    @abstractmethod
    def get_data_sample(self):
        pass

    @abstractmethod
    def load_data(self, data_path, no_labels=False):
        """
        Returns a list of dictionaries,
        with at least the following keys:
                "input": <input-instance>
                "label": <label>
        The dictionaries can contain other keys as well
        which will be saved in the cache
        """
        pass
    @abstractmethod
    def load_train_data(self, train_data_path):
        """
        Returns a list of dictionaries,
        with at least the following keys:
                "input": <input-instance>
                "label": <label>
        The dictionaries can contain other keys as well
        which will be saved in the cache
        """
        pass

    @abstractmethod
    def prepare_fewshots(self, target_data, train_data, n_shots):
        """
        Returns a list of dictionaries,
        with at least the following keys:
                "input": <input-instance>
                "label": <label>
        The dictionaries can contain other keys as well
        which will be saved in the cache
        """
        pass
