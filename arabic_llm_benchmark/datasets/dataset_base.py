from abc import ABC, abstractmethod


class DatasetBase(ABC):
    def __init__(self, **kwargs):
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
