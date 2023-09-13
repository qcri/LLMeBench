import random
from abc import ABC, abstractmethod

import numpy as np


class TaskBase(ABC):
    """
    Base class for tasks

    Implementations of this class need to implement one method, `evaluate()`
    which takes true and predicted labels, and returns some score over
    these.

    Attributes
    ----------
    dataset : DatasetBase
        The dataset that is currently being evaluated by the task
    seed : int
        Seed for initializing pseudo random generators for reproducible
        results

    Methods
    -------
    get_random_prediction(label_set):
        Helper method to choose a random classification label

    get_random_continuous_prediction(score_range):
        Helper method to choose a random regression prediction

    create_random_binary_array(score_range):
        Helper method to generate random multi-label binary array

    evaluate(true_labels, predicted_labels):
        Method to evaluate the predictions and return appropriate scores
    """

    def __init__(self, dataset, seed=2023, **kwargs):
        self.dataset = dataset

        random.seed(seed)
        np.random.seed(seed)

    def get_random_prediction(self, label_set):
        """
        Helper method to choose a random classification label

        Arguments
        ---------
        label_set : set
            Set of unique labels valid for the task

        Returns
        -------
        label : mixed
            A label chosen at random from the `label_set`
        """
        return random.choice(list(label_set))

    def get_random_continuous_prediction(self, score_range):
        """
        Helper method to choose a random regression prediction

        Arguments
        ---------
        score_range : tuple
            Tuple (min_val, max_val) that defines the range from
            which a random number will be chosen

        Returns
        -------
        score : float
            A number chosen at random between `min_val` and `max_val`
        """
        return random.uniform(score_range[0], score_range[1])

    def create_random_binary_array(self, length):
        """
        Helper method to generate random multi-label binary array

        Arguments
        ---------
        length : int
            Length of the generated array

        Returns
        -------
        binary_array : list
            List of length `length` where each element is either 0 or 1
            at random.
        """
        return np.random.randint(low=0, high=2, size=(length,))

    @abstractmethod
    def evaluate(self, true_labels, predicted_labels):
        """
        Method to evaluate the predictions and return appropriate scores

        Arguments
        ---------
        true_labels : list
            List of labels (should match "label" key from the dataset's
            `get_data_sample()`)
        predicted_labels : list
            List of predicted labels (should match "label" key from the
             dataset's `get_data_sample()` in structure and type)

        Returns
        -------
        scores : dict
            Dictionary of one or more elements, each representing a metric
            computed from the predictions. Examples are "Accuracy", "F1",
            "Pearson correlation" etc.
        """
        pass
