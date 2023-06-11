from sklearn.metrics import jaccard_score

from arabic_llm_benchmark.tasks.task_base import TaskBase

import numpy as np
import random
random.seed(1333)


# Create a binary vector of length 10 with 3 randomly placed ones
# print(num_ones)
# binary_vector = create_random_binary_array(10, num_ones)
class EmotionTask(TaskBase):
    def __init__(self, **kwargs):
        super(EmotionTask, self).__init__(**kwargs)
    def create_random_binary_array(self, length, num_ones):
        np.random.seed(1223)
        # Create a zero-initialized array of the given length
        arr = np.zeros(length, dtype=int)
        # Randomly select indices
        random_indices = np.random.choice(length, num_ones, replace=False)
        # Set the value at these indices to 1
        arr[random_indices] = 1
        return arr

    def evaluate(self, true_labels, predicted_labels):
        num_ones=random.randint(1, 11)
        predicted_labels = [
            p if p else self.create_random_binary_array(11, num_ones)
            for p in predicted_labels
        ]
        return {
            "Jaccard Score": jaccard_score(
                true_labels, predicted_labels, average="macro"
            )
        }
