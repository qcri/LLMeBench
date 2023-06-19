import random

import numpy as np
from sklearn.metrics import jaccard_score

from arabic_llm_benchmark.tasks.task_base import TaskBase


class EmotionTask(TaskBase):
    def __init__(self, **kwargs):
        super(EmotionTask, self).__init__(**kwargs)

    def evaluate(self, true_labels, predicted_labels):
        predicted_labels = [
            p if p else np.random.randint(low=0, high=2, size=(11,))
            for p in predicted_labels
        ]
        return {
            "Jaccard Score": jaccard_score(
                true_labels, predicted_labels, average="macro"
            )
        }
