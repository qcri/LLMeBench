import numpy as np

from llmebench.tasks.task_base import TaskBase


class STSTask(TaskBase):
    def __init__(self, **kwargs):
        super(STSTask, self).__init__(**kwargs)

    def evaluate(self, true_scores, predicted_scores):
        score_range = [0, 5]
        predicted_scores = [
            p if p is not None else self.get_random_continuous_prediction(score_range)
            for p in predicted_scores
        ]
        # Pearson Correction is the off-diagonal of the symmetric correlation 2x2 matrix
        return {"PC": np.corrcoef(true_scores, predicted_scores)[0, 1]}
