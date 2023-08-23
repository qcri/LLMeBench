from sklearn.metrics import f1_score

from llmebench.tasks.task_base import TaskBase


class Q2QSimDetectionTask(TaskBase):
    def __init__(self, **kwargs):
        super(Q2QSimDetectionTask, self).__init__(**kwargs)

    def evaluate(self, true_labels, predicted_labels):
        predicted_labels = [
            p if p else self.get_random_prediction(set(true_labels))
            for p in predicted_labels
        ]

        return {"Micro F1": f1_score(true_labels, predicted_labels, average="micro")}
