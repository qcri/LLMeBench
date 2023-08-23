from sklearn.metrics import accuracy_score

from llmebench.tasks.task_base import TaskBase


class XNLITask(TaskBase):
    def __init__(self, **kwargs):
        super(XNLITask, self).__init__(**kwargs)

    def evaluate(self, true_labels, predicted_labels):
        predicted_labels = [
            p if p else self.get_random_prediction(set(true_labels))
            for p in predicted_labels
        ]

        acc = accuracy_score(true_labels, predicted_labels)

        return {"Accuracy": acc}
