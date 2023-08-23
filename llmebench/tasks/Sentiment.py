from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from llmebench.tasks.task_base import TaskBase


class SentimentTask(TaskBase):
    def __init__(self, **kwargs):
        super(SentimentTask, self).__init__(**kwargs)

    def evaluate(self, true_labels, predicted_labels):
        predicted_labels = [
            p if p else self.get_random_prediction(set(true_labels))
            for p in predicted_labels
        ]
        return {
            "Macro F1": f1_score(true_labels, predicted_labels, average="macro"),
            "Micro F1": f1_score(true_labels, predicted_labels, average="micro"),
            "Acc": accuracy_score(true_labels, predicted_labels),
            "Weighted Precision": precision_score(
                true_labels, predicted_labels, average="weighted"
            ),
            "Weighted Recall": recall_score(
                true_labels, predicted_labels, average="weighted"
            ),
            "Weighted F1": f1_score(true_labels, predicted_labels, average="weighted"),
        }
