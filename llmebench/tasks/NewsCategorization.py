from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

from llmebench.tasks.task_base import TaskBase


class NewsCategorizationTask(TaskBase):
    def __init__(self, **kwargs):
        super(NewsCategorizationTask, self).__init__(**kwargs)

    def evaluate(self, gold_labels, pred_labels):
        pred_labels = [
            p if p else self.get_random_prediction(set(gold_labels))
            for p in pred_labels
        ]

        acc = accuracy_score(gold_labels, pred_labels)
        precision = precision_score(gold_labels, pred_labels, average="macro")
        recall = recall_score(gold_labels, pred_labels, average="macro")
        f1 = f1_score(gold_labels, pred_labels, average="macro")
        results = {
            "accuracy": acc,
            "macro-precision": precision,
            "macro-recall": recall,
            "macro-f1": f1,
            "msg": "performance with respect macro-F1.",
        }

        return results
