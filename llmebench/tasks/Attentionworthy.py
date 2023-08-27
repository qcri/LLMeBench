from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from llmebench.tasks.task_base import TaskBase


class AttentionworthyTask(TaskBase):
    def __init__(self, **kwargs):
        super(AttentionworthyTask, self).__init__(**kwargs)

    def evaluate(self, gold_labels, pred_labels):
        pred_labels = [
            p if p else self.get_random_prediction(set(gold_labels))
            for p in pred_labels
        ]
        acc = accuracy_score(gold_labels, pred_labels)
        precision = precision_score(gold_labels, pred_labels, average="weighted")
        recall = recall_score(gold_labels, pred_labels, average="weighted")
        f1 = f1_score(gold_labels, pred_labels, average="weighted")
        results = {
            "accuracy": acc,
            "w-precision": precision,
            "w-recall": recall,
            "w-f1": f1,
            "msg": "performance with respect weighted-F1. W-F1 - official measure. Ref: CheckThat-2022",
        }

        return results
