from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from llmebench.tasks.task_base import TaskBase


class FactualityTask(TaskBase):
    def __init__(self, **kwargs):
        super(FactualityTask, self).__init__(**kwargs)

    def evaluate(self, gold_labels, pred_labels):
        pred_labels = [
            p if p else self.get_random_prediction(set(gold_labels))
            for p in pred_labels
        ]
        acc = accuracy_score(gold_labels, pred_labels)
        precision = precision_score(gold_labels, pred_labels, average="macro")
        recall = recall_score(gold_labels, pred_labels, average="macro")
        f1 = f1_score(gold_labels, pred_labels, average="macro")

        w_precision = precision_score(gold_labels, pred_labels, average="weighted")
        w_recall = recall_score(gold_labels, pred_labels, average="weighted")
        w_f1 = f1_score(gold_labels, pred_labels, average="weighted")

        results = {
            "accuracy": acc,
            "macro-precision": precision,
            "macro-recall": recall,
            "macro-f1": f1,
            "w-precision": w_precision,
            "w-recall": w_recall,
            "w-f1": w_f1,
        }

        return results
