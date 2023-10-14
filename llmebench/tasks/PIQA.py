from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from llmebench.tasks.task_base import TaskBase


class PIQATask(TaskBase):
    def __init__(self, **kwargs):
        super(PIQATask, self).__init__(**kwargs)

    def evaluate(self, gold_labels, pred_labels):
        pred_labels = [
            p if p else self.get_random_prediction(set(gold_labels))
            for p in pred_labels
        ]
        acc = accuracy_score(gold_labels, pred_labels)
        precision = precision_score(
            gold_labels, pred_labels, pos_label="1", average="binary"
        )
        recall = recall_score(gold_labels, pred_labels, pos_label="1", average="binary")
        f1 = f1_score(gold_labels, pred_labels, pos_label="1", average="binary")
        results = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        return results
