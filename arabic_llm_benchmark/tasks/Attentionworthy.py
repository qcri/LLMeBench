from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from arabic_llm_benchmark.tasks.task_base import TaskBase


class CheckWorthinessTask(TaskBase):
    def evaluate(self, gold_labels, pred_labels):
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
            # "msg": "performance with respect to the positive class. F1 (Pos) - official measure. Ref: CheckThat-2022"
        }

        return results
