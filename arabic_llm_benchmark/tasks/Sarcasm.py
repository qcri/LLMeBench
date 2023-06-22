from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from arabic_llm_benchmark.tasks.task_base import TaskBase


class SarcasmTask(TaskBase):
    def __init__(self, **kwargs):
        super(SarcasmTask, self).__init__(**kwargs)

    def evaluate(self, gold_labels, pred_labels):
        pred_labels = [
            p if p else self.get_random_prediction(set(gold_labels))
            for p in pred_labels
        ]

        f1 = f1_score(gold_labels, pred_labels, pos_label="TRUE", average="binary")
        # f1 = f1_score(gold_labels, pred_labels, average="macro")

        results = {
            "f1": f1,
        }

        return results
