from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from llmebench.tasks.task_base import TaskBase


class ClassificationTask(TaskBase):
    def __init__(self, **kwargs):
        super(ClassificationTask, self).__init__(**kwargs)

    def evaluate(self, true_labels, predicted_labels):
        predicted_labels = [
            p if p is not None else self.get_random_prediction(set(true_labels))
            for p in predicted_labels
        ]

        acc_score = accuracy_score(true_labels, predicted_labels)
        macro_precision = precision_score(
            true_labels, predicted_labels, average="macro"
        )
        macro_recall = recall_score(true_labels, predicted_labels, average="macro")
        macro_f1 = f1_score(true_labels, predicted_labels, average="macro")

        micro_precision = precision_score(
            true_labels, predicted_labels, average="micro"
        )
        micro_recall = recall_score(true_labels, predicted_labels, average="micro")
        micro_f1 = f1_score(true_labels, predicted_labels, average="micro")

        w_precision = precision_score(true_labels, predicted_labels, average="weighted")
        w_recall = recall_score(true_labels, predicted_labels, average="weighted")
        w_f1 = f1_score(true_labels, predicted_labels, average="weighted")

        return {
            "Accuracy": acc_score,
            "Macro precision": macro_precision,
            "Macro recall": macro_recall,
            "Macro F1": macro_f1,
            "Micro precision": micro_precision,
            "Micro recall": micro_recall,
            "Micro F1": micro_f1,
            "Weighted Precision": w_precision,
            "Weighted Recall": w_recall,
            "Weighted F1": w_f1,
        }
