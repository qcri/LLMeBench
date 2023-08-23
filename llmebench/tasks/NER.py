from sklearn.metrics import f1_score

from llmebench.tasks.task_base import TaskBase


class NERTask(TaskBase):
    def __init__(self, **kwargs):
        super(NERTask, self).__init__(**kwargs)

    def _clean_ground_truth(self, ground_truth):
        cleaned_version = []
        for i, elem in enumerate(ground_truth):
            if "I-MIS" in elem:
                cleaned_version.append("I-MISC")
            elif "B-MIS" in elem:
                cleaned_version.append("B-MISC")
            else:
                cleaned_version.append(elem)
        return cleaned_version

    def evaluate(self, true_labels, predicted_labels):
        all_ground_truths = []
        all_predictions = []

        for true_label, pred_labels in zip(true_labels, predicted_labels):
            ground_truth = self._clean_ground_truth(true_label.split())

            if pred_labels is None or len(pred_labels) == 0:
                pred_labels = ["O"] * len(ground_truth)

            if len(ground_truth) == len(pred_labels):
                all_ground_truths.extend(ground_truth)
                all_predictions.extend(pred_labels)

            elif len(pred_labels) < len(ground_truth):
                while len(pred_labels) < len(ground_truth):
                    pred_labels += ["O"]
                all_ground_truths.extend(ground_truth)
                all_predictions.extend(pred_labels)
            else:
                pred_labels = pred_labels[: len(ground_truth)]
                all_ground_truths.extend(ground_truth)
                all_predictions.extend(pred_labels)

        return {
            "Macro F1": f1_score(all_ground_truths, all_predictions, average="macro")
        }
