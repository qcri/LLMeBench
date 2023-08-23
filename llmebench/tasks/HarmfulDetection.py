from sklearn.metrics import f1_score

from llmebench.tasks.task_base import TaskBase


class HarmfulDetectionTask(TaskBase):
    def __init__(self, **kwargs):
        super(HarmfulDetectionTask, self).__init__(**kwargs)

    def evaluate(self, true_labels, predicted_labels):
        # Handle cases when model fails!
        predicted_labels = [
            p if p else self.get_random_prediction(set(true_labels))
            for p in predicted_labels
        ]

        f1 = f1_score(true_labels, predicted_labels, pos_label="1", average="binary")

        return {"F1 (POS)": f1}
