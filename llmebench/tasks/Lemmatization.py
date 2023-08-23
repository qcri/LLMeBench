from sklearn.metrics import accuracy_score, f1_score

from llmebench.tasks.task_base import TaskBase


class LemmatizationTask(TaskBase):
    def __init__(self, **kwargs):
        super(LemmatizationTask, self).__init__(**kwargs)

    def evaluate(self, true_labels, predicted_labels):
        # Replace failed predictions with unlemmatized word
        for idx, pred in enumerate(predicted_labels):
            if pred is None:
                predicted_labels[idx] = (None, true_labels[idx][0])

        # Trim labels to relevant pieces
        true_labels = [t_l for _, t_l in true_labels]
        predicted_labels = [p_l for _, p_l in predicted_labels]

        return {"accuracy_score": accuracy_score(true_labels, predicted_labels)}
