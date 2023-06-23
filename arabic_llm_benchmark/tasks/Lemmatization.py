from sklearn.metrics import accuracy_score, f1_score

from arabic_llm_benchmark.tasks.task_base import TaskBase


class LemmatizationTask(TaskBase):
    def __init__(self, **kwargs):
        super(LemmatizationTask, self).__init__(**kwargs)

    def evaluate(self, true_labels, predicted_labels):
        # Replace failed predictions with unlemmatized word
        for idx, pred in enumerate(predicted_labels):
            if pred is None:
                predicted_labels[idx] = true_labels[idx][0]

        # Trim gold labels to lemmatized words
        true_labels = [t_l for _, t_l in true_labels]

        len1 = len(true_labels)
        len2 = len(predicted_labels)
        if len1 < len2:
            # Trim predicted_labels
            predicted_labels = predicted_labels[len1]
        elif len1 > len2:
            # Pad predicted_labels
            for j in range(len1 - len2):
                predicted_labels.add("ERROR")

        return {"accuracy_score": accuracy_score(true_labels, predicted_labels)}
