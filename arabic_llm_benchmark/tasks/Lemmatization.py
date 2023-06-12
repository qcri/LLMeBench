from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from arabic_llm_benchmark.tasks.task_base import TaskBase


class LemmatizationTask(TaskBase):
    def __init__(self, **kwargs):
        super(LemmatizationTask, self).__init__(**kwargs)

    def evaluate(self, true_labels, predicted_labels):
        #predicted_labels = [
        #    p if p else self.get_random_prediction(set(true_labels))
        #    for p in predicted_labels
        #]
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
