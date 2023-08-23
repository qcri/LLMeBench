import itertools

from sklearn import preprocessing
from sklearn.metrics import f1_score

from llmebench.tasks.task_base import TaskBase


class PropagandaMultilabelTask(TaskBase):
    def __init__(self, **kwargs):
        super(PropagandaMultilabelTask, self).__init__(**kwargs)

    def evaluate(self, true_labels, predicted_labels):
        # Handle cases when model fails!
        # Flatten true labels as it is a list of lists
        predicted_labels = [p if p else ["no technique"] for p in predicted_labels]

        # Need the pre-defined list of techniques
        techniques = self.dataset.get_predefined_techniques()

        # Binarize labels and use them for multi-label evaluation
        mlb = preprocessing.MultiLabelBinarizer()
        mlb.fit([techniques])
        gold = mlb.transform(true_labels)
        pred = mlb.transform(predicted_labels)

        micro_f1 = f1_score(gold, pred, average="micro")

        return {"Micro F1": micro_f1}
