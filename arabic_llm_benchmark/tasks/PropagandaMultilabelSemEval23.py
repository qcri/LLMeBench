import itertools

from sklearn import preprocessing
from sklearn.metrics import f1_score, roc_auc_score

from arabic_llm_benchmark.tasks.task_base import TaskBase


class PropagandaMultilabelSemEval23Task(TaskBase):
    def __init__(self, **kwargs):
        super(PropagandaMultilabelSemEval23Task, self).__init__(**kwargs)

    def evaluate(self, true_labels, predicted_labels):
        # Handle cases when model fails!
        # Flatten true labels as it is a list of lists

        predicted_labels = [p if p else ["no_technique"] for p in predicted_labels]

        # Need the pre-defined list of techniques
        techniques = self.dataset.get_predefined_techniques()

        # Binarize labels and use them for multi-label evaluation
        mlb = preprocessing.MultiLabelBinarizer(classes=techniques)

        mlb.fit([techniques])
        gold = mlb.transform(true_labels)
        pred = mlb.transform(predicted_labels)

        micro_f1 = f1_score(gold, pred, average="micro")
        macro_f1 = f1_score(gold, pred, average="macro")
        # roc_auc = roc_auc_score(gold, pred, average="micro")

        return {"Micro F1": micro_f1, "Macro F1": macro_f1}
