from sklearn import preprocessing
from sklearn.metrics import f1_score

from llmebench.tasks.task_base import TaskBase


class MultilabelPropagandaTask(TaskBase):
    def __init__(self, **kwargs):
        super(MultilabelPropagandaTask, self).__init__(**kwargs)

    def evaluate(self, true_labels, predicted_labels):
        # Need the pre-defined list of techniques
        techniques = self.dataset.get_predefined_techniques()

        # To generalize task to multiple datasets, since we use "no technique" as the random label next
        no_technique_label = "no_technique"
        for tch in techniques:
            if "technique" in tch:
                no_technique_label = tch
                break

        # Handle cases when model fails!
        # use no_technique_label as the random label
        predicted_labels = [p if p else [no_technique_label] for p in predicted_labels]

        # Flatten true labels as it is a list of lists
        # Binarize labels and use them for multi-label evaluation
        mlb = preprocessing.MultiLabelBinarizer(classes=techniques)
        mlb.fit([techniques])
        gold = mlb.transform(true_labels)
        pred = mlb.transform(predicted_labels)

        micro_f1 = f1_score(gold, pred, average="micro")
        macro_f1 = f1_score(gold, pred, average="macro")

        return {"Micro F1": micro_f1, "Macro F1": macro_f1}
