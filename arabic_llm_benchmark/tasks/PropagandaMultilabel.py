import itertools

from sklearn import preprocessing
from sklearn.metrics import f1_score

from arabic_llm_benchmark.tasks.task_base import TaskBase


class PropagandaMultilabelTask(TaskBase):
    def __init__(self, **kwargs):
        # Get the path to the file listing the target techniques
        self.techniques_fpath = kwargs["techniques_path"]
        super(PropagandaMultilabelTask, self).__init__(**kwargs)

    def evaluate(self, true_labels, predicted_labels):
        # Handle cases when model fails!
        # Flatten true labels as it is a list of lists
        predicted_labels = [
            p
            if p
            else self.get_random_prediction(
                set(itertools.chain.from_iterable(true_labels))
            )
            for p in predicted_labels
        ]

        # Load a pre-defined list of propaganda techniques
        with open(self.techniques_fpath, "r", encoding="utf-8") as f:
            techniques = [label.strip() for label in f.readlines()]

        # Binarize labels and use them for multi-label evaluation
        mlb = preprocessing.MultiLabelBinarizer()
        mlb.fit([techniques])
        gold = mlb.transform(true_labels)
        pred = mlb.transform(predicted_labels)

        micro_f1 = f1_score(gold, pred, average="micro")

        return {"Micro F1": micro_f1}
