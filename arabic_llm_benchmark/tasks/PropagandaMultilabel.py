from sklearn.metrics import f1_score

from arabic_llm_benchmark.tasks.task_base import TaskBase
from sklearn import preprocessing

class PropagandaMultilabelTask(TaskBase):
    def __init__(self, **kwargs):
        # Get the path to the file listing
        self.techniques_fpath = kwargs["techniques_path"]
        super(PropagandaMultilabelTask, self).__init__(**kwargs)

    def evaluate(self, true_labels, predicted_labels):
        # Handle cases when model fails!
        predicted_labels = [p if p else ["no technique"] for p in predicted_labels]

        # Load a pre-defined list of propaganda techniques
        techniques = []

        with open(self.techniques_fpath, 'r', encoding="utf-8") as f:
            for label in f.readlines():
                label = label.strip()
                if label:
                    techniques.append(label.strip())

        # Binarize classes and use them for multi-label evaluation
        mlb = preprocessing.MultiLabelBinarizer()
        mlb.fit([techniques])
        gold = mlb.transform(true_labels)
        pred = mlb.transform(predicted_labels)

        micro_f1 = f1_score(gold, pred, average="micro")

        return {"Micro F1": micro_f1}
