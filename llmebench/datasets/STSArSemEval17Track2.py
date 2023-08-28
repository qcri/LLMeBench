from llmebench.datasets.dataset_base import DatasetBase


class STSArSemEval17Track2Dataset(DatasetBase):
    def __init__(self, **kwargs):
        super(STSArSemEval17Track2Dataset, self).__init__(**kwargs)

    def citation(self):
        return """
        @inproceedings{cer2017semeval,
            title={SemEval-2017 Task 1: Semantic Textual Similarity Multilingual and Cross-lingual Focused Evaluation},
            author={Cer, Daniel and Diab, Mona and Agirre, Eneko E and Lopez-Gazpio, I{\~n}igo and Specia, Lucia},
            booktitle={The 11th International Workshop on Semantic Evaluation (SemEval-2017)},
            pages={1--14},
            year={2017}
        }"""

    def get_data_sample(self):
        return {"input": "الجملة بالعربية\tالجملة english", "label": 1.2}

    def load_train_data(self, data_path):
        data = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\r\n").strip()
                _, score, s1, s2 = line.split("\t")

                data.append({"input": s1 + "\t" + s2, "label": float(score)})

        return data

    def load_data(self, data_path):
        # A trick to check if load_data is called for test or train data
        if "sentences_path" in data_path:
            sentences = []
            with open(data_path["sentences_path"], encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\r\n")
                    sentences.append(line)

            labels = []
            with open(data_path["gt_data_path"]) as f:
                for line in f:
                    line = float(line.rstrip("\r\n"))
                    labels.append(line)

            return [{"input": s, "label": l} for (s, l) in zip(sentences, labels)]

        else:
            return self.load_train_data(data_path)