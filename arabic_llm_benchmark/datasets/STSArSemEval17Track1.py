import pandas as pd

from arabic_llm_benchmark.datasets.dataset_base import DatasetBase


class STSArSemEval17Track1Dataset(DatasetBase):
    def __init__(self, **kwargs):
        super(STSArSemEval17Track1Dataset, self).__init__(**kwargs)

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
        return {"input": "الجملة بالعربية", "label": 1.2}

    def load_data(self, data_path):
        input_data_path = data_path + "/STS2017.eval.v1.1/STS.input.track1.ar-ar.txt"
        gt_data_path = data_path + "/STS2017.gs/STS.gs.track1.ar-ar.txt"

        sentences = []
        with open(input_data_path) as f:
            for line in f:
                line = line.rstrip("\r\n")
                sentences.append(line)

        labels = []
        with open(gt_data_path) as f:
            for line in f:
                line = float(line.rstrip("\r\n"))
                labels.append(line)

        return [{"input": s, "label": l} for (s, l) in zip(sentences, labels)]
