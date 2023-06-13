import pandas as pd

from arabic_llm_benchmark.datasets.dataset_base import DatasetBase


class STSTrack2Dataset(DatasetBase):
    def __init__(self, **kwargs):
        # custom_param_1/2 are passed from `dataset_args` in the benchmark
        # config
        super(STSTrack2Dataset, self).__init__(**kwargs)

    def citation(self):
        # This function returns a string with the bib entry for the dataset
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
        # This function loads the data and _must_ return a list of
        # dictionaries, where each dictionary has atleast two keys
        #   "input": this will be sent to the prompt generator
        #   "label": this will be used for evaluation
        # return False
        input_data_path = data_path + "/STS2017.eval.v1.1/STS.input.track2.ar-en.txt"
        gt_data_path = data_path + "/STS2017.gs/STS.gs.track2.ar-en.txt"

        sentences = []
        with open(input_data_path) as f:
            for line in f:
                line = line.rstrip("\r\n")
                sentences.append(line)
        f.close()
        labels = []
        with open(gt_data_path) as f:
            for line in f:
                line = float(line.rstrip("\r\n"))
                labels.append(line)
            f.close()

        return [{"input": s, "label": l} for (s, l) in zip(sentences, labels)]

    def load_train_data(self, train_data_path):
        return []

    def prepare_fewshots(self, target_data, train_data, n_shots):
        return []