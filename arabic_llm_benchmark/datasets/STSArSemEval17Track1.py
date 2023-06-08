import pandas as pd

from arabic_llm_benchmark.datasets.dataset_base import DatasetBase


class STSTrack1Dataset(DatasetBase):
    def __init__(self, **kwargs):
        # custom_param_1/2 are passed from `dataset_args` in the benchmark
        # config
        super(STSTrack1Dataset, self).__init__(**kwargs)

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
        rt = 1
        paths = data_path.split(";")
        data_path = paths[0]
        gt_data_path = paths[1]
        if rt == 0:
            data = pd.read_csv(data_path, sep="\t")
            return data
        else:
            sentences = []
            with open(data_path) as f:
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
