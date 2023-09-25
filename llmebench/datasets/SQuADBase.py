import json

from llmebench.datasets.dataset_base import DatasetBase


class SQuADBase(DatasetBase):
    def __init__(self, **kwargs):
        super(SQuADBase, self).__init__(**kwargs)

    @staticmethod
    def get_data_sample():
        return {
            "input": {
                "context": "context for the questions. Usually a snippet of a wikipedia article",
                "question": "question to be answered",
                "question_id": "a unique question id",
            },
            "label": "answer 1",
        }

    def load_data(self, data_path, no_labels=False):
        data_path = self.resolve_path(data_path)
        data = []

        with open(data_path, "r") as reader:
            dataset = json.load(reader)["data"]

        for article in dataset:
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]
                for qa in paragraph["qas"]:
                    question = qa["question"]
                    question_id = qa["id"]
                    answers = qa["answers"]

                    label = list(map(lambda x: x["text"], qa["answers"]))

                    sample = {
                        "context": context,
                        "question": question,
                        "question_id": question_id,
                    }

                    data.append(
                        {
                            "input": sample,
                            "label": label,
                        }
                    )
        return data
