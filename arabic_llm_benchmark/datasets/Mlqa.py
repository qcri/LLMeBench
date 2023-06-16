from arabic_llm_benchmark.datasets.dataset_base import DatasetBase
import json

class MlqaDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(MlqaDataset, self).__init__(**kwargs)

    def citation(self):
        return """ @article{lewis2019mlqa,
        title=MLQA: Evaluating Cross-lingual Extractive Question Answering,
        author={Lewis, Patrick and Ouguz, Barlas and Rinott, Ruty and Riedel, Sebastian and Schwenk, Holger},
        journal={arXiv preprint arXiv:1910.07475},
        year={2019} """

    def get_data_sample(self):
        return {"input": {"context": "context for the questions. Usually a snippet of a wikipedia article", 
                          "question": "question to be answered", 
                          "question_id": "a unique question id"}, 
                
                "label": "answer text"}

    def load_data(self, data_path, no_labels=False):
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

                    sample = {"context":context, "question": question, "question_id": question_id}

                    data.append( 
                        {"input":sample, "label": answers[0]["text"]}
                    )
        return data