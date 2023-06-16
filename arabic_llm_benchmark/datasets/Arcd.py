from arabic_llm_benchmark.datasets.dataset_base import DatasetBase
import json

class ArcdDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(ArcdDataset, self).__init__(**kwargs)

    def citation(self):
        return """
                @misc{mozannar2019neural,
            title={Neural Arabic Question Answering}, 
            author={Hussein Mozannar and Karl El Hajal and Elie Maamary and Hazem Hajj},
            year={2019},
            eprint={1906.05394},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
        }
        }"""

    def get_data_sample(self):
        return {"input": {"context": "context for the questions. Usually a snippet of a wikipedia article", 
                          "question": "question to be answered", 
                          "question_id": "a unique question id"}, 
                
                "label": {"text": "answer text", 'answer_start': 0}}

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
                        {"input":sample, "label": answers}
                    )
        return data