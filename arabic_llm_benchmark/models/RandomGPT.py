from arabic_llm_benchmark.models.model_base import ModelBase

import random

class GPTResponseMock(dict):
    def __init__(self, text):
        self.choices = [{"text": text}]
        dict.__init__(self, choices=self.choices)

class RandomGPTModel(ModelBase):
    def __init__(self, class_labels, **kwargs):
        self.class_labels = class_labels

        super(RandomGPTModel, self).__init__(**kwargs)

    def prompt(self, **kwargs):
        return GPTResponseMock(random.choice(self.class_labels))
