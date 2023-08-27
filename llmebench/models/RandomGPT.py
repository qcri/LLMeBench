import random

from llmebench.models.model_base import ModelBase


class GPTResponseMock(dict):
    def __init__(self, text):
        self.choices = [{"text": text}]
        dict.__init__(self, choices=self.choices)


class RandomGPTException(Exception):
    pass


class RandomGPTModel(ModelBase):
    def __init__(self, class_labels, **kwargs):
        self.class_labels = class_labels

        super(RandomGPTModel, self).__init__(
            retry_exceptions=(RandomGPTException,), **kwargs
        )

    def summarize_response(self, response):
        if (
            "choices" in response
            and isinstance(response["choices"], list)
            and len(response["choices"]) > 0
            and "text" in response["choices"][0]
        ):
            return response["choices"][0]["text"]

        return None

    def prompt(self, processed_input):
        if random.random() < 0.5:
            raise RandomGPTException()
        return GPTResponseMock(random.choice(self.class_labels))
