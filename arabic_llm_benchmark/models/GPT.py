from arabic_llm_benchmark.models.model_base import ModelBase

import openai

class GPTModel(ModelBase):
    def __init__(self, api_type, api_base, api_version, api_key, engine_name, **kwargs):
        openai.api_type = api_type
        openai.api_base = api_base
        openai.api_version = api_version
        openai.api_key = api_key

        self.system_message_template = "<|im_start|>system\n{}\n<|im_end|>"
        self.message_template = "\n<|im_start|>{}\n{}\n<|im_end|>"
        self.engine_name = engine_name

        super(GPTModel, self).__init__(retry_exceptions=(openai.InvalidRequestError,openai.error.Timeout), **kwargs)

    # defining a function to create the prompt from the system and user messages
    def create_prompt(self, system_message, messages):
        prompt = self.system_message_template.format(system_message)
        
        for message in messages:
            prompt += self.message_template.format(message['sender'], message['text'])
        prompt += "\n<|im_start|>assistant\n"
        return prompt

    def prompt(self, **kwargs):
        system_message = kwargs["system_message"]
        messages = kwargs["messages"]
        prompt = self.create_prompt(system_message, messages)
        response = openai.Completion.create(
            engine=self.engine_name,
            prompt=prompt,
            temperature=0.7,
            max_tokens=800,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["<|im_end|>"]
        )
        
        return response


