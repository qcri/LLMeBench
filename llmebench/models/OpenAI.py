import openai

from llmebench.models.model_base import ModelBase


class LegacyOpenAIModel(ModelBase):
    def __init__(
        self,
        api_type,
        api_base,
        api_version,
        api_key,
        engine_name,
        temperature=0,
        top_p=0.95,
        max_tokens=800,
        frequency_penalty=0,
        presence_penalty=0,
        **kwargs
    ):
        # API parameters
        openai.api_type = api_type
        openai.api_base = api_base
        openai.api_version = api_version
        openai.api_key = api_key
        self.engine_name = engine_name

        # GPT parameters
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

        self.system_message_template = "<|im_start|>system\n{}\n<|im_end|>"
        self.message_template = "\n<|im_start|>{}\n{}\n<|im_end|>"

        super(LegacyOpenAIModel, self).__init__(
            retry_exceptions=(openai.error.Timeout, openai.error.RateLimitError),
            **kwargs
        )

    # defining a function to create the prompt from the system and user messages
    def create_prompt(self, system_message, messages):
        prompt = self.system_message_template.format(system_message)

        for message in messages:
            prompt += self.message_template.format(message["sender"], message["text"])
        prompt += "\n<|im_start|>assistant\n"
        return prompt

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
        system_message = processed_input["system_message"]
        messages = processed_input["messages"]
        prompt = self.create_prompt(system_message, messages)
        response = openai.Completion.create(
            engine=self.engine_name,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            stop=["<|im_end|>"],
        )

        return response


class OpenAIModel(ModelBase):
    def __init__(
        self,
        api_type,
        api_base,
        api_version,
        api_key,
        engine_name,
        temperature=0,
        top_p=0.95,
        max_tokens=800,
        frequency_penalty=0,
        presence_penalty=0,
        **kwargs
    ):
        # API parameters
        openai.api_type = api_type
        openai.api_base = api_base
        openai.api_version = api_version
        openai.api_key = api_key
        self.engine_name = engine_name

        # GPT parameters
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

        super(OpenAIModel, self).__init__(
            retry_exceptions=(openai.error.Timeout, openai.error.RateLimitError),
            **kwargs
        )

    def summarize_response(self, response):
        if (
            "choices" in response
            and isinstance(response["choices"], list)
            and len(response["choices"]) > 0
            and "message" in response["choices"][0]
            and "content" in response["choices"][0]["message"]
            and response["choices"][0]["message"]["role"] == "assistant"
        ):
            return response["choices"][0]["message"]["content"]

        return None

    def prompt(self, processed_input):
        response = openai.ChatCompletion.create(
            engine=self.engine_name,
            messages=processed_input,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            stop=None,
        )

        return response
