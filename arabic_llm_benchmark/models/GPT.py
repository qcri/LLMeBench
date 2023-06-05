import openai

from arabic_llm_benchmark.models.model_base import ModelBase


class GPTModel(ModelBase):
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

        super(GPTModel, self).__init__(
            retry_exceptions=(openai.error.Timeout,), **kwargs
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
