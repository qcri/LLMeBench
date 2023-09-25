import os

from llmebench.models.OpenAI import OpenAIModel


class FastChatModel(OpenAIModel):
    def __init__(self, api_base=None, api_key=None, model_name=None, **kwargs):
        api_base = api_base or os.getenv("OPENAI_API_BASE")
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        model_name = model_name or os.getenv("OPENAI_MODEL")
        if api_base is None:
            raise Exception(
                "API url must be provided as model config or environment variable (`OPENAI_API_BASE`)"
            )
        if api_key is None:
            raise Exception(
                "API url must be provided as model config or environment variable (`OPENAI_API_KEY`)"
            )
        if model_name is None:
            raise Exception(
                "API url must be provided as model config or environment variable (`OPENAI_MODEL`)"
            )
        # checks for valid config settings)
        super(FastChatModel, self).__init__(
            api_base=api_base, api_key=api_key, model_name=model_name, **kwargs
        )
