import os

from llmebench.models.OpenAI import OpenAIModel


class FastChatModel(OpenAIModel):
    """
    FastChat Model interface. Can be used for models hosted using FastChat
    https://github.com/lm-sys/FastChat

    Accepts all arguments used by `OpenAIModel`, and overrides the arguments listed
    below with FastChat-specific variables.

    See the [https://github.com/lm-sys/FastChat/blob/main/docs/model_support.md](model_support)
    page in FastChat's documentation for supported models and instructions on extending
    to custom models.

    Arguments
    ---------
    api_base : str
        URL where the model is hosted. If not provided, the implementation will look at
        environment variable `FASTCHAT_API_BASE`
    api_key : str
        Authentication token for the API. If not provided, the implementation will derive it
        from environment variable `FASTCHAT_API_KEY`
    model_name : str
        Name of the model to use. If not provided, the implementation will derive it from
        environment variable `FASTCHAT_MODEL`
    """

    def __init__(self, api_base=None, api_key=None, model_name=None, **kwargs):
        api_base = api_base or os.getenv("FASTCHAT_API_BASE")
        api_key = api_key or os.getenv("FASTCHAT_API_KEY")
        model_name = model_name or os.getenv("FASTCHAT_MODEL")
        if api_base is None:
            raise Exception(
                "API url must be provided as model config or environment variable (`FASTCHAT_API_BASE`)"
            )
        if api_key is None:
            raise Exception(
                "API url must be provided as model config or environment variable (`FASTCHAT_API_KEY`)"
            )
        if model_name is None:
            raise Exception(
                "API url must be provided as model config or environment variable (`FASTCHAT_MODEL`)"
            )
        # checks for valid config settings)
        super(FastChatModel, self).__init__(
            api_base=api_base, api_key=api_key, model_name=model_name, **kwargs
        )
