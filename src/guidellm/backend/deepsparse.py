from typing import Any, AsyncGenerator, List, Optional

from deepsparse import Pipeline
from loguru import logger
from transformers import AutoTokenizer

from guidellm.backend import Backend, GenerativeResponse
from guidellm.core import TextGenerationRequest

__all__ = ["DeepsparseBackend"]


@Backend.register("deepsparse")
class DeepsparseBackend(Backend):
    """
    An Deepsparse backend implementation for the generative AI result.
    """

    def __init__(self, model: Optional[str] = None, **request_args):
        self.request_args = request_args
        self.pipeline: Pipeline = Pipeline.create(
            task="sentiment-analysis",
            model_path=model or self.default_model,
        )

    async def make_request(
        self, request: TextGenerationRequest
    ) -> AsyncGenerator[GenerativeResponse, None]:
        """
        Make a request to the Deepsparse Python API client.

        :param request: The result request to submit.
        :type request: TextGenerationRequest
        :return: An iterator over the generative responses.
        :rtype: Iterator[GenerativeResponse]
        """

        logger.debug(
            f"Making request to Deepsparse backend with prompt: {request.prompt}"
        )

        token_count = 0
        for response in self.pipeline.generations:
            if not (token := response.text):
                yield GenerativeResponse(
                    type_="final",
                    prompt=request.prompt,
                    prompt_token_count=request.prompt_token_count,
                    output_token_count=token_count,
                )
                break
            else:
                token_count += 1
                yield GenerativeResponse(
                    type_="token_iter",
                    add_token=token,
                    prompt=request.prompt,
                    prompt_token_count=request.prompt_token_count,
                    output_token_count=token_count,
                )


    def available_models(self) -> List[str]:
        """
        Get the available models for the backend.

        :return: A list of available models.
        :rtype: List[str]
        """

        # WARNING: The default model from the documentation is defined here

        return [
            "zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none"
        ]

    def model_tokenizer(self, model: str) -> Optional[Any]:
        """
        Get the tokenizer for a model.

        :param model: The model to get the tokenizer for.
        :type model: str
        :return: The tokenizer for the model, or None if it cannot be created.
        :rtype: Optional[Any]
        """

        try:
            tokenizer = AutoTokenizer.from_pretrained(model)
            logger.info(f"Tokenizer created for model: {model}")
            return tokenizer
        except Exception as err:  # noqa: BLE001
            logger.warning(f"Could not create tokenizer for model {model}: {err}")
            return None

    def _token_count(self, text: str) -> int:
        token_count = len(text.split())
        logger.debug(f"Token count for text '{text}': {token_count}")
        return token_count
