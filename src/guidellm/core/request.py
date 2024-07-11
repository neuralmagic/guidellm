import uuid
from typing import Dict, Optional, Any

from guidellm.core.serializable import Serializable


class TextGenerationRequest(Serializable):
    """
    A class to represent a text generation request for generative AI workloads.
    """

    id: str
    prompt: str
    prompt_token_count: Optional[int]
    generated_token_count: Optional[int]
    params: Dict[str, Any]

    def __init__(
        self,
        prompt: str,
        prompt_token_count: Optional[int] = None,
        generated_token_count: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
    ):
        super().__init__(
            id=str(uuid.uuid4()) if id is None else id,
            prompt=prompt,
            prompt_token_count=prompt_token_count,
            generated_token_count=generated_token_count,
            params=params or {},
        )
