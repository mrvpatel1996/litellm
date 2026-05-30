"""``litellm.embedding`` handler for the in-process fastembed provider (dense)."""

from typing import List

from litellm.llms.fastembed.embed import get_fastembed_service
from litellm.types.utils import EmbeddingResponse, Usage


def fastembed_embedding(
    model: str,
    input: List[str],
    model_response: EmbeddingResponse,
) -> EmbeddingResponse:
    """Generate dense embeddings in-process via fastembed.

    ``model`` arrives without the ``fastembed/`` prefix (stripped by
    ``get_llm_provider``), e.g. ``"BAAI/bge-small-en-v1.5"``.
    """
    if isinstance(input, str):
        input = [input]

    service = get_fastembed_service(dense_model=model)
    vectors = service.embed_dense(input)

    model_response.data = [
        {"object": "embedding", "index": i, "embedding": vec}
        for i, vec in enumerate(vectors)
    ]
    model_response.model = model
    model_response.object = "list"
    # fastembed runs locally and does not report token usage.
    model_response.usage = Usage(prompt_tokens=0, total_tokens=0)
    return model_response
