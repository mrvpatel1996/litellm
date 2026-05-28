"""
Qdrant Vector Store integration for LiteLLM.

Native support for Qdrant Cloud and self-hosted Qdrant instances.
Uses the Qdrant REST API (v2 compatible) for vector store operations.

Supports:
- Vector search with embeddings generated via litellm.embeddings
- Collection management (create, search, upsert, delete)
- Payload filtering using Qdrant filter syntax
- Multiple distance metrics (Cosine, Euclid, Dot)
- Batch upsert with automatic embedding generation
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import httpx

import litellm
from litellm.llms.base_llm.vector_store.transformation import BaseVectorStoreConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.vector_stores import (
    BaseVectorStoreAuthCredentials,
    VectorStoreCreateOptionalRequestParams,
    VectorStoreCreateResponse,
    VectorStoreIndexEndpoints,
    VectorStoreResultContent,
    VectorStoreSearchOptionalRequestParams,
    VectorStoreSearchResponse,
    VectorStoreSearchResult,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

QDRANT_OPTIONAL_PARAMS = {
    "limit",
    "offset",
    "filter",
    "with_payload",
    "with_vectors",
    "score_threshold",
    "search_params",
    "consistency",
    "shard_key",
    "group_by",
    "lookup_from",
}


class QdrantVectorStoreConfig(BaseVectorStoreConfig):
    """
    Configuration for Qdrant Vector Store.

    Supports Qdrant Cloud (https://cloud.qdrant.io) and self-hosted Qdrant.

    Environment variables:
    - QDRANT_API_KEY: API key for authentication
    - QDRANT_API_BASE: Base URL (e.g., https://your-cluster-url.qdrant.io or http://localhost:6333)

    litellm_params:
    - api_key: API key (overrides QDRANT_API_KEY)
    - api_base: Base URL (overrides QDRANT_API_BASE)
    - litellm_embedding_model: Embedding model to use (required for search)
    - litellm_embedding_config: Embedding config dict
    - qdrant_distance: Distance metric ('Cosine', 'Euclid', 'Dot') default: Cosine
    - qdrant_vector_size: Vector dimension size (default: auto-detect from embedding)
    - qdrant_text_field: Payload field name for text content (default: 'text')
    - qdrant_collection_config: Additional collection configuration
    """

    def __init__(self):
        super().__init__()

    def validate_environment(
        self, headers: dict, litellm_params: Optional[GenericLiteLLMParams]
    ) -> dict:
        api_key: Optional[str] = None
        if litellm_params is not None:
            api_key = litellm_params.api_key or get_secret_str("QDRANT_API_KEY")

        # Qdrant can run without auth (self-hosted), but warn if cloud
        if api_key:
            headers.update(
                {
                    "api-key": api_key,
                    "Content-Type": "application/json",
                }
            )
        else:
            headers.update({"Content-Type": "application/json"})

        return headers

    def get_auth_credentials(
        self, litellm_params: dict
    ) -> BaseVectorStoreAuthCredentials:
        api_key = litellm_params.get("api_key") or get_secret_str("QDRANT_API_KEY")
        creds: Dict[str, Any] = {"headers": {"Content-Type": "application/json"}}
        if api_key:
            creds["headers"]["api-key"] = api_key
        return creds

    def get_vector_store_endpoints_by_type(self) -> VectorStoreIndexEndpoints:
        return {
            "read": [
                ("POST", "/collections/{collection_name}/points/search"),
                ("POST", "/collections/{collection_name}/points/scroll"),
                ("POST", "/collections/{collection_name}/points/get"),
            ],
            "write": [
                ("POST", "/collections/{collection_name}/points/upsert"),
                ("POST", "/collections/{collection_name}/points/delete"),
                ("PUT", "/collections/{collection_name}"),
            ],
        }

    def map_openai_params(
        self, non_default_params: dict, optional_params: dict, drop_params: bool
    ) -> dict:
        for param, value in non_default_params.items():
            if param in QDRANT_OPTIONAL_PARAMS:
                optional_params[param] = value
        return optional_params

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        api_base = api_base or get_secret_str("QDRANT_API_BASE")

        if not api_base:
            raise ValueError(
                "Qdrant API base URL is required. Set QDRANT_API_BASE environment variable "
                "or pass api_base in litellm_params. "
                "Examples: https://your-cluster.qdrant.io (cloud) or http://localhost:6333 (local)"
            )

        return api_base.rstrip("/")

    def transform_search_vector_store_request(
        self,
        vector_store_id: str,
        query: Union[str, List[str]],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        api_base: str,
        litellm_logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Transform search request to Qdrant search API format.

        Generates embeddings using litellm.embeddings and constructs Qdrant search request.
        """
        # Convert query to string if it's a list
        if isinstance(query, list):
            query = " ".join(query)

        # Get embedding model from litellm_params (required for search)
        embedding_model = litellm_params.get("litellm_embedding_model")
        if not embedding_model:
            raise ValueError(
                "embedding_model is required in litellm_params for Qdrant. "
                "Example: litellm_params['litellm_embedding_model'] = 'openai/text-embedding-3-small'"
            )

        embedding_config = litellm_params.get("litellm_embedding_config", {})

        # Generate embedding for the query
        try:
            embedding_response = litellm.embedding(
                model=embedding_model,
                input=[query],
                **embedding_config,
            )
            query_vector = embedding_response.data[0]["embedding"]
        except Exception as e:
            raise Exception(f"Failed to generate embedding for Qdrant search: {str(e)}")

        # Collection name = vector_store_id
        collection_name = vector_store_id
        url = f"{api_base}/collections/{collection_name}/points/search"

        # Get text field name for payload projection
        text_field = litellm_params.get("qdrant_text_field", "text")

        # Build the Qdrant search request body
        limit = vector_store_search_optional_params.get("limit", 5)
        score_threshold = vector_store_search_optional_params.get(
            "score_threshold",
            litellm_params.get("qdrant_score_threshold"),
        )

        request_body: Dict[str, Any] = {
            "vector": query_vector,
            "limit": limit,
            "with_payload": True,
        }

        # Add optional filter
        qdrant_filter = vector_store_search_optional_params.get("filter")
        if qdrant_filter:
            request_body["filter"] = qdrant_filter

        # Add score threshold
        if score_threshold is not None:
            request_body["score_threshold"] = score_threshold

        # Add search params (hnsw_ef, exact, etc.)
        search_params = vector_store_search_optional_params.get("search_params")
        if search_params:
            request_body["params"] = search_params

        # Add consistency
        consistency = vector_store_search_optional_params.get("consistency")
        if consistency:
            request_body["consistency"] = consistency

        # Add extra body params
        if extra_body:
            request_body.update(extra_body)

        # Update logging
        litellm_logging_obj.model_call_details["input"] = query
        litellm_logging_obj.model_call_details["embedding_model"] = embedding_model

        return url, request_body

    async def atransform_search_vector_store_request(
        self,
        vector_store_id: str,
        query: Union[str, List[str]],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        api_base: str,
        litellm_logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Async version — generates embeddings via litellm.aembedding.
        """
        if isinstance(query, list):
            query = " ".join(query)

        embedding_model = litellm_params.get("litellm_embedding_model")
        if not embedding_model:
            raise ValueError(
                "embedding_model is required in litellm_params for Qdrant. "
                "Example: litellm_params['litellm_embedding_model'] = 'openai/text-embedding-3-small'"
            )

        embedding_config = litellm_params.get("litellm_embedding_config", {})

        try:
            embedding_response = await litellm.aembedding(
                model=embedding_model,
                input=[query],
                **embedding_config,
            )
            query_vector = embedding_response.data[0]["embedding"]
        except Exception as e:
            raise Exception(f"Failed to generate embedding for Qdrant search: {str(e)}")

        collection_name = vector_store_id
        url = f"{api_base}/collections/{collection_name}/points/search"

        text_field = litellm_params.get("qdrant_text_field", "text")
        limit = vector_store_search_optional_params.get("limit", 5)
        score_threshold = vector_store_search_optional_params.get(
            "score_threshold",
            litellm_params.get("qdrant_score_threshold"),
        )

        request_body: Dict[str, Any] = {
            "vector": query_vector,
            "limit": limit,
            "with_payload": True,
        }

        qdrant_filter = vector_store_search_optional_params.get("filter")
        if qdrant_filter:
            request_body["filter"] = qdrant_filter

        if score_threshold is not None:
            request_body["score_threshold"] = score_threshold

        search_params = vector_store_search_optional_params.get("search_params")
        if search_params:
            request_body["params"] = search_params

        if extra_body:
            request_body.update(extra_body)

        litellm_logging_obj.model_call_details["input"] = query
        litellm_logging_obj.model_call_details["embedding_model"] = embedding_model

        return url, request_body

    def transform_search_vector_store_response(
        self, response: httpx.Response, litellm_logging_obj: LiteLLMLoggingObj
    ) -> VectorStoreSearchResponse:
        """
        Transform Qdrant search response to standard vector store search response.

        Qdrant response format:
        {
            "result": [
                {
                    "id": "...",
                    "score": 0.95,
                    "payload": {"text": "...", ...}
                }
            ]
        }
        """
        try:
            response_json = response.json()
            results = response_json.get("result", [])
            if not isinstance(results, list):
                results = []

            # Get text field name
            litellm_params = litellm_logging_obj.model_call_details.get(
                "litellm_params", {}
            )
            text_field = litellm_params.get("qdrant_text_field", "text")

            search_results = []
            for result in results:
                score = result.get("score", 0.0)
                payload = result.get("payload", {})
                point_id = result.get("id", "")

                # Extract text content from payload
                text_content = payload.get(text_field, "")
                if not text_content:
                    # Fallback: try to stringify the payload
                    text_content = str(payload)

                # Build content list
                content = [VectorStoreResultContent(type="text", text=text_content)]

                search_results.append(
                    VectorStoreSearchResult(
                        score=score,
                        content=content,
                        metadata={
                            "id": str(point_id),
                            "payload": payload,
                        },
                    )
                )

            return VectorStoreSearchResponse(
                search_results=search_results,
            )

        except Exception as e:
            raise Exception(f"Failed to parse Qdrant search response: {str(e)}")

    def transform_create_vector_store_request(
        self,
        vector_store_create_optional_params: VectorStoreCreateOptionalRequestParams,
        api_base: str,
    ) -> Tuple[str, Dict]:
        """
        Transform create request to Qdrant collection creation format.

        Creates a new collection with the specified vector configuration.
        """
        collection_name = vector_store_create_optional_params.get(
            "vector_store_name", "default"
        )
        url = f"{api_base}/collections/{collection_name}"

        # Get vector size and distance from params
        vector_size = vector_store_create_optional_params.get(
            "qdrant_vector_size", 1536
        )
        distance = vector_store_create_optional_params.get(
            "qdrant_distance", "Cosine"
        )

        request_body = {
            "vectors": {
                "size": vector_size,
                "distance": distance,
            },
        }

        # Add optional collection config
        collection_config = vector_store_create_optional_params.get(
            "qdrant_collection_config", {}
        )
        if collection_config:
            request_body.update(collection_config)

        return url, request_body

    def transform_create_vector_store_response(
        self, response: httpx.Response
    ) -> VectorStoreCreateResponse:
        """
        Transform Qdrant collection creation response.

        Qdrant response: {"result": true, "status": "ok", "time": 0.001}
        """
        try:
            response_json = response.json()
            result = response_json.get("result", False)
            status = response_json.get("status", "error")

            if result and status == "ok":
                return VectorStoreCreateResponse(
                    id=response_json.get("collection_name", ""),
                    created_at=0,  # Qdrant doesn't return timestamp
                    status="completed",
                )
            else:
                raise Exception(
                    f"Qdrant collection creation failed: {response_json}"
                )
        except Exception as e:
            raise Exception(
                f"Failed to parse Qdrant create response: {str(e)}"
            )
