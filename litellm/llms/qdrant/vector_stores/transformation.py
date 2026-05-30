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

from litellm.llms.base_llm.vector_store.transformation import BaseVectorStoreConfig
from litellm.llms.fastembed.embed import (
    DEFAULT_DENSE_MODEL,
    DEFAULT_SPARSE_MODEL,
    get_fastembed_service,
)
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

    Embeddings are generated in-process via fastembed (no external embedding
    service). Search uses the Qdrant Query API and supports dense-only or
    hybrid (dense + sparse, fused with RRF) retrieval.

    litellm_params:
    - api_key: API key (overrides QDRANT_API_KEY)
    - api_base: Base URL (overrides QDRANT_API_BASE)
    - qdrant_search_mode: 'dense' (default) or 'hybrid' (dense + sparse RRF)
    - fastembed_dense_model: dense model (default: BAAI/bge-small-en-v1.5)
    - fastembed_sparse_model: sparse model (default: prithivida/Splade_PP_en_v1)
    - qdrant_dense_vector_name: named dense vector (default: 'dense')
    - qdrant_sparse_vector_name: named sparse vector (default: 'sparse')
    - qdrant_distance: Distance metric ('Cosine', 'Euclid', 'Dot') default: Cosine
    - qdrant_vector_size: Vector dimension size (default: 384 for BGE-small)
    - qdrant_text_field: Payload field name for text content (default: 'text')
    - qdrant_score_threshold: Minimum score filter
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

    def _build_query_request(
        self,
        vector_store_id: str,
        optional_params: dict,
        litellm_params: dict,
        dense_vec: List[float],
        sparse_vec: Optional[Dict[str, Any]],
        api_base: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """Build a Qdrant Query API request body for dense or hybrid search.

        Dense mode searches the named dense vector directly. Hybrid mode
        prefetches the dense + sparse vectors and fuses them server-side with
        Reciprocal Rank Fusion (RRF).
        """
        collection_name = vector_store_id
        url = f"{api_base}/collections/{collection_name}/points/query"
        limit = optional_params.get("limit", 5)
        mode = litellm_params.get("qdrant_search_mode", "dense")
        dense_name = litellm_params.get("qdrant_dense_vector_name", "dense")
        sparse_name = litellm_params.get("qdrant_sparse_vector_name", "sparse")

        body: Dict[str, Any] = {"limit": limit, "with_payload": True}

        if mode == "hybrid":
            body["prefetch"] = [
                {"query": dense_vec, "using": dense_name, "limit": limit},
                {"query": sparse_vec, "using": sparse_name, "limit": limit},
            ]
            body["query"] = {"fusion": "rrf"}
        else:
            body["query"] = dense_vec
            body["using"] = dense_name

        qdrant_filter = optional_params.get("filter")
        if qdrant_filter:
            body["filter"] = qdrant_filter

        score_threshold = optional_params.get(
            "score_threshold", litellm_params.get("qdrant_score_threshold")
        )
        if score_threshold is not None:
            body["score_threshold"] = score_threshold

        search_params = optional_params.get("search_params")
        if search_params:
            body["params"] = search_params

        return url, body

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
        if isinstance(query, list):
            query = " ".join(query)

        mode = litellm_params.get("qdrant_search_mode", "dense")
        dense_model = litellm_params.get("fastembed_dense_model", DEFAULT_DENSE_MODEL)
        sparse_model = litellm_params.get(
            "fastembed_sparse_model", DEFAULT_SPARSE_MODEL
        )
        service = get_fastembed_service(dense_model, sparse_model)

        dense_vec = service.embed_dense([query])[0]
        sparse_vec = service.embed_sparse([query])[0] if mode == "hybrid" else None

        url, request_body = self._build_query_request(
            vector_store_id=vector_store_id,
            optional_params=vector_store_search_optional_params,
            litellm_params=litellm_params,
            dense_vec=dense_vec,
            sparse_vec=sparse_vec,
            api_base=api_base,
        )

        if extra_body:
            request_body.update(extra_body)

        litellm_logging_obj.model_call_details["input"] = query
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

        mode = litellm_params.get("qdrant_search_mode", "dense")
        dense_model = litellm_params.get("fastembed_dense_model", DEFAULT_DENSE_MODEL)
        sparse_model = litellm_params.get(
            "fastembed_sparse_model", DEFAULT_SPARSE_MODEL
        )
        service = get_fastembed_service(dense_model, sparse_model)

        dense_vec = (await service.aembed_dense([query]))[0]
        sparse_vec = (
            (await service.aembed_sparse([query]))[0] if mode == "hybrid" else None
        )

        url, request_body = self._build_query_request(
            vector_store_id=vector_store_id,
            optional_params=vector_store_search_optional_params,
            litellm_params=litellm_params,
            dense_vec=dense_vec,
            sparse_vec=sparse_vec,
            api_base=api_base,
        )

        if extra_body:
            request_body.update(extra_body)

        litellm_logging_obj.model_call_details["input"] = query
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
            result = response_json.get("result", [])
            # Query API returns {"result": {"points": [...]}}; the legacy search
            # API returns {"result": [...]}.
            if isinstance(result, dict):
                results = result.get("points", [])
            elif isinstance(result, list):
                results = result
            else:
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
        params = vector_store_create_optional_params
        collection_name = params.get("vector_store_name", "default")
        url = f"{api_base}/collections/{collection_name}"

        vector_size = params.get("qdrant_vector_size", 384)
        distance = params.get("qdrant_distance", "Cosine")
        dense_name = params.get("qdrant_dense_vector_name", "dense")
        sparse_name = params.get("qdrant_sparse_vector_name", "sparse")
        mode = params.get("qdrant_search_mode", "dense")

        request_body: Dict[str, Any] = {
            "vectors": {
                dense_name: {
                    "size": vector_size,
                    "distance": distance,
                    "hnsw_config": {"m": 16, "ef_construct": 100},
                }
            }
        }
        if mode == "hybrid":
            request_body["sparse_vectors"] = {sparse_name: {}}

        # Add optional collection config (e.g. quantization, on-disk payload)
        collection_config = params.get("qdrant_collection_config", {})
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
                raise Exception(f"Qdrant collection creation failed: {response_json}")
        except Exception as e:
            raise Exception(f"Failed to parse Qdrant create response: {str(e)}")
