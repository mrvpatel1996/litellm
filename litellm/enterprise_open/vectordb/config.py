"""
Vector DB Configuration Models.

Pydantic models for configuring Qdrant, Milvus, and pgvector
through the enterprise config.yaml.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class VectorDBBackend(str, Enum):
    """Supported vector database backends."""

    QDRANT = "qdrant"
    MILVUS = "milvus"
    PGVECTOR = "pgvector"


class QdrantConfig(BaseModel):
    """Qdrant vector database configuration."""

    api_base: str = Field(
        ...,
        description="Qdrant API base URL. e.g. https://cluster.qdrant.io or http://localhost:6333",
    )
    api_key: Optional[str] = Field(
        None,
        description="API key for authentication. Not required for local self-hosted.",
    )
    distance: str = Field(
        "Cosine",
        description="Distance metric: Cosine, Euclid, or Dot",
    )
    default_vector_size: int = Field(
        1536,
        description="Default vector dimension size",
    )
    text_field: str = Field(
        "text",
        description="Payload field name for text content",
    )
    timeout: int = Field(
        30,
        description="Request timeout in seconds",
    )
    prefer_grpc: bool = Field(
        False,
        description="Use gRPC instead of REST API",
    )


class MilvusConfig(BaseModel):
    """Milvus vector database configuration."""

    api_base: str = Field(
        ...,
        description="Milvus API base URL. e.g. https://instance.api.gcp-us-west1.zillizcloud.io",
    )
    api_key: Optional[str] = Field(
        None,
        description="API key for authentication",
    )
    db_name: Optional[str] = Field(
        None,
        description="Database name (Milvus 2.3+)",
    )
    text_field: str = Field(
        "text",
        description="Field name for text content in Milvus collection",
    )
    timeout: int = Field(
        30,
        description="Request timeout in seconds",
    )


class PgVectorConfig(BaseModel):
    """pgvector configuration."""

    api_base: str = Field(
        ...,
        description="pgvector-compatible API base URL (litellm-pgvector server)",
    )
    api_key: Optional[str] = Field(
        None,
        description="API key for authentication",
    )
    database_url: Optional[str] = Field(
        None,
        description="Direct PostgreSQL connection string (for direct mode)",
    )
    table_prefix: str = Field(
        "litellm_vectors",
        description="Prefix for vector tables",
    )
    embedding_dim: int = Field(
        1536,
        description="Vector dimension size",
    )


class VectorDBHealthStatus(BaseModel):
    """Health status response for a vector DB backend."""

    backend: VectorDBBackend
    connected: bool
    latency_ms: Optional[float] = None
    collections_count: Optional[int] = None
    error: Optional[str] = None
    version: Optional[str] = None


class VectorDBCollectionInfo(BaseModel):
    """Information about a vector DB collection/index."""

    name: str
    backend: VectorDBBackend
    vector_size: int
    distance: str
    points_count: Optional[int] = None
    status: str = "green"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VectorDBRoutingConfig(BaseModel):
    """
    Configuration for routing requests to different vector DB backends.

    Allows specifying which backend to use per model or per team.
    """

    default_backend: VectorDBBackend = Field(
        VectorDBBackend.QDRANT,
        description="Default vector DB backend",
    )
    model_routing: Dict[str, VectorDBBackend] = Field(
        default_factory=dict,
        description="Map of model name patterns to backends. e.g. {'openai/*': 'qdrant'}",
    )
    team_routing: Dict[str, VectorDBBackend] = Field(
        default_factory=dict,
        description="Map of team IDs to preferred backends",
    )
    fallback_backend: Optional[VectorDBBackend] = Field(
        None,
        description="Fallback backend if primary is unavailable",
    )
    health_check_interval: int = Field(
        60,
        description="Health check interval in seconds",
    )


class EnterpriseVectorDBConfig(BaseModel):
    """
    Full enterprise vector DB configuration.

    Use in config.yaml under enterprise_settings.vectordb:

    Example:
      enterprise_settings:
        vectordb:
          backends:
            qdrant:
              api_base: "https://cluster.qdrant.io"
              api_key: "os.environ/QDRANT_API_KEY"
            milvus:
              api_base: "https://instance.zillizcloud.io"
              api_key: "os.environ/MILVUS_API_KEY"
            pgvector:
              api_base: "http://localhost:8001"
          routing:
            default_backend: qdrant
    """

    backends: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration per backend. Keys: qdrant, milvus, pgvector",
    )
    routing: VectorDBRoutingConfig = Field(
        default_factory=VectorDBRoutingConfig,
        description="Routing configuration",
    )

    def get_qdrant_config(self) -> Optional[QdrantConfig]:
        cfg = self.backends.get("qdrant")
        return QdrantConfig(**cfg) if cfg else None

    def get_milvus_config(self) -> Optional[MilvusConfig]:
        cfg = self.backends.get("milvus")
        return MilvusConfig(**cfg) if cfg else None

    def get_pgvector_config(self) -> Optional[PgVectorConfig]:
        cfg = self.backends.get("pgvector")
        return PgVectorConfig(**cfg) if cfg else None
