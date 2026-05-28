"""
Enterprise Vector DB Management.

Unified management layer for multiple vector database backends:
- Qdrant (Cloud + Self-hosted)
- Milvus (Zilliz Cloud + Self-hosted)
- pgvector (PostgreSQL + pgvector extension)

Provides:
- Centralized vector store configuration
- Health monitoring endpoints
- Connection pooling and management
- Collection/index lifecycle management
"""

from litellm.enterprise_open.vectordb.router import vectordb_router

__all__ = ["vectordb_router"]
