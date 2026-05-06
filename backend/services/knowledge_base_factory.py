"""
KnowledgeBase Factory — 🏭 Namespace-Scoped KB Instance Manager

Creates and caches namespace-scoped KnowledgeBase instances.
All instances share the same Pinecone index connection to avoid
redundant connections — only the namespace differs.

Design:
  - Single Pinecone index connection shared across all instances
  - Per-namespace KnowledgeBase instances cached for reuse
  - Thread-safe for concurrent requests with different namespaces
  - Default namespace loaded from config (PINECONE_NAMESPACE)

Usage:
  factory = KnowledgeBaseFactory()
  await factory.initialize()

  # Get default namespace KB
  kb = factory.get()

  # Get tenant-scoped KB
  kb = factory.get("client-abc")
"""

import logging

from services.knowledge_base import KnowledgeBase
from config import pinecone_config

logger = logging.getLogger(__name__)


class KnowledgeBaseFactory:
    """Factory that creates and caches namespace-scoped KnowledgeBase instances.

    Shares the same Pinecone index connection across all instances
    to avoid redundant connections. Each namespace gets its own
    KnowledgeBase instance with scoped upsert/query/delete operations.

    Attributes:
        _default_kb: The default KnowledgeBase instance (from config).
        _cache: Cache of KnowledgeBase instances keyed by namespace.
    """

    def __init__(self) -> None:
        """Initialize the factory with empty cache."""
        self._default_kb: KnowledgeBase | None = None
        self._cache: dict[str, KnowledgeBase] = {}

    async def initialize(self) -> bool:
        """Initialize the default KB and establish the shared index connection.

        Creates the default KnowledgeBase using the namespace from
        config (PINECONE_NAMESPACE), initializes the Pinecone index,
        and caches it.

        Returns:
            True if initialization succeeded, False otherwise.
        """
        self._default_kb = KnowledgeBase(namespace=pinecone_config.namespace)
        success = await self._default_kb.initialize()

        if success:
            self._cache[pinecone_config.namespace] = self._default_kb
            logger.info(
                "KnowledgeBaseFactory initialized — default namespace: '%s'",
                pinecone_config.namespace,
            )

        return success

    def get(self, namespace: str | None = None) -> KnowledgeBase | None:
        """Get a namespace-scoped KnowledgeBase instance.

        Returns a cached instance if one exists for the namespace,
        otherwise creates a new one sharing the same Pinecone index
        connection as the default instance.

        Args:
            namespace: The Pinecone namespace to scope queries to.
                       Uses the config default if None.

        Returns:
            A KnowledgeBase instance scoped to the namespace,
            or None if the factory is not initialized.
        """
        if not self._default_kb:
            logger.warning("KnowledgeBaseFactory not initialized — returning None")
            return None

        ns = namespace or pinecone_config.namespace

        if ns in self._cache:
            return self._cache[ns]

        # Create a new KB instance sharing the same Pinecone connection
        kb = KnowledgeBase(namespace=ns)
        kb._pc = self._default_kb._pc
        kb._index = self._default_kb._index

        self._cache[ns] = kb
        logger.info("Created scoped KnowledgeBase for namespace: '%s'", ns)

        return self._cache[ns]

    @property
    def default(self) -> KnowledgeBase | None:
        """Get the default KnowledgeBase instance (config namespace)."""
        return self._default_kb
