"""Query-service configuration, separated from CSV import settings."""

import os
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class QueryConfig:
    """Defaults target services running on the same Windows host."""

    neo4j_uri: str = field(default="bolt://localhost:7687", repr=False)
    neo4j_username: str = field(default="neo4j", repr=False)
    neo4j_password: str = field(default="", repr=False)
    neo4j_database: str = "neo4j"
    ollama_url: str = "http://localhost:11434"
    qwen_model: str = "qwen2.5:7b-instruct"
    lightrag_url: str = "http://localhost:9621"
    lightrag_key: str = field(default="", repr=False)
    api_key: str = field(default="", repr=False)
    timezone: str = "Asia/Seoul"
    timeout: float = 120
    graph_chars: int = 16000
    document_chars: int = 8000

    def __post_init__(self) -> None:
        """Validate budgets and timezone before starting clients."""
        import math
        from urllib.parse import urlsplit

        ZoneInfo(self.timezone)
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("QUERY_TIMEOUT must be positive and finite")
        if (
            not 1000 <= self.graph_chars <= 30000
            or not 1000 <= self.document_chars <= 16000
        ):
            raise ValueError("Context character budgets are out of range")
        for url in (self.ollama_url, self.lightrag_url):
            if urlsplit(url).scheme not in ("http", "https"):
                raise ValueError("Service URLs must use HTTP(S)")

    @classmethod
    def from_env(cls) -> "QueryConfig":
        """Read only query-service environment variables; never require a CSV."""
        return cls(
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_username=os.getenv("NEO4J_USERNAME", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", ""),
            neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
            ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/"),
            qwen_model=os.getenv("QWEN_MODEL", "qwen2.5:7b-instruct"),
            lightrag_url=os.getenv("LIGHTRAG_URL", "http://localhost:9621").rstrip("/"),
            lightrag_key=os.getenv("LIGHTRAG_QUERY_API_KEY", ""),
            api_key=os.getenv("MIDDLEWARE_API_KEY", ""),
            timezone=os.getenv("SOURCE_TIMEZONE", "Asia/Seoul"),
            timeout=float(os.getenv("QUERY_TIMEOUT", "120")),
            graph_chars=int(os.getenv("GRAPH_CONTEXT_CHARS", "16000")),
            document_chars=int(os.getenv("DOCUMENT_CONTEXT_CHARS", "8000")),
        )
