"""Start with python -m middleware.query_api; interactive requests at /docs."""

import hmac
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Header, HTTPException

from .forecast import TFTClient
from .query_config import QueryConfig
from .query_models import AskRequest
from .rag_service import DocumentRetriever, QwenClient, RagService, UpstreamError


def create_app(
    service: RagService | None = None, config: QueryConfig | None = None
) -> FastAPI:
    """Own client lifecycles and permit dependency injection for offline tests."""
    from dotenv import load_dotenv

    load_dotenv(
        os.getenv("MIDDLEWARE_ENV_FILE", Path(__file__).with_name(".env")),
        override=False,
    )
    config = config or QueryConfig.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Create clients lazily; /health reflects process liveness, not readiness."""
        if service is not None:
            app.state.service = service
            yield
            return
        from neo4j import AsyncGraphDatabase

        from .graph_retriever import GraphRetriever

        driver = AsyncGraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_username, config.neo4j_password),
            connection_timeout=10,
            max_transaction_retry_time=10,
        )
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(config.timeout, connect=10), trust_env=False
            ) as http:
                app.state.service = RagService(
                    GraphRetriever(driver, config),
                    QwenClient(http, config),
                    DocumentRetriever(http, config),
                    config,
                    forecaster=TFTClient(http, config),
                )
                yield
        finally:
            await driver.close()

    app = FastAPI(title="Port-MIS Graph + LightRAG Qwen API", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        """Report liveness without exposing configuration or making upstream calls."""
        return {"status": "alive", "upstreams_verified": False}

    @app.post("/ask")
    async def ask(
        request: AskRequest, x_api_key: str | None = Header(default=None)
    ) -> dict:
        """Run constrained retrieval and answer generation with sanitized errors."""
        if config.api_key and not hmac.compare_digest(x_api_key or "", config.api_key):
            raise HTTPException(status_code=401, detail="Invalid middleware API key")
        try:
            return await app.state.service.ask(request)
        except UpstreamError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from None

    return app


def main() -> None:
    """Run on loopback by default, separately from the existing LightRAG server."""
    import uvicorn

    app = create_app()
    uvicorn.run(
        app,
        host=os.getenv("MIDDLEWARE_HOST", "127.0.0.1"),
        port=int(os.getenv("MIDDLEWARE_PORT", "9622")),
    )


if __name__ == "__main__":
    main()
