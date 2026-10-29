"""Local-only proxy that connects the existing WebUI to the Port-MIS API.

The Docker WebUI remains the source of all pages, regular LightRAG API
responses, and AIS tracks. Only ``/portmis/ask`` is routed directly to the
host-side middleware, which allows the TFT service to stay bound to the
loopback interface.
"""

from __future__ import annotations

import argparse

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


def create_app(webui_url: str, portmis_url: str) -> FastAPI:
    app = FastAPI(title="LightRAG local WebUI bridge", docs_url=None, redoc_url=None)
    webui_url = webui_url.rstrip("/")
    portmis_url = portmis_url.rstrip("/")

    @app.on_event("startup")
    async def startup() -> None:
        app.state.client = httpx.AsyncClient(follow_redirects=False, timeout=300.0)

    @app.on_event("shutdown")
    async def shutdown() -> None:
        await app.state.client.aclose()

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    )
    async def proxy(request: Request, path: str) -> Response:
        if path == "portmis/ask":
            target = f"{portmis_url}/ask"
        else:
            target = f"{webui_url}/{path}"

        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() != "host"
        }
        upstream = await app.state.client.request(
            request.method,
            target,
            params=request.query_params,
            headers=headers,
            content=await request.body(),
        )
        response_headers = {
            key: value
            for key, value in upstream.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS
            and key.lower() not in {"content-length", "content-encoding"}
        }
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=response_headers,
            media_type=None,
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9624)
    parser.add_argument("--webui-url", default="http://127.0.0.1:9621")
    parser.add_argument("--portmis-url", default="http://127.0.0.1:9622")
    args = parser.parse_args()

    uvicorn.run(
        create_app(args.webui_url, args.portmis_url),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
