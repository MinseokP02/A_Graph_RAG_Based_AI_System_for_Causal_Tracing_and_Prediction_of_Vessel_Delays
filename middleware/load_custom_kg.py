"""Offline import using the server's own storage and embedding configuration.

Stop the normal server before running this module against the same storage.
"""

import asyncio
import json
from pathlib import Path


async def main():
    from lightrag.api import lightrag_server as server
    from lightrag.api.config import global_args

    captured = []
    original = server.LightRAG

    def construct(*args, **kwargs):
        rag = original(*args, **kwargs)
        captured.append(rag)
        return rag

    server.LightRAG = construct
    app = server.create_app(global_args)
    async with app.router.lifespan_context(app):
        rag = captured[0]
        payload = json.loads(
            Path("/app/data/inputs/portmis_custom_kg.json").read_text(encoding="utf-8")
        )
        await rag.ainsert_custom_kg(payload, full_doc_id="portmis-neo4j-snapshot")
        Path("/app/data/rag_storage/portmis_custom_kg_import.json").write_text(
            json.dumps({k: len(v) for k, v in payload.items()}), encoding="utf-8"
        )
        print("PORTMIS_CUSTOM_KG_IMPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
