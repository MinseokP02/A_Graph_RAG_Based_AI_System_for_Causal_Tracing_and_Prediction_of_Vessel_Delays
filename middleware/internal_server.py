"""Start the image's server with the optional internal retrieval hook."""

import runpy

from lightrag import LightRAG
from lightrag.portmis import internal_query_enabled, query_with_portmis


def main():
    original = LightRAG.aquery_llm

    async def integrated(self, query, param=None, system_prompt=None, **kwargs):
        from lightrag import QueryParam

        param = param if param is not None else QueryParam()
        if internal_query_enabled() and param.mode != "bypass":
            return await query_with_portmis(self, query, param, system_prompt)
        return await original(self, query, param, system_prompt, **kwargs)

    LightRAG.aquery_llm = integrated
    runpy.run_module("lightrag.api.lightrag_server", run_name="__main__")


if __name__ == "__main__":
    main()
