import os
import asyncio
from pydantic import BaseModel, Field, PrivateAttr
from typing import Optional, Type
from langchain_community.tools import BaseTool
from langchain.callbacks.manager import (
    CallbackManagerForToolRun,
    AsyncCallbackManagerForToolRun,
)
from gestell import Gestell


class GestellSearchInput(BaseModel):
    """Input for Gestell search tool (query to search the collection)."""

    query: str = Field(
        ...,
        description="Search query to find relevant information in the Gestell collection",
    )


class GestellSearchTool(BaseTool):
    """LangChain tool to search a Gestell collection for relevant documents."""

    name: str = "gestell_search"
    description: str = (
        "Search the Gestell data collection for relevant content by query."
    )
    args_schema: Type[BaseModel] = GestellSearchInput

    # Private attributes won’t be treated as Pydantic fields
    _gestell: Gestell = PrivateAttr()
    _collection_id: str = PrivateAttr()

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        collection_id: Optional[str] = None,
    ):
        # Initialize BaseModel internals
        super().__init__()

        # Load credentials
        api_key = api_key or os.getenv("GESTELL_API_KEY")
        if not api_key:
            raise ValueError(
                "Gestell API key must be provided (via argument or GESTELL_API_KEY env var)."
            )
        self._gestell = Gestell(key=api_key)

        coll = collection_id or os.getenv("GESTELL_COLLECTION_ID")
        if not coll:
            raise ValueError(
                "A Gestell collection_id is required (via argument or GESTELL_COLLECTION_ID env var)."
            )
        self._collection_id = coll

    def _run(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForToolRun = None,
    ) -> str:
        """Synchronously execute a search query against the Gestell collection."""
        try:
            result = asyncio.run(
                self._gestell.query.search(
                    collection_id=self._collection_id, prompt=query
                )
            )
        except Exception as e:
            raise RuntimeError(f"Gestell search failed: {e}")
        return str(result)

    async def _arun(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForToolRun = None,
    ) -> str:
        """Asynchronously execute a search query against the Gestell collection."""
        try:
            result = await self._gestell.query.search(
                collection_id=self._collection_id, prompt=query
            )
        except Exception as e:
            raise RuntimeError(f"Gestell search failed: {e}")
        return str(result)
