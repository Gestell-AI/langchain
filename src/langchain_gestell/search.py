import os
import asyncio
from typing import Optional, Type, List, Dict
from pydantic import BaseModel, Field, PrivateAttr
from langchain_community.tools import BaseTool
from langchain.callbacks.manager import (
    CallbackManagerForToolRun,
    AsyncCallbackManagerForToolRun,
)
from gestell import Gestell
from langchain_gestell.util import validate_collection_id


class GestellSearchInput(BaseModel):
    """
    Input schema for the Gestell search tool, defining all optional
    parameters for the `search` endpoint.
    """

    query: str = Field(..., description="The search query or prompt text to execute.")
    collection_id: Optional[str] = Field(
        None,
        description="The UUID of the collection to query (or set via GESTELL_COLLECTION_ID).",
    )
    category_id: Optional[str] = Field(
        None, description="An optional category UUID to filter results by."
    )
    method: Optional[str] = Field(
        None,
        description="The search method to use: choose between 'fast', 'normal', and 'precise'.",
    )
    search_type: Optional[str] = Field(
        None,
        description="The search type to specify: choose between 'keywords', 'phrase', and 'summary'.",
    )
    vector_depth: Optional[int] = Field(
        None, description="Depth of vector-based retrieval."
    )
    node_depth: Optional[int] = Field(
        None, description="Depth of node-based retrieval."
    )
    max_queries: Optional[int] = Field(
        None, description="Maximum number of sub-queries to run."
    )
    max_results: Optional[int] = Field(
        None, description="Maximum number of results to return."
    )
    include_content: Optional[bool] = Field(
        None, description="Whether to include the full content in each result."
    )
    include_edges: Optional[bool] = Field(
        None, description="Whether to include edge metadata in results."
    )


class GestellSearchTool(BaseTool):
    """
    LangChain tool to search a Gestell collection for relevant documents,
    exposing all supported query parameters.
    """

    name: str = "gestell_search"
    description: str = (
        "Search the Gestell data collection for relevant content by query, "
        "with full control over retrieval parameters."
    )
    args_schema: Type[BaseModel] = GestellSearchInput

    _gestell: Gestell = PrivateAttr()
    _collection_id: str = PrivateAttr()

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        collection_id: Optional[str] = None,
    ):
        super().__init__()
        api_key = api_key or os.getenv("GESTELL_API_KEY")
        if not api_key:
            raise ValueError(
                "Gestell API key must be provided (via argument or GESTELL_API_KEY env var)."
            )
        self._gestell = Gestell(key=api_key)

        coll = collection_id or os.getenv("GESTELL_COLLECTION_ID")
        self._collection_id = coll

    def _run(
        self,
        query: str,
        collection_id: Optional[str] = None,
        category_id: Optional[str] = None,
        method: Optional[str] = None,
        search_type: Optional[str] = None,
        vector_depth: Optional[int] = None,
        node_depth: Optional[int] = None,
        max_queries: Optional[int] = None,
        max_results: Optional[int] = None,
        include_content: Optional[bool] = None,
        include_edges: Optional[bool] = None,
        run_manager: CallbackManagerForToolRun = None,
    ) -> List[Dict]:
        """
        Synchronously execute a search query on the Gestell collection.
        """
        try:
            resp = asyncio.run(
                self._gestell.query.search(
                    collection_id=validate_collection_id(
                        self._collection_id, collection_id
                    ),
                    prompt=query,
                    category_id=category_id,
                    method=method,
                    type=search_type,
                    vectorDepth=vector_depth,
                    nodeDepth=node_depth,
                    maxQueries=max_queries,
                    maxResults=max_results,
                    includeContent=include_content,
                    includeEdges=include_edges,
                )
            )
        except Exception as e:
            raise RuntimeError(f"Gestell search failed: {e}")
        return resp.result

    async def _arun(
        self,
        query: str,
        collection_id: Optional[str] = None,
        category_id: Optional[str] = None,
        method: Optional[str] = None,
        search_type: Optional[str] = None,
        vector_depth: Optional[int] = None,
        node_depth: Optional[int] = None,
        max_queries: Optional[int] = None,
        max_results: Optional[int] = None,
        include_content: Optional[bool] = None,
        include_edges: Optional[bool] = None,
        run_manager: AsyncCallbackManagerForToolRun = None,
    ) -> List[Dict]:
        """
        Asynchronously execute a search query on the Gestell collection.
        """
        try:
            resp = await self._gestell.query.search(
                collection_id=self.validate_collection_id(collection_id),
                prompt=query,
                category_id=category_id,
                method=method,
                type=search_type,
                vectorDepth=vector_depth,
                nodeDepth=node_depth,
                maxQueries=max_queries,
                maxResults=max_results,
                includeContent=include_content,
                includeEdges=include_edges,
            )
        except Exception as e:
            raise RuntimeError(f"Gestell search failed: {e}")
        return resp.result
