import os
import asyncio
from typing import Optional, Type, List
from pydantic import BaseModel, Field, PrivateAttr
from langchain_community.tools import BaseTool
from langchain.callbacks.manager import (
    CallbackManagerForToolRun,
    AsyncCallbackManagerForToolRun,
)
from gestell import Gestell

from langchain_gestell.util import validate_collection_id


class GestellPromptInput(BaseModel):
    """
    Input schema for the Gestell prompt tool, defining all optional
    parameters for the `prompt` endpoint.
    """

    prompt: str = Field(..., description="The prompt or query text to execute.")
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
    template: Optional[str] = Field(
        None, description="A prompt template to apply before sending."
    )
    cot: Optional[bool] = Field(
        None, description="Whether to enable chain-of-thought reasoning."
    )
    messages: Optional[List[dict]] = Field(
        None, description="The message history for streaming chat contexts."
    )


class GestellPromptTool(BaseTool):
    """
    LangChain tool to prompt a Gestell collection for an answer using its data,
    exposing all supported query parameters.
    """

    name: str = "gestell_prompt"
    description: str = (
        "Use the Gestell collection to answer a question or fulfill an instruction, "
        "with full control over retrieval parameters."
    )
    args_schema: Type[BaseModel] = GestellPromptInput

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
        prompt: str,
        collection_id: Optional[str] = None,
        category_id: Optional[str] = None,
        method: Optional[str] = None,
        search_type: Optional[str] = None,
        vector_depth: Optional[int] = None,
        node_depth: Optional[int] = None,
        max_queries: Optional[int] = None,
        max_results: Optional[int] = None,
        template: Optional[str] = None,
        cot: Optional[bool] = None,
        messages: Optional[List[dict]] = None,
        run_manager: CallbackManagerForToolRun = None,
    ) -> str:
        """
        Synchronously execute a prompt query on the Gestell collection.
        """
        try:
            response = asyncio.run(
                self._gestell.query.prompt(
                    collection_id=validate_collection_id(
                        self._collection_id, collection_id
                    ),
                    prompt=prompt,
                    category_id=category_id,
                    method=method,
                    type=search_type,
                    vectorDepth=vector_depth,
                    nodeDepth=node_depth,
                    maxQueries=max_queries,
                    maxResults=max_results,
                    template=template,
                    cot=cot,
                    messages=messages,
                )
            )
        except Exception as e:
            raise RuntimeError(f"Gestell prompt failed: {e}")
        return str(response)

    async def _arun(
        self,
        prompt: str,
        collection_id: Optional[str] = None,
        category_id: Optional[str] = None,
        method: Optional[str] = None,
        search_type: Optional[str] = None,
        vector_depth: Optional[int] = None,
        node_depth: Optional[int] = None,
        max_queries: Optional[int] = None,
        max_results: Optional[int] = None,
        template: Optional[str] = None,
        cot: Optional[bool] = None,
        messages: Optional[List[dict]] = None,
        run_manager: AsyncCallbackManagerForToolRun = None,
    ) -> str:
        """
        Asynchronously execute a prompt query on the Gestell collection.
        """
        try:
            response = await self._gestell.query.prompt(
                collection_id=self.validate_collection_id(collection_id),
                prompt=prompt,
                category_id=category_id,
                method=method,
                type=search_type,
                vectorDepth=vector_depth,
                nodeDepth=node_depth,
                maxQueries=max_queries,
                maxResults=max_results,
                template=template,
                cot=cot,
                messages=messages,
            )
        except Exception as e:
            raise RuntimeError(f"Gestell prompt failed: {e}")
        return str(response)
