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


class GestellPromptInput(BaseModel):
    """Input for Gestell prompt tool (a question or task to answer using the collection)."""

    prompt: str = Field(
        ...,
        description="Question or instruction to be answered using the Gestell collection",
    )


class GestellPromptTool(BaseTool):
    """LangChain tool to prompt a Gestell collection for an answer using its data."""

    name: str = "gestell_prompt"
    description: str = (
        "Use the Gestell collection to answer a question or fulfill an instruction."
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
        if not coll:
            raise ValueError(
                "A Gestell collection_id is required (via argument or GESTELL_COLLECTION_ID env var)."
            )
        self._collection_id = coll

    def _run(
        self,
        prompt: str,
        *,
        run_manager: CallbackManagerForToolRun = None,
    ) -> str:
        """Synchronously execute a prompt query on the Gestell collection."""
        try:
            response = asyncio.run(
                self._gestell.query.prompt(
                    collection_id=self._collection_id, prompt=prompt
                )
            )
        except Exception as e:
            raise RuntimeError(f"Gestell prompt failed: {e}")
        return str(response)

    async def _arun(
        self,
        prompt: str,
        *,
        run_manager: AsyncCallbackManagerForToolRun = None,
    ) -> str:
        """Asynchronously execute a prompt query on the Gestell collection."""
        try:
            response = await self._gestell.query.prompt(
                collection_id=self._collection_id, prompt=prompt
            )
        except Exception as e:
            raise RuntimeError(f"Gestell prompt failed: {e}")
        return str(response)
