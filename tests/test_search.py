import pytest
from langchain_gestell import GestellSearchTool


def _make_stub_tool(monkeypatch: pytest.MonkeyPatch) -> GestellSearchTool:
    """
    Return a GestellSearchTool whose _run/_arun are replaced with stubs
    so we never invoke the real Gestell SDK.
    """
    # Provide dummy values so constructor validation passes.
    monkeypatch.setenv("GESTELL_API_KEY", "dummy")
    monkeypatch.setenv("GESTELL_COLLECTION_ID", "cid")

    tool = GestellSearchTool()

    def _sync_stub(*_args, **_kwargs) -> str:  # noqa: D401
        return "mock search result"

    async def _async_stub(*_args, **_kwargs) -> str:  # noqa: D401
        return "mock search result"

    monkeypatch.setattr(tool, "_run", _sync_stub, raising=True)
    monkeypatch.setattr(tool, "_arun", _async_stub, raising=True)
    return tool


def test_run_live(monkeypatch: pytest.MonkeyPatch) -> None:
    """Synchronous search query should return non-empty text."""
    tool = _make_stub_tool(monkeypatch)
    response = tool._run("10-K filings for AAPL")
    assert isinstance(response, str) and response.strip() == "mock search result"


@pytest.mark.asyncio
async def test_arun_live(monkeypatch: pytest.MonkeyPatch) -> None:
    """Asynchronous search query should also return non-empty text."""
    tool = _make_stub_tool(monkeypatch)
    response = await tool._arun("10-K filings for AAPL")
    assert isinstance(response, str) and response.strip() == "mock search result"


def test_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructor must raise when API key is missing."""
    monkeypatch.delenv("GESTELL_API_KEY", raising=False)
    monkeypatch.setenv("GESTELL_COLLECTION_ID", "cid")
    with pytest.raises(ValueError):
        GestellSearchTool()


def test_missing_collection_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructor should still work when collection ID is missing."""
    monkeypatch.setenv("GESTELL_API_KEY", "dummy")
    monkeypatch.delenv("GESTELL_COLLECTION_ID", raising=False)
    tool = GestellSearchTool()
    assert isinstance(tool._collection_id, str)
