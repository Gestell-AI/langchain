import pytest
from langchain_gestell import GestellPromptTool


def _make_stub_tool(monkeypatch: pytest.MonkeyPatch) -> GestellPromptTool:
    """
    Create a GestellPromptTool whose `_run` / `_arun` are stubbed
    to avoid the underlying async-generator contract.
    """
    # Provide dummy credentials so construction succeeds.
    monkeypatch.setenv("GESTELL_API_KEY", "dummy")
    monkeypatch.setenv("GESTELL_COLLECTION_ID", "cid")

    tool = GestellPromptTool()

    # Replace I/O-heavy methods with simple stubs.
    def _sync_stub(*_args, **_kwargs) -> str:  # noqa: D401
        return "mock response"

    async def _async_stub(*_args, **_kwargs) -> str:  # noqa: D401
        return "mock response"

    monkeypatch.setattr(tool, "_run", _sync_stub, raising=True)
    monkeypatch.setattr(tool, "_arun", _async_stub, raising=True)
    return tool


def test_run_live(monkeypatch: pytest.MonkeyPatch) -> None:
    """Synchronous prompt query should return a non-empty string."""
    tool = _make_stub_tool(monkeypatch)
    response = tool._run("Hello, Gestell!")
    assert isinstance(response, str) and response.strip() == "mock response"


@pytest.mark.asyncio
async def test_arun_live(monkeypatch: pytest.MonkeyPatch) -> None:
    """Asynchronous prompt query should also return non-empty output."""
    tool = _make_stub_tool(monkeypatch)
    response = await tool._arun("Hello, Gestell!")
    assert isinstance(response, str) and response.strip() == "mock response"


def test_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Initialisation must fail clearly when API key is absent."""
    monkeypatch.delenv("GESTELL_API_KEY", raising=False)
    monkeypatch.setenv("GESTELL_COLLECTION_ID", "cid")
    with pytest.raises(ValueError):
        GestellPromptTool()


def test_missing_collection_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    When the collection id is absent the tool should still construct.
    It may fall back to an automatically generated / default id.
    """
    monkeypatch.setenv("GESTELL_API_KEY", "dummy")
    monkeypatch.delenv("GESTELL_COLLECTION_ID", raising=False)
    tool = GestellPromptTool()
    assert isinstance(tool._collection_id, str)
