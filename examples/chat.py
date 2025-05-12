import os
from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_gestell import GestellSearchTool, GestellPromptTool

load_dotenv()


def build_tools() -> list[BaseTool]:
    """
    Instantiate Gestell tools from environment variables.

    @returns A list of configured BaseTool instances for Gestell search & prompt.
    """
    api_key = os.getenv("GESTELL_API_KEY")
    collection_id = os.getenv("GESTELL_COLLECTION_ID")
    if not (api_key and collection_id):
        raise RuntimeError("Please set both GESTELL_API_KEY and GESTELL_COLLECTION_ID.")
    return [
        GestellSearchTool(api_key=api_key, collection_id=collection_id),
        GestellPromptTool(api_key=api_key, collection_id=collection_id),
    ]


def main():
    llm = ChatOpenAI(
        temperature=0.2,
        model_name="gpt-4o-mini",  # or any other supported chat model
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )

    agent = create_react_agent(model=llm, tools=build_tools())

    print("Gestell Chat Agent ready. Ctrl-C to exit.")
    try:
        while True:
            user_input = input("\nYou ▸ ").strip()
            if not user_input:
                continue

            print("\nAssistant ▸ ", end="", flush=True)
            agent.invoke({"messages": [{"role": "user", "content": user_input}]})
            print()

    except (KeyboardInterrupt, EOFError):
        print("\n\nThanks for trying Gestell, learn more at: https://gestell.ai")


if __name__ == "__main__":
    main()
