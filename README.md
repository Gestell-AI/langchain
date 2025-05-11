# Gestell's Langchain Integration

![license](https://img.shields.io/badge/license-MIT-blue)
![python-version](https://img.shields.io/badge/python-3-blue)
![version](https://img.shields.io/badge/version-0.1.0-blue)

![Preview](./preview.gif)

Query and prompt your `gestell` collections with `langchain`

## Quick Start

Step 1: Install the langchain integration package:

```bash
pip install gestell-langchain
uv add gestell-langchain
```

Step 2: Integrate it with your agent:

```python
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_gestell import GestellSearchTool, GestellPromptTool

llm = ChatOpenAI()

agent = create_react_agent(
    model=llm,
    tools=[
      # Add api_key and collection_id if not in your env
      GestellSearchTool(api_key="... OPTIONAL", collection_id="... OPTIONAL"),
      GestellPromptTool()
    ],
)

```

You can view a more comprehensive demo [here](./examples/chat.py).

Open an issue if you would like more comprehensive demos or need clarity on how things work.
