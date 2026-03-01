"""
Deep Agent – quickstart example
================================
A research agent that can plan tasks, search the web (mock), write files,
and spawn subagents – all powered by deepagents + LangGraph.

Uses Groq as the LLM backend (llama-3.3-70b-versatile by default).

Requirements
------------
Copy .env.example to .env and fill in your GROQ_API_KEY before running.
Get a free key at https://console.groq.com/

Run
---
    uv run python main.py
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from deepagents import create_deep_agent

# --------------------------------------------------------------------------- #
# Load API keys from .env
# --------------------------------------------------------------------------- #
load_dotenv()

_required_keys = ["GROQ_API_KEY"]
_missing = [k for k in _required_keys if not os.environ.get(k)]
if _missing:
    raise EnvironmentError(
        f"Missing required environment variables: {', '.join(_missing)}\n"
        "Copy .env.example to .env and fill in your Groq API key.\n"
        "Get a free key at https://console.groq.com/"
    )


# --------------------------------------------------------------------------- #
# Define custom tools
# --------------------------------------------------------------------------- #

def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    # Replace with a real weather API call if needed
    return f"It's currently 22°C and sunny in {city}."


def calculate(expression: str) -> str:
    """Safely evaluate a simple arithmetic expression and return the result."""
    allowed_chars = set("0123456789 +-*/().")
    if not all(c in allowed_chars for c in expression):
        return "Error: only basic arithmetic is supported."
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"Error: {exc}"


# --------------------------------------------------------------------------- #
# Build the deep agent
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """\
You are a helpful and thorough assistant.

When given a multi-step task you should:
1. Use write_todos to plan your work.
2. Use the provided tools to gather information.
3. Use write_file / read_file to manage long intermediate results.
4. Spawn sub-agents via the task tool for complex subtasks.
5. Return a clear, concise final answer.
"""

# Groq model – llama-3.3-70b-versatile supports tool calling
# Other good options: "llama-3.1-8b-instant", "mixtral-8x7b-32768"
groq_model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)

agent = create_deep_agent(
    model=groq_model,
    tools=[get_weather, calculate],
    system_prompt=SYSTEM_PROMPT,
)


# --------------------------------------------------------------------------- #
# Streaming helper
# --------------------------------------------------------------------------- #

def stream_agent(user_message: str) -> str:
    """Run the agent with streaming output; return the final text response."""
    print(f"\n{'='*60}")
    print(f"USER: {user_message}")
    print("="*60)

    final_content = ""
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": user_message}]},
        stream_mode="updates",
    ):
        for node, update in chunk.items():
            msgs = update.get("messages", [])
            for msg in msgs:
                # Print tool calls / results as they arrive
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"  [tool call] {tc['name']}({tc['args']})")
                elif hasattr(msg, "name") and msg.name:
                    print(f"  [tool result from {msg.name}]: {str(msg.content)[:120]}")
                elif hasattr(msg, "content") and msg.content:
                    final_content = msg.content
    
    print(f"\nASSISTANT: {final_content}")
    return final_content


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    # Example 1 – simple tool use
    stream_agent("What is the weather in Tokyo and what is 123 * 456?")

    # Example 2 – multi-step planning
    stream_agent(
        "Plan a comparison of the weather in London, Paris, and New York,"
        " then calculate the average temperature if London=15°C, Paris=18°C,"
        " New York=12°C."
    )


if __name__ == "__main__":
    main()

