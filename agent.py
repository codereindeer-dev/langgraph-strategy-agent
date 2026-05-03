"""
CH01 — single-node StateGraph with Claude.

A graph with one LLM node, no tools. Demonstrates the four LangGraph
basics: State (TypedDict + reducer), node (state -> state delta),
edges (START / END), and compile() / invoke().

Run:
    pip install -r requirements.txt
    cp .env.example .env  # fill in ANTHROPIC_API_KEY
    python agent.py
"""

from typing_extensions import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()


class State(TypedDict):
    """The graph's shared state. `add_messages` is a reducer — when a node
    returns `{"messages": [new_msg]}`, LangGraph appends instead of overwriting."""
    messages: Annotated[list, add_messages]


llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=1024)


def llm_node(state: State) -> dict:
    """One node: read all messages, ask Claude, return the new assistant message.
    The reducer on `messages` handles appending into state."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


graph = StateGraph(State)
graph.add_node("llm", llm_node)
graph.add_edge(START, "llm")  # entry point
graph.add_edge("llm", END)    # exit point
agent = graph.compile()

print(agent.get_graph().draw_mermaid())


if __name__ == "__main__":
    print("[init] CH01 — single-node graph (no tools, single-turn)")
    print("[hint] /exit to quit\n")
    while True:
        try:
            user_in = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_in:
            continue
        if user_in in ("/exit", "/quit"):
            break

        # Each invoke starts a fresh state — single-turn for now.
        # Multi-turn comes in CH03 via checkpointing.
        result = agent.invoke({"messages": [{"role": "user", "content": user_in}]})
        print(f"claude> {result['messages'][-1].content}\n")
