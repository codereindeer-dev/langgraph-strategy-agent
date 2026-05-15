"""
CH02 — StateGraph with tools (ReAct loop).

Adds two mock tools (`get_price_data`, `compute_sma`), binds them to the LLM,
and routes via a conditional edge so the graph loops `llm -> tools -> llm`
until the model stops calling tools.

Run:
    pip install -r requirements.txt
    cp .env.example .env  # fill in ANTHROPIC_API_KEY
    python agent.py
"""

import hashlib
import json
import random
import sys
from typing_extensions import Annotated, TypedDict

# Force UTF-8 stdout so emoji / CJK in LLM replies don't crash on Windows cp950.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv(override=True)  # let .env win over an empty shell var


class State(TypedDict):
    """Same shape as CH01. `add_messages` also handles `ToolMessage` — that's
    why the State barely changes when tools show up."""
    messages: Annotated[list, add_messages]


# ---------- Tools (mocked, no external deps) ----------------------------------
# Deterministic mock data: same (ticker, days) -> same prices. Lets you reason
# about the graph's behaviour without worrying about real network calls.

@tool(parse_docstring=True)
def get_price_data(ticker: str, days: int = 30) -> str:
    """Fetch recent daily close prices for a ticker symbol.

    Args:
        ticker: Ticker symbol, e.g. "AAPL" or "TSLA".
        days: How many trading days back to fetch. Defaults to 30.
    """
    # Stable seed across runs (Python's hash() is salted per process).
    seed = int(hashlib.md5(f"{ticker}|{days}".encode()).hexdigest()[:8], 16)
    random.seed(seed)
    base = 100.0
    prices = []
    for _ in range(days):
        base *= 1 + random.uniform(-0.02, 0.02)
        prices.append(round(base, 2))
    return json.dumps({"ticker": ticker, "prices": prices})


@tool(parse_docstring=True)
def compute_sma(prices: list[float], window: int) -> str:
    """Compute the simple moving average over a list of prices.

    Args:
        prices: List of close prices, oldest first.
        window: SMA window size, e.g. 5 or 20.
    """
    if window <= 0 or window > len(prices):
        return f"ERROR: invalid window {window} for {len(prices)} prices"
    sma = [
        round(sum(prices[i - window + 1 : i + 1]) / window, 2)
        for i in range(window - 1, len(prices))
    ]
    return json.dumps({"window": window, "sma": sma})


TOOLS = [get_price_data, compute_sma]


# ---------- Model + nodes -----------------------------------------------------

# bind_tools attaches the tool schemas to every llm.invoke() call. The model
# now knows the tools exist and can emit tool_calls.
llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=1024).bind_tools(TOOLS)


def llm_node(state: State) -> dict:
    """Same shape as CH01 — but the response may now include tool_calls."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# Prebuilt: runs every tool_call on the last AIMessage and appends ToolMessages.
tool_node = ToolNode(TOOLS)


# ---------- Graph -------------------------------------------------------------

graph = StateGraph(State)
graph.add_node("llm", llm_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "llm")
# tools_condition inspects the last message:
#   - has tool_calls  -> route to "tools"
#   - otherwise       -> route to END
graph.add_conditional_edges("llm", tools_condition)
# Cyclic edge: after tools run, go back to llm to let it react to results.
graph.add_edge("tools", "llm")

agent = graph.compile()

print(agent.get_graph().draw_mermaid())


# ---------- REPL --------------------------------------------------------------

def _pretty_print(messages: list) -> None:
    """Walk the final messages list so the reader sees the loop unfolding."""
    for m in messages:
        kind = m.__class__.__name__
        if kind == "HumanMessage":
            print(f"[user]      {m.content}")
        elif kind == "AIMessage":
            # AIMessage.content can be str or list of blocks (text + tool_use)
            if isinstance(m.content, str):
                if m.content.strip():
                    print(f"[assistant] {m.content}")
            else:
                for block in m.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        print(f"[assistant] {block['text']}")
            for tc in getattr(m, "tool_calls", []) or []:
                print(f"[tool_call] {tc['name']}({tc['args']})")
        elif kind == "ToolMessage":
            print(f"[tool_res]  {m.name} -> {m.content}")


if __name__ == "__main__":
    print("[init] CH02 — tools + conditional edges (ReAct loop)")
    print("[hint] try: '抓 AAPL 30 天，再算 5 日 SMA，告訴我最後一個 SMA'")
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

        # Single-turn per invoke — multi-turn waits for CH03 checkpointing.
        result = agent.invoke({"messages": [{"role": "user", "content": user_in}]})
        _pretty_print(result["messages"])
        print()
