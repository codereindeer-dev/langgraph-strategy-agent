"""
CH03 — checkpointing + multi-turn + time travel.

Compiles the CH02 graph with an `InMemorySaver`. Every node boundary
auto-saves a state snapshot, keyed by `thread_id`. The REPL keeps one
`thread_id` across turns -> multi-turn falls out for free. Slash
commands let you inspect history and fork from any historical
checkpoint.

Run:
    pip install -r requirements.txt
    cp .env.example .env  # fill in ANTHROPIC_API_KEY
    python agent.py
"""

import hashlib
import json
import random
import sys
import uuid
from typing_extensions import Annotated, TypedDict

# Force UTF-8 stdout so emoji / CJK in LLM replies don't crash on Windows cp950.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv(override=True)


class State(TypedDict):
    """Same shape as CH01 / CH02 — checkpointing doesn't change State."""
    messages: Annotated[list, add_messages]


# ---------- Tools (same as CH02) ----------------------------------------------

@tool(parse_docstring=True)
def get_price_data(ticker: str, days: int = 30) -> str:
    """Fetch recent daily close prices for a ticker symbol.

    Args:
        ticker: Ticker symbol, e.g. "AAPL" or "TSLA".
        days: How many trading days back to fetch. Defaults to 30.
    """
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


# ---------- Model + nodes (same as CH02) --------------------------------------

llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=1024).bind_tools(TOOLS)


def llm_node(state: State) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


tool_node = ToolNode(TOOLS)


# ---------- Graph + checkpointer ----------------------------------------------
# The graph topology is identical to CH02. The only new thing is
# `checkpointer=...` on compile().

graph = StateGraph(State)
graph.add_node("llm", llm_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", tools_condition)
graph.add_edge("tools", "llm")

checkpointer = InMemorySaver()
agent = graph.compile(checkpointer=checkpointer)

print(agent.get_graph().draw_mermaid())


# ---------- REPL helpers ------------------------------------------------------

def _short(cid: str | None, n: int = 8) -> str:
    """Last `n` chars of a checkpoint UUIDv6 — the random/sequence tail.
    The first segment is a millisecond-precision timestamp, so adjacent
    checkpoints collide there; the tail is what distinguishes them."""
    if not cid:
        return "?"
    return cid[-n:]


def _summarize_msg(m) -> str:
    """One-line label for a message, used in history listings."""
    kind = m.__class__.__name__
    if kind == "HumanMessage":
        return f"user: {m.content[:40]}"
    if kind == "AIMessage":
        tool_calls = getattr(m, "tool_calls", None) or []
        if tool_calls:
            names = ", ".join(tc["name"] for tc in tool_calls)
            return f"ai:   tool_call -> {names}"
        text = m.content if isinstance(m.content, str) else ""
        return f"ai:   {text[:40]}"
    if kind == "ToolMessage":
        return f"tool: {m.name} -> {str(m.content)[:30]}"
    return f"{kind}: ..."


def _print_history(thread_id: str) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    snapshots = list(agent.get_state_history(config))
    if not snapshots:
        print("(no history for this thread)")
        return
    print(f"--- history for thread {thread_id} ({len(snapshots)} checkpoints, newest first) ---")
    print(f"  {'ckpt':<10} {'next':<12} {'msgs':<4}  last")
    for s in snapshots:
        cid = s.config["configurable"].get("checkpoint_id")
        next_nodes = ",".join(s.next) if s.next else "END"
        msgs = s.values.get("messages", [])
        last_label = _summarize_msg(msgs[-1]) if msgs else "-"
        print(f"  {_short(cid):<10} {next_nodes:<12} {len(msgs):<4}  {last_label}")
    print()


def _print_new_messages(prev_count: int, full_messages: list) -> None:
    """Print only messages added in the last invoke (so multi-turn isn't noisy)."""
    for m in full_messages[prev_count:]:
        kind = m.__class__.__name__
        if kind == "HumanMessage":
            print(f"[user]      {m.content}")
        elif kind == "AIMessage":
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
            content = str(m.content)
            preview = content if len(content) <= 120 else content[:120] + "..."
            print(f"[tool_res]  {m.name} -> {preview}")


def _new_thread_id() -> str:
    return "demo-" + uuid.uuid4().hex[:6]


# ---------- REPL --------------------------------------------------------------

if __name__ == "__main__":
    thread_id = _new_thread_id()
    pending_fork_prefix: str | None = None

    print("[init] CH03 — checkpointing + multi-turn + time travel")
    print(f"[init] thread_id = {thread_id}")
    print("[hint] commands:")
    print("       /new          start a fresh thread (new conversation)")
    print("       /history      list checkpoints for the current thread")
    print("       /fork <ckpt>  next message replays from that checkpoint")
    print("       /exit         quit")
    print()

    while True:
        try:
            prompt = f"({thread_id[-6:]}*) you> " if pending_fork_prefix else f"({thread_id[-6:]}) you> "
            user_in = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_in:
            continue

        # ----- slash commands -----
        if user_in in ("/exit", "/quit"):
            break

        if user_in == "/new":
            thread_id = _new_thread_id()
            pending_fork_prefix = None
            print(f"[switched to new thread: {thread_id}]\n")
            continue

        if user_in == "/history":
            _print_history(thread_id)
            continue

        if user_in.startswith("/fork"):
            parts = user_in.split(maxsplit=1)
            if len(parts) < 2:
                print("usage: /fork <ckpt_suffix>\n"
                      "       (paste the short id shown by /history, then "
                      "type the new user message normally)")
                continue
            pending_fork_prefix = parts[1].strip()
            print(f"[fork armed: next message will replay from checkpoint "
                  f"ending in {pending_fork_prefix!r}]\n")
            continue

        # ----- normal turn (or forked turn) -----
        config = {"configurable": {"thread_id": thread_id}}

        if pending_fork_prefix:
            # Find the matching historical checkpoint by id suffix
            # (matches the short form shown by /history).
            snapshots = list(agent.get_state_history(config))
            match = next(
                (s for s in snapshots
                 if (s.config["configurable"].get("checkpoint_id") or "")
                 .endswith(pending_fork_prefix)),
                None,
            )
            if not match:
                print(f"[no checkpoint matching {pending_fork_prefix!r}; aborting fork]\n")
                pending_fork_prefix = None
                continue
            # Replaying with a config that includes checkpoint_id makes the new
            # checkpoints descend from `match` instead of the latest tip.
            config = match.config
            print(f"[forking from {_short(match.config['configurable']['checkpoint_id'])}]")
            pending_fork_prefix = None

        # Track message count before invoke so we only print what's new.
        try:
            prev = agent.get_state(config).values.get("messages", [])
        except Exception:
            prev = []
        prev_count = len(prev) + 1  # +1 because we're about to add this user message

        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_in}]},
            config=config,
        )
        _print_new_messages(prev_count, result["messages"])
        print()
