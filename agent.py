"""
CH04 — streaming.

Same graph as CH03 (llm + tools + checkpointer). The only change is how we
consume the output: instead of `invoke()` blocking until the whole graph
finishes, we use `stream()` / `astream_events()` to surface intermediate
events. The REPL exposes a `/mode` slash command so you can swap between:

    updates    — one chunk per node, only the channel diff (default)
    values     — full state after each step
    messages   — LLM tokens stream out live (typing animation)
    debug      — every internal event (task start/end, checkpoint writes)
    events     — astream_events v2: per-LLM-token + per-tool start/end

Run:
    pip install -r requirements.txt
    cp .env.example .env  # fill in ANTHROPIC_API_KEY
    python agent.py
"""

import asyncio
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
    """Same shape as CH01-CH03 — streaming doesn't change State."""
    messages: Annotated[list, add_messages]


# ---------- Tools (same as CH02/CH03) -----------------------------------------

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


# ---------- Model + nodes (same as CH02/CH03) ---------------------------------

llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=1024).bind_tools(TOOLS)


def llm_node(state: State) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


tool_node = ToolNode(TOOLS)


# ---------- Graph + checkpointer (same as CH03) -------------------------------

graph = StateGraph(State)
graph.add_node("llm", llm_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", tools_condition)
graph.add_edge("tools", "llm")

checkpointer = InMemorySaver()
agent = graph.compile(checkpointer=checkpointer)

print(agent.get_graph().draw_mermaid())


# ---------- Stream-mode formatters --------------------------------------------

STREAM_MODES = ("updates", "values", "messages", "debug", "events")


def _short(cid: str | None, n: int = 8) -> str:
    if not cid:
        return "?"
    return cid[-n:]


def _summarize_msg(m) -> str:
    kind = m.__class__.__name__
    if kind == "HumanMessage":
        return f"user: {m.content[:40]}"
    if kind in ("AIMessage", "AIMessageChunk"):
        tool_calls = getattr(m, "tool_calls", None) or []
        if tool_calls:
            names = ", ".join(tc["name"] for tc in tool_calls)
            return f"ai:   tool_call -> {names}"
        text = m.content if isinstance(m.content, str) else ""
        return f"ai:   {text[:40]}"
    if kind == "ToolMessage":
        return f"tool: {m.name} -> {str(m.content)[:30]}"
    return f"{kind}: ..."


def _stream_sync(user_in: str, config: dict, mode: str) -> None:
    """stream_mode in {updates, values, messages, debug}."""
    payload = {"messages": [{"role": "user", "content": user_in}]}
    step = 0

    for chunk in agent.stream(payload, config=config, stream_mode=mode):
        step += 1

        if mode == "updates":
            # chunk: {node_name: {channel: diff}}
            for node, diff in chunk.items():
                for m in diff.get("messages", []):
                    print(f"[{step:>2}] [{node:>5}] {_summarize_msg(m)}")

        elif mode == "values":
            # chunk: full state dict (after this step)
            msgs = chunk.get("messages", [])
            last = _summarize_msg(msgs[-1]) if msgs else "-"
            print(f"[{step:>2}] [values] msgs={len(msgs):<2} last={last}")

        elif mode == "messages":
            # chunk: (BaseMessageChunk, metadata)
            msg_chunk, meta = chunk
            node = meta.get("langgraph_node", "?")
            kind = msg_chunk.__class__.__name__

            if kind in ("AIMessageChunk", "AIMessage"):
                content = msg_chunk.content
                if isinstance(content, str):
                    if content:
                        print(content, end="", flush=True)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            print(block.get("text", ""), end="", flush=True)
                for tc in getattr(msg_chunk, "tool_call_chunks", []) or []:
                    if tc.get("name"):
                        print(f"\n[tool_call] {tc['name']} args=", end="", flush=True)
                    if tc.get("args"):
                        print(tc["args"], end="", flush=True)
            elif kind == "ToolMessage":
                content = str(msg_chunk.content)
                preview = content if len(content) <= 100 else content[:100] + "..."
                print(f"\n[tool_res ] {msg_chunk.name} -> {preview}")

        elif mode == "debug":
            kind = chunk.get("type", "?")
            step_n = chunk.get("step", "?")
            payload_inner = chunk.get("payload", {})
            name = payload_inner.get("name", "")
            print(f"[{step:>3}] [debug] type={kind:<13} step={step_n}  name={name}")

    if mode == "messages":
        print()  # final newline after token stream


async def _stream_events(user_in: str, config: dict) -> None:
    """astream_events v2 — finer than stream_mode='messages': also gives
    on_tool_start / on_tool_end / on_chain_* / on_chat_model_start, etc."""
    payload = {"messages": [{"role": "user", "content": user_in}]}
    streaming_text = False

    async for ev in agent.astream_events(payload, config=config, version="v2"):
        kind = ev["event"]
        name = ev.get("name", "")
        data = ev.get("data", {})

        if kind == "on_chat_model_start":
            print(f"\n[event] on_chat_model_start  name={name}")
            streaming_text = False
        elif kind == "on_chat_model_stream":
            chunk = data.get("chunk")
            if not chunk:
                continue
            content = chunk.content
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text += block.get("text", "")
            if text:
                if not streaming_text:
                    print("[event] on_chat_model_stream  ", end="", flush=True)
                    streaming_text = True
                print(text, end="", flush=True)
        elif kind == "on_chat_model_end":
            if streaming_text:
                print()
                streaming_text = False
            print(f"[event] on_chat_model_end    name={name}")
        elif kind == "on_tool_start":
            print(f"[event] on_tool_start        name={name}  input={data.get('input')}")
        elif kind == "on_tool_end":
            output_obj = data.get("output")
            output_str = str(getattr(output_obj, "content", output_obj))
            preview = output_str if len(output_str) <= 80 else output_str[:80] + "..."
            print(f"[event] on_tool_end          name={name}  output={preview}")

    if streaming_text:
        print()


def _run_turn(user_in: str, config: dict, mode: str) -> None:
    if mode == "events":
        asyncio.run(_stream_events(user_in, config))
    else:
        _stream_sync(user_in, config, mode)


# ---------- REPL helpers (same as CH03) ---------------------------------------

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


def _new_thread_id() -> str:
    return "demo-" + uuid.uuid4().hex[:6]


# ---------- REPL --------------------------------------------------------------

if __name__ == "__main__":
    thread_id = _new_thread_id()
    pending_fork_prefix: str | None = None
    current_mode = "updates"

    print("[init] CH04 — streaming")
    print(f"[init] thread_id   = {thread_id}")
    print(f"[init] stream mode = {current_mode}  (try /mode for others)")
    print("[hint] commands:")
    print("       /mode [name]  show or switch stream mode")
    print(f"                     options: {', '.join(STREAM_MODES)}")
    print("       /new          start a fresh thread")
    print("       /history      list checkpoints for the current thread")
    print("       /fork <ckpt>  next message replays from that checkpoint")
    print("       /exit         quit")
    print()

    while True:
        try:
            prompt = (
                f"({thread_id[-6:]}|{current_mode}{'*' if pending_fork_prefix else ''}) you> "
            )
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

        if user_in.startswith("/mode"):
            parts = user_in.split(maxsplit=1)
            if len(parts) < 2:
                print(f"[stream mode = {current_mode}]  options: {', '.join(STREAM_MODES)}\n")
                continue
            new_mode = parts[1].strip()
            if new_mode not in STREAM_MODES:
                print(f"[unknown mode {new_mode!r}; options: {', '.join(STREAM_MODES)}]\n")
                continue
            current_mode = new_mode
            print(f"[stream mode -> {current_mode}]\n")
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
            config = match.config
            print(f"[forking from {_short(match.config['configurable']['checkpoint_id'])}]")
            pending_fork_prefix = None

        _run_turn(user_in, config, current_mode)
        print()
