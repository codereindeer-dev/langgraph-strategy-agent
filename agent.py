"""
CH06 — `interrupt()` HITL (human-in-the-loop).

Same graph as CH05 (llm + ToolNode + backtest_subgraph + finalize),
plus one new node `human_approval` that gates the fan-out:

    llm -> human_approval -> {backtest_subgraph (Send fan-out) | llm}

When the LLM calls `compare_strategies`, the graph routes to
`human_approval` (instead of dispatching `Send` directly as in CH05).
That node calls `interrupt({...})` with a payload describing the
proposed run; the graph pauses, the REPL prompts the human, and the
human's reply is delivered back to the node via `Command(resume=...)`.
If approved, control flows to `Send` fan-out as before; if rejected,
the node injects a rejection `ToolMessage` and routes back to `llm` so
the model can re-propose.

New concepts vs CH05:
- `interrupt(payload)`: pauses the running graph at the call site and
  bubbles `payload` up to the caller. On resume, the call returns the
  resume value as if it had been computed locally.
- `Command(resume=value)`: passed to `agent.stream(...)` to continue a
  paused graph from where it stopped. Works hand-in-hand with the
  checkpointer (CH03) — without a checkpointer, there is nothing to
  resume from.
- Conditional edge after a HITL node: the same `Send`-or-named-node
  pattern as CH05's `route_after_llm`, but driven by the approval
  outcome instead of the LLM's tool_call.

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
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, Send, interrupt

load_dotenv(override=True)


# ---------- Reducer for the fan-in channel ------------------------------------

_CLEAR = "CLEAR"


def _bt_reducer(left, right):
    """Append `right` to `left`; "CLEAR" sentinel resets to empty list."""
    if right == _CLEAR:
        return []
    return (left or []) + list(right or [])


class State(TypedDict):
    """Parent-graph state. `backtest_results` is the fan-in channel."""
    messages: Annotated[list, add_messages]
    backtest_results: Annotated[list, _bt_reducer]


# ---------- Tools the LLM can see ---------------------------------------------

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


@tool(parse_docstring=True)
def compare_strategies(ticker: str, windows: list[int]) -> str:
    """Backtest a price-vs-SMA crossover on 60 days of synthetic prices for the same ticker across multiple SMA windows IN PARALLEL, and return per-window metrics (total return %, Sharpe, number of trades). Use this when the user wants to compare two or more SMA window sizes.

    Args:
        ticker: Ticker symbol, e.g. "AAPL".
        windows: SMA window sizes to compare, e.g. [20, 50, 100].
    """
    # Body never runs: the conditional edge after `llm` intercepts this
    # tool call and dispatches a `Send` per window into the subgraph.
    raise NotImplementedError("intercepted by graph router")


# Tools the LLM is told about (schema only for compare_strategies):
LLM_TOOLS = [get_price_data, compute_sma, compare_strategies]
# Tools the regular ToolNode actually executes:
EXEC_TOOLS = [get_price_data, compute_sma]


# ---------- Backtest subgraph -------------------------------------------------

BACKTEST_DAYS = 60


class BacktestState(TypedDict):
    """Subgraph state. The `Send` payload populates ticker/window/tool_call_id;
    the rest is filled in as the subgraph runs.
    """
    ticker: str
    window: int
    tool_call_id: str
    prices: list[float]
    sma: list[float]
    # Same channel name as the parent so the result flows back automatically.
    backtest_results: Annotated[list, _bt_reducer]


def bt_fetch(state: BacktestState) -> dict:
    seed = int(
        hashlib.md5(f"{state['ticker']}|{BACKTEST_DAYS}".encode()).hexdigest()[:8],
        16,
    )
    random.seed(seed)
    base = 100.0
    prices = []
    for _ in range(BACKTEST_DAYS):
        base *= 1 + random.uniform(-0.02, 0.02)
        prices.append(round(base, 2))
    return {"prices": prices}


def bt_sma(state: BacktestState) -> dict:
    w = state["window"]
    prices = state["prices"]
    if w <= 0 or w > len(prices):
        return {"sma": []}
    sma = [
        round(sum(prices[i - w + 1 : i + 1]) / w, 2)
        for i in range(w - 1, len(prices))
    ]
    return {"sma": sma}


def bt_score(state: BacktestState) -> dict:
    """Toy backtest: long when close > SMA, flat otherwise."""
    prices = state["prices"]
    sma = state["sma"]
    w = state["window"]

    if not sma:
        result = {
            "ticker": state["ticker"],
            "window": w,
            "tool_call_id": state["tool_call_id"],
            "total_return_pct": 0.0,
            "sharpe": 0.0,
            "n_trades": 0,
            "note": "window too large for available prices",
        }
        return {"backtest_results": [result]}

    rets: list[float] = []
    pos = 0
    trades = 0
    # prices[i + w - 1] aligns with sma[i]; we trade on the next day's return.
    for i in range(len(sma) - 1):
        p_idx = i + w - 1
        new_pos = 1 if prices[p_idx] > sma[i] else 0
        if new_pos != pos:
            trades += 1
        pos = new_pos
        day_ret = (prices[p_idx + 1] / prices[p_idx] - 1) * pos
        rets.append(day_ret)

    total_return = round(sum(rets) * 100, 2)
    if rets:
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / len(rets)
        std = var ** 0.5
        sharpe = round(mean / std * (252 ** 0.5), 2) if std > 0 else 0.0
    else:
        sharpe = 0.0

    result = {
        "ticker": state["ticker"],
        "window": w,
        "tool_call_id": state["tool_call_id"],
        "total_return_pct": total_return,
        "sharpe": sharpe,
        "n_trades": trades,
    }
    return {"backtest_results": [result]}


_bt_graph = StateGraph(BacktestState)
_bt_graph.add_node("fetch", bt_fetch)
_bt_graph.add_node("sma", bt_sma)
_bt_graph.add_node("score", bt_score)
_bt_graph.add_edge(START, "fetch")
_bt_graph.add_edge("fetch", "sma")
_bt_graph.add_edge("sma", "score")
_bt_graph.add_edge("score", END)
backtest_subgraph = _bt_graph.compile()


# ---------- Parent graph: llm + tools + subgraph + finalize -------------------

llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=1024).bind_tools(LLM_TOOLS)


def llm_node(state: State) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


tool_node = ToolNode(EXEC_TOOLS)


def route_after_llm(state: State):
    """Custom router replacing CH04's `tools_condition`.

    Changed from CH05: instead of returning `Send`s directly, route
    `compare_strategies` calls to `human_approval` so the graph pauses
    for HITL gating before fan-out.

    - no tool_calls            -> END
    - compare_strategies call  -> "human_approval" (HITL gate)
    - any other tool call      -> "tools" (ToolNode)
    """
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    if not tool_calls:
        return END

    has_compare = any(tc["name"] == "compare_strategies" for tc in tool_calls)
    has_regular = any(tc["name"] != "compare_strategies" for tc in tool_calls)

    if has_compare:
        return "human_approval"
    if has_regular:
        return "tools"
    return END


def human_approval(state: State) -> dict:
    """HITL gate: pause before fan-out so a human can approve / reject.

    The node calls `interrupt({...})` with a payload describing what the
    LLM wants to run. The caller (REPL) sees the payload via the
    `__interrupt__` stream chunk, prompts the user, and resumes the graph
    with `Command(resume=<reply>)`. The resume value becomes the return
    value of `interrupt()`.

    - "approve" (or "yes") -> return {} so `route_after_approval` falls
      through to the Send fan-out (same as CH05 would have done).
    - anything else        -> treat as rejection; inject a `ToolMessage`
      naming the original tool_call_id so the LLM sees feedback and can
      re-propose.
    """
    print("Enter human_approval")

    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    compare_tc = next(
        (tc for tc in tool_calls if tc["name"] == "compare_strategies"),
        None,
    )
    if not compare_tc:
        # Shouldn't happen: this node is only reached when route_after_llm
        # sees a compare_strategies call pending.
        return {}

    args = compare_tc.get("args") or {}
    ticker = args.get("ticker", "")
    windows = args.get("windows", []) or []

    # `interrupt` pauses the graph. The payload here is what the caller
    # sees in the __interrupt__ chunk; the caller resumes by sending a
    # string (or any JSON-able value) back via Command(resume=...).
    decision = interrupt(
        {
            "type": "approve_compare_strategies",
            "ticker": ticker,
            "windows": windows,
            "tool_call_id": compare_tc["id"],
            "prompt": (
                f"LLM wants to run compare_strategies(ticker={ticker!r}, "
                f"windows={windows}). Approve? Reply 'yes' to proceed, or "
                f"anything else to reject with feedback."
            ),
        }
    )

    # Normalize the resume value.
    text = decision if isinstance(decision, str) else json.dumps(decision)
    head = text.strip().lower()
    if head in ("y", "yes", "ok", "approve", "approved"):
        return {}  # approved -> route_after_approval will Send fan-out.

    # Anything else is a rejection. Strip an optional leading "no" /
    # "no:" / "reject:" so the rest is treated as feedback.
    feedback = text.strip()
    for prefix in ("reject:", "rejected:", "no:", "no", "reject", "rejected"):
        if feedback.lower().startswith(prefix):
            feedback = feedback[len(prefix):].lstrip(": ").strip()
            break
    if not feedback:
        feedback = "User rejected the proposal without giving a reason."

    rejection = ToolMessage(
        content=json.dumps({"rejected": True, "feedback": feedback}),
        tool_call_id=compare_tc["id"],
        name="compare_strategies",
    )
    return {"messages": [rejection]}


def route_after_approval(state: State):
    """After `human_approval`, dispatch based on what was just appended.

    - last msg is a rejection ToolMessage -> back to "llm" so the model
      can react to the feedback (and possibly re-propose).
    - last msg is still the original AIMessage (no rejection injected)
      -> approved, emit Send fan-out to backtest_subgraph.
    """
    last = state["messages"][-1]
    if last.__class__.__name__ == "ToolMessage":
        return "llm"

    tool_calls = getattr(last, "tool_calls", None) or []
    sends: list[Send] = []
    for tc in tool_calls:
        if tc["name"] != "compare_strategies":
            continue
        args = tc.get("args", {}) or {}
        ticker = args.get("ticker", "")
        for w in args.get("windows", []) or []:
            sends.append(
                Send(
                    "backtest_subgraph",
                    {
                        "ticker": ticker,
                        "window": int(w),
                        "tool_call_id": tc["id"],
                    },
                )
            )
    return sends or "llm"


def finalize(state: State) -> dict:
    """Fan-in: reduce aggregated backtest_results into one ToolMessage."""
    results = state.get("backtest_results", []) or []
    if not results:
        return {}
    tool_call_id = results[0]["tool_call_id"]
    sorted_results = sorted(results, key=lambda r: r["window"])
    summary = {
        "ticker": sorted_results[0]["ticker"],
        "compared_windows": [r["window"] for r in sorted_results],
        "results": [
            {k: v for k, v in r.items() if k != "tool_call_id"}
            for r in sorted_results
        ],
    }
    msg = ToolMessage(
        content=json.dumps(summary),
        tool_call_id=tool_call_id,
        name="compare_strategies",
    )
    return {"messages": [msg], "backtest_results": _CLEAR}


graph = StateGraph(State)
graph.add_node("llm", llm_node)
graph.add_node("tools", tool_node)
graph.add_node("human_approval", human_approval)
graph.add_node("backtest_subgraph", backtest_subgraph)
graph.add_node("finalize", finalize)

graph.add_edge(START, "llm")
graph.add_conditional_edges(
    "llm",
    route_after_llm,
    {
        "tools": "tools",
        "human_approval": "human_approval",
        END: END,
    },
)
graph.add_edge("tools", "llm")
graph.add_conditional_edges(
    "human_approval",
    route_after_approval,
    {
        "llm": "llm",
        "backtest_subgraph": "backtest_subgraph",
    },
)
graph.add_edge("backtest_subgraph", "finalize")
graph.add_edge("finalize", "llm")

# Checkpointer is required for interrupt() / Command(resume=...) to work:
# the graph pauses by writing a checkpoint and resumes by reading it.
checkpointer = InMemorySaver()
agent = graph.compile(checkpointer=checkpointer)

# xray=1 expands the subgraph in the mermaid output.
print(agent.get_graph(xray=1).draw_mermaid())


# ---------- Stream formatters (CH04 + subgraph namespace) ---------------------

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
        return f"tool: {m.name} -> {str(m.content)[:40]}"
    return f"{kind}: ..."


def _ns_label(ns: tuple) -> str:
    """Format namespace tuple from stream(subgraphs=True).

    () for parent, ("backtest_subgraph:UUID",) for subgraph branches.
    We show the node name + last 4 chars of the UUID so parallel
    branches are distinguishable.
    """
    if not ns:
        return "main"
    parts = []
    for elem in ns:
        if ":" in elem:
            name, _, tail = elem.partition(":")
            parts.append(f"{name}#{tail[-4:]}")
        else:
            parts.append(elem)
    return "/".join(parts)


def _fmt_bt(r: dict) -> str:
    if "note" in r:
        return f"bt w={r.get('window'):<3} ({r['note']})"
    return (
        f"bt w={r.get('window'):<3} "
        f"ret={r.get('total_return_pct'):>6}% "
        f"sharpe={r.get('sharpe'):>5} "
        f"trades={r.get('n_trades')}"
    )


def _print_stream_chunk(ns_lbl: str, chunk, mode: str, step: int) -> int:
    """Render one stream chunk for the given mode. Returns the new step counter."""
    if mode == "updates":
        for node, diff in chunk.items():
            if not isinstance(diff, dict):
                continue
            printed_here = False
            for m in diff.get("messages", []) or []:
                step += 1
                printed_here = True
                print(f"[{step:>2}] [{ns_lbl:>22}] [{node:>14}] {_summarize_msg(m)}")
            for r in diff.get("backtest_results", []) or []:
                if isinstance(r, dict):
                    step += 1
                    printed_here = True
                    print(f"[{step:>2}] [{ns_lbl:>22}] [{node:>14}] {_fmt_bt(r)}")
            if not printed_here:
                other = [k for k in diff.keys()
                         if k not in ("messages", "backtest_results")]
                if other:
                    step += 1
                    print(f"[{step:>2}] [{ns_lbl:>22}] [{node:>14}] wrote: {','.join(other)}")

    elif mode == "values":
        step += 1
        msgs = chunk.get("messages", []) if isinstance(chunk, dict) else []
        bt = chunk.get("backtest_results", []) if isinstance(chunk, dict) else []
        last = _summarize_msg(msgs[-1]) if msgs else "-"
        print(
            f"[{step:>2}] [{ns_lbl:>22}] [values] "
            f"msgs={len(msgs):<2} bt={len(bt):<2} last={last}"
        )

    elif mode == "messages":
        msg_chunk, meta = chunk
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
            print(f"\n[tool_res ] [{ns_lbl}] {msg_chunk.name} -> {preview}")

    elif mode == "debug":
        step += 1
        kind = chunk.get("type", "?")
        step_n = chunk.get("step", "?")
        payload_inner = chunk.get("payload", {}) or {}
        name = payload_inner.get("name", "")
        print(
            f"[{step:>3}] [{ns_lbl:>22}] [debug] "
            f"type={kind:<13} step={step_n}  name={name}"
        )

    return step


def _stream_sync(user_in: str, config: dict, mode: str) -> None:
    """stream_mode in {updates, values, messages, debug} with subgraphs=True.

    Handles `interrupt()` by detecting `__interrupt__` chunks in the stream:
    when seen, the function prompts the user and resumes via
    `Command(resume=...)`. The resume itself is another `agent.stream(...)`
    call from the same checkpoint thread, so streaming output is contiguous.
    """
    payload = {"messages": [{"role": "user", "content": user_in}]}
    step = 0
    saw_interrupt = True   # enter the loop at least once

    while saw_interrupt:
        saw_interrupt = False
        interrupt_value = None

        for ns, chunk in agent.stream(
            payload, config=config, stream_mode=mode, subgraphs=True
        ):
            # Interrupt signal: a special chunk keyed by "__interrupt__".
            # Subgraph or not, we always exit the inner stream and ask
            # for a resume value.
            if isinstance(chunk, dict) and "__interrupt__" in chunk:
                tup = chunk["__interrupt__"]
                if tup:
                    interrupt_value = tup[0].value
                saw_interrupt = True
                # Don't print the interrupt as a normal chunk; the prompt
                # below shows what the human needs to decide.
                continue

            ns_lbl = _ns_label(ns)
            step = _print_stream_chunk(ns_lbl, chunk, mode, step)

        if saw_interrupt:
            if mode == "messages":
                print()   # close any pending typing line
            print(_format_hitl_prompt(interrupt_value))
            try:
                reply = input("[hitl] reply> ").strip()
            except (EOFError, KeyboardInterrupt):
                reply = "no: user aborted at HITL prompt"
                print()
            if not reply:
                reply = "no: empty reply, defaulting to reject"
            # Resume the paused graph. Command(resume=...) is the payload
            # for the next stream() call; the graph reads it from the
            # checkpoint and the interrupt() call returns this value.
            payload = Command(resume=reply)

    if mode == "messages":
        print()


def _format_hitl_prompt(iv) -> str:
    """Render the interrupt payload as a short multi-line prompt."""
    if isinstance(iv, dict):
        return (
            "\n[HITL] " + (iv.get("prompt") or "Approve?")
            + f"\n       ticker  = {iv.get('ticker')!r}"
            + f"\n       windows = {iv.get('windows')}"
        )
    return f"\n[HITL] {iv!r}"


# Subgraph internal nodes whose start/end we want to surface in events mode.
_SUBGRAPH_NODES = {"backtest_subgraph", "fetch", "sma", "score", "finalize"}


async def _drain_events(payload, config: dict) -> dict | None:
    """Consume one round of astream_events. Returns the interrupt value
    if the graph paused (so the caller can resume), or None when done."""
    streaming_text = False
    interrupt_value: dict | None = None

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
        elif kind == "on_chain_start" and name in _SUBGRAPH_NODES:
            print(f"[event] on_chain_start       name={name}")
        elif kind == "on_chain_end" and name in _SUBGRAPH_NODES:
            print(f"[event] on_chain_end         name={name}")
        elif kind == "on_chain_end" and name == "human_approval":
            # If this end event carries an interrupt, surface it.
            output_obj = data.get("output")
            if isinstance(output_obj, dict) and "__interrupt__" in output_obj:
                tup = output_obj["__interrupt__"]
                if tup:
                    interrupt_value = tup[0].value
            print(f"[event] on_chain_end         name=human_approval")

    if streaming_text:
        print()

    # Interrupt may also surface via the snapshot — fall back to checking
    # the current state's `next` field to be robust.
    if interrupt_value is None:
        snap = agent.get_state(config)
        if snap.next and "human_approval" in snap.next:
            for task in snap.tasks or ():
                for iv in getattr(task, "interrupts", ()) or ():
                    interrupt_value = iv.value
                    break
                if interrupt_value is not None:
                    break

    return interrupt_value


async def _stream_events(user_in: str, config: dict) -> None:
    """astream_events v2 — subgraph events nest automatically.

    Handles `interrupt()` the same way as `_stream_sync`: drains events
    until the round ends, then if the graph is paused at `human_approval`,
    prompts the human and resumes via `Command(resume=...)`.
    """
    payload = {"messages": [{"role": "user", "content": user_in}]}
    while True:
        interrupt_value = await _drain_events(payload, config)
        if interrupt_value is None:
            break
        print(_format_hitl_prompt(interrupt_value))
        try:
            reply = input("[hitl] reply> ").strip()
        except (EOFError, KeyboardInterrupt):
            reply = "no: user aborted at HITL prompt"
            print()
        if not reply:
            reply = "no: empty reply, defaulting to reject"
        payload = Command(resume=reply)


def _run_turn(user_in: str, config: dict, mode: str) -> None:
    if mode == "events":
        asyncio.run(_stream_events(user_in, config))
    else:
        _stream_sync(user_in, config, mode)


# ---------- REPL helpers (same as CH03/CH04) ----------------------------------

def _print_history(thread_id: str) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    snaps = {
        s.config["configurable"]["checkpoint_id"]: s
        for s in agent.get_state_history(config)
    }
    if not snaps:
        print("(no history for this thread)")
        return

    def parent_of(cid: str) -> str | None:
        p = (snaps[cid].parent_config or {}).get("configurable", {}).get("checkpoint_id")
        return p if p in snaps else None

    def label(cid: str) -> str:
        s = snaps[cid]
        msgs = s.values.get("messages", []) if isinstance(s.values, dict) else []
        last = _summarize_msg(msgs[-1]) if msgs else "-"
        nxt = ",".join(s.next) if s.next else "END"
        return f"{_short(cid):<10} next={nxt:<22} msgs={len(msgs):<2}  {last}"

    # `git log --graph --all` style. get_state_history returns checkpoints
    # newest-first; the branch structure lives in each snapshot's parent_config.
    # We walk newest -> oldest keeping one "lane" per live branch tip:
    #   *        a checkpoint in this lane
    #   | *      a checkpoint in a side branch (parent lane still pending)
    #   |/       a fork point: the side lane rejoins its parent's lane
    lanes: list[str | None] = []   # lanes[i] = the cid that lane i is waiting to emit
    rows: list[tuple[str, str | None]] = []
    for cid in snaps:
        cols = [i for i, x in enumerate(lanes) if x == cid]
        if not cols:                       # a branch tip: open a new lane
            lanes.append(cid)
            cols = [len(lanes) - 1]
        col = cols[0]
        # A fork point is awaited by several lanes; collapse the extras first.
        for d in sorted((i for i in cols if i != col), reverse=True):
            lanes[d] = None
            merge = ["|" if lanes[i] is not None else " " for i in range(d)]
            rows.append((" ".join(merge) + "/", None))
            while lanes and lanes[-1] is None:
                lanes.pop()
        cells = ["*" if i == col else ("|" if x is not None else " ")
                 for i, x in enumerate(lanes)]
        rows.append((" ".join(cells), label(cid)))
        parent = parent_of(cid)
        lanes[col] = parent
        if parent is None:                 # reached a root: close the lane
            lanes[col] = None
            while lanes and lanes[-1] is None:
                lanes.pop()

    width = max(len(g) for g, _ in rows)
    print(f"--- history graph for thread {thread_id} "
          f"({len(snaps)} checkpoints, newest at top) ---")
    for graph, lbl in rows:
        print(graph if lbl is None else f"{graph:<{width}}  {lbl}")
    print()


def _new_thread_id() -> str:
    return "demo-" + uuid.uuid4().hex[:6]


# ---------- REPL --------------------------------------------------------------

if __name__ == "__main__":
    thread_id = _new_thread_id()
    pending_fork_prefix: str | None = None
    current_mode = "updates"

    print("[init] CH06 — interrupt() HITL (human-in-the-loop)")
    print(f"[init] thread_id   = {thread_id}")
    print(f"[init] stream mode = {current_mode}  (try /mode for others)")
    print("[hint] try:")
    print("       比較 AAPL 的 SMA 20、50、100 三組策略")
    print("       (the graph will pause at human_approval; reply 'yes' or 'no: ...')")
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
