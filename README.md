# langgraph-strategy-agent

> 🌐 **English** (you are reading) ・ [繁體中文](README.zh-TW.md)

A LangGraph teaching project —— each commit corresponds to one chapter, progressively building from the first `StateGraph` up through streaming, checkpointer, time travel, and HITL. The vehicle is an "AI strategy researcher": user describes a strategy idea → LLM uses tools to fetch data / compute indicators → eventually pipes through `backtesting.py` for code generation → runs backtest → iterates on results.

Sister project: [minimal-agent](https://github.com/codereindeer-dev/minimal-agent) (no framework, written from scratch with the raw Anthropic SDK). Both projects share the same scope (agent loop + tools + memory + HITL), so reading them side by side makes the framework's abstractions and their trade-offs concrete.

> ⚠️ **This is a teaching / research tool, not investment advice**. LLM-generated strategies may suffer from lookahead bias, overfitting, and other problems; past performance does not predict future returns. Do not put any of this code on a live account.

---

## What it is

One `agent.py` file. Each commit is a minimal runnable version of one LangGraph concept:

- **CH01** — Single-node `StateGraph` + `TypedDict` + `add_messages` reducer
- **CH02** — Add `@tool` tools (`get_price_data` / `compute_sma`, md5-seeded for reproducibility) + `tools_condition` conditional edge + `tools → llm` loop (ReAct loop)
- **CH03** — `InMemorySaver` checkpointer; state snapshot auto-written at every node boundary. Multi-turn conversation, `/history` to view the checkpoint chain, `/fork` to replay from any historical point (time travel)
- **CH04** — `agent.stream()` / `astream_events()`, five stream modes (updates / values / messages / debug / events) for real-time visibility into in-graph events
- **CH05** — Subgraph + `Send` API: `compare_strategies` tool triggers N parallel `backtest_subgraph` fan-outs (each runs fetch → sma → score), `finalize` reduces fan-in results into a single `ToolMessage`. Custom reducer + `stream(subgraphs=True)` to surface parallel execution
- **CH06** — `interrupt()` HITL: a `human_approval` node sits before fan-out, pausing the graph for user approval / rejection, then `Command(resume=...)` continues. Rejection path feeds feedback back to the LLM as a `ToolMessage` so the LLM can re-propose

End goal (planned): `backtesting.py` integration to emit runnable backtest scripts.

---

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY
python agent.py
```

---

## REPL slash commands

| Command | What it does |
|---------|--------------|
| `/mode [name]` | Show or switch stream mode (`updates` / `values` / `messages` / `debug` / `events`) |
| `/new` | Start a new `thread_id` (discard previous conversation history) |
| `/history` | List all checkpoints for the current thread (short id, next node, message count, last entry summary) |
| `/fork <ckpt>` | Replay the next message from the specified checkpoint (time travel) |
| `/exit` | Quit |

---

## Stream modes (CH04)

| Mode | What you see |
|------|--------------|
| `updates` | Channel diff after each node runs (default, most common) |
| `values` | Full state snapshot at every step |
| `messages` | LLM tokens streamed live, like a typing animation |
| `debug` | Graph-internal events: task start/end, checkpoint write |
| `events` | `astream_events v2`: per-token + per-tool start/end, finest grain |

---

## Tools (mock)

| Tool | Behavior |
|------|----------|
| `get_price_data(ticker, days=30)` | Uses `md5(ticker\|days)` seeded pseudo-random walk to produce daily closes. Reproducible, no external API |
| `compute_sma(prices, window)` | Standard SMA (mean over the window) |
| `compare_strategies(ticker, windows)` *(CH05)* | LLM-only schema —— actual execution is intercepted by a conditional edge and fans out to N `backtest_subgraph` branches running a "price vs SMA" crossover strategy, returning per-window total return / Sharpe / number of trades. From CH06 onward this also goes through a `human_approval` HITL gate: the graph pauses for user approval before fan-out |

Tools are minimal teaching-grade versions —— hooking up yfinance / broker APIs is a job for later chapters. The focus is graph architecture and agent behavior, not the data source.

---

## Project structure

```
agent.py            # All the logic; each commit replaces this file
requirements.txt    # langgraph + langchain-anthropic + python-dotenv
README.md           # You are reading this (English)
README.zh-TW.md     # 繁體中文版
```

---

## Commit evolution

| Commit | Chapter | What was added |
|--------|---------|----------------|
| `ad8cb9d` | scaffold | requirements / .gitignore / README skeleton |
| `bd68c24` | **CH01** | Single-node `StateGraph`: START → `llm` → END, `TypedDict` + `add_messages` reducer, stateless single-turn |
| `64628c7` | **CH02** | Add ReAct loop: `@tool` tools, `tools_condition` conditional edge, `tools → llm` loop, Windows UTF-8 stdout fix |
| `9680148` | **CH03** | `InMemorySaver` checkpointer → multi-turn conversation + `/history` to view checkpoint chain + `/fork` time travel |
| `1e35afb` | **CH04** | `stream()` / `astream_events()`, five stream modes, `/mode` to switch live |
| `fdd2440` | **CH05** | Subgraph + `Send` API parallel fan-out: `compare_strategies` triggers N `backtest_subgraph` invocations, `finalize` fans them in to a single `ToolMessage`, custom reducer, `subgraphs=True` to see parallel execution |
| `0cc83b8` | **CH06** | `interrupt()` HITL: `human_approval` node pauses the graph before fan-out, `Command(resume=...)` continues; rejection path uses `ToolMessage` to feed feedback back so the LLM re-proposes; checkpoint preserves the pause point and supports cross-process resume |

How to read along: `git checkout bd68c24` for the simplest version (single-node graph), then walk forward with `git log -p` chapter by chapter. Each commit is a single, independently understandable conceptual addition.

### Planned

- **CH07** — `backtesting.py` integration: LLM produces a `Strategy` subclass → sandbox execution → return metrics (HITL from CH06 carries forward, so the user reviews code before it runs)
- **CH08** — Persistent checkpointer (SQLite / Postgres) + cross-session resume

---

## Who this is for

- You already understand the basic agent loop concept (recommend reading [minimal-agent](https://github.com/codereindeer-dev/minimal-agent) first)
- You want to know what LangGraph adds: checkpoint, time travel, stream modes, subgraphs, `interrupt()` HITL
- You have Python fluency and can read backtest code (you don't need to write strategies —— the AI does that —— but you do need to read the code)
- You want to see the trade-offs of framework abstraction: comparing minimal-agent's hand-written approach against the same functionality in LangGraph
