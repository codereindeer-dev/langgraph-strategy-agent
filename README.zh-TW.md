# langgraph-strategy-agent

> 🌐 [English](README.md) ・ **繁體中文**（你正在讀）

LangGraph 教學專案 —— 每個 commit 對應一個章節，從第一個 `StateGraph` 一路堆到 streaming、checkpointer、time travel、HITL。題目用「AI 策略研究員」當載體：使用者描述策略想法 → LLM 用工具抓資料/算指標 → 之後接 `backtesting.py` 產 code → 跑回測 → 看結果迭代。

姊妹專案：[minimal-agent](https://github.com/codereindeer-dev/minimal-agent)（不用 framework，純 Anthropic SDK 從零寫的版本）。兩個專案題目相同（agent loop + tools + memory + HITL），對照著看可以清楚看出 framework 抽象帶來的取捨。

> ⚠️ **這是教學 / 研究工具，不是投資建議**。LLM 產出的策略可能有 lookahead bias、overfit、其他問題；過去績效不代表未來。不要拿這些 code 直接上實盤。

---

## 它是什麼

一個 `agent.py`，每個 commit 是一個 LangGraph 概念的最小可跑版本：

- **CH01** — 單節點 `StateGraph` + `TypedDict` + `add_messages` reducer
- **CH02** — 加 `@tool` 工具（`get_price_data` / `compute_sma`，md5 seed 可重現）+ `tools_condition` 條件邊 + `tools → llm` 迴圈（ReAct loop）
- **CH03** — `InMemorySaver` checkpointer，每個 node 邊界自動存 state snapshot。多輪對話、`/history` 看 checkpoint 鏈、`/fork` 從任一歷史點 replay（time travel）
- **CH04** — `agent.stream()` / `astream_events()`，五種 stream mode（updates / values / messages / debug / events）即時看圖內事件
- **CH05** — Subgraph + `Send` API：`compare_strategies` 工具觸發 N 個 `backtest_subgraph` 平行 fan-out（fetch → sma → score 各自跑），`finalize` 把結果 fan-in 成一個 `ToolMessage`。自訂 reducer + `stream(subgraphs=True)` 看平行執行
- **CH06** — `interrupt()` HITL：fan-out 前插入 `human_approval` node，graph 暫停等使用者批准 / 拒絕，`Command(resume=...)` 接續。拒絕路徑用 `ToolMessage` 餵 feedback 回 LLM、LLM 重新提案

目標終局（規劃中）：`backtesting.py` 整合產出可執行回測腳本。

---

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env   # 填入 ANTHROPIC_API_KEY
python agent.py
```

---

## REPL slash commands

| 指令 | 作用 |
|------|------|
| `/mode [name]` | 顯示或切換 stream mode（`updates` / `values` / `messages` / `debug` / `events`） |
| `/new` | 開新 `thread_id`（捨棄舊的對話歷史） |
| `/history` | 列出目前 thread 的所有 checkpoint（短 id、next 節點、訊息數、最後一筆摘要） |
| `/fork <ckpt>` | 下一則訊息從指定 checkpoint replay（time travel） |
| `/exit` | 離開 |

---

## Stream modes（CH04）

| Mode | 看什麼 |
|------|--------|
| `updates` | 每個 node 執行完輸出 channel diff（預設、最常用） |
| `values` | 每一步完整 state 快照 |
| `messages` | LLM token 即時噴出，像打字動畫 |
| `debug` | 圖內部事件：task start/end、checkpoint write |
| `events` | `astream_events v2`：per-token + per-tool start/end，最細 |

---

## 工具（mock）

| Tool | 行為 |
|------|------|
| `get_price_data(ticker, days=30)` | 用 `md5(ticker\|days)` seed 的偽隨機 walk 產出 daily close。可重現、無外部 API |
| `compute_sma(prices, window)` | 標準 SMA（window 內平均） |
| `compare_strategies(ticker, windows)` *(CH05)* | LLM-only schema —— 實際執行被 conditional edge 攔截，fan-out 成 N 個 `backtest_subgraph` 分支跑「價格 vs SMA」交叉策略，回 per-window 的報酬率 / Sharpe / 交易次數。CH06 起多了 `human_approval` HITL gate：fan-out 前先暫停問人類批准 |

工具是教學用的最小可跑版本 —— 真實接 yfinance / 券商 API 是之後章節的事。重點在 graph 架構與 agent 行為，不在資料源。

---

## 專案結構

```
agent.py            # 全部邏輯，每個 commit 替換掉這份檔案
requirements.txt    # langgraph + langchain-anthropic + python-dotenv
README.md           # 你正在讀這個
```

---

## Commit 演進

| Commit | 章節 | 加了什麼 |
|--------|------|---------|
| `ad8cb9d` | scaffold | requirements / .gitignore / README 骨架 |
| `bd68c24` | **CH01** | 單節點 `StateGraph`：START → `llm` → END，`TypedDict` + `add_messages` reducer，stateless 單輪 |
| `64628c7` | **CH02** | 加 ReAct loop：`@tool` 工具、`tools_condition` 條件邊、`tools → llm` 迴圈、Windows UTF-8 stdout fix |
| `9680148` | **CH03** | `InMemorySaver` checkpointer → 多輪對話 + `/history` 看 checkpoint 鏈 + `/fork` time travel |
| `1e35afb` | **CH04** | `stream()` / `astream_events()`，五種 stream mode、`/mode` 即時切換 |
| `fdd2440` | **CH05** | Subgraph + `Send` API 平行 fan-out：`compare_strategies` 工具觸發 N 個 `backtest_subgraph`、`finalize` fan-in 成單一 `ToolMessage`、自訂 reducer、`subgraphs=True` 看平行執行 |
| `0cc83b8` | **CH06** | `interrupt()` HITL：`human_approval` node 在 fan-out 前暫停 graph、`Command(resume=...)` 接續；拒絕路徑用 `ToolMessage` 餵 feedback 讓 LLM 重新提案；checkpoint 保留暫停點、可跨 process resume |

照著讀的方式：`git checkout bd68c24` 看最簡單的版本（單節點圖），然後一路 `git log -p` 往新的 commit diff 過去，每個 chapter 都是一個明確、可獨立理解的概念加法。

### 規劃中

- **CH07** — `backtesting.py` 整合：LLM 產出 `Strategy` 子類 → 沙箱執行 → 回傳 metrics（HITL 沿用 CH06、執行前讓人類 review code）
- **CH08** — 持久化 checkpointer（SQLite / Postgres）+ 跨 session 接續

---

## 適合誰讀

- 已經理解 agent loop 的基本概念（建議先看 [minimal-agent](https://github.com/codereindeer-dev/minimal-agent)）
- 想知道 LangGraph 多給了什麼：checkpoint、time travel、stream mode、subgraph、`interrupt()` HITL
- 已經有 Python 基礎、看得懂回測概念（不需要會寫策略，AI 會寫；但要看得懂 code）
- 想看 framework 抽象帶來的 trade-off：對照 minimal-agent 同樣功能要手寫多少
