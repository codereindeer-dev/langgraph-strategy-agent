# langgraph-strategy-agent

LangGraph 教學專案 —— 用 graph + HITL 編排 AI 當「策略研究員」：使用者描述策略想法 → LLM 產出 `backtesting.py` 策略 code → 跑回測 → 看結果迭代。每個 commit 對應一個教學章節，從第一個 StateGraph 一路加到 subgraph 平行 + `interrupt()` HITL。

姊妹專案：[minimal-agent](https://github.com/codereindeer-dev/minimal-agent)（不用 framework，純 Anthropic SDK 從零寫的版本）。

> ⚠️ **這是教學 / 研究工具，不是投資建議**。 LLM 產出的策略可能有 lookahead bias、overfit、其他問題；過去績效不代表未來。 不要拿這些 code 直接上實盤。

---

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env   # 填入 ANTHROPIC_API_KEY
```

各章節的執行方式請看對應 commit 的程式碼。

---

## 章節進度

待章節陸續完成後填入。

---

## 適合誰讀

- 已經理解 agent loop 的基本概念（建議先看 minimal-agent 系列）
- 想知道 LangGraph 多給了什麼：checkpoint、time travel、平行 subgraph、`interrupt()` HITL
- 已經有 Python 基礎、看得懂回測概念（不需要會寫策略，AI 會寫；但要看得懂 code）
- 想看 framework 抽象帶來的 trade-off
