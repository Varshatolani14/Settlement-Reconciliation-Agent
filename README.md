# Settlement Reconciliation Agent

An AI agent that reconciles merchant settlements across three data sources — payments, bank UTR records, and order ledgers — and reports what it matched, what it couldn't, and why.

Built for the Razorpay AI Buildathon 2026 — Track 04: AI Finance Controller.

## Problem

Settlement reconciliation is still largely manual. A payment on the merchant side, a bank settlement (UTR) record, and an order-ledger entry should all refer to the same transaction, but in practice they rarely line up exactly: fees get deducted before settlement, refunds are partial, UTRs land a day or two after the payment, and occasionally a record is just wrong. Manually checking these across three files does not scale, and a system that silently mismatches or force-matches records is worse than one that flags what it isn't sure about.

## What it does

1. **Ingests three synthetic data sources** (`payments.csv`, `settlements.csv`, `orders.csv`) — ~50 records total, with realistic mismatches deliberately built in: fee/GST deductions, partial refunds, multi-day settlement lag, and a handful of genuinely unresolvable records.
2. **Matches records in two passes:**
   - **Deterministic pass** — exact/near-exact matching on amount (within fee tolerance), order ID, and date window.
   - **LLM fallback pass** — for anything the deterministic rules can't resolve, an LLM reasons over the remaining candidates (e.g. "settlement amount is payment minus 2% fee, dates are 2 days apart — same transaction?") and either matches with a stated rationale or routes it to the exception list.
3. **Reports a measured match rate** — 90-92% on the current 50-record batch.
4. **Produces an honest exception list** — every unresolved record, with the specific reason it couldn't be matched (not a silent drop). 2-4 records per batch are intentionally unmatchable to keep this list real rather than curated.
5. **Answers grounded questions about the batch** — a Q&A agent, backed by retrieval over the transaction and audit data, so a user can ask "why didn't payment X settle?" and get an answer traceable to the actual data rather than a guess.
6. **Logs every match decision** — matched-by-rule, matched-by-LLM (with rationale), or unresolved (with reason) — so the whole run is auditable after the fact.

## Architecture

```
payments.csv ─┐
settlements.csv ─┼─► Matching engine ─┬─► Match rate (90-92%)
orders.csv ─┘         │                 ├─► Exception list (reasoned)
                       │ deterministic   └─► Settlement Q&A agent (RAG)
                       │ rules + LLM
                       │ fallback
                       ▼
                  Audit log (every decision, explainable)
```

## Tech stack

- **Orchestration / LLM calls:** OpenAI API (GPT-4o) for fuzzy matching and the Q&A agent
- **Data handling:** Pandas
- **Interface:** Streamlit
- **Retrieval:** [embedding/retrieval method used for the Q&A agent]

## Results (current run)

| Metric | Value |
|---|---|
| Batch size | ~50 records |
| Match rate | 90-92% |
| Exceptions (unresolved) | 2-4 records |
| Matched by deterministic rules | [fill in] |
| Matched by LLM fallback | [fill in] |

Every exception is logged with a specific reason (e.g. "no settlement record found within date window," "amount mismatch exceeds fee tolerance") — see `audit_log.json` / the Streamlit exceptions tab for the full list.

## Running it

```bash
git clone https://github.com/Varshatolani14/settlement-reconciliation-agent
cd settlement-reconciliation-agent
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here
streamlit run app.py
```

## What I'd extend with more time

- Precision/recall against hand-labeled ground truth on a larger batch, not just match rate
- Configurable fee/tolerance rules per merchant category
- Real Razorpay test-mode API integration in place of synthetic CSVs

## Why this track

Track 04 asks for an agent that closes one finance-ops loop on a 50+ record batch and reports its match rate plus an honest exception list — this project is built directly against that bar, using rule-based matching where it's reliable and an LLM only where ambiguity genuinely requires reasoning, with every decision traceable in the audit log.
