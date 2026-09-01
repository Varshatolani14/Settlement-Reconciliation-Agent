"""
Settlement Q&A agent (RAG).

Builds one short document per payment (its joined payment / order / settlement
view plus the reconciliation decision) and per audit entry, embeds them, and at
question time retrieves the top-k by cosine similarity and hands them to GPT-4o
to answer — so "why didn't payment PAY-9042 settle?" is answered from the actual
records, not a guess.

Retrieval: OpenAI ``text-embedding-3-small`` + cosine similarity when a key is
present; a bag-of-words cosine fallback otherwise.
"""

from __future__ import annotations

import pandas as pd

from .llm import LLM, QAAnswer, cosine


class SettlementQA:
    def __init__(self, llm: LLM | None = None) -> None:
        self.llm = llm or LLM()
        self.docs: list[str] = []
        self._vectors: list[list[float]] = []

    # ------------------------------------------------------------------ indexing
    def build_index(
        self,
        results: pd.DataFrame,
        payments: pd.DataFrame,
        orders: pd.DataFrame,
        audit: list[dict],
    ) -> None:
        order_by_id = {r["order_id"]: r for r in orders.to_dict("records")}
        pay_by_id = {r["payment_id"]: r for r in payments.to_dict("records")}
        docs: list[str] = []

        for r in results.to_dict("records"):
            if r.get("payment_id"):
                p = pay_by_id.get(r["payment_id"], {})
                o = order_by_id.get(r.get("order_id"), {})
                docs.append(
                    f"[RECONCILIATION] Payment {r['payment_id']} (order {r.get('order_id')}, "
                    f"merchant {r['merchant_id']}, {r.get('payment_date')}, ₹{_f(r.get('amount'))}, "
                    f"method {p.get('method', '?')}). Order status {o.get('status', '?')}. "
                    f"Outcome: {r['status']} / {r['category']} via {r['method']} "
                    f"(confidence {r.get('confidence')}). "
                    f"Settlement: {r.get('utr') or 'none'} "
                    f"gross ₹{_f(r.get('gross_amount'))}, fee ₹{_f(r.get('fee'))}, "
                    f"GST ₹{_f(r.get('gst'))}, net ₹{_f(r.get('net_amount'))}, "
                    f"variance ₹{_f(r.get('variance'))}, dated {r.get('settlement_date') or 'n/a'}. "
                    f"Reason: {r.get('rationale')}"
                )
            else:  # orphan settlement
                docs.append(
                    f"[ORPHAN SETTLEMENT] UTR {r.get('utr')} for merchant {r['merchant_id']} "
                    f"gross ₹{_f(r.get('gross_amount'))} dated {r.get('settlement_date')}. "
                    f"{r.get('rationale')}"
                )

        for a in audit:
            docs.append(
                f"[AUDIT {a['timestamp']}] {a['event']} {a['ref']} — {a['detail']} "
                f"({a['method']}, {a['status']})"
            )

        self.docs = docs
        self._vectors = self.llm.embed(docs) if docs else []

    # ---------------------------------------------------------------- retrieval
    def retrieve(self, question: str, k: int = 6) -> list[str]:
        if not self.docs:
            return []
        qv = self.llm.embed([question])[0]
        scored = sorted(
            zip(self.docs, self._vectors),
            key=lambda pair: cosine(qv, pair[1]),
            reverse=True,
        )
        return [d for d, _ in scored[:k]]

    # -------------------------------------------------------------------- answer
    def ask(self, question: str, k: int = 6) -> QAAnswer:
        ctx = self.retrieve(question, k=k)
        return self.llm.answer_question(question, ctx)


def _f(v) -> str:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "-"
        return f"{float(v):.2f}"
    except Exception:
        return str(v)
