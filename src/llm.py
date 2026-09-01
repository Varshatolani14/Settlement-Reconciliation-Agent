"""
Thin wrapper around the OpenAI API (GPT-4o) used for:
  1. the LLM fallback pass in the matching engine (fuzzy match reasoning)
  2. the settlement Q&A agent
  3. embeddings for the Q&A retrieval step

If no OPENAI_API_KEY is present the wrapper degrades to a deterministic
offline heuristic so the app still runs end to end for a demo. Every result
carries an ``engine`` field ("gpt-4o" or "offline-heuristic") so the UI and the
audit log can state honestly which path produced a decision.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field

try:  # optional convenience
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")


@dataclass
class MatchVerdict:
    match: bool
    settlement_utr: str | None
    rationale: str
    confidence: int
    engine: str


@dataclass
class QAAnswer:
    answer: str
    engine: str
    used_context: list[str] = field(default_factory=list)


class LLM:
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self._client = None
        if self.api_key and self.api_key != "your_key_here":
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.api_key)
            except Exception:
                self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def engine_name(self) -> str:
        return MODEL if self.available else "offline-heuristic"

    # ------------------------------------------------------------------ matching
    def resolve_match(self, payment: dict, candidates: list[dict]) -> MatchVerdict:
        """Ask the model whether the payment corresponds to one of the candidate
        settlement records, and why."""
        if not candidates:
            return MatchVerdict(
                False, None,
                "No settlement record found within the extended date window.",
                95, self.engine_name,
            )

        if self.available:
            try:
                return self._resolve_match_llm(payment, candidates)
            except Exception as exc:  # fall through to heuristic on any API error
                verdict = _heuristic_match(payment, candidates)
                verdict.rationale += f" (LLM call failed: {exc}; used offline heuristic)"
                verdict.engine = "offline-heuristic"
                return verdict
        return _heuristic_match(payment, candidates)

    def _resolve_match_llm(self, payment: dict, candidates: list[dict]) -> MatchVerdict:
        sys = (
            "You are a settlement reconciliation analyst. You are given one merchant "
            "payment and a short list of candidate bank settlement (UTR) records. "
            "Fees and GST are deducted before settlement, refunds reduce the settled "
            "gross, and UTRs can land a few days after the payment. Decide whether the "
            "payment settles as exactly one of the candidates. Respond with STRICT JSON: "
            '{"match": bool, "settlement_utr": string|null, "rationale": string, '
            '"confidence": integer 0-100}. Keep the rationale to one sentence and cite '
            "the numbers you used."
        )
        user = json.dumps({"payment": payment, "candidate_settlements": candidates}, default=str)
        resp = self._client.chat.completions.create(
            model=MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        )
        raw = resp.choices[0].message.content
        data = json.loads(raw)
        return MatchVerdict(
            match=bool(data.get("match")),
            settlement_utr=data.get("settlement_utr"),
            rationale=str(data.get("rationale", "")).strip(),
            confidence=int(data.get("confidence", 80)),
            engine=MODEL,
        )

    # ----------------------------------------------------------------------- Q&A
    def answer_question(self, question: str, context_docs: list[str]) -> QAAnswer:
        if self.available and context_docs:
            try:
                sys = (
                    "You are the Settlement Reconciliation Q&A agent. Answer ONLY from the "
                    "provided context blocks, which are the reconciliation records and audit "
                    "log for this batch. If the answer is not in the context, say so. Be "
                    "concise and quote the reference IDs and amounts you rely on."
                )
                user = "CONTEXT:\n" + "\n---\n".join(context_docs) + f"\n\nQUESTION: {question}"
                resp = self._client.chat.completions.create(
                    model=MODEL,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": sys},
                        {"role": "user", "content": user},
                    ],
                )
                return QAAnswer(resp.choices[0].message.content.strip(), MODEL, context_docs)
            except Exception as exc:
                return QAAnswer(
                    _heuristic_answer(question, context_docs)
                    + f"\n\n_(LLM call failed: {exc}; answered from retrieved records.)_",
                    "offline-heuristic",
                    context_docs,
                )
        return QAAnswer(_heuristic_answer(question, context_docs), "offline-heuristic", context_docs)

    # ---------------------------------------------------------------- embeddings
    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.available:
            try:
                resp = self._client.embeddings.create(model=EMBED_MODEL, input=texts)
                return [d.embedding for d in resp.data]
            except Exception:
                pass
        return [_bow_vector(t) for t in texts]


# --------------------------------------------------------------------------- #
# Offline heuristics (used when no API key / API failure)
# --------------------------------------------------------------------------- #
def _heuristic_match(payment: dict, candidates: list[dict]) -> MatchVerdict:
    amount = float(payment["amount"])
    p_date = _as_date(payment["payment_date"])
    best = None
    best_reason = ""

    for c in candidates:
        gross = float(c["gross_amount"])
        c_date = _as_date(c["settlement_date"])
        lag = (c_date - p_date).days if (c_date and p_date) else 99
        diff = round(amount - gross, 2)

        # amount matches within a rupee -> settlement lag case
        if abs(diff) <= 1.0 and 0 <= lag <= 7:
            best, best_reason = c, (
                f"Settlement gross ₹{gross:.2f} equals payment ₹{amount:.2f}; "
                f"UTR landed {lag} day(s) after the payment (settlement lag)."
            )
            break
        # settlement gross is smaller by 10-60% -> partial refund case
        if 0 < diff <= amount * 0.6 and diff >= amount * 0.10 and 0 <= lag <= 7:
            best, best_reason = c, (
                f"Settlement gross ₹{gross:.2f} = payment ₹{amount:.2f} minus a partial "
                f"refund of ₹{diff:.2f}; UTR {lag} day(s) later — same transaction."
            )
            break

    if best is not None:
        return MatchVerdict(True, best["utr"], best_reason, 82, "offline-heuristic")

    nearest = min(candidates, key=lambda c: abs(amount - float(c["gross_amount"])))
    gap = round(abs(amount - float(nearest["gross_amount"])), 2)
    return MatchVerdict(
        False, None,
        f"Amount mismatch exceeds fee tolerance — closest settlement is UTR "
        f"{nearest['utr']} at ₹{float(nearest['gross_amount']):.2f} (gap ₹{gap:.2f}).",
        88, "offline-heuristic",
    )


def _heuristic_answer(question: str, context_docs: list[str]) -> str:
    if not context_docs:
        return "No matching records were retrieved for that question."
    ref = None
    m = re.search(r"\b(PAY|ORD|UTR)[-\s]?\d{2,}\b", question, re.I)
    if m:
        ref = m.group(0).upper().replace(" ", "-")
    header = f"Closest records for **{ref}**:" if ref else "Closest records retrieved:"
    return header + "\n\n" + "\n\n".join(f"- {d}" for d in context_docs[:4])


# --------------------------------------------------------------------------- #
# tiny bag-of-words vectoriser for offline retrieval
# --------------------------------------------------------------------------- #
_TOKEN = re.compile(r"[a-z0-9]+")


def _bow_vector(text: str) -> list[float]:
    counts: dict[str, float] = {}
    for tok in _TOKEN.findall(text.lower()):
        counts[tok] = counts.get(tok, 0.0) + 1.0
    norm = math.sqrt(sum(v * v for v in counts.values())) or 1.0
    # stable ordering via hashing into a fixed 512-dim space
    vec = [0.0] * 512
    for tok, v in counts.items():
        vec[hash(tok) % 512] += v / norm
    return vec


def _as_date(value):
    from datetime import date, datetime

    if isinstance(value, date):
        return value
    try:
        return datetime.fromisoformat(str(value)[:10]).date()
    except Exception:
        return None


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)
