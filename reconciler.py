"""
The matching engine.

Two passes, exactly as described in the README:

  1. Deterministic pass
     Exact / near-exact matching on order id, amount (within fee tolerance),
     and a settlement date window. Reliable, so it runs first and takes
     everything it can.

  2. LLM fallback pass
     Whatever the rules can't resolve is handed to the LLM (GPT-4o), which
     reasons over the remaining candidate settlements — "gross is payment
     minus a 2% fee, dates are 2 days apart, same transaction?" — and either
     matches with a stated rationale or routes the record to the exception
     list with a specific reason.

Output: a per-payment results table, an exception list, per-run stats, and a
list of audit records (one per decision).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd

from .llm import LLM

# deterministic-pass parameters
DATE_WINDOW_DAYS = 3          # UTR may land up to 3 days after the payment (ref match)
RULE_AMOUNT_WINDOW_DAYS = 2   # tighter window when matching on amount alone (no ref)
DEFAULT_FEE_TOLERANCE = 0.03  # 3% of payment amount absorbed as fee/GST by the rule
ABS_TOLERANCE = 1.0           # plus a flat ₹1 for rounding

# LLM-pass candidate net
LLM_DATE_WINDOW_DAYS = 7
LLM_AMOUNT_BAND = 0.60        # consider settlements whose gross is within 60% of the payment


@dataclass
class ReconResult:
    results: pd.DataFrame
    exceptions: pd.DataFrame
    stats: dict
    audit: list[dict] = field(default_factory=list)


def _ref(value) -> str:
    """Normalise a payment_ref cell: blank / NaN / 'nan' / 'none' all mean 'no ref'."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(value).strip()
    return "" if s.lower() in {"", "nan", "none", "null"} else s


def _as_date(value) -> date | None:
    if isinstance(value, date):
        return value
    try:
        return datetime.fromisoformat(str(value)[:10]).date()
    except Exception:
        return None


def reconcile(
    payments: pd.DataFrame,
    settlements: pd.DataFrame,
    orders: pd.DataFrame,
    llm: LLM | None = None,
    fee_tolerance: float = DEFAULT_FEE_TOLERANCE,
    merchant_filter: str | None = None,
) -> ReconResult:
    llm = llm or LLM()
    pays = payments.copy()
    setts = settlements.copy()
    ords = orders.copy()

    if merchant_filter and merchant_filter != "ALL":
        pays = pays[pays["merchant_id"] == merchant_filter].reset_index(drop=True)
        setts = setts[setts["merchant_id"] == merchant_filter].reset_index(drop=True)
        ords = ords[ords["merchant_id"] == merchant_filter].reset_index(drop=True)

    order_by_id = {r["order_id"]: r for r in ords.to_dict("records")}
    settle_records = setts.to_dict("records")
    matched_utrs: set[str] = set()
    rows: list[dict] = []
    audit: list[dict] = []

    deferred: list[dict] = []

    # ------------------------------------------------------------------ pass 1
    for pay in pays.to_dict("records"):
        pid = pay["payment_id"]
        amount = float(pay["amount"])
        p_date = _as_date(pay["payment_date"])
        order = order_by_id.get(pay["order_id"])
        tol = max(amount * fee_tolerance, ABS_TOLERANCE)

        # Sweep A — trust an explicit payment_ref link first (most reliable).
        ref_hits = []
        for s in settle_records:
            if s["utr"] in matched_utrs or s["merchant_id"] != pay["merchant_id"]:
                continue
            if _ref(s.get("payment_ref")) != pid:
                continue
            s_date = _as_date(s["settlement_date"])
            lag = (s_date - p_date).days if (s_date and p_date) else 99
            ref_hits.append((abs(lag), abs(float(s["gross_amount"]) - amount), s))

        # Sweep B — no ref: amount within the fee tolerance inside a tight
        # window, against settlements that carry no ref of their own (so we
        # never steal a ref-linked record from the payment it belongs to).
        amt_hits = []
        if not ref_hits:
            for s in settle_records:
                if s["utr"] in matched_utrs or s["merchant_id"] != pay["merchant_id"]:
                    continue
                if _ref(s.get("payment_ref")):
                    continue
                s_date = _as_date(s["settlement_date"])
                lag = (s_date - p_date).days if (s_date and p_date) else 99
                if lag < 0 or lag > RULE_AMOUNT_WINDOW_DAYS:
                    continue
                gap = abs(float(s["gross_amount"]) - amount)
                if gap <= tol:
                    amt_hits.append((gap, lag, s))

        if ref_hits:
            rule_hit = min(ref_hits, key=lambda t: (t[0], t[1]))[2]
            rule_basis = "payment_ref + date window"
        elif amt_hits:
            # accept only an unambiguous winner — one candidate, or a gap
            # clearly smaller than the runner-up — otherwise defer to the LLM
            amt_hits.sort(key=lambda t: (t[0], t[1]))
            if len(amt_hits) == 1 or (amt_hits[1][0] - amt_hits[0][0]) > ABS_TOLERANCE:
                rule_hit = amt_hits[0][2]
                rule_basis = "amount within fee tolerance + date window"
            else:
                deferred.append(pay)
                continue
        else:
            deferred.append(pay)
            continue

        gross = float(rule_hit["gross_amount"])
        fee = float(rule_hit["fee"])
        gst = float(rule_hit["gst"])
        net = float(rule_hit["net_amount"])
        variance = round(amount - gross, 2)
        category = "Exact Match" if abs(variance) < 0.01 and fee == 0 else "Fee / GST Deduction"
        matched_utrs.add(rule_hit["utr"])
        detail = (
            f"Matched to {rule_hit['utr']} on {rule_basis}. "
            f"Gross ₹{gross:.2f}, fee ₹{fee:.2f}, GST ₹{gst:.2f}, net ₹{net:.2f}."
        )
        rows.append(
            {
                "payment_id": pid,
                "order_id": pay["order_id"],
                "merchant_id": pay["merchant_id"],
                "payment_date": pay["payment_date"],
                "amount": amount,
                "utr": rule_hit["utr"],
                "settlement_date": rule_hit["settlement_date"],
                "gross_amount": gross,
                "fee": fee,
                "gst": gst,
                "net_amount": net,
                "variance": variance,
                "status": "MATCHED_RULE",
                "category": category,
                "method": "deterministic-rule",
                "confidence": 99,
                "rationale": detail,
                "order_status": (order or {}).get("status", "UNKNOWN"),
            }
        )
        audit.append(_audit_row("MATCH_RULE", pid, "deterministic-rule", detail, "MATCHED"))

    # ------------------------------------------------------------------ pass 2
    # Propose a 1:1 payment->settlement mapping over everything the rules left
    # open, scored by fee/refund/lag plausibility and assigned greedily (best
    # score first). The LLM then confirms or rejects each proposed pair and
    # supplies the rationale; unassigned payments go to the LLM with no
    # candidate and fall through to the exception list.
    open_setts = [s for s in settle_records if s["utr"] not in matched_utrs]
    sett_by_utr = {s["utr"]: s for s in open_setts}

    scored_pairs: list[tuple[float, str, str]] = []
    for pay in deferred:
        p_date = _as_date(pay["payment_date"])
        amount = float(pay["amount"])
        for s in open_setts:
            if s["merchant_id"] != pay["merchant_id"]:
                continue
            s_date = _as_date(s["settlement_date"])
            lag = (s_date - p_date).days if (s_date and p_date) else 99
            if not (0 <= lag <= LLM_DATE_WINDOW_DAYS):
                continue
            sc = _pair_score(amount, float(s["gross_amount"]), lag)
            if sc > 0:
                scored_pairs.append((sc, pay["payment_id"], s["utr"]))

    scored_pairs.sort(reverse=True)
    assigned: dict[str, dict] = {}
    used_setts: set[str] = set()
    for _sc, pid_, utr_ in scored_pairs:
        if pid_ in assigned or utr_ in used_setts:
            continue
        assigned[pid_] = sett_by_utr[utr_]
        used_setts.add(utr_)

    def _cand(s: dict, p_date) -> dict:
        s_date = _as_date(s["settlement_date"])
        return {
            "utr": s["utr"],
            "settlement_date": s["settlement_date"],
            "gross_amount": float(s["gross_amount"]),
            "fee": float(s["fee"]),
            "gst": float(s["gst"]),
            "net_amount": float(s["net_amount"]),
            "payment_ref": _ref(s.get("payment_ref")),
            "days_after_payment": (s_date - p_date).days if (s_date and p_date) else None,
        }

    for pay in deferred:
        pid = pay["payment_id"]
        amount = float(pay["amount"])
        p_date = _as_date(pay["payment_date"])
        order = order_by_id.get(pay["order_id"])

        proposed = assigned.get(pid)
        candidates = []
        if proposed is not None:
            candidates.append(_cand(proposed, p_date))
        # a couple of alternates for context, so the LLM can still say "no"
        for s in open_setts:
            if len(candidates) >= 3:
                break
            if s["merchant_id"] != pay["merchant_id"] or s["utr"] in used_setts:
                continue
            s_date = _as_date(s["settlement_date"])
            lag = (s_date - p_date).days if (s_date and p_date) else 99
            if 0 <= lag <= LLM_DATE_WINDOW_DAYS:
                candidates.append(_cand(s, p_date))

        verdict = llm.resolve_match(
            {
                "payment_id": pid,
                "order_id": pay["order_id"],
                "merchant_id": pay["merchant_id"],
                "payment_date": pay["payment_date"],
                "amount": amount,
                "order_status": (order or {}).get("status", "UNKNOWN"),
            },
            candidates,
        )

        if verdict.match and verdict.settlement_utr:
            s = next((c for c in candidates if c["utr"] == verdict.settlement_utr), None)
            if s is None:
                s = candidates[0]
            gross = float(s["gross_amount"])
            variance = round(amount - gross, 2)
            lag = s.get("days_after_payment") or 0
            if variance > 0.01:
                category = "Partial Refund"
            elif lag > RULE_AMOUNT_WINDOW_DAYS:
                category = "Settlement Lag"
            else:
                category = "Fee / GST Deduction"
            matched_utrs.add(s["utr"])
            rows.append(
                {
                    "payment_id": pid,
                    "order_id": pay["order_id"],
                    "merchant_id": pay["merchant_id"],
                    "payment_date": pay["payment_date"],
                    "amount": amount,
                    "utr": s["utr"],
                    "settlement_date": s["settlement_date"],
                    "gross_amount": gross,
                    "fee": float(s["fee"]),
                    "gst": float(s["gst"]),
                    "net_amount": float(s["net_amount"]),
                    "variance": variance,
                    "status": "MATCHED_LLM",
                    "category": category,
                    "method": f"llm-fallback ({verdict.engine})",
                    "confidence": verdict.confidence,
                    "rationale": verdict.rationale,
                    "order_status": (order or {}).get("status", "UNKNOWN"),
                }
            )
            audit.append(
                _audit_row(
                    "MATCH_LLM", pid, f"llm-fallback ({verdict.engine})",
                    f"Matched to {s['utr']}. {verdict.rationale}", "MATCHED",
                )
            )
        else:
            rows.append(
                {
                    "payment_id": pid,
                    "order_id": pay["order_id"],
                    "merchant_id": pay["merchant_id"],
                    "payment_date": pay["payment_date"],
                    "amount": amount,
                    "utr": None,
                    "settlement_date": None,
                    "gross_amount": None,
                    "fee": None,
                    "gst": None,
                    "net_amount": None,
                    "variance": None,
                    "status": "EXCEPTION",
                    "category": "No Settlement Found",
                    "method": f"llm-fallback ({verdict.engine})",
                    "confidence": verdict.confidence,
                    "rationale": verdict.rationale,
                    "order_status": (order or {}).get("status", "UNKNOWN"),
                }
            )
            audit.append(
                _audit_row(
                    "EXCEPTION", pid, f"llm-fallback ({verdict.engine})",
                    verdict.rationale, "UNRESOLVED",
                )
            )

    # ------------------------------------------------------ orphan settlements
    orphan_rows: list[dict] = []
    for s in settle_records:
        if s["utr"] in matched_utrs:
            continue
        reason = (
            f"Settlement {s['utr']} (₹{float(s['gross_amount']):.2f}, {s['settlement_date']}) "
            f"is present in the bank records but no payment in this batch corresponds to it."
        )
        orphan_rows.append(
            {
                "payment_id": None,
                "order_id": None,
                "merchant_id": s["merchant_id"],
                "payment_date": None,
                "amount": None,
                "utr": s["utr"],
                "settlement_date": s["settlement_date"],
                "gross_amount": float(s["gross_amount"]),
                "fee": float(s["fee"]),
                "gst": float(s["gst"]),
                "net_amount": float(s["net_amount"]),
                "variance": None,
                "status": "EXCEPTION",
                "category": "Orphan Settlement",
                "method": "deterministic-rule",
                "confidence": 97,
                "rationale": reason,
                "order_status": "N/A",
            }
        )
        audit.append(_audit_row("EXCEPTION", s["utr"], "deterministic-rule", reason, "UNRESOLVED"))

    results = pd.DataFrame(rows + orphan_rows)
    exceptions = results[results["status"] == "EXCEPTION"].reset_index(drop=True)

    n_payments = len(pays)
    matched = int((results["status"].isin(["MATCHED_RULE", "MATCHED_LLM"])).sum())
    by_rule = int((results["status"] == "MATCHED_RULE").sum())
    by_llm = int((results["status"] == "MATCHED_LLM").sum())
    n_exceptions = int((results["status"] == "EXCEPTION").sum())
    denom = n_payments + len(orphan_rows)
    match_rate = round(100 * matched / denom, 1) if denom else 0.0

    stats = {
        "batch_size": denom,
        "payments": n_payments,
        "matched": matched,
        "matched_by_rule": by_rule,
        "matched_by_llm": by_llm,
        "exceptions": n_exceptions,
        "match_rate": match_rate,
        "gross_volume": round(float(pays["amount"].sum()), 2),
        "settled_net": round(float(results["net_amount"].fillna(0).sum()), 2),
        "discrepancy_value": round(
            float(results.loc[results["status"] != "MATCHED_RULE", "variance"].fillna(0).abs().sum()), 2
        ),
        "llm_engine": llm.engine_name,
        "llm_available": llm.available,
    }

    audit.insert(
        0,
        _audit_row(
            "RUN_START", f"BATCH-{n_payments}", "reconciliation-engine",
            f"Two-pass reconciliation over {n_payments} payments / {len(settle_records)} settlements. "
            f"Fee tolerance {fee_tolerance:.0%} + ₹{ABS_TOLERANCE:.2f}, date window {DATE_WINDOW_DAYS}d. "
            f"LLM engine: {llm.engine_name}.",
            "SUCCESS",
        ),
    )
    audit.append(
        _audit_row(
            "RUN_SUMMARY", f"BATCH-{n_payments}", "reconciliation-engine",
            f"Match rate {match_rate}% — {by_rule} by rule, {by_llm} by LLM, {n_exceptions} exceptions.",
            "SUCCESS",
        )
    )

    return ReconResult(results=results, exceptions=exceptions, stats=stats, audit=audit)


def _pair_score(amount: float, gross: float, lag: int) -> float:
    """Plausibility that a deferred payment settles as this open settlement.
    Higher is better; 0 means "not a candidate". Used to build the greedy 1:1
    map the LLM fallback pass then confirms."""
    diff = amount - gross
    if abs(diff) <= ABS_TOLERANCE:              # fee/GST deducted post-settlement, or pure lag
        return 1.00 - 0.03 * min(lag, 10)
    if amount * 0.08 <= diff <= amount * 0.60:  # settlement gross = payment minus a partial refund
        frac = diff / amount
        tidy = min(abs(frac - t) for t in (0.25, 0.30, 0.40, 0.50))
        return 0.60 - 0.03 * min(lag, 10) - 0.4 * tidy
    return 0.0


def _audit_row(event: str, ref: str, method: str, detail: str, status: str) -> dict:
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event": event,
        "ref": ref,
        "actor": "reconciliation-engine",
        "method": method,
        "detail": detail,
        "status": status,
    }
