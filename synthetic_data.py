"""
Synthetic data generator for the three reconciliation sources.

Produces ~50 records total across:
  - orders.csv       : the merchant order ledger
  - payments.csv     : payments captured on the merchant side
  - settlements.csv  : bank settlement / UTR records

Realistic mismatches are deliberately built in, exactly as described in the README:
  - fee / GST deductions before settlement
  - partial refunds (settlement gross = payment minus refund)
  - multi-day settlement lag (UTR lands 1-4 days later)
  - a handful of genuinely unresolvable records (2-4 per batch)

The generator is seeded so every run of the batch is reproducible.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

import pandas as pd

SEED = 2026
BATCH_START = date(2026, 8, 21)

MERCHANTS = [
    {"merchant_id": "MERCH-001", "name": "Acme Global E-Commerce", "currency": "INR", "fee_pct": 0.020},
    {"merchant_id": "MERCH-002", "name": "Nova Retail Holdings", "currency": "INR", "fee_pct": 0.023},
    {"merchant_id": "MERCH-003", "name": "Fintech Pay Gateway", "currency": "INR", "fee_pct": 0.018},
]

GST_ON_FEE = 0.18  # GST is charged on the processing fee
# Bank records with no corresponding payment in the batch. Kept at 0 so the batch
# lands on the README's "2-4 unresolved records" exactly; the engine still
# detects and reports orphan settlements when uploaded data contains them.
ORPHAN_SETTLEMENTS = 0
METHODS = ["UPI", "CARD", "NETBANKING", "WALLET"]
BANKS = ["HDFC", "ICICI", "AXIS", "SBI"]


def _round2(x: float) -> float:
    return round(float(x) + 1e-9, 2)


def _merchant_for(i: int) -> dict:
    return MERCHANTS[i % len(MERCHANTS)]


def generate_batch(seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Return {'orders': df, 'payments': df, 'settlements': df} for one batch."""
    rng = random.Random(seed)

    orders: list[dict] = []
    payments: list[dict] = []
    settlements: list[dict] = []

    # ---- plan the 50-payment batch -------------------------------------------------
    # bucket           count   resolves via
    # clean             30     deterministic rule (order id + amount + window)
    # fee_variance       6     deterministic rule (net within fee tolerance)
    # missing_ref        4     deterministic rule (amount + window, no payment_ref)
    # partial_refund     3     LLM fallback (gross = payment - refund)
    # settlement_lag     3     LLM fallback (amount matches, dates 3-4 days apart, no ref)
    # unmatchable        4     exception  (no settlement record at all)
    plan = (
        ["clean"] * 30
        + ["fee_variance"] * 6
        + ["missing_ref"] * 4
        + ["partial_refund"] * 3
        + ["settlement_lag"] * 3
        + ["unmatchable"] * 4
    )
    rng.shuffle(plan)

    for idx, kind in enumerate(plan, start=1):
        m = _merchant_for(idx)
        order_id = f"ORD-{4000 + idx}"
        pay_id = f"PAY-{9000 + idx}"
        pay_day = BATCH_START + timedelta(days=rng.randint(0, 6))
        # spread amounts widely so no two payments collide inside the rule's
        # rounding tolerance (which would let the amount-only sweep mis-link them)
        base_amount = _round2(rng.randint(200, 9000) + rng.choice([0.00, 0.49, 0.75, 0.99]))
        method = rng.choice(METHODS)
        fee_pct = m["fee_pct"]

        # ---- order ledger row ---------------------------------------------------
        order_amount = base_amount
        order_status = "PAID"
        if kind == "partial_refund":
            order_status = "PARTIALLY_REFUNDED"

        orders.append(
            {
                "order_id": order_id,
                "merchant_id": m["merchant_id"],
                "order_date": pay_day.isoformat(),
                "order_amount": order_amount,
                "currency": m["currency"],
                "customer_id": f"CUST-{rng.randint(10000, 99999)}",
                "status": order_status,
            }
        )

        # ---- payment row ------------------------------------------------------
        payments.append(
            {
                "payment_id": pay_id,
                "order_id": order_id,
                "merchant_id": m["merchant_id"],
                "payment_date": pay_day.isoformat(),
                "amount": base_amount,
                "currency": m["currency"],
                "method": method,
                "gateway_desc": f"{method} CAPTURE {m['name'].split()[0].upper()} {order_id}",
            }
        )

        if kind == "unmatchable":
            # No settlement record is ever produced for this payment.
            continue

        # ---- settlement row ---------------------------------------------------
        utr = f"UTR{seed}{idx:03d}"
        gross = base_amount
        refund_amt = 0.0
        lag_days = rng.randint(1, 2)
        payment_ref = pay_id

        if kind == "partial_refund":
            refund_amt = _round2(base_amount * rng.choice([0.25, 0.30, 0.40]))
            gross = _round2(base_amount - refund_amt)
            payment_ref = ""  # refund breaks the clean ref linkage
        elif kind == "settlement_lag":
            lag_days = rng.randint(4, 5)
            payment_ref = ""
        elif kind == "missing_ref":
            payment_ref = ""
        elif kind == "fee_variance":
            # slightly irregular fee (tiered pricing) — still inside tolerance
            fee_pct = fee_pct + rng.choice([0.002, -0.001, 0.0015])

        fee = _round2(gross * fee_pct + rng.choice([0.0, 0.30, 0.50]))
        gst = _round2(fee * GST_ON_FEE)
        net = _round2(gross - fee - gst)
        settle_day = pay_day + timedelta(days=lag_days)

        settlements.append(
            {
                "utr": utr,
                "merchant_id": m["merchant_id"],
                "settlement_date": settle_day.isoformat(),
                "gross_amount": gross,
                "fee": fee,
                "gst": gst,
                "net_amount": net,
                "bank": rng.choice(BANKS),
                "payment_ref": payment_ref,
            }
        )

    # ---- 1 orphan settlement: present in bank records, matches no payment -----
    for j in range(1, 1 + ORPHAN_SETTLEMENTS):
        m = _merchant_for(j)
        settle_day = BATCH_START + timedelta(days=rng.randint(1, 6))
        gross = _round2(rng.choice([1234, 5678, 3210]) + rng.randint(0, 88))
        fee = _round2(gross * m["fee_pct"])
        gst = _round2(fee * GST_ON_FEE)
        settlements.append(
            {
                "utr": f"UTR{seed}ORPH{j}",
                "merchant_id": m["merchant_id"],
                "settlement_date": settle_day.isoformat(),
                "gross_amount": gross,
                "fee": fee,
                "gst": gst,
                "net_amount": _round2(gross - fee - gst),
                "bank": rng.choice(BANKS),
                "payment_ref": "",
            }
        )

    rng.shuffle(settlements)

    return {
        "orders": pd.DataFrame(orders),
        "payments": pd.DataFrame(payments),
        "settlements": pd.DataFrame(settlements),
    }


def write_batch(data_dir: str = "data", seed: int = SEED) -> None:
    import os

    os.makedirs(data_dir, exist_ok=True)
    batch = generate_batch(seed)
    for name, df in batch.items():
        df.to_csv(os.path.join(data_dir, f"{name}.csv"), index=False)
        print(f"wrote {data_dir}/{name}.csv  ({len(df)} rows)")


if __name__ == "__main__":
    write_batch()
