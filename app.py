"""
Settlement Reconciliation Agent — Streamlit interface.

Reconciles merchant settlements across three data sources (payments, bank UTR
records, order ledgers) in two passes — deterministic rules, then an LLM
(GPT-4o) fallback — and reports what it matched, what it couldn't, and why.

Run:  streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.audit import AuditLog
from src.llm import LLM
from src.qa_agent import SettlementQA
from src.reconciler import DEFAULT_FEE_TOLERANCE, reconcile
from src.synthetic_data import generate_batch

DATA_DIR = Path("data")
REQUIRED = {
    "payments": ["payment_id", "order_id", "merchant_id", "payment_date", "amount"],
    "settlements": ["utr", "merchant_id", "settlement_date", "gross_amount", "fee", "gst", "net_amount"],
    "orders": ["order_id", "merchant_id", "order_date", "order_amount", "status"],
}

st.set_page_config(
    page_title="Settlement Reconciliation Agent",
    page_icon="🧾",
    layout="wide",
)

# --------------------------------------------------------------------------- #
# styling — dark FinOps console, modelled on the reference mock
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

      .stApp { background:#020617; color:#e2e8f0;
               font-family:'Inter',ui-sans-serif,system-ui,sans-serif; }
      header[data-testid="stHeader"] { background:transparent; }
      .block-container { padding-top:1.4rem; max-width:1240px; }
      section[data-testid="stSidebar"] { background:#0b1220; border-right:1px solid #1e293b; }
      h1,h2,h3,h4 { color:#f1f5f9 !important; letter-spacing:-0.01em; }
      code, .mono { font-family:'JetBrains Mono',monospace; }
      .muted { color:#64748b; font-size:.8rem; }

      /* ---- top header bar ---- */
      .app-header {
        display:flex; align-items:center; gap:14px;
        background:linear-gradient(90deg,#0f172a, rgba(15,23,42,.6) 55%, rgba(76,29,149,.28));
        border:1px solid #1e293b; border-radius:16px; padding:14px 18px; margin-bottom:14px;
      }
      .app-logo {
        height:44px; width:44px; border-radius:13px; flex:none;
        background:linear-gradient(135deg,#7c3aed,#4f46e5 55%,#34d399);
        display:flex; align-items:center; justify-content:center; font-size:22px;
        box-shadow:0 0 24px -6px rgba(99,102,241,.5);
      }
      .app-title { font-size:1.15rem; font-weight:700; color:#f8fafc; }
      .app-sub   { font-size:.76rem; color:#94a3b8; margin-top:1px; }
      .app-ctx   { margin-left:auto; text-align:right; font-size:.72rem; color:#94a3b8;
                   font-family:'JetBrains Mono',monospace; }
      .badge { display:inline-block; padding:2px 9px; border-radius:999px; font-size:.66rem;
               font-weight:600; background:rgba(139,92,246,.12); color:#c4b5fd;
               border:1px solid rgba(139,92,246,.28); margin-left:8px; vertical-align:middle; }

      /* ---- summary banner ---- */
      .banner {
        display:grid; grid-template-columns:repeat(4,1fr); gap:0;
        background:linear-gradient(90deg,#0f172a,#0f172a 60%,rgba(76,29,149,.22));
        border:1px solid #1e293b; border-radius:16px; padding:16px 6px; margin-bottom:16px;
      }
      .banner > div { padding:0 18px; border-right:1px solid rgba(51,65,85,.6); }
      .banner > div:last-child { border-right:none; }
      .banner .k { font-size:.68rem; color:#94a3b8; text-transform:uppercase; letter-spacing:.06em; }
      .banner .v { font-size:.9rem; color:#e2e8f0; font-weight:600; margin-top:4px; }
      .banner .v.violet { color:#c4b5fd; }
      .banner a { color:#818cf8; text-decoration:none; font-family:'JetBrains Mono',monospace; font-size:.8rem; }

      /* ---- KPI cards ---- */
      .kpi {
        background:linear-gradient(180deg,rgba(30,41,59,.7),rgba(15,23,42,.7));
        border:1px solid #1e293b; border-radius:16px; padding:16px 18px; height:100%;
        backdrop-filter:blur(8px);
      }
      .kpi-top { display:flex; align-items:center; justify-content:space-between; }
      .kpi .label { font-size:.68rem; text-transform:uppercase; letter-spacing:.08em; color:#94a3b8; }
      .kpi .chip  { height:34px; width:34px; border-radius:10px; display:flex; align-items:center;
                    justify-content:center; font-size:17px; }
      .kpi .value { font-size:1.7rem; font-weight:700; color:#f8fafc; margin-top:10px; }
      .kpi .sub   { font-size:.74rem; color:#64748b; margin-top:3px; }
      .kpi .bar   { height:6px; border-radius:999px; background:#1e293b; margin-top:10px; overflow:hidden; }
      .kpi .bar > span { display:block; height:100%; border-radius:999px;
                         background:linear-gradient(90deg,#10b981,#34d399); }
      .kpi.violet  { border-left:3px solid #8b5cf6; } .kpi.violet  .chip { background:rgba(139,92,246,.14); color:#c4b5fd; }
      .kpi.emerald { border-left:3px solid #10b981; } .kpi.emerald .chip { background:rgba(16,185,129,.14); color:#34d399; }
      .kpi.amber   { border-left:3px solid #f59e0b; } .kpi.amber   .chip { background:rgba(245,158,11,.14); color:#fbbf24; }
      .kpi.rose    { border-left:3px solid #f43f5e; } .kpi.rose    .chip { background:rgba(244,63,94,.14);  color:#fb7185; }

      .pill { display:inline-block; padding:2px 10px; border-radius:999px; font-size:.7rem; font-weight:600; }
      .pill.ok   { background:rgba(16,185,129,.12); color:#34d399; border:1px solid rgba(16,185,129,.3);}
      .pill.warn { background:rgba(245,158,11,.12); color:#fbbf24; border:1px solid rgba(245,158,11,.3);}
      .pill.bad  { background:rgba(244,63,94,.12);  color:#fb7185; border:1px solid rgba(244,63,94,.3);}

      .trace { background:#0b1220; border:1px solid #1e293b; border-radius:12px;
               padding:14px 16px; margin-bottom:10px; }
      .trace .step { font-weight:700; font-size:.8rem; }

      /* ---- tab nav ---- */
      .stTabs [data-baseweb="tab-list"] { gap:4px; border-bottom:1px solid #1e293b; }
      .stTabs [data-baseweb="tab"] {
        background:transparent; color:#94a3b8; font-size:.82rem; font-weight:500;
        padding:8px 14px; border-radius:8px 8px 0 0;
      }
      .stTabs [data-baseweb="tab"]:hover { color:#e2e8f0; background:rgba(148,163,184,.06); }
      .stTabs [aria-selected="true"] {
        color:#c4b5fd !important; border-bottom:2px solid #8b5cf6; background:rgba(139,92,246,.08);
      }

      /* ---- buttons ---- */
      .stButton > button, .stDownloadButton > button {
        border-radius:10px; border:1px solid #334155; font-weight:600; font-size:.82rem;
        background:#1e293b; color:#e2e8f0;
      }
      .stButton > button:hover, .stDownloadButton > button:hover { border-color:#8b5cf6; color:#fff; }
      .stButton > button[kind="primary"] {
        background:linear-gradient(90deg,#7c3aed,#4f46e5 55%,#2563eb); border:none; color:#fff;
        box-shadow:0 6px 18px -6px rgba(79,70,229,.5);
      }

      /* ---- inputs & panels ---- */
      [data-testid="stDataFrame"] { border:1px solid #1e293b; border-radius:12px; }
      div[data-baseweb="select"] > div, div[data-baseweb="input"] > div,
      .stTextInput input, .stNumberInput input {
        background:#0b1220 !important; border-color:#334155 !important; color:#e2e8f0 !important;
      }
      [data-testid="stMetricValue"] { color:#f8fafc; }
    </style>
    """,
    unsafe_allow_html=True,
)

llm = LLM()


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #
def _read_dir() -> dict[str, pd.DataFrame] | None:
    try:
        return {
            "payments": pd.read_csv(DATA_DIR / "payments.csv"),
            "settlements": pd.read_csv(DATA_DIR / "settlements.csv"),
            "orders": pd.read_csv(DATA_DIR / "orders.csv"),
        }
    except FileNotFoundError:
        return None


def get_frames() -> dict[str, pd.DataFrame]:
    if "frames" not in st.session_state:
        st.session_state.frames = _read_dir() or generate_batch()
        st.session_state.source = "data/ CSVs" if _read_dir() else "generated in-memory"
    return st.session_state.frames


def run_reconciliation() -> None:
    f = get_frames()
    res = reconcile(
        f["payments"], f["settlements"], f["orders"],
        llm=llm,
        fee_tolerance=st.session_state.get("tol", DEFAULT_FEE_TOLERANCE),
        merchant_filter=st.session_state.get("merchant", "ALL"),
    )
    audit = AuditLog()
    audit.extend(res.audit)
    try:
        audit.save()
    except Exception:
        pass
    qa = SettlementQA(llm)
    qa.build_index(res.results, f["payments"], f["orders"], res.audit)
    st.session_state.result = res
    st.session_state.qa = qa


# --------------------------------------------------------------------------- #
# sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### 🧾 Settlement Reconciliation Agent")
    st.caption("Razorpay AI Buildathon 2026 · Track 04 — AI Finance Controller")

    frames = get_frames()
    merchants = ["ALL"] + sorted(frames["payments"]["merchant_id"].unique().tolist())
    st.selectbox("Merchant scope", merchants, key="merchant")
    st.selectbox(
        "Settlement cycle",
        ["T+1 Standard Payout", "T+0 Same-day", "T+2 Cross-border"],
        key="cycle",
        help="Display context only — the engine matches on the actual settlement dates.",
    )
    st.slider(
        "Deterministic fee tolerance", 0.0, 0.06,
        float(DEFAULT_FEE_TOLERANCE), 0.005, format="%.3f", key="tol",
        help="Share of the payment the rule pass may absorb as fee/GST before deferring to the LLM.",
    )

    st.divider()
    if llm.available:
        st.markdown(f"<span class='pill ok'>LLM online · {llm.engine_name}</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='pill warn'>LLM offline · deterministic heuristic fallback</span>", unsafe_allow_html=True)
        st.caption("Set `OPENAI_API_KEY` to route the fallback pass and Q&A through GPT-4o.")

    st.write("")
    if st.button("▶  Run reconciliation", type="primary", use_container_width=True):
        run_reconciliation()
    st.caption(f"Data source: {st.session_state.get('source', 'data/ CSVs')}")

    st.divider()
    st.markdown(
        "[GitHub · Settlement-Reconciliation-Agent]"
        "(https://github.com/Varshatolani14/Settlement-Reconciliation-Agent)"
    )

# first load -> run once (fast in offline mode)
if "result" not in st.session_state:
    run_reconciliation()

res = st.session_state.result
qa: SettlementQA = st.session_state.qa
stats = res.stats
results = res.results
frames = get_frames()


# --------------------------------------------------------------------------- #
# header bar + summary banner + KPIs   (modelled on the reference mock)
# --------------------------------------------------------------------------- #
_engine_pill = (
    f"<span style='color:#34d399'>● {stats['llm_engine']}</span>"
    if stats["llm_available"]
    else f"<span style='color:#fbbf24'>● {stats['llm_engine']}</span>"
)
st.markdown(
    f"""
    <div class="app-header">
      <div class="app-logo">🧾</div>
      <div>
        <div class="app-title">Settlement Reconciliation Agent
          <span class="badge">Razorpay AI Buildathon 2026 · Track 04</span>
        </div>
        <div class="app-sub">Two-pass matching engine · deterministic rules + GPT-4o fallback · reasoned exceptions · grounded Q&amp;A · audit log</div>
      </div>
      <div class="app-ctx">
        merchant: {st.session_state.get('merchant', 'ALL')}<br>
        cycle: {st.session_state.get('cycle', 'T+1 Standard Payout')}<br>
        engine: {_engine_pill}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="banner">
      <div><div class="k">System architecture</div>
        <div class="v">AI Financial Operations · reconciliation loop</div></div>
      <div><div class="k">Selected track</div>
        <div class="v violet">Track 04 — AI Finance Controller</div></div>
      <div><div class="k">Engine capabilities</div>
        <div class="v">Deterministic rules · GPT-4o fuzzy match · RAG Q&amp;A · explainable audit log</div></div>
      <div><div class="k">GitHub repository</div>
        <div class="v"><a href="https://github.com/Varshatolani14/Settlement-Reconciliation-Agent" target="_blank">
        Varshatolani14/Settlement-Reconciliation-Agent ↗</a></div></div>
    </div>
    """,
    unsafe_allow_html=True,
)


def kpi(col, cls, chip, label, value, sub, bar=None):
    bar_html = f"<div class='bar'><span style='width:{bar}%'></span></div>" if bar is not None else ""
    col.markdown(
        f"<div class='kpi {cls}'>"
        f"<div class='kpi-top'><span class='label'>{label}</span><span class='chip'>{chip}</span></div>"
        f"<div class='value'>{value}</div><div class='sub'>{sub}</div>{bar_html}</div>",
        unsafe_allow_html=True,
    )


c1, c2, c3, c4 = st.columns(4)
kpi(c1, "violet", "💰", "Gross settlement volume", f"₹{stats['gross_volume']:,.0f}",
    f"{stats['payments']} payments · {stats['batch_size']} record batch")
kpi(c2, "emerald", "✅", "Match rate", f"{stats['match_rate']}%",
    f"{stats['matched']} / {stats['batch_size']} matched", bar=stats["match_rate"])
kpi(c3, "amber", "⚖️", "Matched by rule / LLM", f"{stats['matched_by_rule']} / {stats['matched_by_llm']}",
    "deterministic pass · LLM fallback pass")
kpi(c4, "rose", "⚠️", "Exceptions (unresolved)", f"{stats['exceptions']}",
    f"₹{stats['discrepancy_value']:,.0f} discrepancy value")
st.write("")

tabs = st.tabs([
    "Dashboard", "Reconciliation workspace", "AI reasoning & exceptions",
    "Settlement Q&A", "Risk signals", "Data & ingest", "Audit log",
])

# --------------------------------------------------------------------------- #
# 1 · Dashboard
# --------------------------------------------------------------------------- #
with tabs[0]:
    left, right = st.columns([2, 1])

    with left:
        st.markdown("#### Gross captured vs. net settled, by date")
        pay = frames["payments"].copy()
        pay["payment_date"] = pd.to_datetime(pay["payment_date"])
        gross_by_day = pay.groupby("payment_date")["amount"].sum().rename("Gross captured")

        matched = results[results["utr"].notna() & results["settlement_date"].notna()].copy()
        net_by_day = pd.Series(dtype=float, name="Net settled")
        if not matched.empty:
            matched["settlement_date"] = pd.to_datetime(matched["settlement_date"])
            net_by_day = matched.groupby("settlement_date")["net_amount"].sum().rename("Net settled")

        chart_df = pd.concat([gross_by_day, net_by_day], axis=1).sort_index().fillna(0)
        chart_df.columns = ["Gross captured", "Net settled"]
        st.line_chart(chart_df, height=280, color=["#8b5cf6", "#10b981"])

    with right:
        st.markdown("#### Discrepancy categorization")
        counts = results["category"].value_counts()
        total = int(counts.sum()) or 1
        palette = {
            "Exact Match": "#10b981", "Fee / GST Deduction": "#f59e0b",
            "Partial Refund": "#f43f5e", "Settlement Lag": "#8b5cf6",
            "No Settlement Found": "#64748b", "Orphan Settlement": "#38bdf8",
        }
        stops, acc, legend = [], 0.0, []
        for name, n in counts.items():
            colr = palette.get(name, "#94a3b8")
            start = acc / total * 360
            acc += n
            end = acc / total * 360
            stops.append(f"{colr} {start:.1f}deg {end:.1f}deg")
            legend.append(
                f"<div style='display:flex;align-items:center;gap:7px;margin:3px 0;font-size:.76rem'>"
                f"<span style='width:10px;height:10px;border-radius:3px;background:{colr}'></span>"
                f"<span style='color:#cbd5e1'>{name}</span>"
                f"<span style='color:#64748b;margin-left:auto'>{int(n)}</span></div>"
            )
        st.markdown(
            f"<div style='display:flex;gap:18px;align-items:center'>"
            f"<div style='width:150px;height:150px;border-radius:50%;flex:none;"
            f"background:conic-gradient({','.join(stops)});"
            f"-webkit-mask:radial-gradient(circle 42px at 50% 50%,transparent 98%,#000 100%);"
            f"mask:radial-gradient(circle 42px at 50% 50%,transparent 98%,#000 100%)'></div>"
            f"<div style='flex:1'>{''.join(legend)}</div></div>",
            unsafe_allow_html=True,
        )
        top_cause = counts.index[0] if len(counts) else "—"
        st.caption(f"Top cause: **{top_cause}** · {int(counts.iloc[0]) if len(counts) else 0} of {total} records")

    st.markdown("#### Engine highlights")
    h1, h2, h3 = st.columns(3)
    lag = int((results["category"] == "Settlement Lag").sum())
    refund = int((results["category"] == "Partial Refund").sum())
    orphan = int((results["category"] == "Orphan Settlement").sum())
    h1.markdown(
        f"<div class='kpi violet'><div class='label'>Deterministic pass</div>"
        f"<div class='value'>{stats['matched_by_rule']}</div>"
        f"<div class='sub'>matched on order id + amount within fee tolerance + date window</div></div>",
        unsafe_allow_html=True)
    h2.markdown(
        f"<div class='kpi emerald'><div class='label'>LLM fallback pass</div>"
        f"<div class='value'>{stats['matched_by_llm']}</div>"
        f"<div class='sub'>{refund} partial refund · {lag} settlement lag reasoned & matched with rationale</div></div>",
        unsafe_allow_html=True)
    exc_sub = f"{stats['exceptions'] - orphan} payment(s) with no settlement in the date window"
    if orphan:
        exc_sub += f" · {orphan} orphan settlement(s)"
    exc_sub += " — each logged with its reason"
    h3.markdown(
        f"<div class='kpi rose'><div class='label'>Honest exception list</div>"
        f"<div class='value'>{stats['exceptions']}</div>"
        f"<div class='sub'>{exc_sub}</div></div>",
        unsafe_allow_html=True)

    st.caption(
        f"LLM engine: **{stats['llm_engine']}** · fee tolerance "
        f"{st.session_state.get('tol', DEFAULT_FEE_TOLERANCE):.1%} + ₹1 rounding · "
        "settlement date window 3 days (rule) / 7 days (LLM)."
    )


# --------------------------------------------------------------------------- #
# 2 · Reconciliation workspace
# --------------------------------------------------------------------------- #
with tabs[1]:
    fc1, fc2, fc3 = st.columns([2, 2, 2])
    q = fc1.text_input("Search", placeholder="payment id, order id, UTR, amount…")
    status_opts = ["ALL"] + sorted(results["status"].unique().tolist())
    status_f = fc2.selectbox("Status", status_opts)
    cat_opts = ["ALL"] + sorted(results["category"].unique().tolist())
    cat_f = fc3.selectbox("Category", cat_opts)

    view = results.copy()
    if status_f != "ALL":
        view = view[view["status"] == status_f]
    if cat_f != "ALL":
        view = view[view["category"] == cat_f]
    if q:
        ql = q.lower()
        mask = view.apply(lambda r: ql in " ".join(str(v).lower() for v in r.values), axis=1)
        view = view[mask]

    st.caption(f"Showing {len(view)} of {len(results)} records")
    show_cols = [
        "payment_id", "order_id", "merchant_id", "payment_date", "amount",
        "utr", "settlement_date", "gross_amount", "fee", "gst", "net_amount",
        "variance", "status", "category", "method", "confidence",
    ]
    st.dataframe(
        view[show_cols],
        use_container_width=True, hide_index=True, height=460,
        column_config={
            "amount": st.column_config.NumberColumn("amount", format="₹%.2f"),
            "gross_amount": st.column_config.NumberColumn("gross", format="₹%.2f"),
            "fee": st.column_config.NumberColumn("fee", format="₹%.2f"),
            "gst": st.column_config.NumberColumn("gst", format="₹%.2f"),
            "net_amount": st.column_config.NumberColumn("net", format="₹%.2f"),
            "variance": st.column_config.NumberColumn("variance", format="₹%.2f"),
            "confidence": st.column_config.ProgressColumn("confidence", min_value=0, max_value=100, format="%d"),
        },
    )
    with st.expander("Row-level rationale (every decision)"):
        for _, r in view.iterrows():
            ref = r["payment_id"] or r["utr"]
            st.markdown(f"**{ref}** · `{r['status']}` · {r['category']} — {r['rationale']}")


# --------------------------------------------------------------------------- #
# 3 · AI reasoning & exceptions
# --------------------------------------------------------------------------- #
with tabs[2]:
    st.markdown("#### Two-pass matching — decision trace")
    st.markdown(
        "<span class='muted'>Pass 1 takes everything the deterministic rules can prove. "
        "Pass 2 hands the rest to the LLM, which reasons over the remaining candidate "
        "settlements and either matches with a stated rationale or routes the record to "
        "the exception list with a specific reason.</span>",
        unsafe_allow_html=True,
    )
    st.write("")

    focus = results[results["status"].isin(["MATCHED_LLM", "EXCEPTION"])].copy()
    if focus.empty:
        st.info("Every record was resolved by the deterministic pass in this scope.")
    else:
        labels = {
            (r["payment_id"] or r["utr"]): f"{r['payment_id'] or r['utr']} · {r['status']} · {r['category']}"
            for _, r in focus.iterrows()
        }
        pick = st.selectbox("Record", list(labels), format_func=lambda k: labels[k])
        row = focus[(focus["payment_id"] == pick) | (focus["utr"] == pick)].iloc[0]

        p = frames["payments"][frames["payments"]["payment_id"] == row["payment_id"]]
        pay_amount = float(p["amount"].iloc[0]) if not p.empty else None

        st.markdown(
            "<div class='trace'><div class='step' style='color:#a78bfa'>Pass 1 · deterministic rules</div>"
            "<div class='muted'>Order id + amount within fee tolerance + 3-day settlement window. "
            "Result: <b>not resolved</b> — deferred to the LLM pass.</div></div>",
            unsafe_allow_html=True,
        )
        amt_txt = f"₹{pay_amount:,.2f}" if pay_amount is not None else "—"
        gross_txt = f"₹{row['gross_amount']:,.2f}" if pd.notna(row["gross_amount"]) else "—"
        st.markdown(
            f"<div class='trace'><div class='step' style='color:#fbbf24'>Pass 2 · LLM fallback "
            f"({row['method']})</div>"
            f"<div class='muted'>Payment {amt_txt} vs candidate settlement {gross_txt}"
            f"{'' if pd.isna(row['variance']) else f' · variance ₹{row['variance']:,.2f}'}. "
            f"Confidence {row['confidence']}%.</div></div>",
            unsafe_allow_html=True,
        )
        verdict_cls = "bad" if row["status"] == "EXCEPTION" else "ok"
        verdict_txt = "ROUTED TO EXCEPTION LIST" if row["status"] == "EXCEPTION" else f"MATCHED → {row['utr']}"
        st.markdown(
            f"<div class='trace'><div class='step'>Verdict &nbsp;"
            f"<span class='pill {verdict_cls}'>{verdict_txt}</span></div>"
            f"<div style='margin-top:6px;color:#cbd5e1'>{row['rationale']}</div></div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("#### Exception list — every unresolved record, with its reason")
    exc = res.exceptions.copy()
    if exc.empty:
        st.success("No exceptions in this scope.")
    else:
        exc_view = exc[["payment_id", "utr", "merchant_id", "amount", "gross_amount",
                        "category", "rationale"]]
        st.dataframe(exc_view, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# 4 · Settlement Q&A
# --------------------------------------------------------------------------- #
with tabs[3]:
    st.markdown("#### Grounded Q&A over the batch")
    st.markdown(
        "<span class='muted'>Retrieval over the reconciliation records and audit log, "
        "answered by GPT-4o (or an offline extractive fallback). Answers cite the "
        "reference ids and amounts they rely on.</span>",
        unsafe_allow_html=True,
    )
    exc_ids = res.exceptions["payment_id"].dropna().tolist()
    default_q = (
        f"Why did payment {exc_ids[0]} not settle?" if exc_ids
        else "Which payments were matched only by the LLM fallback, and why?"
    )
    question = st.text_input("Question", value=default_q)
    ccol1, ccol2 = st.columns([1, 3])
    ask = ccol1.button("Ask", type="primary")
    ccol2.caption("Examples: “what caused the partial-refund mismatches?” · “list the orphan settlements” · “how many matched by rule vs LLM?”")

    if ask and question.strip():
        with st.spinner("Retrieving records and reasoning…"):
            ans = qa.ask(question)
        st.markdown(f"<span class='pill ok'>engine · {ans.engine}</span>", unsafe_allow_html=True)
        st.markdown(ans.answer)
        with st.expander(f"Retrieved context ({len(ans.used_context)} blocks)"):
            for d in ans.used_context:
                st.markdown(f"- {d}")


# --------------------------------------------------------------------------- #
# 5 · Risk signals  (supplementary pandas heuristics)
# --------------------------------------------------------------------------- #
with tabs[4]:
    st.markdown("#### Continuous risk heuristics & fraud detection")
    st.caption(
        "Supplementary pandas heuristics over the same payment batch — velocity "
        "spikes, duplicate debits and Z-score (>2) outliers. Not part of the "
        "match / exception loop; surfaced for the reviewer."
    )
    pay = frames["payments"].copy()
    pay["z"] = pay.groupby("merchant_id")["amount"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) or 1.0)
    )
    outliers = pay[pay["z"].abs() > 2][["payment_id", "merchant_id", "amount", "z"]].round({"z": 2})
    dup = pay[pay.duplicated(["merchant_id", "amount", "payment_date"], keep=False)][
        ["payment_id", "merchant_id", "amount", "payment_date"]
    ].sort_values(["merchant_id", "amount"])
    vel = (
        pay.groupby(["merchant_id", "payment_date"]).size().reset_index(name="count")
    )
    vel = vel[vel["count"] >= 5]

    def rule_card(col, cls, title, desc, hits):
        state = "ACTIVE" if hits else "CLEAR"
        col.markdown(
            f"<div class='kpi {cls}'>"
            f"<div class='kpi-top'><span class='label'>{title}</span>"
            f"<span class='pill {'warn' if hits else 'ok'}'>{state}</span></div>"
            f"<div class='value'>{hits}</div>"
            f"<div class='sub'>{desc}</div></div>",
            unsafe_allow_html=True,
        )

    r1, r2, r3 = st.columns(3)
    rule_card(r1, "amber", "Velocity spike detector", "5+ payments from one merchant on the same day", len(vel))
    rule_card(r2, "rose", "Duplicate debit alarm", "same merchant, amount and day", len(dup))
    rule_card(r3, "violet", "Z-score outliers", "amount > 2σ from the merchant mean", len(outliers))
    st.write("")

    incidents = []
    for _, x in vel.iterrows():
        incidents.append({"heuristic": "Velocity spike", "ref": f"{x['merchant_id']} / {x['payment_date']}",
                          "metric": f"{int(x['count'])} payments in one day", "action": "Flagged for review"})
    for _, x in dup.iterrows():
        incidents.append({"heuristic": "Duplicate debit", "ref": x["payment_id"],
                          "metric": f"{x['merchant_id']} · ₹{x['amount']:.2f}", "action": "Held — manual check"})
    for _, x in outliers.iterrows():
        incidents.append({"heuristic": "Z-score outlier", "ref": x["payment_id"],
                          "metric": f"z = {x['z']:.2f} · ₹{x['amount']:.2f}", "action": "Monitor"})

    st.markdown("**Risk incidents**")
    if incidents:
        st.dataframe(pd.DataFrame(incidents), use_container_width=True, hide_index=True)
    else:
        st.caption("No velocity, duplicate or outlier signals in the current batch — all three heuristics clear.")


# --------------------------------------------------------------------------- #
# 6 · Data & ingest
# --------------------------------------------------------------------------- #
with tabs[5]:
    st.markdown("#### Three data sources")
    st.caption(
        "Upload your own `payments.csv`, `settlements.csv` and `orders.csv`, or "
        "regenerate the synthetic ~50-record batch (fees/GST, partial refunds, "
        "settlement lag and a handful of genuinely unresolvable records built in)."
    )

    up = st.columns(3)
    new_frames: dict[str, pd.DataFrame] = {}
    for i, name in enumerate(["payments", "settlements", "orders"]):
        f = up[i].file_uploader(f"{name}.csv", type="csv", key=f"up_{name}")
        if f is not None:
            df = pd.read_csv(f)
            missing = [c for c in REQUIRED[name] if c not in df.columns]
            if missing:
                up[i].error(f"missing columns: {', '.join(missing)}")
            else:
                new_frames[name] = df
                up[i].success(f"{len(df)} rows")

    b1, b2 = st.columns(2)
    if b1.button("Load uploaded CSVs", disabled=len(new_frames) != 3, use_container_width=True):
        st.session_state.frames = new_frames
        st.session_state.source = "uploaded CSVs"
        run_reconciliation()
        st.rerun()
    if b2.button("Regenerate synthetic batch", use_container_width=True):
        b = generate_batch()
        st.session_state.frames = b
        st.session_state.source = "generated in-memory"
        run_reconciliation()
        st.rerun()

    st.divider()
    for name in ["payments", "settlements", "orders"]:
        with st.expander(f"{name}.csv  ·  {len(frames[name])} rows"):
            st.dataframe(frames[name], use_container_width=True, hide_index=True, height=260)


# --------------------------------------------------------------------------- #
# 7 · Audit log
# --------------------------------------------------------------------------- #
with tabs[6]:
    st.markdown("#### Audit log — every decision, explainable")
    audit_df = pd.DataFrame(res.audit)
    st.dataframe(audit_df, use_container_width=True, hide_index=True, height=420)

    d1, d2, d3 = st.columns(3)
    d1.download_button(
        "Download audit_log.json", res_json := json.dumps(res.audit, indent=2),
        file_name="audit_log.json", mime="application/json", use_container_width=True,
    )
    recon_csv = results.to_csv(index=False)
    d2.download_button(
        "Download reconciliation.csv", recon_csv,
        file_name="reconciliation.csv", mime="text/csv", use_container_width=True,
    )
    exc_csv = res.exceptions.to_csv(index=False)
    d3.download_button(
        "Download exceptions.csv", exc_csv,
        file_name="exceptions.csv", mime="text/csv", use_container_width=True,
    )

    st.caption(
        f"Run summary — batch {stats['batch_size']} · match rate {stats['match_rate']}% · "
        f"{stats['matched_by_rule']} by rule · {stats['matched_by_llm']} by LLM · "
        f"{stats['exceptions']} exceptions · engine {stats['llm_engine']}."
    )
