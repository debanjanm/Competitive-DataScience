import json
import logging

import streamlit as st

from graph import graph
from tools import load_companies, load_questionnaire

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

st.set_page_config(page_title="GreenGate", page_icon="\U0001F33F", layout="wide")

BAND_COLORS = {
    "Eligible for Green Finance": "#2F7A4A",
    "Conditionally Eligible": "#A8730F",
    "Further Review Required": "#B4531B",
    "Not Eligible": "#9E2B2B",
}

CSS = """
<style>
:root {
    --ground: #F6F8F7;
    --surface: #FFFFFF;
    --surface-2: #EFF3F1;
    --ink: #131C18;
    --ink-soft: #3C4B45;
    --muted: #64756E;
    --line: #DCE5E1;
    --line-strong: #C3D0CA;
    --accent: #0B5F63;
    --accent-soft: #E2EFEE;
}
@media (prefers-color-scheme: dark) {
    :root {
        --ground: #0E1512; --surface: #161F1B; --surface-2: #1C2621;
        --ink: #E8EFEB; --ink-soft: #BCC9C3; --muted: #8B9C95;
        --line: #24322C; --line-strong: #33443C;
        --accent: #5FBDB4; --accent-soft: #14302E;
    }
}
.gg-serif { font-family: Georgia, "Iowan Old Style", "Times New Roman", serif; }
.gg-mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }

.gg-masthead {
    display: flex; align-items: baseline; justify-content: space-between;
    gap: 16px; flex-wrap: wrap; border-bottom: 1px solid var(--line);
    padding-bottom: 14px; margin-bottom: 18px;
}
.gg-brand { font-size: 22px; letter-spacing: -.01em; margin: 0; }
.gg-brand span { color: var(--accent); }
.gg-tagline { margin: 0; color: var(--muted); font-size: 13px; max-width: 46ch; }

.gg-pill {
    display: inline-flex; align-items: center; gap: 7px;
    border-radius: 100px; padding: 6px 14px; font-size: 13px; font-weight: 600;
    white-space: nowrap; border: 1px solid var(--band); color: var(--band);
    background: color-mix(in srgb, var(--band) 10%, transparent);
}
.gg-pill i { width: 7px; height: 7px; border-radius: 50%; background: currentColor; display: inline-block; }

.gg-tile-k {
    font-size: 10px; letter-spacing: .12em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 4px;
}
.gg-tile-v { font-size: 30px; line-height: 1; font-weight: 600; font-variant-numeric: tabular-nums; }

.gg-lede {
    margin: 14px 0 0; padding: 14px 18px; border-radius: 3px;
    background: var(--surface-2); border: 1px solid var(--line);
    font-size: 15px; line-height: 1.6; color: var(--ink-soft); max-width: 78ch;
}
.gg-lede b {
    display: block; font-size: 10px; font-weight: 500; letter-spacing: .12em;
    text-transform: uppercase; color: var(--muted); margin-bottom: 6px;
}

.gg-sec {
    font-size: 10.5px; font-weight: 500; letter-spacing: .12em; text-transform: uppercase;
    color: var(--muted); margin: 18px 0 8px; padding-bottom: 6px;
    border-bottom: 1px solid var(--line);
}
.gg-pts { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 8px; }
.gg-pts li {
    position: relative; padding-left: 18px; font-size: 14px; line-height: 1.5;
    color: var(--ink-soft); max-width: 82ch;
}
.gg-pts li::before {
    content: ""; position: absolute; left: 3px; top: 7px;
    width: 6px; height: 6px; border-radius: 1px; background: var(--mark, var(--muted));
}
.gg-strong li::before { background: #2F7A4A; }
.gg-gap li::before { background: #B4531B; }

.gg-qid {
    font-size: 11px; color: var(--accent); background: var(--accent-soft);
    border-radius: 2px; padding: 1px 5px; white-space: nowrap;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}

.gg-instrument {
    background: var(--accent-soft); border: 1px solid var(--line);
    border-left: 3px solid var(--accent); border-radius: 2px; padding: 12px 16px;
}
.gg-instrument strong { font-size: 18px; }
.gg-instrument p { margin: 4px 0 0; font-size: 14px; color: var(--ink-soft); }

.gg-quote {
    display: block; margin-top: 5px; padding-left: 9px;
    border-left: 2px solid var(--line-strong); font-size: 12.5px;
    color: var(--muted); font-style: italic;
}
.gg-status { font-size: 12px; font-weight: 600; white-space: nowrap; }

.gg-co {
    border: 1px solid var(--line); border-left: 3px solid var(--band);
    border-radius: 3px; padding: 12px 14px; background: var(--surface);
}
.gg-co-name { font-size: 13px; font-weight: 600; }
.gg-co-score { font-size: 24px; font-weight: 600; color: var(--band); line-height: 1.1; }
.gg-co-band { font-size: 11px; color: var(--muted); }

[class*="st-key-case_"] button {
    width: 100% !important; text-align: left !important; white-space: pre-line !important;
    line-height: 1.45 !important; font-size: 12.5px !important;
    border: 1px solid var(--line) !important; border-radius: 3px !important;
    background: var(--surface) !important; color: var(--ink) !important;
    padding: 10px 12px !important; box-shadow: none !important;
    transition: transform .12s, border-color .12s, background .12s !important;
}
[class*="st-key-case_"] button p { font-size: 12.5px !important; }
[class*="st-key-case_"] button:hover {
    border-color: var(--line-strong) !important; transform: translateY(-1px);
}

.gg-progress {
    border: 1px solid var(--line); border-radius: 3px; background: var(--surface);
    padding: 18px 20px 20px;
}
.gg-progress-head {
    display: flex; align-items: baseline; justify-content: space-between;
    margin-bottom: 12px;
}
.gg-progress-label {
    font-size: 15px; font-weight: 600; color: var(--ink);
}
.gg-progress-pct {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 13px; color: var(--muted); font-variant-numeric: tabular-nums;
}
.gg-progress-track {
    height: 6px; background: var(--surface-2); border-radius: 100px;
    overflow: hidden; margin-bottom: 16px;
}
.gg-progress-fill {
    height: 100%; border-radius: 100px; background: var(--accent);
    transition: width .5s cubic-bezier(.4,0,.2,1);
}
.gg-steps {
    display: grid; grid-template-columns: repeat(6, 1fr); gap: 8px;
}
.gg-step {
    display: flex; flex-direction: column; gap: 6px; align-items: flex-start;
}
.gg-step-dot {
    width: 20px; height: 20px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 600; flex: none;
    border: 1.5px solid var(--line-strong); color: var(--muted);
    background: var(--surface);
}
.gg-step-dot.done {
    background: var(--accent); border-color: var(--accent); color: var(--surface);
}
.gg-step-dot.active {
    border-color: var(--accent); color: var(--accent);
    animation: gg-pulse 1.4s ease-in-out infinite;
}
.gg-step-label {
    font-size: 10.5px; letter-spacing: .02em; color: var(--muted);
    line-height: 1.3;
}
.gg-step.done .gg-step-label, .gg-step.active .gg-step-label { color: var(--ink-soft); }

@keyframes gg-pulse {
    0%, 100% { box-shadow: 0 0 0 0 color-mix(in srgb, var(--accent) 35%, transparent); }
    50% { box-shadow: 0 0 0 4px color-mix(in srgb, var(--accent) 0%, transparent); }
}
@media (prefers-reduced-motion: reduce) {
    .gg-progress-fill { transition: none; }
    .gg-step-dot.active { animation: none; }
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


def cite(text):
    """Wrap (E1) / (S1, S2) style citations in styled spans."""
    import re
    return re.sub(
        r"\(([EGS]\d[^)]*)\)",
        lambda m: f'(<span class="gg-qid">{m.group(1)}</span>)',
        text,
    )


def status_meta(status):
    return {
        "Fully Addressed": ("#2F7A4A", "●"),
        "Partially Addressed": ("#A8730F", "●"),
        "Not Disclosed": ("#9E2B2B", "●"),
        "Not Applicable": ("#64756E", "●"),
    }.get(status, ("#64756E", "●"))


def render_masthead():
    st.markdown(
        """
        <div class="gg-masthead">
          <div>
            <h1 class="gg-brand gg-serif">Green<span>Gate</span></h1>
            <p class="gg-tagline">Green &amp; transition finance eligibility screening.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_strip(history, selected_index):
    """Case history strip. Each card is a real button - click to switch the
    result view below to that past run, without re-running the pipeline."""
    if not history:
        return None

    st.markdown('<div class="gg-sec">Case history &middot; click to view</div>', unsafe_allow_html=True)

    cols = st.columns(len(history))
    clicked = None
    dynamic_css = ""

    for i, (col, run) in enumerate(zip(cols, history)):
        band = run["verdict"]
        color = BAND_COLORS.get(band, "#64756E")
        key = f"case_{i}"
        is_selected = i == selected_index

        dynamic_css += (
            f'.st-key-{key} button {{ border-left: 3px solid {color} !important; }}\n'
        )
        if is_selected:
            dynamic_css += (
                f'.st-key-{key} button[kind="primary"] {{'
                f'  background: color-mix(in srgb, {color} 14%, var(--surface)) !important;'
                f'  border-color: {color} !important;'
                f'  box-shadow: 0 1px 2px rgba(19,28,24,.06), 0 6px 16px -10px {color} !important;'
                f"}}\n"
            )

        with col:
            with st.container(key=key):
                label = f"{run['company_name']}\n{run['scores']['overall']:.1f}\n{band}"
                if st.button(
                    label,
                    key=f"{key}_btn",
                    type="primary" if is_selected else "secondary",
                    use_container_width=True,
                ):
                    clicked = i

    st.markdown(f"<style>{dynamic_css}</style>", unsafe_allow_html=True)
    return clicked


def disclosure_rate(answers, questionnaire):
    total_w = sum(q["weight"] for q in questionnaire)
    disclosed_w = sum(
        q["weight"] for q in questionnaire
        if answers.get(q["id"], {}).get("status") in ("Fully Addressed", "Partially Addressed")
    )
    return round((disclosed_w / total_w) * 100, 1) if total_w else 0.0


def render_result(result, questionnaire):
    band = result["verdict"]
    color = BAND_COLORS.get(band, "#64756E")
    scores = result["scores"]
    rec = result["recommendation"]

    st.markdown(f'<div style="--band:{color}">', unsafe_allow_html=True)

    top_l, top_r = st.columns([3, 1])
    with top_l:
        st.markdown(
            f'<h2 class="gg-serif" style="margin-bottom:2px">{result["company_name"]}</h2>'
            f'<div style="color:var(--muted);font-size:12.5px">'
            f'{result["metadata"].get("industry","")} &middot; '
            f'{result["metadata"].get("country","")} &middot; '
            f'FY{result["metadata"].get("reporting_year","")} &middot; '
            f'{len(result.get("documents", []))} documents assessed</div>',
            unsafe_allow_html=True,
        )
    with top_r:
        st.markdown(
            f'<div class="gg-pill" style="--band:{color}"><i></i>{band}</div>',
            unsafe_allow_html=True,
        )

    m1, m2, m3 = st.columns([1, 1, 2])
    with m1:
        st.markdown(
            f'<div class="gg-tile-k">Overall score</div>'
            f'<div class="gg-tile-v gg-mono" style="color:{color}">{scores["overall"]:.1f}'
            f'<span style="font-size:15px;color:var(--muted);font-weight:400"> / 100</span></div>',
            unsafe_allow_html=True,
        )
    with m2:
        d = disclosure_rate(result["answers"], questionnaire)
        st.markdown(
            f'<div class="gg-tile-k">Disclosure rate</div>'
            f'<div class="gg-tile-v gg-mono">{d:.0f}<span style="font-size:15px;color:var(--muted);font-weight:400">%</span></div>',
            unsafe_allow_html=True,
        )
    with m3:
        for cat, val in scores["categories"].items():
            st.markdown(
                f'<div style="display:grid;grid-template-columns:100px 1fr 40px;gap:8px;'
                f'align-items:center;font-size:12px;margin-bottom:6px">'
                f'<span style="color:var(--ink-soft)">{cat.capitalize()}</span>'
                f'<span style="height:7px;background:var(--surface-2);border-radius:100px;overflow:hidden;display:block">'
                f'<span style="display:block;height:100%;width:{val}%;background:var(--accent);border-radius:100px"></span></span>'
                f'<span class="gg-mono" style="text-align:right;color:var(--ink-soft)">{val:.1f}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        f'<div class="gg-lede"><b>Executive summary</b>{cite(rec.get("executive_summary",""))}</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["Assessment", "Financing view", "Evidence"])

    with tab1:
        st.markdown('<div class="gg-sec">Strengths</div>', unsafe_allow_html=True)
        st.markdown(
            '<ul class="gg-pts gg-strong">' +
            "".join(f"<li>{cite(s)}</li>" for s in rec.get("strengths", [])) +
            "</ul>",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="gg-sec">Disclosure gaps</div>', unsafe_allow_html=True)
        st.markdown(
            '<ul class="gg-pts gg-gap">' +
            "".join(f"<li>{cite(g)}</li>" for g in rec.get("gaps", [])) +
            "</ul>",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="gg-sec">Recommended actions</div>', unsafe_allow_html=True)
        st.markdown(
            '<ol class="gg-pts">' +
            "".join(f"<li>{cite(a)}</li>" for a in rec.get("actions", [])) +
            "</ol>",
            unsafe_allow_html=True,
        )

    with tab2:
        st.markdown('<div class="gg-sec">Eligibility rationale</div>', unsafe_allow_html=True)
        st.markdown(
            f'<p style="max-width:82ch;color:var(--ink-soft);font-size:14.5px">{cite(result.get("reasoning",""))}</p>',
            unsafe_allow_html=True,
        )
        gfr = rec.get("green_finance_recommendation", {})
        st.markdown('<div class="gg-sec">Financing recommendation</div>', unsafe_allow_html=True)
        if isinstance(gfr, dict):
            st.markdown(
                f'<div class="gg-instrument gg-serif"><strong>{gfr.get("instrument","")}</strong>'
                f'<p class="gg-sans">{cite(gfr.get("rationale",""))}</p></div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="gg-sec">Conditions &amp; covenants</div>', unsafe_allow_html=True)
            st.markdown(
                '<ol class="gg-pts">' +
                "".join(f"<li>{cite(c)}</li>" for c in gfr.get("conditions", [])) +
                "</ol>",
                unsafe_allow_html=True,
            )
            st.markdown('<div class="gg-sec">Monitoring KPIs</div>', unsafe_allow_html=True)
            st.markdown(
                '<ul class="gg-pts">' +
                "".join(f"<li>{cite(k)}</li>" for k in gfr.get("monitoring_kpis", [])) +
                "</ul>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f"<p>{cite(str(gfr))}</p>", unsafe_allow_html=True)

    with tab3:
        st.markdown(
            '<div class="gg-sec">Question-level assessment &middot; every status traced to source text</div>',
            unsafe_allow_html=True,
        )
        answers = result["answers"]
        facts_by_id = {
            f["id"]: f for f in result.get("extracted_context", {}).get("facts", [])
        }
        for q in questionnaire:
            ans = answers.get(q["id"])
            if not ans:
                continue
            color_s, dot = status_meta(ans["status"])
            evidence_lines = ""
            for fid in ans.get("evidence", []):
                fact = facts_by_id.get(fid)
                if fact:
                    evidence_lines += (
                        f'<span class="gg-quote">&ldquo;{fact["evidence"]}&rdquo; '
                        f'<span class="gg-mono">&mdash; {fact["document"]}</span></span>'
                    )
            st.markdown(
                f'<div style="display:grid;grid-template-columns:44px 1fr 130px 34px;gap:12px;'
                f'padding:10px 0;border-bottom:1px solid var(--line);align-items:start">'
                f'<span class="gg-qid">{q["id"]}</span>'
                f'<span style="font-size:13.5px;color:var(--ink-soft)">{q["question"]}'
                f'<br>{ans.get("reason","")}{evidence_lines}</span>'
                f'<span class="gg-status" style="color:{color_s}">{dot} {ans["status"]}</span>'
                f'<span class="gg-mono" style="text-align:right;color:var(--muted)">{q["weight"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

    st.download_button(
        "Download summary report",
        data=json.dumps(result, indent=2, default=str),
        file_name=f"{result['company_name'].replace(' ', '_')}_greengate_report.json",
        mime="application/json",
    )


NODE_LABELS = {
    "input_request": "Intake",
    "document_analyzer": "Document analysis",
    "question_answering": "Questionnaire assessment",
    "score_calculation": "Score calculation",
    "recommendation": "Recommendation",
    "verdict": "Verdict",
}


def node_message(node_name, full_state, questionnaire):
    """Build a real-time status line from what the node actually produced."""
    if node_name == "input_request":
        docs = full_state.get("documents", [])
        name = full_state.get("company_name", "").rstrip(".")
        return f"Loaded {len(docs)} document(s) for {name}."

    if node_name == "document_analyzer":
        facts = full_state.get("extracted_context", {}).get("facts", [])
        return f"Extracted {len(facts)} ESG fact(s) from the source documents."

    if node_name == "question_answering":
        answered = len(full_state.get("answers", {}))
        total = len(questionnaire)
        return f"Answered {answered} of {total} questionnaire items from cited evidence."

    if node_name == "score_calculation":
        scores = full_state.get("scores", {})
        cats = scores.get("categories", {})
        cat_str = ", ".join(f"{k[0].upper()} {v:.0f}" for k, v in cats.items())
        return f"Computed score: {scores.get('overall', 0):.1f}/100 ({cat_str})."

    if node_name == "recommendation":
        rec = full_state.get("recommendation", {})
        return (
            f"Drafted recommendation: {len(rec.get('strengths', []))} strengths, "
            f"{len(rec.get('gaps', []))} gaps, {len(rec.get('actions', []))} actions."
        )

    if node_name == "verdict":
        return f"Verdict: {full_state.get('verdict', '')}."

    return f"{NODE_LABELS.get(node_name, node_name)} complete."


NODE_ORDER = list(NODE_LABELS.keys())


def render_progress(active_index, done_count, current_message):
    total = len(NODE_ORDER)
    pct = round((done_count / total) * 100)
    current_label = NODE_LABELS[NODE_ORDER[active_index]] if active_index < total else "Complete"

    steps_html = ""
    for i, key in enumerate(NODE_ORDER):
        if i < done_count:
            state_cls, dot = "done", "&#10003;"
        elif i == active_index:
            state_cls, dot = "active", str(i + 1)
        else:
            state_cls, dot = "", str(i + 1)
        steps_html += (
            f'<div class="gg-step {state_cls}">'
            f'<span class="gg-step-dot {state_cls}">{dot}</span>'
            f'<span class="gg-step-label">{NODE_LABELS[key]}</span>'
            f"</div>"
        )

    return (
        '<div class="gg-progress">'
        '<div class="gg-progress-head">'
        f'<span class="gg-progress-label">{current_label}</span>'
        f'<span class="gg-progress-pct">{done_count}/{total} &middot; {pct}%</span>'
        "</div>"
        '<div class="gg-progress-track">'
        f'<div class="gg-progress-fill" style="width:{pct}%"></div>'
        "</div>"
        f'<div class="gg-steps">{steps_html}</div>'
        f'<div style="margin-top:14px;font-size:13px;color:var(--ink-soft)">{current_message}</div>'
        "</div>"
    )


def main():
    render_masthead()

    if "history" not in st.session_state:
        st.session_state.history = []
    if "selected_index" not in st.session_state:
        st.session_state.selected_index = None

    clicked = render_strip(st.session_state.history, st.session_state.selected_index)
    if clicked is not None:
        st.session_state.selected_index = clicked
        st.rerun()

    companies = load_companies()
    questionnaire = load_questionnaire()

    with st.form("screen_form"):
        st.markdown('<div class="gg-sec">Screen a company</div>', unsafe_allow_html=True)
        company_name = st.selectbox("Company", list(companies.keys()))
        comment = st.text_area(
            "Comment / context (optional)",
            placeholder="e.g. Renewable energy project financing, proceeds allocated to...",
        )
        submitted = st.form_submit_button("Submit for screening")

    if submitted:
        initial_state = {
            "company_name": company_name,
            "documents": [],
            "document_texts": [],
            "extracted_context": {},
            "answers": {},
            "scores": {},
            "recommendation": {},
            "verdict": "",
            "reasoning": "",
            "metadata": {},
        }

        full_state = dict(initial_state)
        progress_slot = st.empty()

        progress_slot.markdown(
            render_progress(0, 0, "Starting screening..."),
            unsafe_allow_html=True,
        )

        for i, update in enumerate(graph.stream(initial_state, stream_mode="updates")):
            node_name, node_output = next(iter(update.items()))
            full_state.update(node_output)
            message = node_message(node_name, full_state, questionnaire)
            done_count = i + 1
            next_index = min(done_count, len(NODE_ORDER) - 1)
            progress_slot.markdown(
                render_progress(next_index, done_count, message),
                unsafe_allow_html=True,
            )

        progress_slot.markdown(
            render_progress(len(NODE_ORDER), len(NODE_ORDER), "Screening complete."),
            unsafe_allow_html=True,
        )

        result = full_state

        st.session_state.history.insert(0, result)
        st.session_state.selected_index = 0
        st.rerun()

    if st.session_state.history:
        index = st.session_state.selected_index
        if index is None or index >= len(st.session_state.history):
            index = 0
        st.divider()
        render_result(st.session_state.history[index], questionnaire)


if __name__ == "__main__":
    main()
