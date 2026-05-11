"""Deeper analysis: what *kinds* of tokens reconstruct poorly?

Reads results parquet, classifies tokens along several axes (orthography,
surface form, AV fallback templates, etc.), and reports cos by category.

Usage:
    python analyze_deep.py results.parquet --out report.md
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ─── Token classification ─────────────────────────────────────────────────────

# Cheap regexes — applied to the raw token *string* (with leading space if any).
RE_WS_ONLY      = re.compile(r"^\s+$")
RE_PUNCT_ONLY   = re.compile(r"^[\W_]+$", re.UNICODE)  # \W = non-word
RE_DIGIT_ONLY   = re.compile(r"^\s*[\d]+$")
RE_ALL_UPPER    = re.compile(r"^[A-Z]+$")
RE_LEAD_SPACE   = re.compile(r"^[  ]")           # leading space = word-initial in BPE
RE_HAS_NONASCII = re.compile(r"[^\x00-\x7F]")
RE_CODE_CHARS   = re.compile(r"[{}\[\]()<>;|&^=+\\*/%@#$~]")  # code-y punctuation


def classify_token(s: str) -> dict:
    """Return a flat dict of categorical features for a token string."""
    if not isinstance(s, str):
        s = str(s)
    stripped = s.strip()
    n_chars = len(s)
    n_stripped = len(stripped)
    has_lead_space = bool(RE_LEAD_SPACE.match(s))
    is_ws = bool(RE_WS_ONLY.match(s))
    is_punct = (not is_ws) and bool(RE_PUNCT_ONLY.match(s))
    is_digit = bool(RE_DIGIT_ONLY.match(s))
    is_nonascii = bool(RE_HAS_NONASCII.search(s))
    has_code = bool(RE_CODE_CHARS.search(s))
    first_char = stripped[:1]
    is_alpha_initial_upper = first_char.isupper()
    is_alpha_initial_lower = first_char.isalpha() and not is_alpha_initial_upper
    is_all_upper = bool(stripped) and bool(RE_ALL_UPPER.fullmatch(stripped))
    is_single_char = n_stripped == 1
    is_short = n_stripped <= 3
    is_long = n_stripped >= 8
    # "Word-likeness" — alphabetic, ≥ 2 chars, leading space → typical content word.
    is_wordy = (
        n_stripped >= 2 and stripped.isalpha() and has_lead_space
    )

    # Bucketed token category — non-overlapping, mutually exclusive.
    if is_ws:
        cat = "whitespace"
    elif is_punct:
        cat = "punct_only"
    elif is_digit:
        cat = "digit_only"
    elif is_nonascii:
        cat = "non_ascii"
    elif is_single_char:
        cat = "single_char_alpha"
    elif has_code:
        cat = "code_syntax"
    elif is_alpha_initial_upper and not has_lead_space:
        # E.g. "How", "What", "Android" — sentence-initial or BPE-mid-word continuation
        # of a capitalised token.
        cat = "cap_no_space"
    elif is_alpha_initial_upper and has_lead_space:
        cat = "cap_with_space"
    elif is_wordy:
        cat = "word_lower_with_space"
    elif is_alpha_initial_lower and not has_lead_space:
        cat = "subword_continuation"   # mid-BPE piece, e.g. "ll" in "I'll", "ing"
    else:
        cat = "other"

    return dict(
        cat=cat,
        n_chars=n_chars,
        n_stripped=n_stripped,
        has_lead_space=has_lead_space,
        is_alpha_initial_upper=is_alpha_initial_upper,
        is_all_upper=is_all_upper,
        is_single_char=is_single_char,
        is_short=is_short,
        is_long=is_long,
    )


# ─── AV fallback detection ────────────────────────────────────────────────────

# Hand-spotted from the dry-run worst-cos tail. The first two are the ones
# we saw verbatim across many samples.
FALLBACK_PATTERNS = {
    "fragmented_academic":  re.compile(r"Fragmented, incoherent academic text pattern", re.I),
    "tech_article_research_paper": re.compile(
        r"Technical article structure[^.]*blog post[^.]*research paper", re.I),
    "fibonacci": re.compile(r"Fibonacci sequence", re.I),
    "tutorial_struct_research_paper": re.compile(
        r"Technical tutorial structure[^.]*specific research paper", re.I),
    "kalman": re.compile(r"Kalman filter", re.I),
    "no_explanation_tags": None,    # special-case: av_parsed == False
}


def add_fallback_flags(df: pd.DataFrame) -> pd.DataFrame:
    expls = df["explanation"].fillna("").astype(str)
    for name, pat in FALLBACK_PATTERNS.items():
        if pat is None:
            df[f"fb_{name}"] = ~df["av_parsed"].astype(bool)
        else:
            df[f"fb_{name}"] = expls.str.contains(pat)
    df["fb_any"] = df[[c for c in df.columns if c.startswith("fb_")]].any(axis=1)
    return df


# ─── Reporting helpers ────────────────────────────────────────────────────────


def report_table(df: pd.DataFrame, by: str, title: str, *,
                 min_count: int = 30) -> list[str]:
    """Cos statistics for a categorical column."""
    g = df.groupby(by, observed=True).agg(
        n=("cos", "size"),
        cos_mean=("cos", "mean"),
        cos_median=("cos", "median"),
        cos_p10=("cos", lambda s: float(np.percentile(s, 10))),
        norm_mean=("vec_norm", "mean"),
    ).reset_index()
    g = g[g["n"] >= min_count]
    g = g.sort_values("cos_mean")
    lines = [f"\n## {title}\n"]
    lines.append(f"| {by} | n | cos_mean | cos_median | cos_p10 | norm_mean |")
    lines.append("|---|---|---|---|---|---|")
    for r in g.itertuples():
        lines.append(
            f"| `{getattr(r, by)}` | {r.n} | {r.cos_mean:.3f} | {r.cos_median:.3f} "
            f"| {r.cos_p10:.3f} | {r.norm_mean:.0f} |"
        )
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("--out", default="-")
    ap.add_argument("--top-tokens", type=int, default=40)
    ap.add_argument("--norm-cutoff", type=float, default=120_000,
                    help="Above this L2-norm, treat as 'sink-token' cluster "
                         "and report separately.")
    args = ap.parse_args()

    print(f"[load] {args.parquet}")
    t = pq.read_table(args.parquet, columns=[
        "sample_idx", "token_id", "token_str", "position", "seq_len",
        "vec_norm", "text_preview", "explanation", "av_parsed",
        "mse_nrm", "cos",
    ])
    df = t.to_pandas()
    n_total = len(df)
    print(f"[load] {n_total} rows")

    # Token features
    feats = pd.DataFrame.from_records(
        [classify_token(s) for s in df["token_str"]]
    )
    df = pd.concat([df, feats], axis=1)

    # AV fallback flags
    df = add_fallback_flags(df)

    # Explanation length
    df["expl_len_chars"] = df["explanation"].fillna("").astype(str).str.len()
    df["expl_len_words"] = df["explanation"].fillna("").astype(str).str.split().str.len()

    # Sink cluster split
    df["is_sink"] = df["vec_norm"] >= args.norm_cutoff
    n_sink = int(df["is_sink"].sum())

    out: list[str] = []
    w = out.append

    w(f"# Deep token-level analysis — `{Path(args.parquet).name}`\n")
    w(f"N = **{n_total}** activations.")
    w(f"Sink-token cluster (vec_norm ≥ {args.norm_cutoff:,.0f}): "
      f"**{n_sink}** rows ({100*n_sink/n_total:.2f} %). "
      f"All cross-cuts below report on the **non-sink** subset unless noted.\n")

    main_df = df[~df["is_sink"]].copy()
    sink_df = df[df["is_sink"]].copy()

    # ─── 1. Headline by category ─────────────────────────────────────────────
    out.extend(report_table(
        main_df, "cat",
        "1. Cos by token category (non-sink)",
        min_count=20,
    ))

    # ─── 2. Stripped-length buckets ──────────────────────────────────────────
    main_df["len_bucket"] = pd.cut(
        main_df["n_stripped"], bins=[0, 1, 2, 3, 5, 8, 12, 20, 999],
        labels=["1", "2", "3", "4-5", "6-8", "9-12", "13-20", "21+"],
    )
    out.extend(report_table(
        main_df, "len_bucket",
        "2. Cos by stripped token length (non-sink)",
        min_count=20,
    ))

    # ─── 3. Capitalization × leading-space cross-cut for alpha tokens ────────
    alpha_mask = main_df["token_str"].str.strip().str.match(r"^[A-Za-z]+$").fillna(False)
    a = main_df[alpha_mask].copy()
    a["case_ws"] = (
        np.where(a["has_lead_space"], "lead-space ", "no-space ")
        + np.where(a["is_alpha_initial_upper"], "Cap", "lower")
    )
    out.extend(report_table(
        a, "case_ws",
        "3. Cos by case × leading-space (alphabetic tokens, non-sink)",
        min_count=30,
    ))

    # ─── 4. AV fallback prevalence ───────────────────────────────────────────
    w("\n## 4. AV fallback templates — prevalence and quality\n")
    w("How often does the AV emit each known fallback phrase, "
      "and how does cos differ when it does vs doesn't?\n")
    w("| pattern | n_full | %_full | cos_when_present | cos_when_absent |")
    w("|---|---|---|---|---|")
    for name in FALLBACK_PATTERNS:
        col = f"fb_{name}"
        present = df[df[col]]
        absent  = df[~df[col]]
        w(f"| `{name}` | {len(present)} | {100*len(present)/n_total:.2f} % "
          f"| {present['cos'].mean() if len(present) else float('nan'):.3f} "
          f"| {absent['cos'].mean():.3f} |")

    # ─── 5. Worst tokens by category — drill down ────────────────────────────
    w("\n## 5. Worst individual samples per category (non-sink, cos < 0.7)\n")
    bad = main_df[main_df["cos"] < 0.7].copy()
    w(f"Total non-sink rows with cos < 0.7: **{len(bad)}** "
      f"({100*len(bad)/len(main_df):.2f} % of non-sink).\n")
    if len(bad):
        cat_counts = (bad.groupby("cat").size()
                      .sort_values(ascending=False))
        w("Distribution by category:\n")
        w("| cat | n_bad | n_total_in_cat | bad_rate |")
        w("|---|---|---|---|")
        cat_totals = main_df.groupby("cat").size()
        for c, n in cat_counts.items():
            tot = int(cat_totals.get(c, 0))
            rate = n / tot if tot else 0.0
            w(f"| {c} | {n} | {tot} | {100*rate:.2f} % |")

        w("\n### Examples (cos < 0.5, sorted by cos)\n")
        ex = bad[bad["cos"] < 0.5].sort_values("cos").head(40)
        for r in ex.itertuples():
            w(f"- cos={r.cos:.3f}  cat=`{r.cat}`  ‖v‖={r.vec_norm:.0f}  "
              f"pos={r.position}  tok=`{r.token_str!r}`  "
              f"context=`{(r.text_preview or '')[:90]}…`")

    # ─── 6. Sink cluster characterization ────────────────────────────────────
    w(f"\n## 6. Sink-token cluster (vec_norm ≥ {args.norm_cutoff:,.0f})\n")
    w(f"N = **{len(sink_df)}** rows. Top-1 fallback phrase prevalence:\n")
    w("| pattern | %_in_sink | %_in_main |")
    w("|---|---|---|")
    for name in FALLBACK_PATTERNS:
        col = f"fb_{name}"
        sf = 100 * sink_df[col].mean() if len(sink_df) else 0
        mf = 100 * main_df[col].mean()
        w(f"| `{name}` | {sf:.1f} % | {mf:.1f} % |")
    w(f"\nMean cos in sink cluster: **{sink_df['cos'].mean():.3f}** "
      f"vs **{main_df['cos'].mean():.3f}** in non-sink.\n")
    w("Top-10 sink tokens by frequency:\n")
    sink_top = (sink_df["token_str"].value_counts().head(10))
    w("| token | n |")
    w("|---|---|")
    for tok, n in sink_top.items():
        w(f"| `{tok!r}` | {n} |")

    # ─── 7. AR predicted-norm sanity ─────────────────────────────────────────
    # The recon predicts a unit-direction; cos+mse_nrm captures everything.
    # Just report cos vs explanation length to confirm "AV gave up" hypothesis.
    w("\n## 7. Cos vs AV explanation length (non-sink)\n")
    main_df["expl_word_bucket"] = pd.cut(
        main_df["expl_len_words"].fillna(0),
        bins=[-1, 20, 35, 50, 70, 100, 9999],
        labels=["≤20", "21-35", "36-50", "51-70", "71-100", "100+"],
    )
    out.extend(report_table(
        main_df, "expl_word_bucket",
        "7a. Cos by explanation length (words)",
        min_count=50,
    ))

    # ─── 8. Position re-look, separating sink ────────────────────────────────
    w("\n## 8. Position effect (non-sink, decile bins)\n")
    main_df["pos_bucket"] = pd.qcut(
        main_df["position"], 10, duplicates="drop")
    out.extend(report_table(
        main_df.assign(pos_bucket=main_df["pos_bucket"].astype(str)),
        "pos_bucket",
        "8. Cos by position decile (non-sink)",
        min_count=50,
    ))

    # ─── 9. Single-char alpha tokens — what character matters? ───────────────
    w("\n## 9. Single-character alphabetic tokens — which letters reconstruct worst?\n")
    sc = main_df[main_df["cat"] == "single_char_alpha"].copy()
    if len(sc):
        sc_g = (sc.groupby("token_str").agg(
            n=("cos", "size"), cos_mean=("cos", "mean")
        ).reset_index().sort_values("cos_mean").head(20))
        w("Bottom-20 single-character tokens (n ≥ 3):")
        w("| token | n | cos_mean |")
        w("|---|---|---|")
        for r in sc_g[sc_g["n"] >= 3].itertuples():
            w(f"| `{r.token_str!r}` | {r.n} | {r.cos_mean:.3f} |")

    # ─── 10. AV verbatim duplication — finds memorised templates ─────────────
    w("\n## 10. AV explanations seen multiple times verbatim (≥ 5 copies)\n")
    w("If the AV emits identical text for many distinct activations, that's a "
      "memorised template — likely a fallback when the AV can't extract content.\n")
    expl_first_chars = df["explanation"].fillna("").astype(str).str[:200]
    dup_counts = expl_first_chars.value_counts()
    dup_counts = dup_counts[dup_counts >= 5].head(20)
    w("| n_copies | mean_cos_when_emitted | first 200 chars of explanation |")
    w("|---|---|---|")
    for prefix, n in dup_counts.items():
        m = expl_first_chars == prefix
        cm = df.loc[m, "cos"].mean()
        # Escape pipes & truncate
        clean = (prefix or "").replace("|", "\\|").replace("\n", " ").replace("`", "'")[:200]
        w(f"| {n} | {cm:.3f} | `{clean}` |")

    text = "\n".join(out)
    if args.out == "-":
        print(text)
    else:
        Path(args.out).write_text(text)
        print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
