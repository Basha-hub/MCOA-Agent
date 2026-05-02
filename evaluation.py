#!/usr/bin/env python3
"""
evaluation.py — MCoA Agent vs BeautifulSoup Baseline Benchmark (Step 5)
========================================================================
Runs a head-to-head comparison of two extraction strategies across two
dashboard versions, then prints a formatted summary report and saves all
results to a CSV.

STRATEGY A — BeautifulSoup (baseline):
  A conventional scraper that locates data by querying hard-coded CSS class
  names and HTML tag types (e.g. soup.find("table", class_="revenue-table")).
  Works on dashboard_v1.html (semantic, descriptive class names).
  Fails completely on dashboard_v2.html (obfuscated classes, no <table> tag).

STRATEGY B — MCoA Agent:
  Full visual pipeline: Set-of-Mark annotation → Gemini 2.5 Flash visual
  reasoning → Playwright DOM extraction. Knows nothing about HTML structure.
  Expected to succeed on both dashboard versions with equal accuracy.

TEST CASES (5 queries with known ground-truth answers):
  Each query targets a specific cell in the quarterly revenue table.
  The correct answers are read directly from the HTML source.

METRICS:
  - Accuracy: did the extracted text exactly match the ground-truth answer?
  - Latency: wall-clock seconds per extraction
  - Token usage: Gemini prompt + output token counts per call
  - Estimated cost: USD per call, using published Gemini 2.5 Flash pricing

Usage:
    export GOOGLE_API_KEY="your-key-here"
    python evaluation.py

Optional flags:
    --skip-annotation    Reuse existing annotated_v1.png / annotated_v2.png
                         (saves ~6 seconds if they are already up to date)
    --output <csv>       Where to write the raw results. Default: eval_results.csv
    --model <name>       Gemini model name. Default: gemini-2.5-flash
"""

import os
import sys
import time
import pathlib
import argparse
import datetime

import pandas as pd
from bs4 import BeautifulSoup

# Import our pipeline modules directly — no subprocess overhead.
from som_annotator import annotate as som_annotate
from gemini_agent import query_gemini, load_image_bytes
from extract import extract_text


# ---------------------------------------------------------------------------
# Gemini 2.5 Flash pricing (approximate, Q2 2026).
# Source: https://ai.google.dev/gemini-api/docs/models
# Prices are for prompts ≤200K tokens (standard tier).
# ---------------------------------------------------------------------------
FLASH_INPUT_COST_PER_M  = 0.075   # USD per million input tokens
FLASH_OUTPUT_COST_PER_M = 0.300   # USD per million output tokens


# ---------------------------------------------------------------------------
# Test cases.
#
# Each entry specifies:
#   query         — the natural-language question sent to Gemini
#   segment       — the row label, used by the BS4 scraper for row matching
#   bs4_col_class — the CSS class on the target <td> in dashboard_v1.html;
#                   this is the selector the BS4 scraper hard-codes
#   correct       — the exact string value the cell should contain
#
# All five answers can be verified against dashboard_v1.html:
#   Cloud Platform Services  Q3 Revenue  → $5.10M
#   Enterprise Consulting    Q4 Revenue  → $2.75M
#   Data Licensing           Profit Margin → 44.1%
#   AI Solutions             YoY Growth  → +210.0%
#   Managed Security         Q1 Revenue  → $1.10M
# ---------------------------------------------------------------------------
TEST_CASES = [
    {
        "query":         "Q3 Revenue for Cloud Platform Services",
        "segment":       "Cloud Platform Services",
        "bs4_col_class": "q3-revenue",
        "correct":       "$5.10M",
    },
    {
        "query":         "Q4 Revenue for Enterprise Consulting",
        "segment":       "Enterprise Consulting",
        "bs4_col_class": "q4-revenue",
        "correct":       "$2.75M",
    },
    {
        "query":         "Profit Margin for Data Licensing",
        "segment":       "Data Licensing",
        "bs4_col_class": "profit-margin",
        "correct":       "44.1%",
    },
    {
        "query":         "YoY Growth for AI Solutions",
        "segment":       "AI Solutions",
        "bs4_col_class": "yoy-growth",
        "correct":       "+210.0%",
    },
    {
        "query":         "Q1 Revenue for Managed Security",
        "segment":       "Managed Security",
        "bs4_col_class": "q1-revenue",
        "correct":       "$1.10M",
    },
]


# ---------------------------------------------------------------------------
# BeautifulSoup baseline scraper.
#
# This mirrors what a developer would write after inspecting dashboard_v1.html:
#   1. Find the <table class="revenue-table"> element.
#   2. Walk <tbody> rows; find the row whose .segment-name cell matches the
#      target segment name.
#   3. In that row, find the <td> with the target column class and return its
#      text.
#
# On dashboard_v2.html every step fails:
#   - soup.find("table", class_="revenue-table") returns None (no <table>)
#   - Even if we relaxed that, there are no <td class="segment-name"> cells
#   - The column classes (.q3-revenue etc.) do not exist in v2
#
# Returns (extracted_text | None, latency_seconds).
# ---------------------------------------------------------------------------
def scrape_bs4(html_path: str, segment: str, col_class: str) -> tuple:
    t0 = time.perf_counter()
    try:
        with open(html_path, encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")

        # Step 1: locate the semantic table by its CSS class name
        table = soup.find("table", class_="revenue-table")
        if table is None:
            return None, time.perf_counter() - t0

        # Step 2: find the tbody row whose segment-name cell matches
        tbody = table.find("tbody")
        if tbody is None:
            return None, time.perf_counter() - t0

        for row in tbody.find_all("tr"):
            name_cell = row.find("td", class_="segment-name")
            if name_cell is None:
                continue
            # Use 'in' to handle the "AI Solutions (new)" case where the cell
            # contains extra inline text from the <em> child element
            if segment not in name_cell.get_text():
                continue

            # Step 3: within the matched row, find the target column cell
            target_cell = row.find("td", class_=col_class)
            if target_cell:
                return target_cell.get_text().strip(), time.perf_counter() - t0

        # Row was found but the target column class was absent
        return None, time.perf_counter() - t0

    except Exception:
        return None, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Full MCoA pipeline runner (single query against one HTML file).
#
# Steps:
#   1. Load the pre-generated annotated PNG for this dashboard version.
#   2. Send it to Gemini with the MCoA prompt and receive a target_box_id.
#   3. Use Playwright to open the HTML file and extract the text at that index.
#   4. Compare extracted text to the known correct answer.
#
# Returns a flat dict with all metrics for this single run.
# ---------------------------------------------------------------------------
def run_mcoa(
    html_path: str,
    png_path: str,
    query: str,
    correct: str,
    api_key: str,
    model_name: str,
) -> dict:

    t0 = time.perf_counter()
    result = {
        "extracted": None,
        "box_id":    None,
        "is_correct": False,
        "latency":   0.0,
        "prompt_tok": 0,
        "output_tok": 0,
        "total_tok":  0,
        "cost_usd":   0.0,
        "error":      None,
    }

    try:
        image_bytes = load_image_bytes(png_path)

        gemini_result, _ = query_gemini(
            api_key=api_key,
            image_bytes=image_bytes,
            query=query,
            model_name=model_name,
            verbose=False,       # suppress per-call print output
        )

        result["prompt_tok"] = gemini_result.get("_prompt_tokens", 0)
        result["output_tok"] = gemini_result.get("_output_tokens", 0)
        result["total_tok"]  = gemini_result.get("_total_tokens",  0)
        result["cost_usd"] = (
            (result["prompt_tok"] / 1_000_000) * FLASH_INPUT_COST_PER_M +
            (result["output_tok"] / 1_000_000) * FLASH_OUTPUT_COST_PER_M
        )

        box_id = gemini_result.get("target_box_id")
        if box_id is None:
            result["error"]   = "Gemini returned no target_box_id"
            result["latency"] = time.perf_counter() - t0
            return result

        result["box_id"] = box_id

        extracted = extract_text(html_path, box_id, verbose=False)
        result["extracted"]  = extracted
        result["is_correct"] = (extracted.strip() == correct.strip())

    except Exception as exc:
        result["error"] = str(exc)

    result["latency"] = time.perf_counter() - t0
    return result


# ---------------------------------------------------------------------------
# Output helpers.
# ---------------------------------------------------------------------------

def _pass_fail(value: str | None, correct: str) -> str:
    """Return a fixed-width PASS/FAIL label."""
    if value is None:
        return "FAIL (None)"
    return "PASS" if value.strip() == correct.strip() else f"FAIL ({value})"


def print_section_table(title: str, html_label: str, rows: list) -> None:
    """
    Print one section of the results table (one dashboard version).
    `rows` is a list of dicts with keys: query, correct, bs4, mcoa.
    """
    COL_Q   = 44    # query column width
    COL_EXP = 12
    COL_BS4 = 14
    COL_MCA = 14

    sep = "─" * (COL_Q + COL_EXP + COL_BS4 + COL_MCA + 7)

    print(f"\n{title} — {html_label}")
    print(sep)
    print(
        f" {'#':<3} {'Query':<{COL_Q}} {'Expected':<{COL_EXP}} "
        f"{'BeautifulSoup':<{COL_BS4}} {'MCoA Agent':<{COL_MCA}}"
    )
    print(sep)
    for i, r in enumerate(rows, start=1):
        q_truncated = r["query"][:COL_Q - 1].ljust(COL_Q)
        print(
            f" {i:<3} {q_truncated} {r['correct']:<{COL_EXP}} "
            f"{r['bs4_label']:<{COL_BS4}} {r['mcoa_label']:<{COL_MCA}}"
        )
    print(sep)


def print_summary(
    v1_bs4_rows, v2_bs4_rows,
    v1_mcoa_rows, v2_mcoa_rows,
) -> None:
    """Print the four-column summary with aggregated metrics."""

    def _rate(rows):
        passed = sum(1 for r in rows if r["is_correct"])
        return f"{passed}/{len(rows)} ({100*passed//len(rows)}%)"

    def _mean_lat(rows):
        return f"{sum(r['latency'] for r in rows) / len(rows):.2f}s"

    def _total_tok(rows):
        t = sum(r.get("total_tok", 0) for r in rows)
        return f"{t:,}" if t else "—"

    def _total_cost(rows):
        c = sum(r.get("cost_usd", 0.0) for r in rows)
        return f"${c:.5f}" if c else "—"

    C = 18
    print(f"\n{'═' * (C * 4 + 26)}")
    print("SUMMARY")
    print(f"{'═' * (C * 4 + 26)}")
    print(f"{'Metric':<24} {'BS4 v1':<{C}} {'BS4 v2':<{C}} {'MCoA v1':<{C}} {'MCoA v2':<{C}}")
    print(f"{'─' * (C * 4 + 26)}")
    print(f"{'Accuracy':<24} {_rate(v1_bs4_rows):<{C}} {_rate(v2_bs4_rows):<{C}} {_rate(v1_mcoa_rows):<{C}} {_rate(v2_mcoa_rows):<{C}}")
    print(f"{'Mean Latency':<24} {_mean_lat(v1_bs4_rows):<{C}} {_mean_lat(v2_bs4_rows):<{C}} {_mean_lat(v1_mcoa_rows):<{C}} {_mean_lat(v2_mcoa_rows):<{C}}")
    print(f"{'Total Tokens':<24} {'—':<{C}} {'—':<{C}} {_total_tok(v1_mcoa_rows):<{C}} {_total_tok(v2_mcoa_rows):<{C}}")
    print(f"{'Est. API Cost':<24} {'—':<{C}} {'—':<{C}} {_total_cost(v1_mcoa_rows):<{C}} {_total_cost(v2_mcoa_rows):<{C}}")
    print(f"{'═' * (C * 4 + 26)}")
    print("* Gemini 2.5 Flash pricing: $0.075/M input tokens, $0.30/M output tokens (approx)")


def save_csv(output_path: str, all_rows: list) -> None:
    df = pd.DataFrame(all_rows)
    df.to_csv(output_path, index=False)
    print(f"\nRaw results saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main orchestration.
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:

    # Resolve HTML paths
    v1_html = pathlib.Path("dashboard_v1.html").resolve()
    v2_html = pathlib.Path("dashboard_v2.html").resolve()
    v1_png  = pathlib.Path("annotated_v1.png").resolve()
    v2_png  = pathlib.Path("annotated_v2.png").resolve()

    for p in (v1_html, v2_html):
        if not p.exists():
            print(f"[ERROR] Required file not found: {p}")
            sys.exit(1)

    # ------------------------------------------------------------------
    # Phase 0: Annotate (or reuse existing PNGs)
    # ------------------------------------------------------------------
    print("=" * 72)
    print("MCoA Agent vs BeautifulSoup — Evaluation Benchmark")
    print("=" * 72)

    if args.skip_annotation and v1_png.exists() and v2_png.exists():
        print("[Setup] Reusing existing annotated PNGs (--skip-annotation).")
    else:
        print("[Setup] Generating Set-of-Mark annotated screenshots ...")
        som_annotate(str(v1_html), str(v1_png))
        som_annotate(str(v2_html), str(v2_png))
        print("[Setup] Annotation complete.")

    # ------------------------------------------------------------------
    # Phase 1: Run all test cases
    # ------------------------------------------------------------------
    print(f"\n[Eval] Running {len(TEST_CASES)} test cases × 2 dashboards × 2 methods "
          f"= {len(TEST_CASES) * 4} extractions total.")
    print(f"[Eval] MCoA calls will contact the Gemini API — expect ~8–15s each.\n")

    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        print("[ERROR] GOOGLE_API_KEY is not set. Export it and re-run.")
        sys.exit(1)

    # Accumulate per-row result dicts for the summary + CSV
    v1_bs4_rows   = []
    v2_bs4_rows   = []
    v1_mcoa_rows  = []
    v2_mcoa_rows  = []

    # Per-section display rows (for the per-dashboard tables)
    v1_display = []
    v2_display = []

    all_csv_rows = []

    for i, tc in enumerate(TEST_CASES, start=1):
        q       = tc["query"]
        seg     = tc["segment"]
        col_cls = tc["bs4_col_class"]
        correct = tc["correct"]

        print(f"  [{i}/{len(TEST_CASES)}] {q}")

        # ── BeautifulSoup on v1 ──────────────────────────────────────
        bs4_v1_text, bs4_v1_lat = scrape_bs4(str(v1_html), seg, col_cls)
        bs4_v1_correct = bs4_v1_text is not None and bs4_v1_text.strip() == correct.strip()
        v1_bs4_rows.append({"is_correct": bs4_v1_correct, "latency": bs4_v1_lat})

        # ── BeautifulSoup on v2 ──────────────────────────────────────
        bs4_v2_text, bs4_v2_lat = scrape_bs4(str(v2_html), seg, col_cls)
        bs4_v2_correct = bs4_v2_text is not None and bs4_v2_text.strip() == correct.strip()
        v2_bs4_rows.append({"is_correct": bs4_v2_correct, "latency": bs4_v2_lat})

        # ── MCoA on v1 ───────────────────────────────────────────────
        print(f"         MCoA v1 ...", end=" ", flush=True)
        mcoa_v1 = run_mcoa(str(v1_html), str(v1_png), q, correct, api_key, args.model)
        v1_mcoa_rows.append(mcoa_v1)
        print("PASS" if mcoa_v1["is_correct"] else f"FAIL ({mcoa_v1.get('error') or mcoa_v1['extracted']})")

        # ── MCoA on v2 ───────────────────────────────────────────────
        print(f"         MCoA v2 ...", end=" ", flush=True)
        mcoa_v2 = run_mcoa(str(v2_html), str(v2_png), q, correct, api_key, args.model)
        v2_mcoa_rows.append(mcoa_v2)
        print("PASS" if mcoa_v2["is_correct"] else f"FAIL ({mcoa_v2.get('error') or mcoa_v2['extracted']})")

        # ── Accumulate display rows ───────────────────────────────────
        v1_display.append({
            "query":      q,
            "correct":    correct,
            "bs4_label":  _pass_fail(bs4_v1_text, correct),
            "mcoa_label": "PASS" if mcoa_v1["is_correct"] else "FAIL",
            "is_correct": mcoa_v1["is_correct"],
        })
        v2_display.append({
            "query":      q,
            "correct":    correct,
            "bs4_label":  _pass_fail(bs4_v2_text, correct),
            "mcoa_label": "PASS" if mcoa_v2["is_correct"] else "FAIL",
            "is_correct": mcoa_v2["is_correct"],
        })

        # ── Accumulate CSV rows ───────────────────────────────────────
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        for label, method, html_name, text, correct_flag, lat, ptok, otok, cost in [
            ("bs4",  "BeautifulSoup", "dashboard_v1.html", bs4_v1_text, bs4_v1_correct, bs4_v1_lat, 0, 0, 0.0),
            ("bs4",  "BeautifulSoup", "dashboard_v2.html", bs4_v2_text, bs4_v2_correct, bs4_v2_lat, 0, 0, 0.0),
            ("mcoa", "MCoA",          "dashboard_v1.html", mcoa_v1["extracted"], mcoa_v1["is_correct"], mcoa_v1["latency"], mcoa_v1["prompt_tok"], mcoa_v1["output_tok"], mcoa_v1["cost_usd"]),
            ("mcoa", "MCoA",          "dashboard_v2.html", mcoa_v2["extracted"], mcoa_v2["is_correct"], mcoa_v2["latency"], mcoa_v2["prompt_tok"], mcoa_v2["output_tok"], mcoa_v2["cost_usd"]),
        ]:
            all_csv_rows.append({
                "timestamp":      ts,
                "method":         method,
                "html_file":      html_name,
                "query":          q,
                "correct_answer": correct,
                "extracted_text": text,
                "is_correct":     correct_flag,
                "latency_s":      round(lat, 4),
                "prompt_tokens":  ptok,
                "output_tokens":  otok,
                "cost_usd":       round(cost, 6),
            })

    # ------------------------------------------------------------------
    # Phase 2: Print results
    # ------------------------------------------------------------------
    print_section_table("Test A", "dashboard_v1.html (semantic HTML)", v1_display)
    print_section_table("Test B", "dashboard_v2.html (obfuscated, div-grid)", v2_display)
    print_summary(v1_bs4_rows, v2_bs4_rows, v1_mcoa_rows, v2_mcoa_rows)

    # ------------------------------------------------------------------
    # Phase 3: Save CSV
    # ------------------------------------------------------------------
    save_csv(args.output, all_csv_rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MCoA agent vs BeautifulSoup baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--skip-annotation",
        action="store_true",
        help="Reuse existing annotated PNGs instead of regenerating them.",
    )
    parser.add_argument(
        "--output",
        default="eval_results.csv",
        help="Path for the output CSV. Default: eval_results.csv",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model name. Default: gemini-2.5-flash",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
