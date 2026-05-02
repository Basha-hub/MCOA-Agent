# MCoA GUI Agent — Robust Automated Data Extraction via Multimodal Chain-of-Action

A production-grade autonomous web data extraction agent that replaces brittle CSS-selector-based scrapers with **visual reasoning**. Instead of parsing HTML source code, the agent renders the page as a screenshot — precisely what a human observer perceives — and employs Google Gemini's multimodal capabilities to reason step-by-step about where target data resides on screen.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [The Solution: MCoA Visual Reasoning](#the-solution-mcoa-visual-reasoning)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Environment & Dependencies](#environment--dependencies)
- [Setup Instructions](#setup-instructions)
- [Running the Pipeline](#running-the-pipeline)
- [Evaluation Results](#evaluation-results)
- [Known Limitations](#known-limitations)
- [Possible Extensions](#possible-extensions)
- [Critical Configuration Notes](#critical-configuration-notes)

---

## Problem Statement

Traditional web scraping frameworks such as BeautifulSoup or Selenium operate by querying a page's HTML source using hard-coded CSS class selectors. A canonical example:

```python
# Locates the revenue table by its exact class name in the HTML source
table = soup.find("table", class_="revenue-table")

# Locates the Q3 revenue cell for a specific row by its exact class name
cell = row.find("td", class_="q3-revenue")
```

This approach is structurally fragile. Any of the following routine events — a front-end redesign, a framework migration, a CSS minifier renaming classes to hashed strings, or a developer refactoring component names — will cause every selector to return `None`, silently breaking the entire data pipeline.

The fundamental deficiency is that the scraper is coupled to the **markup**, not to the **meaning** of the content. When the markup changes, so does the contract the scraper depends upon.

---

## The Solution: MCoA Visual Reasoning

This project decouples the extraction logic entirely from the HTML structure by operating at the **pixel level** rather than the source-code level.

The agent's reasoning pipeline proceeds as follows:

1. A real browser (Google Chrome, via Playwright) renders the target HTML file fully — CSS, fonts, layout, and all.
2. A JavaScript overlay annotates every visible text-bearing element with a numbered red bounding box (**Set-of-Mark labelling**).
3. A screenshot of this annotated page is passed to **Google Gemini 2.5 Flash** alongside a structured natural-language query.
4. Gemini executes a **Multimodal Chain-of-Action (MCoA)** reasoning protocol: it orients to the dashboard layout, identifies column headers, identifies row labels, triangulates the intersection cell, and returns a JSON object containing the target box ID.
5. Playwright re-opens the HTML, rebuilds the same element list deterministically, and extracts the inner text of element N.

Because the agent reasons from what is **visually rendered** rather than what is **structurally encoded**, it succeeds regardless of whether the underlying HTML uses `<td class="q3-revenue">` or `<div class="b2nr r9mw_c">`. A cell displaying `$5.10M` will be found correctly either way.

---

## Architecture

The system is divided into three conceptual layers:

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1 — THE HANDS  (Playwright + Google Chrome)          │
│                                                             │
│  Opens the HTML file in a real headless browser.            │
│  Injects JavaScript to draw numbered bounding boxes over    │
│  every visible text-bearing element (Set-of-Mark).          │
│  Takes a full-page screenshot.                              │
│  Also used at the end to pull the actual text from the DOM. │
└─────────────────────────────────────────────────────────────┘
             │ annotated PNG                ▲ box ID integer
             ▼                             │
┌─────────────────────────────────────────────────────────────┐
│  Layer 2 — THE BRAIN  (Google Gemini 2.5 Flash)             │
│                                                             │
│  Receives the annotated screenshot and a natural-language   │
│  query. Executes a 5-step spatial reasoning protocol.       │
│  Returns a JSON object with target_box_id and chain-of-     │
│  thought reasoning.                                         │
└─────────────────────────────────────────────────────────────┘
             ▲ image + prompt              │ JSON response
             │                             ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 3 — THE GLUE  (Python orchestration)                 │
│                                                             │
│  Coordinates Playwright and Gemini.                         │
│  Parses Gemini's JSON response.                             │
│  Passes the box ID back to Playwright for text extraction.  │
│  Logs results to CSV.                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
mcoa-agent/
│
├── dashboard_v1.html       Semantically-named HTML dashboard (BeautifulSoup-friendly)
├── dashboard_v2.html       Visually identical dashboard with fully obfuscated HTML
│
├── som_annotator.py        Set-of-Mark annotator — injects JS bounding boxes, takes screenshot
├── gemini_agent.py         MCoA visual reasoning agent — sends image to Gemini, parses JSON
├── extract.py              DOM text extractor — maps a box ID back to live DOM inner text
├── evaluation.py           Head-to-head benchmark — BeautifulSoup vs MCoA agent
│
├── annotated_v1.png        Generated annotated screenshot of dashboard_v1 (auto-generated)
├── annotated_v2.png        Generated annotated screenshot of dashboard_v2 (auto-generated)
│
├── results.csv             Running log of individual extractions (auto-generated)
├── eval_results.csv        Full evaluation benchmark output (auto-generated)
│
├── .gitignore              Excludes virtual environment, API keys, and generated artifacts
└── README.md               This document
```

> **Note**: `agent_env/` (the Python virtual environment) is excluded via `.gitignore` and must be reconstructed locally following the setup instructions below.

---

## Environment & Dependencies

| Component | Value |
|---|---|
| Operating System | Ubuntu 22.04+ (x64) |
| Python | 3.10+ |
| Browser Binary | `/opt/google/chrome/chrome` |
| Gemini SDK | `google-genai==1.73.1` |
| Browser Automation | `playwright==1.x` |

### Python Packages

```
playwright          — headless browser automation
google-genai        — Google Gemini API SDK (the current SDK, not the deprecated one)
pandas              — CSV output and data handling
beautifulsoup4      — used exclusively in evaluation.py as the brittle baseline scraper
```

### Critical SDK Note

The package `google-generativeai` (imported as `import google.generativeai as genai`) is **fully deprecated**. Do not install or use it. The current replacement is `google-genai`, which uses a completely different API surface:

```python
# DEPRECATED — do not use
import google.generativeai as genai
genai.configure(api_key=key)

# CURRENT — use this
from google import genai
from google.genai import types
client = genai.Client(api_key=key)
```

---

## Setup Instructions

### Step 1 — Clone the repository

```bash
git clone https://github.com/<your-username>/mcoa-agent.git
cd mcoa-agent
```

### Step 2 — Install Google Chrome

The system depends on Google Chrome specifically. The Snap-packaged Chromium binary at `/usr/bin/chromium-browser` will **not** work because its `snap-confine` wrapper lacks the `cap_dac_override` capability that Playwright requires. Install Chrome directly:

```bash
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo dpkg -i google-chrome-stable_current_amd64.deb
sudo apt-get install -f   # resolves any dependency issues
```

Verify the binary exists at the expected path:

```bash
ls /opt/google/chrome/chrome
```

### Step 3 — Create and activate the virtual environment

```bash
python3 -m venv agent_env
source agent_env/bin/activate
```

### Step 4 — Install Python dependencies

```bash
pip install playwright google-genai pandas beautifulsoup4
```

### Step 5 — Install Playwright's browser drivers

```bash
playwright install chromium
```

Although the scripts use the system Google Chrome binary directly (bypassing Playwright's own Chromium), this step installs supporting libraries that Playwright depends upon at runtime.

### Step 6 — Configure your Google API key

Obtain a Gemini API key from [Google AI Studio](https://aistudio.google.com). Export it as an environment variable — do not hard-code it into any file:

```bash
export GOOGLE_API_KEY="your-key-here"
```

To make this persistent across terminal sessions, append the export line to your `~/.bashrc` or `~/.zshrc`.

---

## Running the Pipeline

### Individual Steps

**Step 1 — Generate annotated screenshots**

This script opens the HTML file in Chrome, injects the Set-of-Mark JavaScript overlay that draws numbered red bounding boxes on every visible text-bearing element, and saves the result as a PNG.

```bash
python som_annotator.py dashboard_v1.html annotated_v1.png
python som_annotator.py dashboard_v2.html annotated_v2.png
```

**Step 2 — Query Gemini for a target box ID**

This script sends the annotated screenshot to Gemini along with a natural-language query. Gemini executes its five-step spatial reasoning protocol and returns a JSON object identifying the box ID of the element that contains the requested data value.

```bash
python gemini_agent.py \
    --image annotated_v1.png \
    --query "Q3 Revenue for Cloud Platform Services"
```

To use the higher-accuracy model:

```bash
python gemini_agent.py \
    --image annotated_v1.png \
    --query "Q3 Revenue for Cloud Platform Services" \
    --model gemini-2.5-pro
```

Expected output:

```json
{
  "thought_process": ["Step 1: ...", "Step 2: ...", "Step 3: ...", "Step 4: ...", "Step 5: ..."],
  "target_box_id": 31,
  "target_text_preview": "$5.10M",
  "confidence": "high"
}
```

**Step 3 — Extract the text from the DOM**

This script re-opens the HTML file, rebuilds the element index deterministically using the same JavaScript heuristic as the annotator, and extracts the inner text of the element at the given box ID. The result is appended to `results.csv`.

```bash
python extract.py \
    --html dashboard_v1.html \
    --box-id 31 \
    --query "Q3 Revenue for Cloud Platform Services"
```

### Full Evaluation Benchmark

This script runs all five test queries against both dashboard versions using both the BeautifulSoup baseline and the MCoA agent, prints a formatted comparison table, and saves detailed results to `eval_results.csv`.

```bash
# Reuse already-generated annotated PNGs (faster)
python evaluation.py --skip-annotation

# Regenerate annotated PNGs before running (guarantees fresh screenshots)
python evaluation.py

# Use the higher-accuracy Gemini model
python evaluation.py --skip-annotation --model gemini-2.5-pro
```

---

## Evaluation Results

The following results were obtained on the final benchmark run.

### Test A — dashboard_v1.html (Semantic HTML)

| # | Query | Expected | BeautifulSoup | MCoA Agent |
|---|---|---|---|---|
| 1 | Q3 Revenue — Cloud Platform Services | $5.10M | PASS | PASS |
| 2 | Q4 Revenue — Enterprise Consulting | $2.75M | PASS | PASS |
| 3 | Profit Margin — Data Licensing | 44.1% | PASS | PASS |
| 4 | YoY Growth — AI Solutions | +210.0% | PASS | FAIL* |
| 5 | Q1 Revenue — Managed Security | $1.10M | PASS | PASS |

### Test B — dashboard_v2.html (Obfuscated HTML)

| # | Query | Expected | BeautifulSoup | MCoA Agent |
|---|---|---|---|---|
| 1 | Q3 Revenue — Cloud Platform Services | $5.10M | FAIL | PASS |
| 2 | Q4 Revenue — Enterprise Consulting | $2.75M | FAIL | PASS |
| 3 | Profit Margin — Data Licensing | 44.1% | FAIL | PASS |
| 4 | YoY Growth — AI Solutions | +210.0% | FAIL | PASS |
| 5 | Q1 Revenue — Managed Security | $1.10M | FAIL | PASS |

### Summary

| Metric | BS4 v1 | BS4 v2 | MCoA v1 | MCoA v2 |
|---|---|---|---|---|
| Accuracy | 5/5 (100%) | 0/5 (0%) | 4/5 (80%) | 5/5 (100%) |
| Mean Latency | ~0.01s | ~0.00s | ~7.67s | ~15.30s |
| Total Tokens | — | — | 10,426 | 9,967 |
| Estimated API Cost | — | — | $0.00116 | $0.00111 |

> Cost estimates reflect published Gemini 2.5 Flash pricing ($0.075/M input tokens, $0.30/M output tokens). On the free tier, actual charges are $0.00.

### Note on the single MCoA failure (Test A, Query 4)

The query "YoY Growth for AI Solutions" on dashboard_v1 returned `18.7%` (the Profit Margin value) instead of the correct `+210.0%`. Gemini correctly identified the row (AI Solutions) but selected the wrong column — Profit Margin instead of YoY Growth. Both columns occupy the rightmost region of the table and both display percentage values, which creates a spatial disambiguation challenge. This is a genuine model-level reasoning limitation rather than a code defect. Running the same query with `--model gemini-2.5-pro` is expected to resolve it due to that model's stronger spatial reasoning capacity.

---

## Known Limitations

| Limitation | Description |
|---|---|
| Latency | Each MCoA extraction requires a full browser render, a JavaScript injection, a screenshot, and an API call. This yields 7–15 seconds per query versus sub-second for traditional scrapers. The trade-off is resilience. |
| API Rate Limit (Free Tier) | The free tier allows 5 Gemini requests per minute. The evaluation script fires 10 calls in rapid succession and handles this automatically via a retry loop with parsed `retryDelay` backoff. Expect the full benchmark to take approximately 3–4 minutes. |
| Column Disambiguation | When adjacent columns share the same data type (e.g., two percentage columns side by side), the model may confuse them. The five-step prompt scaffold mitigates this but does not eliminate it entirely. |
| Static Files Only | All testing was conducted on local HTML files. Live website scraping would require additional handling for dynamic content, authentication flows, and anti-bot countermeasures. |

---

## Possible Extensions

The following capabilities were deliberately deferred and represent logical next development steps:

**Single-command pipeline wrapper** — Currently the user invokes three scripts sequentially. A `run_pipeline.py` wrapper accepting `--html` and `--query` arguments could consolidate the annotate → reason → extract flow into a single command.

**Prompt improvement for column disambiguation** — Inserting an explicit Step 2.5 into the MCoA prompt instructing Gemini to count columns left-to-right and confirm the column index before identifying the intersection cell would address the YoY Growth confusion case.

**n8n workflow integration** — The original architecture specification described routing extracted data through an n8n webhook to a destination such as a SQL database or Airtable. No n8n code has been written; the Python extraction layer is complete and webhook-ready.

**Live website support** — Replacing `page.goto("file://...")` with `page.goto("https://...")` in `som_annotator.py` would extend the system to live websites. This would require managing dynamic rendering delays (`page.wait_for_load_state("networkidle")`), scroll-based lazy loading, and session cookies.

**Batch evaluation mode** — Gemini's batch API could process multiple queries against a single screenshot in one request, meaningfully reducing both latency and token cost for large evaluation sets.

---

## Critical Configuration Notes

These notes encode decisions that required debugging to resolve. Do not deviate from them.

**Chrome binary path** — Every Playwright `launch()` call must specify:
```python
browser = pw.chromium.launch(
    executable_path='/opt/google/chrome/chrome',
    headless=True,
)
```
The Snap wrapper at `/usr/bin/chromium-browser` will fail with a `snap-confine` capabilities error. Do not use it.

**Gemini model name** — Use `gemini-2.5-flash` as the default model. The model identifier `gemini-1.5-pro` returns a `404 NOT_FOUND` error with the current SDK version.

**`max_output_tokens`** — Must be set to at least `4096`. A value of `1024` causes the model to truncate mid-JSON, producing a parse failure.

**`temperature`** — Set to `0.1` for near-deterministic output. Higher values introduce hallucinated box IDs.

---

## License

This project was developed as an academic data science capstone demonstration. All data used (Meridian Analytics Inc. financials) is entirely fictional and generated for evaluation purposes.
