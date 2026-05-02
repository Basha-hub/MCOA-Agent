#!/usr/bin/env python3
"""
extract.py — DOM Element Text Extractor (Step 4)
=================================================
Takes the target_box_id returned by gemini_agent.py and uses Playwright to
open the original HTML file, rebuild the same ordered Set-of-Mark element
list that som_annotator.py produced, locate the specific DOM element at that
index, and extract its full rendered inner text.

The extracted value is then saved as a row in a CSV file (via Pandas) so that
results from multiple extractions and both dashboard versions can be compared
side-by-side in evaluation.py (Step 5).

Why re-run element collection here instead of using the annotated PNG?
  The annotated PNG is a pixel image — it contains visual evidence of the box
  layout but not the underlying DOM text. To extract the actual data value we
  must go back to the live page and use Playwright's ElementHandle API.
  We re-run the same collection heuristic (direct text nodes, visible, ≥8px)
  against the same HTML file to reconstruct the exact same ordered list of
  elements that the annotator produced, then index into it with box_id - 1.
  Because the heuristic is deterministic and the HTML file is static, the
  element at index N is always the same element that was given Box ID N+1.

This module is also importable: extract_text() can be called directly by
evaluation.py without going through the CLI, allowing batch extraction without
subprocess overhead.

Usage:
    python extract.py --html <html-file> --box-id <N> [--query <text>] [--output <csv>]

Examples:
    python extract.py --html dashboard_v1.html --box-id 31 \\
        --query "Q3 Revenue for Cloud Platform Services"

    python extract.py --html dashboard_v2.html --box-id 31 \\
        --query "Q3 Revenue for Cloud Platform Services" \\
        --output results.csv
"""

import sys
import argparse
import pathlib
import datetime

import pandas as pd
from playwright.sync_api import sync_playwright


# ---------------------------------------------------------------------------
# JavaScript that rebuilds the Set-of-Mark element list in the live page.
#
# This is the same collection heuristic used in som_annotator.py, but
# stripped of the overlay-drawing code — here we only need to reconstruct
# the ordered list so we can index into it.
#
# The collected elements are stored in window.__som_elements__ (a plain array
# of live DOM Element references) so that a follow-up evaluate_handle() call
# can retrieve a specific element by its 0-based index.
#
# Returns the total number of elements collected, which we use for
# bounds-checking the requested box_id.
# ---------------------------------------------------------------------------
COLLECT_ELEMENTS_JS = """
(function () {
    'use strict';

    const allElements = document.querySelectorAll('*');
    const targets = [];

    for (const el of allElements) {

        // (a) Element must have at least one direct non-empty text node.
        //     This matches the exact same rule used in the annotator, ensuring
        //     the index of each element is identical across both scripts.
        let hasDirectText = false;
        for (const child of el.childNodes) {
            if (
                child.nodeType === Node.TEXT_NODE &&
                child.textContent.trim().length > 0
            ) {
                hasDirectText = true;
                break;
            }
        }
        if (!hasDirectText) continue;

        // (b) Element must not be hidden by computed style.
        const style = window.getComputedStyle(el);
        if (
            style.display      === 'none'   ||
            style.visibility   === 'hidden' ||
            parseFloat(style.opacity) === 0
        ) {
            continue;
        }

        // (c) Element must have a non-trivial rendered bounding box.
        //     display:contents elements (e.g. .qr_drow in v2) return a
        //     zero-size rect and are excluded here, just as in the annotator.
        const rect = el.getBoundingClientRect();
        if (rect.width < 8 || rect.height < 8) continue;

        targets.push(el);
    }

    // Expose as a global so evaluate_handle() can reference elements by index
    window.__som_elements__ = targets;

    return targets.length;
})();
"""


def extract_text(html_path: str, box_id: int, verbose: bool = True) -> str:
    """
    Open html_path in Playwright, rebuild the SoM element list, locate the
    element at position box_id (1-indexed), and return its rendered inner text.

    Parameters
    ----------
    html_path : str
        Path to the local HTML file (dashboard_v1.html or dashboard_v2.html).
    box_id : int
        1-based box number as returned by gemini_agent.py.

    Returns
    -------
    str
        The stripped inner text of the target DOM element.

    Raises
    ------
    FileNotFoundError
        If the HTML file does not exist at the given path.
    ValueError
        If box_id is outside the range 1..total_boxes.
    RuntimeError
        If Playwright cannot resolve the element handle to a DOM element.
    """

    html_abs = pathlib.Path(html_path).resolve()
    if not html_abs.exists():
        raise FileNotFoundError(f"HTML file not found: {html_abs}")

    file_url = html_abs.as_uri()

    if verbose:
        print(f"[Extractor] HTML file : {html_abs.name}")
        print(f"[Extractor] Target    : Box ID {box_id}")

    with sync_playwright() as pw:

        # Same browser binary used throughout the project
        browser = pw.chromium.launch(
            executable_path='/opt/google/chrome/chrome',
            headless=True,
        )
        context = browser.new_context(viewport={'width': 1280, 'height': 900})
        page = context.new_page()

        # Load the page using the same wait condition as the annotator so the
        # rendered state is identical to what the annotator saw
        page.goto(file_url, wait_until='networkidle')
        page.wait_for_timeout(400)

        # Rebuild the ordered element list and store it on window.__som_elements__
        total_boxes = page.evaluate(COLLECT_ELEMENTS_JS)
        if verbose:
            print(f"[Extractor] Elements found: {total_boxes}")

        if box_id < 1 or box_id > total_boxes:
            context.close()
            browser.close()
            raise ValueError(
                f"Box ID {box_id} is out of range. "
                f"This page has {total_boxes} annotatable elements (1–{total_boxes})."
            )

        # Retrieve a JSHandle pointing at the DOM element at index box_id-1.
        # evaluate_handle() keeps the object alive in the JS VM and returns a
        # Python proxy (JSHandle) rather than serialising it. We then call
        # as_element() to obtain a typed ElementHandle, which exposes the
        # inner_text() method.
        js_handle = page.evaluate_handle(
            f"window.__som_elements__[{box_id - 1}]"
        )
        element = js_handle.as_element()

        if element is None:
            context.close()
            browser.close()
            raise RuntimeError(
                f"Box ID {box_id} resolved to a non-element JS value. "
                "The SoM collection may have produced an inconsistent result."
            )

        # inner_text() returns the rendered text as a human would read it —
        # it respects CSS visibility, skips hidden child nodes, and normalises
        # whitespace. This is what we want, not .text_content() which includes
        # hidden text and script content.
        raw_text = element.inner_text()
        extracted = raw_text.strip()

        if verbose:
            print(f"[Extractor] Extracted : {repr(extracted)}")

        context.close()
        browser.close()

    return extracted


def save_to_csv(output_path: str, record: dict) -> None:
    """
    Append a single extraction result record to a CSV file.

    If the file already exists, the new row is concatenated onto the existing
    DataFrame and the file is overwritten. If it does not exist, a new file is
    created. This gives us a running log of all extractions across Steps 4 and 5.

    Parameters
    ----------
    output_path : str
        Path to the CSV file (created if absent).
    record : dict
        A flat dict with keys: timestamp, html_file, query, box_id,
        extracted_text, correct_answer (optional), is_correct (optional).
    """

    p = pathlib.Path(output_path)
    new_row = pd.DataFrame([record])

    if p.exists():
        existing = pd.read_csv(p)
        # Align columns — new_row may have columns that the existing CSV lacks
        # (e.g. when evaluation.py adds is_correct). pd.concat handles this
        # by filling missing values with NaN.
        updated = pd.concat([existing, new_row], ignore_index=True)
    else:
        updated = new_row

    updated.to_csv(p, index=False)


def run(html_path: str, box_id: int, query: str, output_path: str, verbose: bool = True) -> str:
    """
    End-to-end extraction + CSV save. Returns the extracted text.
    Callable from evaluation.py without going through the CLI.
    """

    extracted = extract_text(html_path, box_id, verbose=verbose)

    record = {
        "timestamp":      datetime.datetime.now().isoformat(timespec="seconds"),
        "html_file":      pathlib.Path(html_path).name,
        "query":          query,
        "box_id":         box_id,
        "extracted_text": extracted,
    }

    save_to_csv(output_path, record)
    return extracted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract DOM text for a given SoM Box ID from an HTML file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--html",
        required=True,
        help="Path to the HTML file to open (dashboard_v1.html or dashboard_v2.html).",
    )
    parser.add_argument(
        "--box-id",
        type=int,
        required=True,
        dest="box_id",
        help="1-based Box ID returned by gemini_agent.py.",
    )
    parser.add_argument(
        "--query",
        default="",
        help="Original extraction query (recorded in the CSV for traceability).",
    )
    parser.add_argument(
        "--output",
        default="results.csv",
        help="CSV file to append the result to. Created if absent. Default: results.csv",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    args = parse_args()

    print("=" * 60)

    try:
        extracted = run(
            html_path=args.html,
            box_id=args.box_id,
            query=args.query,
            output_path=args.output,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    print("=" * 60)
    print(f"  Box ID         : {args.box_id}")
    print(f"  Query          : {args.query or '(not specified)'}")
    print(f"  Extracted text : {extracted}")
    print("=" * 60)
