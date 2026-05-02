#!/usr/bin/env python3
"""
som_annotator.py — Set-of-Mark (SoM) Visual Grounding Script
=============================================================
Opens a local HTML file in headless Chromium via Playwright, injects a
JavaScript overlay that draws numbered red bounding boxes around every
visible, text-bearing element on the page, then saves a full-page annotated
screenshot as a PNG.

The resulting PNG is the input that the Gemini MCoA agent (Step 3) will
receive. By labelling every visible element with a unique number, the model
has a concrete reference system: it can return "the value I want is in Box 14"
rather than trying to describe an imprecise pixel coordinate, and we can look
up Box 14's corresponding DOM element to extract its text in Step 4.

The annotation strategy is purely visual/structural:
  - It finds elements by checking for direct text-node children and visible
    rendered dimensions — NOT by tag name or CSS class name.
  - This means it works identically on dashboard_v1.html (semantic <table>)
    and dashboard_v2.html (CSS-grid <div> layout with obfuscated class names).

Usage:
    python som_annotator.py <path-to-html-file> [output-png-path]

Examples:
    python som_annotator.py dashboard_v1.html
        → saves annotated_v1.png in the current directory

    python som_annotator.py dashboard_v2.html annotated_v2.png
        → saves annotated_v2.png
"""

import sys
import pathlib
from playwright.sync_api import sync_playwright


# ---------------------------------------------------------------------------
# JavaScript IIFE injected into the live page.
#
# Design goals:
#   1. Work on ANY HTML layout — no assumptions about tag names or classes.
#   2. Mark elements visually with a red border and a numbered label.
#   3. Expose a lookup table (window.__som_targets__) so Python code can
#      map a box number back to the originating DOM element in Step 4.
#   4. Return the total box count to Python for logging.
#
# Element selection heuristic:
#   An element qualifies for annotation if ALL of the following are true:
#     (a) It has at least one direct TEXT_NODE child with non-whitespace text.
#         This filters out pure wrapper/container divs that have no text of
#         their own — only their children do.
#     (b) Its computed style is not hidden (display:none, visibility:hidden,
#         or opacity:0).
#     (c) Its rendered bounding box (getBoundingClientRect) has width and
#         height both >= 8px. This removes invisible or degenerate elements
#         such as collapsed line-breaks or zero-height flex spacers.
#
# Overlay architecture:
#   A single <div id="__som_overlay__"> is attached to <body> with
#   position:absolute, covering the full document. All box markers are
#   children of this container. Using position:absolute (not fixed) means
#   the overlay stretches with the document so it survives the viewport
#   expansion and full_page screenshot that Playwright takes. pointer-events
#   is set to none on everything so the overlay never alters layout or flow.
#
# Coordinate maths:
#   getBoundingClientRect() returns viewport-relative coordinates (i.e.,
#   relative to the current scroll position). To get document-absolute
#   coordinates — which is what we need since the overlay is positioned
#   relative to the document, not the viewport — we add window.scrollX and
#   window.scrollY. For a static, unscrolled page this is 0, but it's
#   included for correctness on any page.
# ---------------------------------------------------------------------------
SOM_INJECTION_JS = """
(function () {
    'use strict';

    // ------------------------------------------------------------------
    // Step 1: Walk the entire DOM and collect qualifying elements.
    // ------------------------------------------------------------------
    function collectTargets() {
        const allElements = document.querySelectorAll('*');
        const results = [];

        for (const el of allElements) {

            // (a) Does this element own at least one direct non-empty text node?
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

            // (b) Is the element visible per computed style?
            const style = window.getComputedStyle(el);
            if (
                style.display      === 'none'   ||
                style.visibility   === 'hidden' ||
                parseFloat(style.opacity) === 0
            ) {
                continue;
            }

            // (c) Does the element have a usable rendered bounding box?
            //     Elements with display:contents (e.g. the .qr_drow wrappers
            //     in dashboard_v2) return a zero-size rect, so they are
            //     naturally excluded here.
            const rect = el.getBoundingClientRect();
            if (rect.width < 8 || rect.height < 8) continue;

            results.push({ el, rect });
        }

        return results;
    }

    // ------------------------------------------------------------------
    // Step 2: Create a full-document-size overlay container.
    //
    //   - Attached to <body> so child positions are relative to the
    //     document origin, not the viewport.
    //   - pointer-events:none on both the container and all children
    //     ensures the overlay is purely cosmetic and never affects layout.
    //   - z-index set to the maximum possible value (2^31 - 1) so the
    //     overlay always renders on top of all page content.
    // ------------------------------------------------------------------
    function createOverlay() {
        // Remove any pre-existing overlay (idempotent re-injection)
        const existing = document.getElementById('__som_overlay__');
        if (existing) existing.remove();

        const overlay = document.createElement('div');
        overlay.id = '__som_overlay__';
        overlay.style.cssText = [
            'position: absolute',
            'top: 0',
            'left: 0',
            'width: 100%',
            'height: 100%',
            'pointer-events: none',
            'z-index: 2147483647',
        ].join('; ');

        // Setting position:relative on body makes <body> the containing block
        // for our overlay's absolute children, which is what we want.
        document.body.style.position = 'relative';
        document.body.appendChild(overlay);
        return overlay;
    }

    // ------------------------------------------------------------------
    // Step 3: Draw one annotation box for a single element.
    //
    //   Each annotation consists of:
    //     - An outer <div> with a 2px red border, sized and positioned
    //       to frame the target element exactly.
    //     - A <span> label in the top-left corner of that border, showing
    //       the box number in white text on a red background pill.
    //
    //   scrollX / scrollY converts viewport-relative rect coordinates
    //   to document-absolute coordinates (necessary for full-page shots).
    // ------------------------------------------------------------------
    function drawAnnotation(overlay, rect, boxNumber) {
        const scrollX = window.scrollX || document.documentElement.scrollLeft || 0;
        const scrollY = window.scrollY || document.documentElement.scrollTop  || 0;

        const docTop  = rect.top  + scrollY;
        const docLeft = rect.left + scrollX;

        // The border frame
        const box = document.createElement('div');
        box.style.cssText = [
            'position: absolute',
            `top: ${docTop}px`,
            `left: ${docLeft}px`,
            `width: ${rect.width}px`,
            `height: ${rect.height}px`,
            'border: 2px solid #e53e3e',
            'box-sizing: border-box',
            'pointer-events: none',
        ].join('; ');

        // The numbered label
        const label = document.createElement('span');
        label.textContent = String(boxNumber);
        label.style.cssText = [
            'position: absolute',
            'top: -2px',
            'left: -2px',
            'background: #e53e3e',
            'color: #ffffff',
            'font-family: monospace, monospace',
            'font-size: 10px',
            'font-weight: bold',
            'line-height: 1',
            'padding: 2px 4px',
            'border-radius: 0 0 3px 0',
            'pointer-events: none',
            'white-space: nowrap',
            'user-select: none',
        ].join('; ');

        box.appendChild(label);
        overlay.appendChild(box);
    }

    // ------------------------------------------------------------------
    // Step 4: Orchestrate — collect, annotate, expose metadata.
    // ------------------------------------------------------------------
    const targets = collectTargets();
    const overlay = createOverlay();

    targets.forEach(({ el, rect }, index) => {
        drawAnnotation(overlay, rect, index + 1);   // 1-indexed box numbers
    });

    // Expose a lookup table on window so Step 4 (extract.py) can map a
    // Gemini-returned box number back to the DOM element it came from.
    // We store tag, class, and a text preview — enough to re-locate the
    // element and extract its full innerText.
    window.__som_targets__ = targets.map(({ el }, i) => ({
        id:      i + 1,
        tag:     el.tagName.toLowerCase(),
        cls:     el.className || '',
        txt:     el.textContent.trim().slice(0, 120),   // truncated preview
    }));

    // Return total count so Python can log it
    return targets.length;

})();
"""


def derive_output_path(html_path: str) -> str:
    """
    Construct a default output PNG filename from the HTML input path.

    The logic strips a leading 'dashboard_' prefix from the stem so that
    the filenames stay concise:
        dashboard_v1.html  →  annotated_v1.png
        dashboard_v2.html  →  annotated_v2.png
        report.html        →  annotated_report.png
    """
    stem = pathlib.Path(html_path).stem
    clean = stem[len("dashboard_"):] if stem.startswith("dashboard_") else stem
    return f"annotated_{clean}.png"


def annotate(html_path: str, output_path: str) -> None:
    """
    End-to-end orchestration:
      1. Validate input path.
      2. Launch headless Chromium via the system binary.
      3. Load the HTML file.
      4. Inject the SoM overlay JavaScript.
      5. Expand the viewport to the full document dimensions.
      6. Capture a full-page screenshot.
      7. Print a diagnostic summary of the annotated elements.
    """

    # Resolve to absolute path and check existence before launching the browser
    html_abs = pathlib.Path(html_path).resolve()
    if not html_abs.exists():
        print(f"[ERROR] HTML file not found: {html_abs}")
        sys.exit(1)

    # Playwright's page.goto() requires a proper URI, not a raw filesystem path
    file_url = html_abs.as_uri()

    print("=" * 60)
    print(f"[SoM Annotator] Input    : {html_abs}")
    print(f"[SoM Annotator] Output   : {output_path}")
    print(f"[SoM Annotator] File URL : {file_url}")
    print("=" * 60)

    with sync_playwright() as pw:

        # -----------------------------------------------------------------
        # Browser launch.
        # '/usr/bin/chromium-browser' is a Snap wrapper that fails in
        # this environment because snap-confine lacks cap_dac_override.
        # The working binary is Google Chrome at /opt/google/chrome/chrome,
        # which shares the same DevTools Protocol and works as a drop-in
        # replacement for Playwright's Chromium driver.
        # -----------------------------------------------------------------
        browser = pw.chromium.launch(
            executable_path='/opt/google/chrome/chrome',
            headless=True,
        )

        # A fresh context with a 1280-wide viewport. Height is a placeholder
        # — we resize to the true document height after the page loads.
        context = browser.new_context(
            viewport={'width': 1280, 'height': 900},
        )
        page = context.new_page()

        # -----------------------------------------------------------------
        # Load the page.
        # wait_until='networkidle' holds execution until there are no more
        # than 0 in-flight network connections for at least 500 ms. For a
        # local HTML file this fires almost instantly, but it's the correct
        # idiom for pages that load CSS or fonts over the network.
        # -----------------------------------------------------------------
        print("[SoM Annotator] Loading page ...")
        page.goto(file_url, wait_until='networkidle')

        # Give any CSS transitions (e.g. gradient backgrounds, box-shadows)
        # time to complete their first paint before we annotate. 400 ms is
        # more than sufficient for a static dashboard.
        page.wait_for_timeout(400)

        # -----------------------------------------------------------------
        # Inject the SoM annotation overlay.
        # page.evaluate() executes the IIFE in the page's JS context and
        # returns its return value — the integer box count — to Python.
        # -----------------------------------------------------------------
        print("[SoM Annotator] Injecting Set-of-Mark overlay ...")
        box_count = page.evaluate(SOM_INJECTION_JS)
        print(f"[SoM Annotator] Total boxes annotated: {box_count}")

        # -----------------------------------------------------------------
        # Resize the viewport to the full rendered document dimensions.
        # This prevents Playwright from cropping the screenshot at the
        # initial 900 px viewport height on tall pages.
        # full_page=True (below) also handles this, but explicitly setting
        # the viewport avoids edge-case clipping in some Chromium builds.
        # -----------------------------------------------------------------
        doc_height = page.evaluate("document.body.scrollHeight")
        doc_width  = page.evaluate("document.body.scrollWidth")
        page.set_viewport_size({
            'width':  max(doc_width,  1280),
            'height': max(doc_height, 900),
        })

        # -----------------------------------------------------------------
        # Take the annotated screenshot.
        # full_page=True captures the entire scrollable document, not just
        # the visible viewport — essential for dashboards that are taller
        # than 900 px.
        # -----------------------------------------------------------------
        print("[SoM Annotator] Capturing full-page screenshot ...")
        page.screenshot(path=output_path, full_page=True)

        # -----------------------------------------------------------------
        # Print a diagnostic sample of the first 10 annotated elements.
        # This gives us a quick sanity-check without opening the image:
        # we can confirm that table cells, KPI values, and headers were
        # picked up rather than invisible structural divs.
        # -----------------------------------------------------------------
        sample_js = """
            (window.__som_targets__ || [])
                .slice(0, 10)
                .map(t =>
                    `  Box ${String(t.id).padStart(3)}: <${t.tag}>`
                    + ` cls="${t.cls.slice(0,30)}"`
                    + ` | "${t.txt.slice(0,50)}"`
                )
                .join('\\n')
        """
        sample = page.evaluate(sample_js)
        print(f"[SoM Annotator] First 10 annotated elements:\n{sample}")

        context.close()
        browser.close()

    print("=" * 60)
    print(f"[SoM Annotator] Screenshot saved: {output_path}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':

    if len(sys.argv) < 2:
        print(__doc__)
        print("\nUsage: python som_annotator.py <html-file> [output.png]")
        sys.exit(1)

    html_input  = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) >= 3 else derive_output_path(html_input)

    annotate(html_input, output_file)
