#!/usr/bin/env python3
"""
gemini_agent.py — Multimodal Chain-of-Action (MCoA) Prompting Agent (Step 3)
=============================================================================
Sends a Set-of-Mark annotated dashboard screenshot to Google Gemini 1.5 Pro
and instructs the model to reason visually — step by step — about which
numbered box contains a specific data value. Returns a structured JSON
response containing the chain-of-thought reasoning and the identified Box ID.

This is the "Brain" layer of the MCoA architecture. It takes the PNG produced
by som_annotator.py (Step 2B) as input and produces a target_box_id as output,
which extract.py (Step 4) then uses to retrieve the actual DOM text via
Playwright.

Why visual reasoning over DOM parsing?
  A traditional scraper queries the DOM by CSS class or tag. When those change
  (as in dashboard_v2.html), the scraper breaks. Gemini looks at the rendered
  image — the same pixels a human would see — and reasons spatially: "the Q3
  column is third from the left; the Cloud Platform Services row is the first
  data row; their intersection is the cell I want." That reasoning remains valid
  regardless of what the underlying HTML looks like.

SDK note:
  This script uses the current `google-genai` package (google.genai), NOT the
  deprecated `google-generativeai` package (google.generativeai). The new SDK
  is client-based: you instantiate a Client object rather than calling module-
  level configure() functions.

Usage:
    python gemini_agent.py --image <annotated-png> --query "<data description>"

Examples:
    python gemini_agent.py --image annotated_v1.png \\
        --query "Q3 Revenue for Cloud Platform Services"

    python gemini_agent.py --image annotated_v2.png \\
        --query "Profit Margin for Data Licensing" \\
        --model gemini-2.5-pro

Prerequisites:
    export GOOGLE_API_KEY="your-key-here"
"""

import os
import sys
import json
import re
import time
import pathlib
import argparse

from google import genai
from google.genai import types


# ---------------------------------------------------------------------------
# MCoA system prompt template.
#
# Design principles:
#   1. Role framing — establishes the model as a spatial reasoning agent,
#      focusing its attention on visual layout rather than general knowledge.
#   2. SoM convention explanation — tells the model exactly how the annotation
#      system works so it does not misinterpret the numbered labels.
#   3. Explicit five-step reasoning scaffold — this is the "Chain of Action"
#      in MCoA. Forcing structured intermediate steps reduces the chance of
#      the model jumping to a wrong answer. The step sequence mirrors how a
#      human navigates a table: orient → find column → find row → intersect.
#   4. Strict JSON output requirement — the prompt explicitly forbids markdown
#      formatting, code fences, and any non-JSON surrounding text. We still
#      strip those in post-processing (see parse_json_response) defensively.
#
# The double braces {{ }} around the JSON schema are Python f-string escapes —
# they render as single braces { } in the final prompt string.
# ---------------------------------------------------------------------------
MCOA_PROMPT_TEMPLATE = """\
You are a visual data extraction agent. You receive a screenshot of a \
web page — it could be a dashboard, a Wikipedia article, a news page, \
a data table, or any other kind of webpage.

Every visible text-bearing element in this screenshot has been annotated \
with a red numbered bounding box — this is called Set-of-Mark (SoM) labelling. \
The small red badge in the top-left corner of each box displays that element's \
unique integer Box ID. Box IDs are sequential integers starting at 1, assigned \
left-to-right, top-to-bottom across the visible viewport.

IMPORTANT — this screenshot shows only ONE viewport of the page. \
If the data you are looking for is NOT visible anywhere in this screenshot, \
you must return -1 as the target_box_id. Do not guess or hallucinate a box \
that does not exist in the image.

Your task is to locate the specific data value described in the QUERY below \
and return the Box ID of the element that contains it, or -1 if the data \
is not visible in this screenshot.

QUERY: {query}

Work through the following five reasoning steps before giving your final answer.

Step 1 — Page orientation:
  What kind of page is this? Describe the major visible sections \
(e.g., article title, infobox, data table, navigation, footer).

Step 2 — Relevant structure identification:
  Is there a table, list, or section that is likely to contain the \
queried data? Describe what you see and which Box IDs are near it.

Step 3 — Row and column mapping (if a table exists):
  If there is a data table visible, identify the relevant row labels and \
column headers with their Box IDs. If there is no relevant table visible, \
note that the data may be below the fold.

Step 4 — Target cell localisation:
  Identify the specific box that contains the answer to the query. \
State its Box ID and the text it contains. If the data is NOT visible \
in this screenshot, explicitly state that and set target_box_id to -1.

Step 5 — Confidence assessment:
  Does the text in the identified box directly answer the query? \
State your confidence as "high", "medium", or "low".

You MUST respond with ONLY a valid JSON object. Do not wrap it in markdown \
code fences. Do not include any text before or after the JSON. Your entire \
response must be directly parseable by json.loads() with no pre-processing.

{{
  "thought_process": [
    "Step 1: <page orientation>",
    "Step 2: <relevant structure identification>",
    "Step 3: <row/column mapping or note that data is below fold>",
    "Step 4: <target box ID and text, or explanation of why -1>",
    "Step 5: <confidence assessment>"
  ],
  "target_box_id": <integer or -1 if not found in this screenshot>,
  "target_text_preview": "<expected text, or 'not visible' if -1>",
  "confidence": "<high|medium|low>"
}}
"""


def load_api_key() -> str:
    """
    Read the Google API key from the GOOGLE_API_KEY environment variable.
    Exits with a clear message if the variable is not set, rather than letting
    the SDK throw a cryptic authentication error downstream.
    """
    key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not key:
        print(
            "[ERROR] GOOGLE_API_KEY environment variable is not set.\n"
            "        Export it before running this script:\n"
            "          export GOOGLE_API_KEY='your-api-key-here'"
        )
        sys.exit(1)
    return key


def load_image_bytes(image_path: str) -> bytes:
    """
    Read the annotated PNG screenshot from disk and return its raw bytes.
    The google.genai SDK accepts images as inline binary data with an explicit
    MIME type via types.Part.from_bytes(), so no image library is needed.
    """
    p = pathlib.Path(image_path).resolve()
    if not p.exists():
        print(f"[ERROR] Annotated image not found: {p}")
        sys.exit(1)
    data = p.read_bytes()
    print(f"[Gemini Agent] Image loaded: {p.name} ({len(data) / 1024:.1f} KB)")
    return data


def build_prompt(query: str) -> str:
    """
    Substitute the extraction query into the MCoA prompt template.
    The template uses double-brace escaping for the JSON schema section, so
    .format() expands only the single {query} placeholder.
    """
    return MCOA_PROMPT_TEMPLATE.format(query=query)


def parse_json_response(raw_text: str) -> dict:
    """
    Extract and parse a JSON object from the model's response text.

    Gemini occasionally wraps its output in markdown code fences even when the
    prompt instructs it not to. Three strategies are tried in order:

      1. Direct parse — treat the entire stripped response as JSON.
      2. Fence strip — remove ```json / ``` wrappers, then parse.
      3. Brace extraction — find the first balanced { ... } block via character
         scanning and parse only that portion.

    If all three fail, an error envelope dict is returned so the caller always
    receives a dict and can handle the failure gracefully.
    """

    stripped = raw_text.strip()

    # Strategy 1: direct parse
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown code fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", stripped, re.IGNORECASE)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 3: extract the first balanced brace block
    start = stripped.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(stripped[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(stripped[start : i + 1])
                    except json.JSONDecodeError:
                        break

    return {
        "error": "JSON parsing failed — all three extraction strategies unsuccessful",
        "raw_response": raw_text,
        "target_box_id": None,
        "thought_process": [],
        "target_text_preview": None,
        "confidence": "low",
    }


def query_gemini(
    api_key: str,
    image_bytes: bytes,
    query: str,
    model_name: str = "gemini-2.5-flash",
    verbose: bool = True,
) -> tuple[dict, str]:
    """
    Send the annotated screenshot and the MCoA prompt to Gemini and return
    the parsed JSON response alongside the raw response text.

    New SDK usage (google.genai):
      - Instantiate a Client with the API key.
      - Call client.models.generate_content() with a contents list that mixes
        a types.Part (the inline image) and a plain string (the prompt).
      - types.Part.from_bytes() wraps raw bytes with a MIME type label so the
        model receives the image as a proper multimodal input.

    The prompt is sent AFTER the image in the contents list. This ordering
    mirrors the natural human workflow of looking at an image before reading
    instructions, and empirically produces slightly better spatial reasoning.

    Generation config:
      temperature=0.1  — very low, trading creativity for JSON reliability
      top_p=0.95       — slight nucleus sampling for non-determinism resilience
      max_output_tokens=1024 — enough for a five-step CoT plus the JSON envelope
    """

    client = genai.Client(api_key=api_key)
    prompt_text = build_prompt(query)

    # Wrap the raw PNG bytes as a typed multimodal Part
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")

    config = types.GenerateContentConfig(
        temperature=0.1,
        top_p=0.95,
        max_output_tokens=4096,
    )

    if verbose:
        print("=" * 60)
        print(f"[Gemini Agent] Model  : {model_name}")
        print(f"[Gemini Agent] Query  : {query}")
        print("[Gemini Agent] Sending request to Gemini API ...")

    # Retry loop for 429 RESOURCE_EXHAUSTED (free-tier rate limit: 5 req/min).
    # The error response includes a retryDelay field — we parse that and wait
    # slightly longer than the suggested delay before each retry attempt.
    MAX_RETRIES = 4
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[image_part, prompt_text],
                config=config,
            )
            break   # success — exit the retry loop
        except Exception as exc:
            err_str = str(exc)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:

                # Daily quota exhausted — retrying is pointless, the quota
                # will not reset until tomorrow. Fail immediately with a clear
                # message so the user is not stuck waiting 3 minutes for nothing.
                if "PerDay" in err_str or "per_day" in err_str.lower():
                    raise RuntimeError(
                        "Daily free-tier quota exhausted (20 requests/day for "
                        "Gemini 2.5 Flash). The quota resets tomorrow. "
                        "To remove this limit, enable billing at "
                        "https://aistudio.google.com and switch to pay-as-you-go."
                    ) from exc

                # Per-minute rate limit — worth retrying after the suggested delay.
                delay_match = re.search(r"retryDelay['\": ]+(\d+)s", err_str)
                wait_secs = int(delay_match.group(1)) + 5 if delay_match else 65
                if attempt < MAX_RETRIES:
                    if verbose:
                        print(
                            f"[Gemini Agent] Rate limit hit (attempt {attempt}/{MAX_RETRIES}). "
                            f"Waiting {wait_secs}s before retry ..."
                        )
                    else:
                        print(
                            f"\n    [rate limit] waiting {wait_secs}s "
                            f"(attempt {attempt}/{MAX_RETRIES}) ...",
                            end=" ", flush=True,
                        )
                    time.sleep(wait_secs)
                else:
                    raise   # exhausted all retries
            else:
                raise       # non-rate-limit error — fail immediately

    # Log token usage and embed counts in the result dict so callers
    # (e.g. evaluation.py) can compute cost without re-parsing the response.
    prompt_tok = output_tok = total_tok = 0
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        meta = response.usage_metadata
        prompt_tok = getattr(meta, "prompt_token_count",     0) or 0
        output_tok = getattr(meta, "candidates_token_count", 0) or 0
        total_tok  = getattr(meta, "total_token_count",      0) or 0
        if verbose:
            print(
                f"[Gemini Agent] Tokens : {prompt_tok} prompt + "
                f"{output_tok} output = {total_tok} total"
            )

    raw_text = response.text
    if verbose:
        print("[Gemini Agent] Response received. Parsing JSON ...")

    parsed = parse_json_response(raw_text)

    # Attach token metadata with underscore-prefix keys so downstream code
    # can distinguish them from model-generated fields.
    parsed["_prompt_tokens"] = prompt_tok
    parsed["_output_tokens"] = output_tok
    parsed["_total_tokens"]  = total_tok

    return parsed, raw_text


def print_result(result: dict, raw_text: str) -> None:
    """
    Display the parsed Gemini response in a structured, readable format.
    Prints each chain-of-thought step on its own line, then highlights the
    final answer: Box ID, expected text, and confidence level.
    """

    print("\n" + "=" * 60)
    print("GEMINI MCoA RESPONSE")
    print("=" * 60)

    if "error" in result:
        print("[PARSE ERROR] Could not extract valid JSON from the model response.")
        print(f"Raw text:\n{raw_text}")
        return

    print("\nChain-of-Thought Reasoning:")
    for step in result.get("thought_process", []):
        # Indent multi-sentence steps cleanly
        print(f"  {step}")

    print("\n" + "-" * 60)
    print(f"  Target Box ID    : {result.get('target_box_id')}")
    print(f"  Expected text    : {result.get('target_text_preview', '(not provided)')}")
    print(f"  Confidence       : {result.get('confidence', 'unknown')}")
    print("-" * 60)

    print("\nFull parsed JSON:")
    print(json.dumps(result, indent=2))
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MCoA visual data extraction agent using Google Gemini.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the Set-of-Mark annotated PNG produced by som_annotator.py.",
    )
    parser.add_argument(
        "--query",
        default="Q3 Revenue for Cloud Platform Services",
        help=(
            "Natural-language description of the data value to locate. "
            'Default: "Q3 Revenue for Cloud Platform Services"'
        ),
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model name to use. Default: gemini-2.5-flash",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    args = parse_args()

    api_key = load_api_key()
    image_bytes = load_image_bytes(args.image)

    result, raw_text = query_gemini(
        api_key=api_key,
        image_bytes=image_bytes,
        query=args.query,
        model_name=args.model,
    )

    print_result(result, raw_text)

    # Non-zero exit code on failure so shell scripts can detect errors
    if "error" in result or result.get("target_box_id") is None:
        sys.exit(1)
