import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types


TARGET_COUNT = 400
BATCH_SIZE = 40
MAX_BATCHES = 20          # safety cap so a bad run cannot loop forever
MAX_RETRIES = 3
TIMEOUT_MS = 120_000      # 2 minutes per request
MODEL = "gemini-4-hydrogen"

BASE_DIR = Path(__file__).parent
OUTPUT_FILE = BASE_DIR / "ideas.csv"
PARTIAL_FILE = BASE_DIR / "ideas_partial.json"
RAW_LOG_FILE = BASE_DIR / "gemini_responses.jsonl"
KEY_FILES = [BASE_DIR / ".env", BASE_DIR / ".env.example"]

FOCUS_AREAS = [
    "campus navigation and finding buildings or rooms",
    "transportation, parking, and getting around Raleigh",
    "dining, meal plans, and food access",
    "academic life, studying, and course logistics",
    "social connections and making friends",
    "time management and daily routines",
    "student resources, advising, and campus services",
    "everyday inconveniences in dorms, laundry, and weather",
    "budgeting, money, and textbooks",
    "health, sleep, and wellbeing habits",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ideas with Gemini in batches.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace ideas.csv and discard any saved partial progress.")
    return parser.parse_args()


def load_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    for path in KEY_FILES:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("GEMINI_API_KEY="):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                if value:
                    print(f"      Using key from {path.name}.", flush=True)
                    return value
    raise RuntimeError("GEMINI_API_KEY not found in environment, .env, or .env.example.")


ABBREVIATIONS = re.compile(
    r"\b(?:e\.g|i\.e|etc|vs|U\.S|a\.m|p\.m|approx|Dr|Mr|Ms|St)\.", re.IGNORECASE
)


def sentence_count(text: str) -> int:
    cleaned = ABBREVIATIONS.sub(lambda m: m.group(0).replace(".", ""), text.strip())
    return len(re.findall(r"[.!?]+[\"')\]]?(?=\s|$)", cleaned))


def title_words(title: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", title.casefold()))


def is_duplicate(title: str, seen_titles: list[str]) -> bool:
    words = title_words(title)
    if not words:
        return True
    for other in seen_titles:
        other_words = title_words(other)
        if not other_words:
            continue
        overlap = len(words & other_words) / len(words | other_words)
        if overlap >= 0.8:
            return True
    return False


def build_prompt(existing_titles: list[str], focus: str) -> str:
    avoid = "\n".join(f"- {t}" for t in existing_titles) or "(none yet)"
    return f"""
Generate exactly {BATCH_SIZE} distinct, practical ideas for simple improvements to
challenges faced by a first-semester NC State freshman in daily life.
For this batch, concentrate on: {focus}.
Ideas may be non-technical. Avoid generic suggestions, repetition, and variations of
the same idea.

Do not repeat or closely vary any of these existing ideas:
{avoid}

Return only a JSON object with an "ideas" array containing exactly {BATCH_SIZE} objects.
Each object must have:
- "title": a short, clear, one-sentence title
- "description": exactly three sentences: the first explains the idea, the second
  explains the problem it addresses, and the third explains how it could help.
  Avoid abbreviations with periods such as "e.g." or "i.e." inside descriptions.
"""


def request_batch(client: genai.Client, prompt: str, batch_no: int) -> list:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    http_options=types.HttpOptions(timeout=TIMEOUT_MS),
                ),
            )
            if not response.text:
                raise ValueError("empty response")

            with RAW_LOG_FILE.open("a", encoding="utf-8") as log:
                log.write(json.dumps({"batch": batch_no, "text": response.text}) + "\n")

            data = json.loads(response.text)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and isinstance(data.get("ideas"), list):
                return data["ideas"]
            raise ValueError('response has no "ideas" array')
        except Exception as error:
            print(f"      Attempt {attempt}/{MAX_RETRIES} failed: {error}", flush=True)
            if attempt < MAX_RETRIES:
                time.sleep(3 * attempt)
    return []


def accept_ideas(raw_ideas: list, ideas: list[dict[str, str]]) -> tuple[int, int]:
    """Validate raw ideas and append good ones to `ideas`. Returns (accepted, rejected)."""
    accepted = rejected = 0
    seen_titles = [i["title"] for i in ideas]
    for item in raw_ideas:
        if len(ideas) >= TARGET_COUNT:
            break
        if not isinstance(item, dict):
            rejected += 1
            continue
        title = item.get("title")
        description = item.get("description")
        if not isinstance(title, str) or not title.strip():
            rejected += 1
            continue
        if not isinstance(description, str) or sentence_count(description) != 3:
            rejected += 1
            continue
        title = " ".join(title.split())
        if is_duplicate(title, seen_titles):
            rejected += 1
            continue
        ideas.append({"title": title, "description": " ".join(description.split())})
        seen_titles.append(title)
        accepted += 1
    return accepted, rejected


def load_partial() -> list[dict[str, str]]:
    if not PARTIAL_FILE.exists():
        return []
    try:
        data = json.loads(PARTIAL_FILE.read_text(encoding="utf-8"))
        return [i for i in data if isinstance(i, dict) and "title" in i and "description" in i]
    except (json.JSONDecodeError, OSError):
        return []


def save_partial(ideas: list[dict[str, str]]) -> None:
    PARTIAL_FILE.write_text(json.dumps(ideas, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv(ideas: list[dict[str, str]]) -> None:
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "title", "description"])
        writer.writeheader()
        for idea_id, idea in enumerate(ideas, start=1):
            writer.writerow({"id": idea_id, **idea})


def generate_all() -> list[dict[str, str]]:
    print("[1/4] Loading API key...", flush=True)
    api_key = load_api_key()
    print(f"      API key found ({len(api_key)} characters).", flush=True)
    client = genai.Client(api_key=api_key)

    ideas = load_partial()
    if ideas:
        print(f"      Resuming with {len(ideas)} ideas from {PARTIAL_FILE.name}.", flush=True)

    print(f"[2/4] Generating {TARGET_COUNT} ideas in batches of {BATCH_SIZE}...", flush=True)
    batch_no = 0
    while len(ideas) < TARGET_COUNT and batch_no < MAX_BATCHES:
        batch_no += 1
        focus = FOCUS_AREAS[(batch_no - 1) % len(FOCUS_AREAS)]
        print(f"  Batch {batch_no} (focus: {focus})", flush=True)

        prompt = build_prompt([i["title"] for i in ideas], focus)
        started = time.time()
        raw = request_batch(client, prompt, batch_no)
        elapsed = time.time() - started

        accepted, rejected = accept_ideas(raw, ideas)
        save_partial(ideas)
        print(f"      +{accepted} accepted, {rejected} rejected, "
              f"{len(ideas)}/{TARGET_COUNT} total ({elapsed:.1f}s)", flush=True)

    if len(ideas) < TARGET_COUNT:
        print(f"      Stopped at {len(ideas)} ideas after {batch_no} batches.", flush=True)
    return ideas


def main() -> int:
    print("Starting idea generation.", flush=True)
    args = parse_args()

    if args.overwrite:
        PARTIAL_FILE.unlink(missing_ok=True)
        RAW_LOG_FILE.unlink(missing_ok=True)
    elif OUTPUT_FILE.exists():
        print(f"{OUTPUT_FILE.name} already exists. Use --overwrite to regenerate it.",
              file=sys.stderr)
        return 1

    try:
        ideas = generate_all()
        if not ideas:
            raise RuntimeError("No valid ideas were generated.")
        print(f"[3/4] Writing {len(ideas)} ideas to {OUTPUT_FILE.name}...", flush=True)
        save_csv(ideas)
    except KeyboardInterrupt:
        print("\nInterrupted. Progress is saved in ideas_partial.json; rerun to resume.",
              file=sys.stderr)
        return 130
    except (RuntimeError, ValueError, OSError) as error:
        print(f"Could not generate ideas: {error}", file=sys.stderr)
        return 1

    if len(ideas) >= TARGET_COUNT:
        PARTIAL_FILE.unlink(missing_ok=True)
    print(f"[4/4] Done. Saved {len(ideas)} ideas to {OUTPUT_FILE}.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())