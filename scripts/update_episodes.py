"""
Incremental updater for the MFM Founder Episode Explorer.

Checks the official RSS feed for episodes not yet in episodes.json,
cleans and classifies ONLY the new ones (not the whole catalog), and
appends them. Designed to run on a schedule via GitHub Actions -- see
.github/workflows/update-episodes.yml

Usage:
    pip install feedparser anthropic
    export ANTHROPIC_API_KEY=sk-ant-...   (or set as a repo secret in CI)
    python scripts/update_episodes.py
"""

import json
import os
import re
import time
import hashlib
from pathlib import Path

import feedparser
import anthropic

FEED_URL = "https://feeds.megaphone.fm/HS2300184645"
EPISODES_JSON = Path(__file__).parent.parent / "episodes.json"
MODEL = "claude-haiku-4-5-20251001"  # cheap + fast, fine for a few new episodes/week

EP_RE = re.compile(r"Episode\s+(\d+)", re.IGNORECASE)
HASH_RE = re.compile(r"^#(\d+)\b")
TIMESTAMP_LINE_RE = re.compile(r"^\(?\d{1,2}:\d{2}(:\d{2})?\)?\s*[-–—:]?\s*.+")
BARE_TIMESTAMP_RE = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?\s+\S")
BOILERPLATE_MARKERS = [
    "Want to be featured in a future episode",
    "Have you joined our private Facebook group",
]

TAXONOMY = {
    "stage": ["idea-validation","side-hustle-part-time","0-to-1-launch","scaling-ops","fundraising","exit-or-acquisition"],
    "function": ["sales-and-closing","marketing-and-growth","hiring-and-team","finance-and-unit-economics","product-and-positioning","operations"],
    "business_model": ["crypto-and-web3","real-estate","ecommerce-and-dtc","content-and-newsletter-business","investing-and-public-markets","blue-collar-and-local-services"],
    "career_personal": ["career-pivot","founder-mental-health","health-and-longevity","motivation-and-mindset","family-and-life-integration","money-mindset-personal-wealth"],
    "format": ["guest-interview","idea-brainstorm","news-reaction","deep-dive-single-topic","historical-biography-deep-dive","greatest-hits-replay"],
}
ALL_TAGS = set(t for group in TAXONOMY.values() for t in group)

TAXONOMY_DESCRIPTION = """
STAGE: idea-validation, side-hustle-part-time, 0-to-1-launch, scaling-ops, fundraising, exit-or-acquisition
FUNCTION: sales-and-closing, marketing-and-growth, hiring-and-team, finance-and-unit-economics, product-and-positioning, operations
BUSINESS_MODEL: crypto-and-web3, real-estate, ecommerce-and-dtc, content-and-newsletter-business, investing-and-public-markets, blue-collar-and-local-services
CAREER_PERSONAL: career-pivot, founder-mental-health, health-and-longevity, motivation-and-mindset, family-and-life-integration, money-mindset-personal-wealth
FORMAT: guest-interview, idea-brainstorm, news-reaction, deep-dive-single-topic, historical-biography-deep-dive, greatest-hits-replay
""".strip()

SYSTEM_PROMPT = f"""You are tagging podcast episodes of "My First Million" for a founder-facing filtering app.

Given an episode title and show notes, select 3-6 tags from this exact taxonomy (use these exact strings, nothing else):

{TAXONOMY_DESCRIPTION}

Rules:
- Pick tags from multiple groups where relevant.
- Only apply career_personal tags when the notes actually reflect that content.
- Respond with ONLY a JSON array of tag strings, nothing else."""


def clean_html(raw_html):
    if not raw_html:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", raw_html)
    text = re.sub(r"</p>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&#8217;|&rsquo;", "'", text)
    text = re.sub(r"&#8220;|&#8221;|&ldquo;|&rdquo;", '"', text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_episode_number(title, full_text):
    m = EP_RE.search(title) or EP_RE.search(full_text[:200])
    if m:
        return int(m.group(1))
    m = HASH_RE.match(title.strip())
    if m:
        return int(m.group(1))
    return None


def extract_clean_show_notes(full_text):
    idx = full_text.find("Show Notes:")
    if idx == -1:
        return None
    tail = full_text[idx + len("Show Notes:"):]
    lines = tail.split("\n")
    kept, started, blanks = [], False, 0
    for line in lines:
        s = line.strip()
        if not s:
            if started:
                blanks += 1
                if blanks > 1:
                    break
            continue
        if TIMESTAMP_LINE_RE.match(s):
            kept.append(s)
            started = True
            blanks = 0
        elif started:
            break
    return "\n".join(kept) if kept else None


def extract_bare_timestamp_notes(full_text):
    lines = [l.strip() for l in full_text.split("\n")]
    kept, started, blanks = [], False, 0
    for line in lines:
        if not line:
            if started:
                blanks += 1
                if blanks > 1:
                    break
            continue
        if BARE_TIMESTAMP_RE.match(line):
            kept.append(line)
            started = True
            blanks = 0
        elif started:
            break
    return "\n".join(kept) if kept else None


def strip_boilerplate(text):
    for marker in BOILERPLATE_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            end = text.find("\n", idx)
            text = text[:idx] + (text[end:] if end != -1 else "")
    return text.strip()


def make_id(title, date):
    basis = f"{title}|{date}"
    return hashlib.sha1(basis.encode()).hexdigest()[:12]


def clean_entry(title, pub_date, link, raw_desc):
    full_text = clean_html(raw_desc)
    ep_num = extract_episode_number(title, full_text)
    show_notes = extract_clean_show_notes(full_text) or extract_bare_timestamp_notes(full_text)

    fallback_desc = None
    if not show_notes:
        raw = strip_boilerplate(full_text)
        for stop in ["\n-----", "\nLinks:", "\nCheck Out"]:
            cut = raw.find(stop)
            if cut != -1:
                raw = raw[:cut]
        fallback_desc = strip_boilerplate(raw).strip()[:1200] or None

    return {
        "id": make_id(title, pub_date),
        "episode_number": ep_num,
        "title": title,
        "date": pub_date,
        "url": link,
        "show_notes": show_notes,
        "fallback_description": fallback_desc,
        "is_special": ep_num is None,
    }


def classify_episode(client, ep, max_retries=3):
    text = ep.get("show_notes") or ep.get("fallback_description") or ""
    message = f"Title: {ep['title']}\n\nShow notes:\n{text}"
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=500,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": message}],
            )
            raw = resp.content[0].text.strip()
            raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
            tags = json.loads(raw)
            return [t for t in tags if t in ALL_TAGS]
        except Exception as e:
            print(f"  attempt {attempt} failed: {e}")
            time.sleep(2 * attempt)
    return []


def main():
    if not EPISODES_JSON.exists():
        raise SystemExit(
            f"{EPISODES_JSON} not found. Seed it first with your initial tagged "
            f"dataset (the output of classify_episodes.py), placed at repo root "
            f"as episodes.json, before running incremental updates."
        )

    with open(EPISODES_JSON, "r", encoding="utf-8") as f:
        existing = json.load(f)
    existing_ids = {e["id"] for e in existing if e.get("id")}
    print(f"Loaded {len(existing)} existing episodes.")

    print(f"Fetching feed: {FEED_URL}")
    feed = feedparser.parse(FEED_URL)
    print(f"Feed has {len(feed.entries)} entries.")

    new_entries = [e for e in feed.entries if make_id(e.get("title","").strip(), e.get("published","")) not in existing_ids]
    print(f"Found {len(new_entries)} new episode(s) since last update.")

    if not new_entries:
        print("Nothing to do.")
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set.")
    client = anthropic.Anthropic(api_key=api_key)

    added = []
    for entry in new_entries:
        title = entry.get("title", "").strip()
        pub_date = entry.get("published", "")
        link = entry.get("link", "")
        raw_desc = entry.get("summary", "") or entry.get("description", "")

        ep = clean_entry(title, pub_date, link, raw_desc)
        print(f"Classifying new episode: {title[:60]}")
        ep["tags"] = classify_episode(client, ep)
        added.append(ep)
        time.sleep(0.3)

    all_episodes = existing + added
    all_episodes.sort(key=lambda e: (e.get("episode_number") or 0))

    with open(EPISODES_JSON, "w", encoding="utf-8") as f:
        json.dump(all_episodes, f, indent=2, ensure_ascii=False)

    print(f"Added {len(added)} new episode(s). Total now: {len(all_episodes)}.")


if __name__ == "__main__":
    main()
