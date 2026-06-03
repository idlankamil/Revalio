"""
data_manager.py — Revalio Stage 1: Data Collection
====================================================
What this file does:
    - Searches YouTube for review videos per product (multi-layer progressive search)
    - Applies hard filters (views, duration, title keywords, non-English) — runs BEFORE LLM filter
    - Runs LLM batch title filter (GPT-4o-mini) — checks relevance and version correctness
    - Classifies each video (full_review / comparison / multi_comparison / early_impression)
    - Fetches raw transcripts and applies quality check (min 3000 chars)
    - Saves clean video data to PostgreSQL


"""

# ============================================================
# IMPORTS
# ============================================================
import os
import re
import json
import time
import sys
import requests
import http.cookiejar

from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from googleapiclient.discovery import build          # YouTube Data API v3 client
from youtube_transcript_api import YouTubeTranscriptApi  # fetches video transcripts
from langdetect import detect, DetectorFactory       # detects language of text

DetectorFactory.seed = 0  # ensures langdetect is deterministic across all runs

sys.path.append(r"D:\REVALIO\Backend")
from models.database import Session, Product, Video  # SQLAlchemy DB models


# ============================================================
# ENVIRONMENT
# ============================================================
load_dotenv(r"D:\REVALIO\Backend\.env")  # loads API keys from .env file

# Collects all YOUTUBE_API_KEY_* keys from .env — supports multiple keys for quota rotation
YOUTUBE_API_KEYS = [
    v for k, v in sorted(os.environ.items())
    if k.startswith("YOUTUBE_API_KEY_") and v
]

if not YOUTUBE_API_KEYS:
    raise ValueError("No YOUTUBE_API_KEY_* found in .env")

_yt_key_index = 0  # tracks which key is currently active

def get_youtube_key():
    return YOUTUBE_API_KEYS[_yt_key_index]

def rotate_youtube_key():
    # Moves to the next API key when current one hits daily quota limit
    global _yt_key_index
    _yt_key_index += 1
    if _yt_key_index >= len(YOUTUBE_API_KEYS):
        raise RuntimeError("[QUOTA] All YouTube API keys have exceeded their quota. Try again tomorrow.")
    print(f"[QUOTA] Rotating to YouTube API key {_yt_key_index + 1}/{len(YOUTUBE_API_KEYS)}")

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")


# ============================================================
# OPENAI CLIENT
# ============================================================
client = OpenAI(api_key=OPENAI_API_KEY)  # single shared GPT client used for LLM filtering


# ============================================================
# SETTINGS
# ============================================================
VIDEOS_PER_PRODUCT     = 25       # how many videos to save per product
SKIP_IF_VIDEOS_GTE     = 25       # skip product if already has this many videos in DB
SEARCH_STOP_THRESHOLD  = 30       # default candidate target (adjusted adaptively after Layer 1)
LLM_CANDIDATE_CAP      = 100      # max candidates sent to LLM (sorted by view count)
DELAY_BETWEEN_VIDEOS   = 45       # seconds to wait between transcript fetches (avoids IP ban)
DELAY_BETWEEN_PRODUCTS = 60       # seconds to wait between products (rate limiting)
MIN_TRANSCRIPT_LENGTH  = 3000     # transcripts shorter than this are too thin for analysis
COOKIE_FILE            = r"D:\REVALIO\Backend\youtube.com_cookies.txt"  # browser cookies to bypass transcript restrictions


# ============================================================
# PRODUCT LIST
# ============================================================
# Each tuple: (product name, category, brand)
# Only products listed here will be processed when run_all() is called
PRODUCTS = [
    # Laptops
    ("Acer Swift X 14 2024", "laptops", "Acer"),
    # Phones
    ("iPhone 15 Pro Max", "phones", "Apple"),
    ("Google Pixel 10 Pro","phones", "Google"),
]


# ============================================================
# VIEW THRESHOLDS
# ============================================================
# Minimum view count per category — filters out low-traffic / low-credibility videos
# Laptops/phones have higher thresholds because they attract more mainstream viewers
# Monitors/keyboards are niche, so a lower bar is acceptable
VIEW_COUNT_THRESHOLD = {
    "laptops": 50000,
    "phones": 50000,
    "monitors": 20000,
    "keyboards": 20000,
}


# ============================================================
# DATABASE HELPERS
# ============================================================
def get_existing_video_count(product_name):
    # Returns how many videos are already saved in DB for this product
    # Used to decide whether to skip or how many more videos to collect
    db = Session()
    try:
        return db.query(Video).join(Product).filter(
            Product.name == product_name
        ).count()
    except Exception as e:
        print(f"[DB ERROR] get_existing_video_count for '{product_name}': {e}")
        return 0
    finally:
        db.close()


def already_in_db(video_id):
    # Checks if this exact video ID is already saved — prevents duplicate entries
    db = Session()
    try:
        exists = db.query(Video).filter_by(video_id=video_id).first()
        return exists is not None
    except Exception as e:
        print(f"[DB ERROR] already_in_db for '{video_id}': {e}")
        return False
    finally:
        db.close()


# ============================================================
# NAME UTILITIES
# ============================================================
def get_short_name(product_name, brand):
    # Strips the brand name from the product name to create a shorter search query
    # e.g. "Apple MacBook Air M3" → "MacBook Air M3"
    pattern = re.compile(re.escape(brand), re.IGNORECASE)
    short = pattern.sub("", product_name).strip()
    return short if short else product_name


def build_search_layers(product_name, brand):
    short = get_short_name(product_name, brand)

    # Each tuple: (query, maxResults)
    # Layer 1: exact full name 
    # Layer 2: short name + review 
    # Layer 3: brand + short name 
    # Layer 4: year variants 
    # Layer 5: bare short name 
    layers = [
        (f'"{product_name}" review', 70),
        (f'"{short}" review', 60),
        (f'"{brand}" "{short}"', 50),
        (f'{short} review 2023 OR 2024 OR 2025 OR 2026', 40),
        (f'{short}', 20),
    ]

    return layers, short


# ============================================================
# ADAPTIVE THRESHOLD
# ============================================================
def get_adaptive_threshold(layer1_count):
    """
    Set search stop threshold based on Layer 1 hard-filtered results.

    >= 15 candidates → high-demand product → stop early at 20
    >= 10 candidates → medium-demand product → standard buffer at 25
    <  10 candidates → low-demand / niche product → search deeper at 30
    """
    # High-demand product — already enough candidates from Layer 1 alone
    if layer1_count >= 15:
        return 70
    # Medium-demand — need a bit more searching
    elif layer1_count >= 10:
        return 100
    # Niche product — go deeper across all layers to find enough videos
    else:
        return 120


# ============================================================
# CLASSIFICATION
# ============================================================
def classify_video(title):
    # Assigns a video type and a content_weight (0.0–1.0) based on title keywords
    # Weight reflects how much we trust this video's opinion for scoring
    title = title.lower()

    # Early impressions are short-term takes — lower weight (0.5)
    if any(k in title for k in ["first impression", "hands on", "hands-on", "impressions"]) \
       or re.search(r"after \d+ (day|week|month)", title):
        return "early_impression", 0.5

    # Multi-comparison videos cover many products — least reliable per product (0.3)
    gen_keywords = ["m1", "m2", "m3", "m4", "2022", "2023", "2024", "2025", "2026"]
    if sum(1 for k in gen_keywords if k in title) >= 3:
        return "multi_comparison", 0.3

    # Head-to-head comparisons are focused but split between two products (0.75)
    if any(k in title for k in ["vs", "versus", "compared", "comparison", " better than ", "winner"]):
        return "comparison", 0.75

    # Full dedicated review — highest trust (1.0)
    return "full_review", 1.0


# ============================================================
# HARD FILTERS
# ============================================================
# Regex pattern covering all major non-Latin scripts
# Used to reject non-English video titles before language detection
NON_ENGLISH_PATTERN = re.compile(
    r'[\u0400-\u04FF'    # Cyrillic (Russian, Ukrainian, etc.)
    r'\u0500-\u052F'     # Cyrillic Supplement
    r'\u0600-\u06FF'     # Arabic
    r'\u0750-\u077F'     # Arabic Supplement
    r'\u08A0-\u08FF'     # Arabic Extended-A
    r'\u0900-\u097F'     # Devanagari (Hindi, Marathi, etc.)
    r'\u0980-\u09FF'     # Bengali
    r'\u0A00-\u0A7F'     # Gurmukhi (Punjabi)
    r'\u0A80-\u0AFF'     # Gujarati
    r'\u0B00-\u0B7F'     # Odia
    r'\u0B80-\u0BFF'     # Tamil
    r'\u0C00-\u0C7F'     # Telugu
    r'\u0C80-\u0CFF'     # Kannada
    r'\u0D00-\u0D7F'     # Malayalam
    r'\u0D80-\u0DFF'     # Sinhala
    r'\u0E00-\u0E7F'     # Thai
    r'\u0E80-\u0EFF'     # Lao
    r'\u0F00-\u0FFF'     # Tibetan
    r'\u1000-\u109F'     # Myanmar (Burmese)
    r'\u10A0-\u10FF'     # Georgian
    r'\u1100-\u11FF'     # Hangul Jamo (Korean)
    r'\u1700-\u171F'     # Tagalog (Filipino)
    r'\u1800-\u18AF'     # Mongolian
    r'\u1C00-\u1C4F'     # Lepcha / Ol Chiki
    r'\u1E00-\u1EFF'     # Latin Extended Additional (Vietnamese diacritics)
    r'\u3040-\u309F'     # Hiragana (Japanese)
    r'\u30A0-\u30FF'     # Katakana (Japanese)
    r'\u3400-\u4DBF'     # CJK Extension A
    r'\u4E00-\u9FFF'     # CJK Unified (Chinese / Japanese Kanji)
    r'\uA960-\uA97F'     # Hangul Jamo Extended-A
    r'\uAC00-\uD7AF'     # Hangul Syllables (Korean)
    r'\uD7B0-\uD7FF'     # Hangul Jamo Extended-B
    r'\uF900-\uFAFF'     # CJK Compatibility Ideographs
    r'\uFB50-\uFDFF'     # Arabic Presentation Forms-A
    r'\uFE70-\uFEFF]'    # Arabic Presentation Forms-B
)

MIN_LANGDETECT_LENGTH = 15  # titles shorter than this are unreliable for langdetect


def has_non_english_title(title):
    """
    Two-layer non-English detection:

    Layer 1 — Unicode regex:
        Catches script-based non-English titles (Arabic, Chinese, Korean, Thai, etc.)
        Fast, free, deterministic.

    Layer 2 — langdetect:
        Catches Latin-script non-English titles (Spanish, French, Indonesian, etc.)
        Only runs if title is long enough to be reliable (>= 15 chars).
        DetectorFactory.seed = 0 ensures consistent results across runs.
    """
    # Layer 1: script-based detection
    if len(NON_ENGLISH_PATTERN.findall(title)) >= 2:
        return True

    # Layer 2: language detection for Latin-script non-English titles
    if len(title) >= MIN_LANGDETECT_LENGTH:
        try:
            return detect(title) != "en"
        except Exception:
            # If langdetect fails (e.g. too ambiguous), don't reject — let it through
            return False

    return False


def passes_hard_filters(video, category):
    # Runs cheap rule-based checks BEFORE calling LLM — saves API cost and time
    title = video["title"].lower()

    # Reject non-English titles — pipeline only supports English transcripts
    if has_non_english_title(video["title"]):
        return False

    # Reject content types that don't contain product opinions/sentiment
    junk = ["shorts", "unboxing", "teardown", "repair", "durability", "tips", "how to"]
    if any(k in title for k in junk):
        return False

    # Reject low-view videos — proxy for low-credibility or low-effort content
    if video["view_count"] < VIEW_COUNT_THRESHOLD.get(category, 20000):
        return False

    # Reject videos under 5 minutes — too short to be a real review
    if video["duration_seconds"] < 300:
        return False

    return True


# ============================================================
# LLM FILTER
# ============================================================
# System prompt instructs GPT to act as a strict relevance judge
# Returns only a JSON array of accepted indices — no extra text
LLM_SYSTEM_PROMPT = """You are a strict video relevance filter for a product review system.

Your job is to decide if a YouTube video title is relevant to a specific target product.

Rules (apply ALL of them, no exceptions):
1. The title MUST refer to the exact product or a clearly equivalent name.
   - Accept minor formatting differences: "MacBook Air M3" = "MacBook Air M3 13-inch"
   - Reject different versions or generations: "Victus 15" is NOT "Victus 16"
   - Reject vague titles that mention the brand but not the specific model
2. The title MUST be about reviewing, testing, or evaluating the product.
   - Accept: review, hands-on, test, comparison, impressions, analysis
   - Reject: how to fix, repair, teardown, unboxing only, setup guide
3. If you are not confident the title matches the target product, REJECT it.

Return ONLY a JSON array of accepted indices. No explanation. No markdown. Example:
[1, 3, 5]
If nothing passes, return: []"""


def llm_filter_titles(videos, product_name):
    if not videos:
        return []

    # Cap at LLM_CANDIDATE_CAP, sorted by view count descending
    # Ensures LLM sees the highest-quality candidates and avoids degraded
    # accuracy on very long prompts
    videos = sorted(videos, key=lambda x: x["view_count"], reverse=True)[:LLM_CANDIDATE_CAP]

    # Build numbered title list for the LLM to evaluate
    title_list = "\n".join(f"{i+1}. {v['title']}" for i, v in enumerate(videos))

    user_prompt = f"""Target product: {product_name}

Titles to evaluate:
{title_list}

Return only the JSON array of accepted indices."""

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0  # temperature=0 → deterministic output, no randomness in filtering decisions
        )

        raw = res.choices[0].message.content.strip()
        indices = json.loads(raw)  # parse the JSON array returned by GPT

        if not isinstance(indices, list):
            print(f"[LLM FILTER] Unexpected response format for '{product_name}' — keeping all candidates")
            return videos

        # Map 1-based indices back to actual video objects
        return [videos[i-1] for i in indices if isinstance(i, int) and 1 <= i <= len(videos)]

    except json.JSONDecodeError as e:
        # GPT returned something that isn't valid JSON — safe fallback: keep all candidates
        print(f"[LLM FILTER] JSON parse failed for '{product_name}': {e} — keeping all candidates")
        return videos
    except Exception as e:
        # API error — don't discard work already done, fall back to unfiltered list
        print(f"[LLM FILTER] API error for '{product_name}': {e} — keeping all candidates")
        return videos


# ============================================================
# YOUTUBE
# ============================================================
def parse_duration(d):
    # Converts YouTube's ISO 8601 duration format (e.g. "PT12M30S") to total seconds
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", d)
    if not m:
        return 0
    h, m_, s = int(m.group(1) or 0), int(m.group(2) or 0), int(m.group(3) or 0)
    return h*3600 + m_*60 + s


def search_youtube_query(query, max_results=25):
    global _yt_key_index

    while _yt_key_index < len(YOUTUBE_API_KEYS):
        try:
            yt = build("youtube", "v3", developerKey=get_youtube_key())

            # Step 1: Search for video IDs matching the query
            res = yt.search().list(
                q=query,
                part="id,snippet",
                type="video",
                maxResults=max_results
            ).execute()

            ids = [i["id"]["videoId"] for i in res["items"]]

            if not ids:
                return []

            # Step 2: Fetch full details (stats + duration) for those video IDs
            # Search API doesn't return view counts or duration — need a second call
            details = yt.videos().list(
                part="snippet,statistics,contentDetails",
                id=",".join(ids)
            ).execute()

            return [
                {
                    "video_id": i["id"],
                    "title": i["snippet"]["title"],
                    "channel": i["snippet"]["channelTitle"],
                    "view_count": int(i["statistics"].get("viewCount", 0)),
                    "published_date": i["snippet"]["publishedAt"],
                    "duration_seconds": parse_duration(i["contentDetails"]["duration"]),
                }
                for i in details["items"]
            ]

        except Exception as e:
            error_str = str(e)
            # Quota exceeded error code is 403 with reason "quotaExceeded" or "dailyLimitExceeded"
            if "quotaExceeded" in error_str or "dailyLimitExceeded" in error_str or "403" in error_str:
                print(f"[QUOTA] Key {_yt_key_index + 1} exhausted — {e}")
                rotate_youtube_key()
                continue  # retry with next key
            else:
                print(f"[YOUTUBE ERROR] Query '{query}': {e}")
                return []

    return []  # all keys exhausted


# ============================================================
# TRANSCRIPT
# ============================================================
def fetch_transcript(video_id):
    # Uses browser cookies to authenticate the transcript request
    # Needed because YouTube restricts transcript access without a logged-in session
    try:
        session = requests.Session()
        cookies = http.cookiejar.MozillaCookieJar(COOKIE_FILE)
        cookies.load(ignore_discard=True, ignore_expires=True)
        session.cookies = cookies

        api = YouTubeTranscriptApi(http_client=session)
        data = api.fetch(video_id)

        # Joins all transcript segments into one continuous string
        return " ".join([x.text for x in data])

    except Exception as e:
        print(f"[TRANSCRIPT ERROR] {video_id}: {e}")
        return None


# ============================================================
# DB SAVE
# ============================================================
def save_to_db(product_name, category, brand, data):
    db = Session()
    try:
        # Check if product already exists — create it if not
        product = db.query(Product).filter_by(name=product_name).first()

        if not product:
            product = Product(name=product_name, category=category, brand=brand)
            db.add(product)
            db.flush()  # flush to get product.id before inserting Video

        video = Video(
            product_id=product.id,
            video_id=data["video_id"],
            title=data["title"],
            channel=data["channel"],
            view_count=data["view_count"],
            published_date=data["published_date"],
            video_classification=data["video_classification"],
            content_weight=data["content_weight"],
            raw_transcript=data["raw_transcript"],
        )

        db.add(video)
        db.commit()

    except Exception as e:
        print(f"[DB ERROR] save_to_db for '{product_name}': {e}")
        db.rollback()  # undo any partial writes if something goes wrong
    finally:
        db.close()


# ============================================================
# MAIN PIPELINE
# ============================================================
def collect_product(product_name, category, brand):
    # Orchestrates the full 3-phase collection process for one product

    existing = get_existing_video_count(product_name)

    # Skip entirely if DB already has enough videos for this product
    if existing >= SKIP_IF_VIDEOS_GTE:
        print(f"[SKIP] '{product_name}' already has {existing} videos in DB")
        return

    needed = VIDEOS_PER_PRODUCT - existing
    print(f"\n[START] '{product_name}' — need {needed} more videos")

    layers, short = build_search_layers(product_name, brand)

    seen_ids   = set()   # tracks video IDs already added — prevents duplicates across layers
    candidates = []
    threshold  = SEARCH_STOP_THRESHOLD  # default, overridden after Layer 1

    # ── PHASE 1: Progressive Search ──────────────────────────
    # Runs up to 5 search queries, from specific to broad
    # Stops early once enough candidates are collected
    for i, (query, max_results) in enumerate(layers):
        print(f"[SEARCH] Layer {i+1}/5 — query: {query} (max {max_results})")

        results  = search_youtube_query(query, max_results)
        filtered = [
            v for v in results
            if v["video_id"] not in seen_ids
            and passes_hard_filters(v, category)
        ]

        for v in filtered:
            seen_ids.add(v["video_id"])
            candidates.append(v)

        print(f"[SEARCH] Layer {i+1} — {len(filtered)} new candidates added (total: {len(candidates)})")

        # After Layer 1, set adaptive threshold based on actual results
        if i == 0:
            threshold = get_adaptive_threshold(len(candidates))
            print(f"[SEARCH] Adaptive threshold set to {threshold} (Layer 1 yielded {len(candidates)} candidates)")

        # Early stop — log when final fallback layer is reached
        if i == 4:
            print(f"[SEARCH] Final fallback layer reached for '{product_name}'")

        if len(candidates) >= threshold:
            print(f"[SEARCH] Threshold {threshold} reached — stopping search at Layer {i+1}")
            break

    if not candidates:
        print(f"[SKIP] '{product_name}' — no candidates found after all layers")
        return

    # ── PHASE 2: LLM Filter (runs ONCE on full pool) ─────────
    # GPT-4o-mini checks each title for relevance and correct product version
    # Runs after hard filters to reduce cost — only quality candidates reach this step
    print(f"[LLM] Filtering {len(candidates)} candidates for '{product_name}'")
    candidates = llm_filter_titles(candidates, product_name)
    print(f"[LLM] {len(candidates)} candidates passed LLM filter")

    if not candidates:
        print(f"[SKIP] '{product_name}' — no candidates survived LLM filter")
        return

    # ── PHASE 3: Transcript Fetch (best videos first) ─────────
    # Sort by view count descending — fetch highest-quality videos first
    # so if we hit the target early, we collected the best ones
    candidates.sort(key=lambda x: x["view_count"], reverse=True)

    collected = 0

    for v in candidates:
        if collected >= needed:
            break

        if already_in_db(v["video_id"]):
            print(f"[SKIP - ALREADY SAVED] '{v['title']}'")
            continue

        cls, weight = classify_video(v["title"])
        transcript  = fetch_transcript(v["video_id"])

        # Quality gate — skip transcripts that are too short to be useful for NLP
        if not transcript or len(transcript) < MIN_TRANSCRIPT_LENGTH:
            print(f"[SKIP] '{v['title']}' — transcript missing or too short")
            continue

        v.update({
            "video_classification": cls,
            "content_weight":       weight,
            "raw_transcript":       transcript,
        })

        save_to_db(product_name, category, brand, v)
        collected += 1
        print(f"[SAVED] ({collected}/{needed}) '{v['title']}'")
        time.sleep(DELAY_BETWEEN_VIDEOS)  # rate limiting — avoids triggering YouTube's bot detection

    print(f"[DONE] '{product_name}' — {collected} videos collected")


# ============================================================
# RUN ALL
# ============================================================
def run_all():
    # Iterates through all products in PRODUCTS list and runs collect_product() for each
    for i, (name, cat, brand) in enumerate(PRODUCTS):
        collect_product(name, cat, brand)

        # Wait between products — avoids hitting API rate limits back-to-back
        if i < len(PRODUCTS) - 1:
            time.sleep(DELAY_BETWEEN_PRODUCTS)


# ============================================================
# ENTRY
# ============================================================
if __name__ == "__main__":
    run_all()