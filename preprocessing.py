"""
preprocessing.py — Revalio Stage 2: GPT Fact Extraction
=========================================================
What this file does:
    1. Loads unprocessed videos from DB (have transcript, no structured facts yet)
    2. Splits raw transcript into overlapping sentence chunks
    3. Sends each chunk to GPT-4o-mini — extracts {aspect, sentiment, evidence, confidence}
    4. Filters out low confidence and invalid facts
    5. Filters out non-English evidence sentences
    6. Applies per-video fact cap (max 50 facts per video)
    7. Applies minimum evidence rule (2+ sentences per aspect to score it)
    8. Saves structured_facts and aspects_mentioned back to DB

Author: Idlan Kamil
Last Updated: April 2026
KB Version: 5

Changes from KB Version 4:
    [FIX 1] GPT prompt — added explicit negative extraction instruction
            GPT was only extracting obvious positives. Now actively instructed
            to look for complaints, limitations, and criticisms.
    [FIX 2] GPT prompt — added mixed sentence splitting rule
            Sentences like "battery is great for light use but terrible for gaming"
            were stored as one fact. GPT now splits these into two separate facts.
            Fixes low BERT agreement caused by ambiguous sentiment labels.
    [FIX 3] Per-video fact cap (MAX_FACTS_PER_VIDEO = 50)
            One video was contributing 26-57% of all facts for a product,
            skewing the final rating. Cap applied before saving to DB.
    [FIX 4] Non-English evidence filter using langdetect
            German and Indonesian evidence sentences were slipping through
            from non-English videos that passed Stage 1 filters. Now each
            evidence sentence is checked and discarded if not English.
"""

import os
import re
import json
from openai import OpenAI
from dotenv import load_dotenv
from langdetect import detect, DetectorFactory
import sys
sys.path.append(r"D:\REVALIO\Backend")
from models.database import Session, Video

DetectorFactory.seed = 0  # makes langdetect deterministic across all runs

# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================
load_dotenv(r"D:\REVALIO\Backend\.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================
# SETTINGS
# ============================================================
CHUNK_SIZE         = 25   # sentences per chunk
CHUNK_OVERLAP      = 7    # sentences of overlap between chunks
MIN_EVIDENCE       = 2    # minimum evidence sentences needed to score an aspect
MAX_FACTS_PER_VIDEO = 50  # [FIX 3] cap per-video facts before saving to DB


# ============================================================
# CATEGORY ASPECTS
# Locked — do not change these
# ============================================================
CATEGORY_ASPECTS = {
    "laptops":   ["performance", "battery", "display", "build", "keyboard", "thermals", "ports", "value"],
    "phones":    ["camera", "battery", "performance", "display", "software", "build", "thermals", "value"],
    "keyboards": ["typing", "sound", "build", "layout", "connectivity", "comfort", "value"],
    "monitors":  ["display", "ergonomics", "connectivity", "build", "value"],
}


# ============================================================
# CONFIDENCE WEIGHTS
# ============================================================
CONFIDENCE_WEIGHTS = {
    "high":   1.0,
    "medium": 0.7,
    "low":    0.0,   # discarded
}


# ============================================================
# STEP 1 — SPLIT TRANSCRIPT INTO CHUNKS
# ============================================================
def chunk_transcript(raw_transcript):
    """
    Splits a raw transcript into overlapping sentence chunks.
    Returns a list of text chunks.
    """
    sentences = re.split(r'(?<=[.!?])\s+', raw_transcript.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    if not sentences:
        return []

    chunks = []
    start  = 0

    while start < len(sentences):
        end   = min(start + CHUNK_SIZE, len(sentences))
        chunk = " ".join(sentences[start:end])
        chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
        if end == len(sentences):
            break

    return chunks


# ============================================================
# STEP 2 — GPT EXTRACTION
# ============================================================
def extract_facts_from_chunk(chunk, category, product_name):
    """
    Sends a transcript chunk to GPT-4o-mini.
    Returns a list of extracted facts as dicts.
    Each fact: {aspect, sentiment, evidence, confidence}

    [FIX 1] Prompt now explicitly instructs GPT to look for negatives.
    [FIX 2] Prompt now instructs GPT to split mixed sentences into two facts.
    """
    aspects     = CATEGORY_ASPECTS.get(category, [])
    aspects_str = ", ".join(aspects)

    prompt = f"""You are an AI system that extracts structured product review information from YouTube transcripts.
Your task is to analyze this transcript chunk from a review of: {product_name} ({category})

ONLY extract opinions about these aspects: {aspects_str}
Ignore anything that is not about these aspects.

For each opinion you find, return a JSON object with exactly these fields:
- "aspect": one of the aspects listed above (lowercase, exactly as written)
- "sentiment": either "positive", "negative", or "neutral"
- "evidence": a short clean sentence (max 15 words) summarizing what the reviewer said
- "confidence": "high" if you are very sure, "medium" if somewhat sure, "low" if unsure

Rules:
- Only extract information that is clearly mentioned in the transcript — do NOT guess or assume
- Do NOT add information that is not present in the transcript
- Only extract aspects from the list above — no other aspects
- Each evidence must be 1 short sentence, max 15 words
- Avoid repeating the same idea twice
- Handle slang and hype words correctly (e.g. "cooked" = bad, "insane" = good, "mid" = mediocre)
- Handle sarcasm carefully
- If an aspect is not mentioned, do not include it
- For monitors, map ALL display-related comments (brightness, refresh rate, color, motion, response time) under "display"

[FIX 1] Negative extraction — IMPORTANT:
- Actively look for complaints, limitations, criticisms, and weaknesses
- Do NOT only extract praise — negatives are just as important as positives
- Common places negatives hide: end of sentence ("...but X is disappointing"), comparisons ("worse than competitor"), qualifications ("good, except for X")

[FIX 2] Mixed sentence splitting — IMPORTANT:
- If a sentence contains BOTH positive and negative information, split it into TWO separate facts
- Example: "battery is great for light use but terrible for gaming"
  → Fact 1: aspect=battery, sentiment=positive, evidence="Battery is great for light use"
  → Fact 2: aspect=battery, sentiment=negative, evidence="Battery is terrible for gaming"
- Never label a mixed sentence as a single fact with one sentiment

Return ONLY a valid JSON array. No explanation, no markdown, no extra text.
If you find nothing relevant, return an empty array: []

Transcript chunk:
{chunk}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=1000,
            temperature=0,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        raw_output = response.choices[0].message.content.strip()
        raw_output = raw_output.replace("```json", "").replace("```", "").strip()
        facts = json.loads(raw_output)

        if not isinstance(facts, list):
            return []

        return facts

    except Exception as e:
        print(f"      [ERROR] GPT extraction failed: {e}")
        return []


# ============================================================
# STEP 3 — FILTER AND VALIDATE FACTS
# ============================================================
def is_english_evidence(text):
    """
    [FIX 4] Checks if an evidence sentence is English using langdetect.
    Returns True if English or detection fails (fail-safe = keep).
    Short sentences under 10 chars are skipped — too short to detect reliably.
    """
    if len(text.strip()) < 10:
        return True  # too short to detect — don't reject

    try:
        return detect(text) == "en"
    except Exception:
        return True  # if detection fails, keep the fact — don't reject


def filter_facts(facts, category):
    """
    Filters out low confidence facts, invalid aspects, duplicates,
    and non-English evidence sentences.

    [FIX 4] Now also discards facts where evidence is not English.
    Returns cleaned list of facts with confidence_weight added.
    """
    valid_aspects = CATEGORY_ASPECTS.get(category, [])
    seen_evidence = set()
    cleaned       = []

    for fact in facts:
        if not all(k in fact for k in ["aspect", "sentiment", "evidence", "confidence"]):
            continue
        if fact["aspect"].lower() not in valid_aspects:
            continue
        if fact["sentiment"].lower() not in ["positive", "negative", "neutral"]:
            continue

        confidence = fact["confidence"].lower()
        if confidence == "low":
            continue

        evidence = fact["evidence"].strip()

        # [FIX 4] Discard non-English evidence sentences
        if not is_english_evidence(evidence):
            print(f"      [FILTER] Non-English evidence discarded: {evidence[:60]}")
            continue

        evidence_lower = evidence.lower()
        if evidence_lower in seen_evidence:
            continue
        seen_evidence.add(evidence_lower)

        fact["aspect"]            = fact["aspect"].lower()
        fact["sentiment"]         = fact["sentiment"].lower()
        fact["confidence"]        = confidence
        fact["confidence_weight"] = CONFIDENCE_WEIGHTS.get(confidence, 0.7)

        cleaned.append(fact)

    return cleaned


# ============================================================
# STEP 4 — CHECK MINIMUM EVIDENCE RULE
# ============================================================
def apply_minimum_evidence_rule(all_facts):
    """
    Groups facts by aspect and removes aspects with too little evidence.
    Returns filtered facts list and a summary of which aspects passed.
    """
    aspect_counts = {}
    for fact in all_facts:
        aspect = fact["aspect"]
        aspect_counts[aspect] = aspect_counts.get(aspect, 0) + 1

    passed_aspects = {k for k, v in aspect_counts.items() if v >= MIN_EVIDENCE}
    filtered_facts = [f for f in all_facts if f["aspect"] in passed_aspects]

    dropped = {k for k in aspect_counts if k not in passed_aspects}
    if dropped:
        print(f"      [INFO] Not enough evidence for: {', '.join(dropped)} (need {MIN_EVIDENCE}+)")

    return filtered_facts, list(passed_aspects)


# ============================================================
# MAIN — PROCESS ONE VIDEO
# ============================================================
def process_video(video, category, product_name):
    """
    Runs full GPT extraction pipeline for one video.
    Saves structured_facts back to the DB.

    [FIX 3] Per-video fact cap applied before saving.
            If a single video produces more than MAX_FACTS_PER_VIDEO facts,
            the lowest-confidence ones are dropped. This prevents one video
            from dominating the final product rating.
    """
    print(f"\n    Processing: {video.title[:60]}")

    # Skip if already processed (has real facts, not empty)
    if video.structured_facts:
        print(f"      [SKIP] Already processed")
        return

    if not video.raw_transcript:
        print(f"      [SKIP] No raw transcript")
        return

    chunks = chunk_transcript(video.raw_transcript)
    print(f"      [CHUNK] Split into {len(chunks)} chunks")

    if not chunks:
        print(f"      [SKIP] Could not chunk transcript")
        return

    all_facts = []
    for i, chunk in enumerate(chunks):
        print(f"      [GPT] Processing chunk {i+1}/{len(chunks)}...")
        facts = extract_facts_from_chunk(chunk, category, product_name)
        all_facts.extend(facts)

    print(f"      [GPT] Extracted {len(all_facts)} raw facts total")

    filtered_facts = filter_facts(all_facts, category)
    print(f"      [FILTER] {len(filtered_facts)} facts after confidence + language filtering")

    # [FIX 3] Apply per-video fact cap
    # Sort by confidence_weight descending — keep highest confidence facts
    if len(filtered_facts) > MAX_FACTS_PER_VIDEO:
        filtered_facts = sorted(
            filtered_facts,
            key=lambda f: f.get("confidence_weight", 0.7),
            reverse=True
        )[:MAX_FACTS_PER_VIDEO]
        print(f"      [CAP] Capped to {MAX_FACTS_PER_VIDEO} facts (was over limit)")

    final_facts, aspects_covered = apply_minimum_evidence_rule(filtered_facts)
    print(f"      [EVIDENCE] Aspects with enough data: {aspects_covered}")

    db = Session()
    try:
        db_video = db.query(Video).filter_by(id=video.id).first()
        db_video.structured_facts  = final_facts
        db_video.aspects_mentioned = aspects_covered
        db.commit()
        print(f"      [DB] Saved {len(final_facts)} structured facts")
    except Exception as e:
        db.rollback()
        print(f"      [ERROR] Failed to save: {e}")
    finally:
        db.close()


# ============================================================
# RUN ALL — PROCESS ALL UNPROCESSED VIDEOS
# ============================================================
def run_all(category=None, product_name=None):
    """
    Runs Stage 2 on all videos that have raw transcripts but no structured facts yet.

    Args:
        category     (str, optional) — filter by category (e.g. "laptops")
        product_name (str, optional) — filter by exact product name

    When called from run_pipeline.py, always pass product_name=product_name
    to scope Stage 2 to the current product only.
    """
    from models.database import Product

    print(f"\n{'='*60}")
    print(f"  REVALIO — GPT Extraction (Stage 2)")
    if product_name:
        print(f"  Scoped to product: {product_name}")
    if category:
        print(f"  Scoped to category: {category}")
    print(f"{'='*60}")

    db = Session()

    try:
        # Start with all videos that have a transcript
        query = db.query(Video).filter(
            Video.raw_transcript.isnot(None),
        )

        # Join Product table ONCE if any product/category filter is needed
        if product_name or category:
            query = query.join(Product, Product.id == Video.product_id)

        if product_name:
            query = query.filter(Product.name == product_name)

        if category:
            query = query.filter(Product.category == category)

        all_videos = query.all()

        # Filter in Python — treats both None AND empty list [] as unprocessed
        videos = [v for v in all_videos if not v.structured_facts]

        if not videos:
            print(f"  No unprocessed videos found.")
            return

        print(f"  Videos to process: {len(videos)}")

        for video in videos:
            product = db.query(Product).filter_by(id=video.product_id).first()
            if not product:
                print(f"  [SKIP] No product found for video {video.video_id}")
                continue
            process_video(video, product.category, product.name)

    finally:
        db.close()

    print(f"\n{'='*60}")
    print(f"  Done! Run check_db.py to see results.")
    print(f"{'='*60}\n")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    run_all()