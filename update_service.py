"""
update_service.py — Revalio: Weekly Update & Manual Refresh Core Logic
=======================================================================
What this file does:
    - Searches YouTube for NEW videos for a single product
    - Applies the same filters as Stage 1 (LLM filter + hard filters + classify)
    - Fetches transcripts for new videos only (max 3 per run — KB rule)
    - Runs Stage 2 (GPT fact extraction) on each new video
    - Runs Stage 3 (BERT scoring) on each new video
    - Blends new scores into the existing product rating using total_weight
    - Updates the product rating, pros/cons, and last_updated_label in DB

    This file is called by TWO things:
        1. tasks/celery_app.py  → weekly automatic job (all 20 products)
        2. routes/refresh.py    → manual refresh button (one product at a time)

    Both call the same function: run_update_for_product(product_id)

Blending formula (from KB):
    new_rating = (old_rating × old_total_weight + new_score × new_video_weight)
               / (old_total_weight + new_video_weight)

    Where total_weight = sum of all video weights that built the current rating.
    This is already saved to DB by nlp_service.py — we just read and update it.

Author: Idlan Kamil
Last Updated: April 2026
KB Version: 7
"""

import os
import sys
import math
import time

from datetime import datetime, timezone
from dotenv import load_dotenv

# ============================================================
# PATH SETUP
# so this file can import from models/ and services/
# ============================================================
sys.path.append(r"D:\REVALIO\Backend")

from models.database import Session, Product, Video

# Import the functions we need from existing services
# We reuse them directly — no code duplication
from services.data_manager import (
    build_search_layers,
    search_youtube_query,
    passes_hard_filters,
    llm_filter_titles,
    classify_video,
    fetch_transcript,
    save_to_db,
)
from services.preprocessing import process_video
from services.nlp_service    import score_video, blend_videos, calculate_overall_rating, extract_pros_cons, get_updated_label

load_dotenv(r"D:\REVALIO\Backend\.env")


# ============================================================
# SETTINGS
# ============================================================
MAX_NEW_VIDEOS        = 3      # KB rule: max 3 new videos per product per run
MIN_TRANSCRIPT_LENGTH = 3000   # same as Stage 1
DELAY_BETWEEN_VIDEOS  = 45     # seconds — same as Stage 1 to avoid YouTube rate limits


# ============================================================
# STEP 1 — SEARCH FOR NEW VIDEOS
# Finds videos that are NOT already in the DB for this product
# ============================================================
def find_new_videos(product_name, category, brand):
    """
    Searches YouTube for new videos for a product.
    Returns a list of video dicts that passed all filters
    AND are not already saved in the DB.

    Uses the same search layers, hard filters, and LLM filter
    as data_manager.py Stage 1 — consistent quality.
    """
    print(f"  [SEARCH] Looking for new videos for '{product_name}'")

    # Get all video_ids already in DB for this product
    # We use these to skip videos we already have
    db = Session()
    try:
        product_row  = db.query(Product).filter_by(name=product_name).first()
        existing_ids = set(
            v.video_id for v in db.query(Video)
            .filter(Video.product_id == product_row.id)
            .all()
        ) if product_row else set()
    except Exception as e:
        print(f"  [DB ERROR] Could not fetch existing video IDs: {e}")
        existing_ids = set()
    finally:
        db.close()

    print(f"  [SEARCH] Product already has {len(existing_ids)} videos in DB — will skip these")

    # Run the same multi-layer search as Stage 1
    layers, short = build_search_layers(product_name, brand)
    seen_ids   = set()
    candidates = []

    for i, (query, max_results) in enumerate(layers):
        results  = search_youtube_query(query, max_results)
        filtered = [
            v for v in results
            if v["video_id"] not in seen_ids          # not a duplicate from another layer
            and v["video_id"] not in existing_ids     # not already saved in DB
            and passes_hard_filters(v, category)      # passes view count, duration, junk keywords
        ]

        for v in filtered:
            seen_ids.add(v["video_id"])
            candidates.append(v)

        # Stop early if we have enough candidates to work with
        if len(candidates) >= 20:
            print(f"  [SEARCH] Got {len(candidates)} new candidates — stopping search early")
            break

    if not candidates:
        print(f"  [SEARCH] No new videos found for '{product_name}' — skipping")
        return []

    # Run LLM filter on the new candidates — same as Stage 1
    print(f"  [LLM FILTER] Filtering {len(candidates)} new candidates...")
    candidates = llm_filter_titles(candidates, product_name)
    print(f"  [LLM FILTER] {len(candidates)} candidates passed")

    return candidates


# ============================================================
# STEP 2 — FETCH AND SAVE NEW VIDEOS
# Fetches transcripts and saves to DB (same as Stage 1)
# Capped at MAX_NEW_VIDEOS (3) per KB rule
# ============================================================
def fetch_and_save_new_videos(product_name, category, brand, candidates):
    """
    Takes the filtered candidates from find_new_videos(),
    fetches transcripts, and saves them to DB.

    Returns a list of Video DB objects that were successfully saved.
    Capped at MAX_NEW_VIDEOS (3) — KB rule.
    """
    # Sort by view count — fetch the most popular new videos first
    candidates.sort(key=lambda x: x["view_count"], reverse=True)

    saved_videos = []

    for v in candidates:
        if len(saved_videos) >= MAX_NEW_VIDEOS:
            break

        # Safety net — check DB one more time before saving
        # Prevents UniqueViolation if existing_ids check in find_new_videos missed anything
        db_check = Session()
        try:
            already_exists = db_check.query(Video).filter_by(video_id=v["video_id"]).first()
            if already_exists:
                print(f"  [SKIP - DUPLICATE] '{v['title'][:60]}' — already in DB")
                continue
        finally:
            db_check.close()

        cls, weight = classify_video(v["title"])
        transcript  = fetch_transcript(v["video_id"])

        if not transcript or len(transcript) < MIN_TRANSCRIPT_LENGTH:
            print(f"  [SKIP] '{v['title'][:60]}' — transcript missing or too short")
            continue

        v.update({
            "video_classification": cls,
            "content_weight":       weight,
            "raw_transcript":       transcript,
        })

        save_to_db(product_name, category, brand, v)
        print(f"  [SAVED] '{v['title'][:60]}' ({cls}, weight {weight})")

        # Fetch the saved Video object from DB so we can pass it to Stage 2/3
        db = Session()
        try:
            saved = db.query(Video).filter_by(video_id=v["video_id"]).first()
            if saved:
                saved_videos.append(saved)
        except Exception as e:
            print(f"  [DB ERROR] Could not fetch saved video: {e}")
        finally:
            db.close()

        time.sleep(DELAY_BETWEEN_VIDEOS)

    print(f"  [FETCH] {len(saved_videos)} new videos saved successfully")
    return saved_videos


# ============================================================
# STEP 3 — RUN STAGE 2 ON NEW VIDEOS
# GPT fact extraction — same as preprocessing.py
# ============================================================
def run_stage2_on_new_videos(new_videos, category, product_name):
    """
    Runs GPT fact extraction (Stage 2) on each new video.
    Reuses process_video() from preprocessing.py directly.

    After this, each new video has structured_facts in DB.
    """
    print(f"  [STAGE 2] Running GPT extraction on {len(new_videos)} new videos...")

    for video in new_videos:
        print(f"  [STAGE 2] Processing: '{video.title[:60]}'")
        process_video(video, category, product_name)

    print(f"  [STAGE 2] Done")


# ============================================================
# STEP 4 — BLEND NEW SCORES INTO EXISTING RATING
# The most important step — this is what makes the update work
# ============================================================
def blend_new_scores_into_existing(product, new_videos):
    """
    Scores new videos using BERT (Stage 3) and blends their scores
    into the existing product rating using the total_weight formula.

    Blending formula (from KB):
        new_rating = (old_rating × old_total_weight + new_score × new_video_weight)
                   / (old_total_weight + new_video_weight)

    This formula is applied per aspect — not just overall.
    Then calculate_overall_rating() combines the updated aspect scores.

    Returns: updated aspect_scores dict, updated overall rating, new total_weight
    """
    print(f"  [BLEND] Blending new videos into existing rating for '{product.name}'")
    print(f"  [BLEND] Existing rating: {product.overall_rating} | Existing total_weight: {round(product.total_weight or 0, 2)}")

    # Reload new videos fresh from DB so structured_facts are populated
    db = Session()
    try:
        refreshed = [db.query(Video).filter_by(id=v.id).first() for v in new_videos]
        refreshed = [v for v in refreshed if v and v.structured_facts]
    finally:
        db.close()

    if not refreshed:
        print(f"  [BLEND] No new videos have structured facts — nothing to blend")
        return None, None, None

    # Score each new video using Stage 3
    new_videos_data      = []
    all_new_scored_facts = []

    for video in refreshed:
        aspect_scores, scored_facts = score_video(video)

        if not aspect_scores:
            print(f"  [BLEND] '{video.title[:50]}' — no scoreable facts, skipping")
            continue

        new_videos_data.append((video, aspect_scores, scored_facts))
        all_new_scored_facts.extend(scored_facts)

        print(f"  [BLEND] New video scored: '{video.title[:50]}'")
        for asp, score in sorted(aspect_scores.items()):
            print(f"           {asp:<15} {score}/10")

    if not new_videos_data:
        print(f"  [BLEND] No new videos could be scored — skipping blend")
        return None, None, None

    # Get the new videos' blended scores and total weight contribution
    new_aspect_scores, new_total_weight = blend_videos(new_videos_data)

    print(f"  [BLEND] New videos combined score: {new_aspect_scores}")
    print(f"  [BLEND] New videos combined weight: {round(new_total_weight, 2)}")

    # Now blend new scores into existing scores using the KB formula:
    #   blended = (old_score × old_weight + new_score × new_weight)
    #           / (old_weight + new_weight)
    old_aspect_scores = product.aspect_scores or {}
    old_total_weight  = product.total_weight  or 0.0

    blended_aspect_scores = {}

    # Collect all aspect names from both old and new
    all_aspects = set(list(old_aspect_scores.keys()) + list(new_aspect_scores.keys()))

    for asp in all_aspects:
        old_score = old_aspect_scores.get(asp)
        new_score = new_aspect_scores.get(asp)

        if old_score is not None and new_score is not None:
            # Both old and new have data — proper blend
            blended = (
                (old_score * old_total_weight) + (new_score * new_total_weight)
            ) / (old_total_weight + new_total_weight)
            blended_aspect_scores[asp] = round(blended, 2)

        elif old_score is not None:
            # Only old data — keep as is (new videos didn't mention this aspect)
            blended_aspect_scores[asp] = old_score

        else:
            # Only new data — use new score directly
            blended_aspect_scores[asp] = new_score

    # Recalculate overall rating from blended aspect scores
    updated_overall      = calculate_overall_rating(blended_aspect_scores, product.category)
    updated_total_weight = old_total_weight + new_total_weight

    print(f"  [BLEND] Updated aspect scores: {blended_aspect_scores}")
    print(f"  [BLEND] Updated overall rating: {updated_overall} (was {product.overall_rating})")
    print(f"  [BLEND] Updated total_weight: {round(updated_total_weight, 2)}")

    return blended_aspect_scores, updated_overall, updated_total_weight, all_new_scored_facts


# ============================================================
# STEP 5 — SAVE UPDATED RATING TO DB
# ============================================================
def save_updated_rating(product_id, blended_aspect_scores, updated_overall,
                        updated_total_weight, all_new_scored_facts, old_scored_facts):
    """
    Saves the blended rating back to the products table.
    Also updates pros/cons by re-ranking ALL facts (old + new combined).
    Updates last_updated_label and last_analyzed timestamps.
    """
    db = Session()
    try:
        product = db.query(Product).filter_by(id=product_id).first()
        if not product:
            print(f"  [DB ERROR] Product {product_id} not found")
            return

        now = datetime.now(timezone.utc)

        # Re-extract pros/cons from combined old + new facts
        # This ensures pros/cons reflect the full picture, not just new videos
        all_facts_combined = (old_scored_facts or []) + (all_new_scored_facts or [])
        pros, cons = extract_pros_cons(all_facts_combined)

        product.overall_rating         = updated_overall
        product.aspect_scores          = blended_aspect_scores
        product.pros                   = pros
        product.cons                   = cons
        product.total_weight           = updated_total_weight
        product.total_reviews_analyzed = (product.total_reviews_analyzed or 0) + len(all_new_scored_facts)
        product.last_analyzed          = now
        product.last_updated_label     = get_updated_label(now)
        product.last_updated           = now
        product.is_updating            = False   # clear the guard flag

        db.commit()
        print(f"  [DB] Rating saved for '{product.name}' ✓")
        print(f"  [DB] New rating: {updated_overall}/10 | Pros: {len(pros)} | Cons: {len(cons)}")

    except Exception as e:
        db.rollback()
        print(f"  [DB ERROR] Failed to save updated rating: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

def _clear_updating_flag(product_id):
    db = Session()
    try:
        product = db.query(Product).filter_by(id=product_id).first()
        if product:
            product.is_updating = False
            db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()

# ============================================================
# MAIN FUNCTION — called by Celery tasks and refresh route
# This is the only function other files need to call
# ============================================================
def run_update_for_product(product_id):
    """
    Runs the full update pipeline for ONE product.

    Called by:
        - tasks/celery_app.py  → weekly_update_all_products() loops all 20 products
        - routes/refresh.py    → manual Refresh Rating button

    Flow:
        1. Load product from DB
        2. Search for new YouTube videos
        3. If none found → skip (saves API quota + GPT cost)
        4. Fetch transcripts for new videos (max 3)
        5. Run Stage 2 — GPT fact extraction
        6. Run Stage 3 — BERT scoring
        7. Blend new scores into existing rating
        8. Save updated rating to DB

    Returns:
        "updated"     — new videos found and rating blended successfully
        "no_new"      — no new videos found, rating unchanged
        "skipped"     — product has no existing rating yet (run full pipeline first)
        "error"       — something went wrong (check logs)
    """
    print(f"\n{'='*60}")
    print(f"  UPDATE — Product ID: {product_id}")
    print(f"{'='*60}")

    # ── Load product from DB ──────────────────────────────────
    db = Session()
    try:
        product = db.query(Product).filter_by(id=product_id).first()
        if not product:
            print(f"  [ERROR] Product {product_id} not found in DB")
            return "error"

        # Safety check — don't update a product that has no rating yet
        # They need to run the full pipeline first
        if product.overall_rating is None:
            print(f"  [SKIP] '{product.name}' has no existing rating — run full pipeline first")
            return "skipped"

        # Snapshot product info while session is open
        product_id   = product.id
        product_name = product.name
        category     = product.category
        brand        = product.brand

    except Exception as e:
        print(f"  [DB ERROR] Failed to load product: {e}")
        return "error"
    finally:
        db.close()

    print(f"  Product : {product_name}")
    print(f"  Category: {category}")

    # ── Step 1: Search for new videos ────────────────────────
    new_candidates = find_new_videos(product_name, category, brand)

    if not new_candidates:
        print(f"  [RESULT] No new videos found — rating unchanged")
        _clear_updating_flag(product_id)
        return "no_new"

    # ── Step 2: Fetch transcripts and save to DB ─────────────
    new_videos = fetch_and_save_new_videos(product_name, category, brand, new_candidates)

    if not new_videos:
        print(f"  [RESULT] No new videos had usable transcripts — rating unchanged")
        _clear_updating_flag(product_id)
        return "no_new"

    # ── Step 3: Run Stage 2 (GPT extraction) ─────────────────
    run_stage2_on_new_videos(new_videos, category, product_name)

    # ── Step 4: Blend new scores into existing rating ─────────
    # Re-load product fresh from DB before blending
    db = Session()
    try:
        product = db.query(Product).filter_by(id=product_id).first()

        # Collect existing scored facts from all OLD videos for pros/cons merge
        old_videos = db.query(Video).filter(
            Video.product_id       == product_id,
            Video.structured_facts != None,
        ).all()

        # We need scored_facts from old videos to re-rank pros/cons after blend
        # Import score_video here to avoid circular — already imported at top
        old_scored_facts = []
        for v in old_videos:
            if v.id not in [nv.id for nv in new_videos]:   # skip the new ones
                _, scored = score_video(v)
                old_scored_facts.extend(scored)

    finally:
        db.close()

    result = blend_new_scores_into_existing(product, new_videos)

    # blend_new_scores_into_existing returns 4 values
    if result[0] is None:
        print(f"  [RESULT] Blend failed — rating unchanged")
        return "error"

    blended_aspect_scores, updated_overall, updated_total_weight, all_new_scored_facts = result

    # ── Step 5: Save updated rating to DB ────────────────────
    save_updated_rating(
        product_id,
        blended_aspect_scores,
        updated_overall,
        updated_total_weight,
        all_new_scored_facts,
        old_scored_facts,
    )

    print(f"\n  [RESULT] '{product_name}' updated successfully!")
    print(f"  [RESULT] New rating: {updated_overall}/10")
    print(f"{'='*60}\n")

    return "updated"
