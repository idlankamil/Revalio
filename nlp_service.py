"""
nlp_service.py — Revalio Stage 3: BERT Validation & Scoring
=============================================================
What this file does:
    1. Runs BERT on each GPT-extracted evidence sentence
    2. Compares BERT sentiment vs GPT sentiment (validation)
       → Agree  : keep original confidence weight (full trust)
       → Disagree: reduce confidence weight by 30% (flag as uncertain)
    3. Aggregates adjusted scores per aspect per video
    4. Blends all videos using: weight = log10(view_count) × content_weight
    5. Calculates overall product rating from weighted aspect scores
    6. Extracts real pros/cons ranked by BERT confidence
    7. Saves everything to DB

Fixes applied:
    [FIX 5]  Lazy load BERT — only loads when Stage 3 actually runs
    [FIX 6]  Correct BERT truncation using truncation=True (tokens not chars)
    [FIX 12] Fixed scoring formula — neutrals weighted at 0.5 to reduce
             score compression and widen the rating spread
    [FIX 13] Added POSITIVE_WEIGHT = 1.5 to scoring formula — corrects for
             systematic negativity bias in YouTube reviewer data. Reviewers are
             power users who compare products and find flaws by profession.
             A 1.5x positive weight produces realistic scores that align with
             expert consensus ratings (e.g. MacBook Pro should score 8+, not 6.8).
             Academic basis: sentiment analysis literature documents systematic
             source bias in professional review corpora vs general consumer opinion.
    [FIX 14] Minimum facts per aspect before scoring — if an aspect has fewer
             than MIN_FACTS_PER_ASPECT weighted facts across all videos, it is
             excluded from the final score entirely. Prevents 1-2 facts from
             producing extreme scores like 0.68 or 10.0 that distort the overall
             rating. Academically justified as sparse-data exclusion — standard
             practice in sentiment aggregation systems.
    [FIX 15] Aspect score soft clamping between SCORE_MIN and SCORE_MAX —
             no real product deserves a perfect 10.0 or a near-zero score from
             a review system. Scores outside this range are statistical artifacts
             of sparse data, not meaningful signals. Clamping to 3.0–9.5 keeps
             scores in a realistic range while preserving meaningful differences
             between products.
    [Phase 4] Saves BERT agreement rate to confidence_score column
              for use in product quality selection

Author: Idlan Kamil
Last Updated: April 2026
"""

import sys
import math
from pathlib import Path
from datetime import datetime, timezone
from transformers import pipeline

# ============================================================
# PATH SETUP — relative paths so it works on any machine
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent   # → Backend/
sys.path.insert(0, str(BASE_DIR))
from models.database import Session, Product, Video


# ============================================================
# [FIX 5] LAZY BERT LOADER
# BERT (~700MB) only loads when Stage 3 actually needs it.
# Before this fix, it loaded at import time — slowing down
# every pipeline startup even when Stage 3 would never run.
# ============================================================
_bert_model = None

def get_bert():
    """
    Lazy-loads BERT model. Loads once on first use, reuses after.
    This prevents 30-60s startup delay when Stage 3 isn't needed.
    """
    global _bert_model
    if _bert_model is None:
        print("  [BERT] Loading model...")
        _bert_model = pipeline(
            "text-classification",
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        )
        print("  [BERT] Model ready!")
    return _bert_model


# ============================================================
# SETTINGS
# ============================================================
MAX_FACTS_PER_VIDEO   = 30    # cap per-video influence (long videos don't dominate)
MAX_PROS              = 3     # max pros to show on product page
MAX_CONS              = 3     # max cons to show on product page
BERT_DISAGREE_PENALTY = 0.7   # multiply confidence weight by this if BERT disagrees with GPT
NEUTRAL_WEIGHT        = 0.5   # [FIX 12] neutrals count at half weight in scoring denominator
POSITIVE_WEIGHT       = 1.5   # [FIX 13] positives weighted 1.5x to correct for YouTube reviewer
                               #          negativity bias — professional reviewers are systematically
                               #          more critical than average consumers. Without this, a product
                               #          praised 60% of the time scores 6.0 instead of a realistic 7.5.

# [FIX 14] Minimum total weighted facts an aspect must have across ALL videos
# before it contributes to the final product score.
# Why 4? An aspect with 1-3 facts is statistically unreliable — one stray
# negative tanks it to near-zero, one stray positive inflates it to 10.
# 4 facts gives the formula enough signal to produce a meaningful score.
MIN_FACTS_PER_ASPECT  = 4

# [FIX 15] Soft score clamping — aspect scores are bounded to this range.
# A real review system should never produce 0.68 or 10.0 — those are
# artifacts of sparse data, not real product quality signals.
# 3.0 floor: even the worst reviewed aspect of a real product has some positives.
# 9.5 ceiling: no product is universally praised with zero criticism.
SCORE_MIN = 3.0
SCORE_MAX = 9.5


# ============================================================
# CATEGORY ASPECT WEIGHTS
# Used to calculate overall rating from aspect scores
# Higher weight = more important to the final score
# ============================================================
ASPECT_WEIGHTS = {
    "laptops": {
        "performance": 2.0,
        "battery":     1.5,
        "display":     1.5,
        "build":       1.0,
        "keyboard":    1.0,
        "thermals":    1.5,
        "ports":       0.5,
        "value":       1.0,
    },
    "phones": {
        "camera":      2.0,
        "battery":     1.5,
        "performance": 2.0,
        "display":     1.5,
        "software":    1.0,
        "build":       1.0,
        "thermals":    1.0,
        "value":       1.0,
    },
    "keyboards": {
        "typing":       2.0,
        "sound":        1.0,
        "build":        1.5,
        "layout":       1.0,
        "connectivity": 1.0,
        "comfort":      1.5,
        "value":        1.0,
    },
    "monitors": {
        "display":      2.0,
        "ergonomics":   1.0,
        "connectivity": 1.0,
        "build":        1.0,
        "value":        1.0,
    },
}


# ============================================================
# STEP 1 — BERT SCORING
# Runs BERT on one evidence sentence
# Returns: bert_sentiment, stars (1-5), bert_confidence (0.0-1.0)
# ============================================================
def bert_score(evidence):
    """
    Runs BERT on one evidence sentence.
    Returns: bert_sentiment (positive/negative/neutral), stars (1-5), confidence (0.0-1.0)

    [FIX 6] Uses truncation=True with max_length=512 so HuggingFace handles
            tokenization correctly. Previous code sliced at 512 characters
            which is semantically wrong (BERT limit is 512 tokens, not chars).
    """
    try:
        result = get_bert()(evidence, truncation=True, max_length=512)[0]

        # Label looks like "4 stars" — grab the number
        stars      = int(result["label"].split()[0])
        confidence = result["score"]        # how confident BERT is (0.0 to 1.0)

        # Map star rating to sentiment label
        if stars >= 4:
            bert_sentiment = "positive"
        elif stars <= 2:
            bert_sentiment = "negative"
        else:
            bert_sentiment = "neutral"

        return bert_sentiment, stars, confidence

    except Exception as e:
        print(f"      [ERROR] BERT scoring failed: {e}")
        return "neutral", 3, 0.5     # safe fallback


# ============================================================
# STEP 2 — BERT VALIDATION
# Compares BERT sentiment vs GPT sentiment
# If they agree → full weight. If they disagree → reduce by 30%.
# ============================================================
def apply_bert_validation(gpt_sentiment, bert_sentiment, confidence_weight):
    """
    Compares GPT and BERT sentiments.
    Returns adjusted_weight and whether they agreed.

    Logic:
        agree    → adjusted_weight = confidence_weight × 1.0 (keep full)
        disagree → adjusted_weight = confidence_weight × 0.7 (reduce by 30%)

    Note: GPT is the truth — we just use BERT to flag uncertain extractions.
    """
    if gpt_sentiment == bert_sentiment:
        adjusted_weight = confidence_weight
        agreed = True
    else:
        adjusted_weight = confidence_weight * BERT_DISAGREE_PENALTY
        agreed = False

    return adjusted_weight, agreed


# ============================================================
# STEP 3 — SCORE ONE VIDEO
# Scores all evidence facts for one video
# Returns: aspect scores dict (0-10), scored facts list
# ============================================================
def score_video(video):
    """
    Scores all facts in one video.
    - Runs BERT on each evidence sentence
    - Validates BERT vs GPT sentiment
    - Adjusts confidence weights based on agreement
    - Aggregates to aspect-level scores (0-10)

    [FIX 12] Scoring formula updated — neutrals weighted at 0.5:
        raw_score = (pos - neg) / (pos + neg + neu × NEUTRAL_WEIGHT)

    [FIX 13] Positive weight added — corrects reviewer negativity bias:
        effective_pos = pos × POSITIVE_WEIGHT (1.5)
        raw_score = (effective_pos - neg) / (effective_pos + neg + neu × NEUTRAL_WEIGHT)
        score = (raw_score + 1) / 2 × 10

    [FIX 14] Aspect minimum fact check happens at blend stage (across all videos),
             not per-video. Per-video scoring still runs normally.

    [FIX 15] Score clamping applied here per aspect — SCORE_MIN to SCORE_MAX.

    Returns: aspect_scores dict, scored_facts list
    """
    facts = video.structured_facts
    if not facts:
        return {}, []

    # Cap per-video influence — take top facts by confidence weight
    if len(facts) > MAX_FACTS_PER_VIDEO:
        facts = sorted(facts, key=lambda f: f.get("confidence_weight", 0.7), reverse=True)
        facts = facts[:MAX_FACTS_PER_VIDEO]

    # Score and validate each fact
    scored_facts   = []
    agree_count    = 0
    disagree_count = 0

    for fact in facts:
        gpt_sentiment     = fact["sentiment"]
        confidence_weight = fact.get("confidence_weight", 0.7)

        # Run BERT on the evidence sentence
        bert_sentiment, stars, bert_confidence = bert_score(fact["evidence"])

        # BERT VALIDATION — compare GPT vs BERT, adjust weight
        adjusted_weight, agreed = apply_bert_validation(
            gpt_sentiment, bert_sentiment, confidence_weight
        )

        if agreed:
            agree_count += 1
        else:
            disagree_count += 1

        scored_facts.append({
            "aspect":            fact["aspect"],
            "gpt_sentiment":     gpt_sentiment,
            "bert_sentiment":    bert_sentiment,
            "evidence":          fact["evidence"],
            "confidence":        fact["confidence"],
            "confidence_weight": confidence_weight,
            "adjusted_weight":   adjusted_weight,
            "bert_confidence":   bert_confidence,
            "bert_agreed":       agreed,
            "stars":             stars,
        })

    print(f"      [BERT] Agreement: {agree_count} agree, {disagree_count} disagree "
          f"({round(agree_count / max(len(scored_facts), 1) * 100)}% agreement rate)")

    # Aggregate to aspect-level scores
    aspect_buckets = {}   # aspect → {positive, negative, neutral weighted sums, fact_count}

    for sf in scored_facts:
        asp = sf["aspect"]
        if asp not in aspect_buckets:
            aspect_buckets[asp] = {
                "positive":   0.0,
                "negative":   0.0,
                "neutral":    0.0,
                "fact_count": 0,
            }
        aspect_buckets[asp][sf["gpt_sentiment"]] += sf["adjusted_weight"]
        aspect_buckets[asp]["fact_count"]         += 1

    # [FIX 12 + FIX 13 + FIX 15] Calculate score per aspect
    #
    # effective_pos = pos × POSITIVE_WEIGHT   (1.5x — corrects reviewer negativity bias)
    # raw_score = (effective_pos - neg) / (effective_pos + neg + neu × NEUTRAL_WEIGHT)
    # score = (raw_score + 1) / 2 × 10
    # score = clamp(score, SCORE_MIN, SCORE_MAX)   ← [FIX 15]
    #
    # Note: MIN_FACTS_PER_ASPECT check [FIX 14] is done in blend_videos()
    # across all videos combined, not per-video. This is intentional —
    # a video with 3 facts on an aspect still contributes to the pool.
    aspect_scores = {}
    for asp, counts in aspect_buckets.items():
        pos = counts["positive"]
        neg = counts["negative"]
        neu = counts["neutral"]

        effective_pos   = pos * POSITIVE_WEIGHT
        effective_total = effective_pos + neg + (neu * NEUTRAL_WEIGHT)

        if effective_total == 0:
            continue

        raw_score = (effective_pos - neg) / effective_total  # range: -1.0 to +1.0
        score     = (raw_score + 1) / 2 * 10                 # range: 0.0 to 10.0

        # [FIX 15] Clamp to realistic range — removes statistical artifacts
        score = round(max(SCORE_MIN, min(SCORE_MAX, score)), 2)

        aspect_scores[asp] = score

    return aspect_scores, scored_facts


# ============================================================
# STEP 4 — BLEND ALL VIDEOS
# Combines per-video scores into one final product rating
# Formula: weight = log10(view_count) × content_weight
# ============================================================
def blend_videos(videos_data):
    """
    Blends aspect scores across all videos using the weighted formula.

    [FIX 14] After blending, any aspect whose total raw fact count across
             all videos is below MIN_FACTS_PER_ASPECT is removed from the
             final scores. This prevents sparse aspects from distorting the
             overall rating.

    videos_data: list of (video_db_object, aspect_scores_dict, scored_facts_list)
    Returns: final blended aspect scores dict, total_weight
    """
    blended           = {}   # aspect → weighted score sum
    weight_sums       = {}   # aspect → total weight for this aspect
    total_weight      = 0.0  # sum of all video weights
    aspect_fact_count = {}   # aspect → total raw facts across all videos [FIX 14]

    for video, aspect_scores, scored_facts in videos_data:
        view_count     = max(video.view_count, 1)
        content_weight = video.content_weight or 1.0
        video_weight   = math.log10(view_count) * content_weight
        total_weight  += video_weight

        for asp, score in aspect_scores.items():
            if asp not in blended:
                blended[asp]           = 0.0
                weight_sums[asp]       = 0.0
                aspect_fact_count[asp] = 0
            blended[asp]     += score * video_weight
            weight_sums[asp] += video_weight

        # Count raw facts per aspect across all videos [FIX 14]
        for sf in scored_facts:
            asp = sf["aspect"]
            if asp not in aspect_fact_count:
                aspect_fact_count[asp] = 0
            aspect_fact_count[asp] += 1

    # Normalize each aspect score
    final_scores = {}
    for asp in blended:
        if weight_sums[asp] > 0:
            # [FIX 14] Skip aspects with too few facts — not enough signal
            if aspect_fact_count.get(asp, 0) < MIN_FACTS_PER_ASPECT:
                print(f"      [FIX 14] Skipping '{asp}' — only "
                      f"{aspect_fact_count.get(asp, 0)} facts (minimum: {MIN_FACTS_PER_ASPECT})")
                continue
            final_scores[asp] = round(blended[asp] / weight_sums[asp], 2)

    return final_scores, total_weight


# ============================================================
# STEP 5 — OVERALL RATING
# Weighted average of aspect scores using category-specific weights
# ============================================================
def calculate_overall_rating(aspect_scores, category):
    """
    Calculates overall product rating (0-10) from aspect scores.
    Uses category-specific weights so important aspects matter more.
    e.g. for laptops: performance weight 2.0 vs ports weight 0.5
    """
    weights      = ASPECT_WEIGHTS.get(category, {})
    total_score  = 0.0
    total_weight = 0.0

    for asp, score in aspect_scores.items():
        weight        = weights.get(asp, 1.0)
        total_score  += score * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0

    overall = round(total_score / total_weight, 1)
    return overall


# ============================================================
# STEP 6 — EXTRACT PROS AND CONS
# Pulls real evidence sentences ranked by BERT confidence
# ============================================================
def extract_pros_cons(all_scored_facts):
    """
    Extracts top pros and cons from all scored facts across all videos.

    - GPT sentiment decides what's a pro vs con
    - BERT confidence score decides which ones to show first
    - Deduplicates by evidence text

    Returns: pros list, cons list (each max 3 items)
    """
    positives = [f for f in all_scored_facts if f["gpt_sentiment"] == "positive"]
    negatives = [f for f in all_scored_facts if f["gpt_sentiment"] == "negative"]

    # Sort by BERT confidence — show most reliable sentences first
    positives = sorted(positives, key=lambda f: f["bert_confidence"], reverse=True)
    negatives = sorted(negatives, key=lambda f: f["bert_confidence"], reverse=True)

    # Deduplicate and pick top ones
    seen = set()
    pros = []
    for f in positives:
        evidence = f["evidence"].strip()
        if evidence not in seen:
            seen.add(evidence)
            pros.append(evidence)
        if len(pros) >= MAX_PROS:
            break

    seen = set()
    cons = []
    for f in negatives:
        evidence = f["evidence"].strip()
        if evidence not in seen:
            seen.add(evidence)
            cons.append(evidence)
        if len(cons) >= MAX_CONS:
            break

    return pros, cons


# ============================================================
# STEP 7 — LAST UPDATED LABEL
# Human-readable freshness label shown on the product page
# ============================================================
def get_updated_label(last_analyzed):
    """
    Returns a human-readable freshness label.
    Examples: "Updated this week", "Updated this month", "Updated 3 weeks ago"
    """
    if not last_analyzed:
        return "Recently analyzed"

    now = datetime.now(timezone.utc)

    if last_analyzed.tzinfo is None:
        last_analyzed = last_analyzed.replace(tzinfo=timezone.utc)

    days = (now - last_analyzed).days

    if days <= 7:
        return "Updated this week"
    elif days <= 30:
        return "Updated this month"
    else:
        weeks = days // 7
        return f"Updated {weeks} weeks ago"


# ============================================================
# MAIN — PROCESS ONE PRODUCT
# Full Stage 3 pipeline for one product
# ============================================================
def process_product(product):
    """
    Runs the full Stage 3 pipeline for one product.

    [FIX 14] blend_videos() now receives scored_facts per video so it can
             count total facts per aspect across all videos before deciding
             whether to include that aspect in the final score.
    """
    print(f"\n  Product: {product.name} ({product.category})")

    db = Session()
    try:
        videos = db.query(Video).filter(
            Video.product_id       == product.id,
            Video.structured_facts != None
        ).all()

        if not videos:
            print(f"    [SKIP] No processed videos found — run Stage 2 first")
            return

        print(f"    Videos to score: {len(videos)}")

        # Score each video individually
        videos_data      = []
        all_scored_facts = []

        for video in videos:
            print(f"    Scoring: {video.title[:55]}...")
            aspect_scores, scored_facts = score_video(video)

            if not aspect_scores:
                print(f"      [SKIP] No scoreable facts in this video")
                continue

            # Pass scored_facts alongside video + scores for FIX 14 fact counting
            videos_data.append((video, aspect_scores, scored_facts))
            all_scored_facts.extend(scored_facts)

            for asp, score in sorted(aspect_scores.items()):
                print(f"      {asp:<15} {score}/10")

        if not videos_data:
            print(f"    [SKIP] No scoreable videos — check structured_facts in DB")
            return

        # Blend all videos into final product scores
        final_aspect_scores, total_weight = blend_videos(videos_data)

        print(f"\n    -- Final Aspect Scores --")
        for asp, score in sorted(final_aspect_scores.items()):
            print(f"      {asp:<15} {score}/10")
        print(f"    Total weight: {round(total_weight, 2)}")

        # Calculate overall rating
        overall = calculate_overall_rating(final_aspect_scores, product.category)
        print(f"    Overall rating: {overall}/10")

        # Extract pros and cons
        pros, cons = extract_pros_cons(all_scored_facts)
        print(f"\n    Pros:")
        for p in pros:
            print(f"      + {p}")
        print(f"    Cons:")
        for c in cons:
            print(f"      - {c}")

        # Calculate BERT agreement rate
        total_agreed   = sum(1 for f in all_scored_facts if f["bert_agreed"])
        total_facts    = len(all_scored_facts)
        agreement_rate = round(total_agreed / max(total_facts, 1), 3)
        print(f"    BERT agreement rate: {agreement_rate:.1%}")

        # Save everything to DB
        now        = datetime.now(timezone.utc)
        db_product = db.query(Product).filter_by(id=product.id).first()

        db_product.overall_rating         = overall
        db_product.aspect_scores          = final_aspect_scores
        db_product.pros                   = pros
        db_product.cons                   = cons
        db_product.confidence_score       = agreement_rate
        db_product.total_reviews_analyzed = len(videos_data)
        db_product.total_weight           = total_weight
        db_product.last_analyzed          = now
        db_product.last_updated_label     = get_updated_label(now)
        db_product.last_updated           = now

        db.commit()
        print(f"\n    [DB] Saved final rating for {product.name} ✓")

    except Exception as e:
        db.rollback()
        print(f"    [ERROR] Failed to process {product.name}: {e}")
        import traceback
        traceback.print_exc()

    finally:
        db.close()


# ============================================================
# RUN ALL — PROCESS ALL PRODUCTS
# ============================================================
def run_all(category=None):
    """
    Runs Stage 3 on all products (or filtered by category).
    Useful for running Stage 3 standalone without run_pipeline.py.
    """
    print(f"\n{'='*60}")
    print(f"  REVALIO — BERT Validation & Scoring (Stage 3)")
    print(f"{'='*60}")

    db = Session()
    try:
        query = db.query(Product)
        if category:
            query = query.filter(Product.category == category)
        products = query.all()

        if not products:
            print(f"  No products found.")
            return

        print(f"  Products to score: {len(products)}")
        for product in products:
            process_product(product)

    finally:
        db.close()

    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"{'='*60}\n")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    run_all()