from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, DateTime, Text, Boolean
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# This is the base class that all our table models inherit from
Base = declarative_base()


# ============================================================
# TABLE 1: products
# Stores one row per product (e.g. MacBook Air M4)
# The AI rating, aspect scores, pros/cons all live here
# ============================================================
class Product(Base):
    __tablename__ = 'products'

    id                     = Column(Integer, primary_key=True)
    name                   = Column(String, nullable=False)
    category               = Column(String, nullable=False)   # laptops, phones, keyboards, monitors
    brand                  = Column(String)

    # Auto-selection flag — set by run_pipeline.py after Stage 2
    # True = this product was selected as top 5 in its category
    # False = not selected (kept in DB but skipped by Stage 3)
    is_selected            = Column(Boolean, default=False)
    is_updating            = Column(Boolean, default=False)
    

    # AI analysis results (filled in by pipeline)
    overall_rating         = Column(Float)                    # 0-10 scale
    aspect_scores          = Column(JSON)  
    specs                  = Column(JSON)                   # {"performance": 9.2, "battery": 7.8, ...}
    pros                   = Column(JSON)                     # real evidence sentences from GPT (2-3)
    cons                   = Column(JSON)                     # real evidence sentences from GPT (2-3)
    confidence_score       = Column(Float)                    # 0.0-1.0, how reliable the rating is
    total_reviews_analyzed = Column(Integer, default=0)

    # Weekly update blending — IMPORTANT
    # Stores the total accumulated weight behind the current rating
    # Used in blending formula: new_rating = (old_rating * total_weight + new_score * new_weight) / (total_weight + new_weight)
    total_weight           = Column(Float, default=0.0)

    # Price comparison (filled in later by scraper)
    shopee_price           = Column(Float)
    lazada_price           = Column(Float)
    shopee_url             = Column(String)
    lazada_url             = Column(String)
    last_price_updated     = Column(DateTime)
    price_unavailable      = Column(Boolean, default=False)

    # Weekly update system fields
    last_analyzed          = Column(DateTime)                 # when did pipeline last run on this product?
    last_updated_label     = Column(String)                   # "Updated this week", "Updated this month"

    created_at             = Column(DateTime, default=datetime.now)
    last_updated           = Column(DateTime, default=datetime.now)
    image_url              = Column(String)


# ============================================================
# TABLE 2: videos
# Stores one row per YouTube video analyzed
# Linked to a product via product_id
# ============================================================
class Video(Base):
    __tablename__ = 'videos'

    id                   = Column(Integer, primary_key=True)
    product_id           = Column(Integer)                    # links to products.id

    video_id             = Column(String, unique=True)        # YouTube video ID (e.g. "G0cmfY7qdmY")
    title                = Column(String)
    channel              = Column(String)
    view_count           = Column(Integer)
    published_date       = Column(String)

    # Stage 1 output — classification and weight
    video_classification = Column(String)                     # "full_review", "comparison", "early_impression"
    content_weight       = Column(Float)                      # 1.0 / 0.75 / 0.5 based on classification

    # Stage 1 output — raw transcript (saved as-is from YouTube, no processing)
    raw_transcript       = Column(Text)                       # raw text from YouTube transcript API

    # Stage 2 output — GPT extracted structured facts
    # Format: [{"aspect": "battery", "sentiment": "negative", "evidence": "...", "confidence": "high"}, ...]
    structured_facts     = Column(JSON)

    # Stage 3 output — final scores for this video
    sentiment_score      = Column(Float)                      # overall sentiment score for this video
    aspects_mentioned    = Column(JSON)                       # which aspects this video covers

    created_at           = Column(DateTime, default=datetime.now)


# ============================================================
# TABLE 3: price_history
# Stores a price snapshot every time we scrape
# Useful for tracking price changes over time
# ============================================================
class PriceHistory(Base):
    __tablename__ = 'price_history'

    id         = Column(Integer, primary_key=True)
    product_id = Column(Integer)
    platform   = Column(String)                               # "shopee" or "lazada"
    price      = Column(Float)
    timestamp  = Column(DateTime, default=datetime.now)


# ============================================================
# DATABASE CONNECTION
# Format: postgresql://username:password@host/database_name
# ============================================================
engine = create_engine('postgresql://postgres:Idlanlol26@localhost/revalio')

# Creates all tables if they don't exist yet
# Safe to run multiple times - won't delete existing data
Base.metadata.create_all(engine)

# Session is what you use to read/write data throughout the app
Session = sessionmaker(bind=engine)