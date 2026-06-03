# chatbot_service.py
# Revalio Chatbot - Hybrid Mode (Option C)
# GPT handles all reasoning. Frontend carries history. No Flask sessions.

import base64
import io
import requests
from PIL import Image
from sqlalchemy.orm import Session
from models.database import Product  # SQLAlchemy model lives in database.py
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ============================================================
# IMAGE UTILS (kept from demo - solid as-is)
# ============================================================

def convert_image_to_base64(image_bytes: bytes) -> str | None:
    """
    Convert raw image bytes to base64 JPEG string.
    Handles RGBA, CMYK, animated, oversized images.
    Adapted from demo - works with bytes instead of file path (FastAPI style).
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))

        # grab first frame if animated
        if hasattr(img, 'is_animated') and img.is_animated:
            img.seek(0)

        # normalize color mode to RGB
        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            if 'transparency' in img.info:
                background.paste(img, mask=img.split()[-1])
            else:
                background.paste(img)
            img = background
        elif img.mode == 'CMYK':
            img = img.convert('RGB')
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # resize if too large
        max_size = 2048
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=90, optimize=True)
        buffer.seek(0)

        return base64.b64encode(buffer.read()).decode('utf-8')

    except Exception as e:
        print(f"❌ Image conversion error: {e}")
        return None


# ============================================================
# DB HELPERS
# ============================================================

def load_selected_products(db: Session) -> list[dict]:
    """
    Load all 20 selected products from PostgreSQL.
    Returns a list of dicts for system prompt injection.
    """
    products = db.query(Product).filter(Product.is_selected == True).all()

    result = []
    for p in products:
        result.append({
            "id": p.id,
            "name": p.name,                    # ← Product.name (not product_name)
            "category": p.category,
            "overall_rating": p.overall_rating,
            "shopee_price": p.shopee_price,
            "lazada_price": p.lazada_price,
            "pros": p.pros,                    # JSON column - list of strings
            "cons": p.cons,                    # JSON column - list of strings
            "aspects": p.aspect_scores,        # ← Product.aspect_scores (not aspects)
        })

    return result


def format_products_for_prompt(products: list[dict]) -> str:
    """
    Format product list into clean text block for GPT system prompt.
    GPT needs to READ this and reason about it - keep it structured.
    """
    lines = []
    for p in products:
        lines.append(
            f"[ID:{p['id']}] {p['name']} | Category: {p['category']} | "
            f"AI Rating: {p['overall_rating']}/10 | "
            f"Shopee: RM{p['shopee_price']} | Lazada: RM{p['lazada_price']} | "
            f"Pros: {p['pros']} | Cons: {p['cons']}"
        )
    return "\n".join(lines)


# ============================================================
# SYSTEM PROMPT BUILDER
# ============================================================

def build_system_prompt(products: list[dict]) -> str:
    """
    Inject product data into GPT system prompt.
    GPT reads this and reasons about it - no keyword matching needed.
    """
    product_block = format_products_for_prompt(products)

    return f"""You are Revalio, an AI shopping assistant for Malaysian university students.
You help students choose the best electronics based on their budget, needs, and real YouTube review data.

PRODUCT DATABASE (these are the only products you can recommend):
{product_block}

RESPONSE FORMAT - VERY IMPORTANT:
You must ALWAYS reply in this exact JSON format, no exceptions:
{{
  "reply": "your natural language response here",
  "product_ids": [list of integer IDs you are recommending, or empty list [] if not recommending]
}}

RULES:
- Only recommend products from the database above. Never invent products.
- Always mention prices in RM when recommending.
- Always mention the AI rating when recommending a product.
- Be concise - students want clear answers, not essays.
- product_ids must only contain IDs from the database above.
- If the user asks about something outside the database, say it's outside Revalio's current scope and product_ids should be [].
- If the user is just chatting (not asking for recommendations), product_ids should be [].
- For image queries, identify what the product is and find the closest match in the database."""


# ============================================================
# CORE GPT CALL FUNCTIONS
# ============================================================

def call_gpt_text(messages: list[dict]) -> str:
    """
    Call GPT-4o-mini for text-only queries.
    Cheaper and fast enough for chat.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "max_tokens": 600,
        "response_format": {"type": "json_object"}  # force JSON output
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload
    )

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    
    print(f"❌ GPT text error: {response.status_code} - {response.text}")
    return '{"reply": "Sorry, I\'m having trouble right now. Please try again!", "product_ids": []}'


def call_gpt_vision(messages: list[dict]) -> str:
    """
    Call GPT-4o for image + text queries.
    More expensive - only used when image is attached.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": messages,
        "max_tokens": 600,
        "response_format": {"type": "json_object"}
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload
    )

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    
    print(f"❌ GPT vision error: {response.status_code} - {response.text}")
    return '{"reply": "Sorry, I couldn\'t process the image. Please try again!", "product_ids": []}'


# ============================================================
# MAIN CHAT FUNCTION (called by FastAPI endpoint)
# ============================================================

import json

def process_chat(
    user_message: str,
    history: list[dict],       # [{"role": "user"/"assistant", "content": "..."}]
    image_bytes: bytes | None, # raw bytes if image attached, else None
    db: Session
) -> dict:
    """
    Main chatbot function. Called by POST /api/chat endpoint.
    
    Returns:
        {
            "reply": "GPT natural language response",
            "product_ids": [3, 7],     # IDs to render as cards (can be empty)
            "products": [...]          # full product data for those IDs
        }
    """
    # step 1: load products from DB and build system prompt
    products = load_selected_products(db)
    system_prompt = build_system_prompt(products)

    # step 2: build messages array
    # structure: [system] + history (last 10 turns) + new user message
    messages = [{"role": "system", "content": system_prompt}]
    messages += history[-10:]  # keep last 10 turns for context window

    # step 3: build user message content
    # if image attached → multimodal content, else plain text
    if image_bytes:
        base64_image = convert_image_to_base64(image_bytes)

        if base64_image:
            user_content = [
                {
                    "type": "text",
                    "text": user_message if user_message else "What product is this? Find the closest match in your database."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        else:
            # image conversion failed - fall back to text only
            user_content = user_message + " (Note: image upload failed to process)"
    else:
        user_content = user_message

    messages.append({"role": "user", "content": user_content})

    # step 4: call GPT - vision model if image, mini if text only
    if image_bytes and base64_image:
        raw_response = call_gpt_vision(messages)
    else:
        raw_response = call_gpt_text(messages)

    # step 5: parse GPT response
    # GPT is forced to return JSON via response_format - but always have a fallback
    try:
        parsed = json.loads(raw_response)
        reply = parsed.get("reply", "I had trouble generating a response.")
        product_ids = parsed.get("product_ids", [])

        # safety check - make sure IDs are actually integers
        product_ids = [int(pid) for pid in product_ids if str(pid).isdigit()]

    except json.JSONDecodeError:
        # GPT broke format somehow - return safe fallback
        reply = raw_response  # show whatever GPT said
        product_ids = []

    # step 6: fetch full product data for recommended IDs
    # frontend needs this to render product cards
    recommended_products = []
    if product_ids:
        db_products = db.query(Product).filter(Product.id.in_(product_ids)).all()
        for p in db_products:
            recommended_products.append({
                "id": p.id,
                "name": p.name,                # ← correct column name
                "category": p.category,
                "brand": p.brand,
                "overall_rating": p.overall_rating,
                "shopee_price": p.shopee_price,
                "lazada_price": p.lazada_price,
                "shopee_url": p.shopee_url,
                "lazada_url": p.lazada_url,
                # no image_url column in schema - frontend can use a placeholder
            })

    return {
        "reply": reply,
        "product_ids": product_ids,
        "products": recommended_products  # frontend renders these as cards
    }
