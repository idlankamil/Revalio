import sys
sys.path.append(r"D:\REVALIO\Backend")

import time
import re
from datetime import datetime
from urllib.parse import quote
from DrissionPage import ChromiumPage, ChromiumOptions

from models.database import Session, Product, PriceHistory


# ============================================================
# SETTINGS
# ============================================================
MAX_RESULTS  = 15
SCROLL_TIMES = 4   # Lazada is slower to render, needs more scrolls

# Base junk keywords — same as Shopee scraper
BASE_JUNK_KEYWORDS = [
    "sleeve", "bag", "adapter", "mouse",
    "screen protector", "protector", "skin", "stand",
    "dock", "cable", "charger", "refurbished", "pre-order", "preorder",
    "pre order", "deposit", "booking", "book now",
    "refurb", "second hand", "secondhand", "used", "compatible with",
    "for laptop", "accessories", "accessory",
    "2nd", "pre-loved", "like new", "grade a", "grade b",
    "demo", "demo set", "demo display", "demo unit", "display unit",
]

# Extra junk keywords per category — same as Shopee scraper
CATEGORY_JUNK_KEYWORDS = {
    "laptops"  : ["case", "cover", "keyboard", "for macbook", "for laptop"],
    "phones"   : ["case", "cover", "tempered glass", "glass", "for iphone", "for samsung", "demo display", "demo set"],
    "keyboards": ["keycap", "wrist rest", "lube", "demo display", "case", "cover"],
    "monitors" : ["mount", "arm", "vesa", "wall mount"],
}


# ============================================================
# RELEVANCE CHECK
# Same logic as Shopee — brand + query words must be present
# ============================================================
def is_relevant(title, query, brand, category=""):
    title_lower = title.lower()

    # Build junk list for this category
    junk_list = BASE_JUNK_KEYWORDS.copy()
    if category in CATEGORY_JUNK_KEYWORDS:
        junk_list += CATEGORY_JUNK_KEYWORDS[category]

    # Hard rule 1 — reject junk
    for junk in junk_list:
        if junk in title_lower:
            return False

    # Hard rule 2 — brand must be present
    if brand.lower() not in title_lower:
        return False

    # Hard rule 3 — ALL meaningful query words must match
    query_words = [w for w in query.lower().split() if len(w) > 1]
    for word in query_words:
        if word not in title_lower:
            return False

    return True


# ============================================================
# SETUP CHROME
# Same DrissionPage setup — real Chrome profile avoids detection
# ============================================================
def create_driver():
    options = ChromiumOptions()
    options.set_browser_path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
    options.set_argument("--window-size=1366,768")
    driver = ChromiumPage(options)
    return driver


# ============================================================
# CLEAN PRICE
# Handles "RM 1,299.00" or "RM1299" — strips everything non-numeric
# ============================================================
def clean_price(price_text):
    if not price_text:
        return None
    cleaned = re.sub(r"[^\d.]", "", price_text.replace(",", ""))
    try:
        return float(cleaned)
    except ValueError:
        return None


# ============================================================
# SAVE LAZADA PRICE TO DB
# Same structure as save_shopee_price — just platform = "lazada"
# ============================================================
def save_lazada_price(product_id, result):
    db = Session()
    try:
        product = db.query(Product).filter(Product.id == product_id).first()
        if not product:
            print(f"[DB] ERROR: Product ID {product_id} not found")
            return

        # Update product row
        product.lazada_price       = result["price"]
        product.lazada_url         = result["url"]
        product.last_price_updated = datetime.now()
        product.price_unavailable  = False

        # Log to price history
        history = PriceHistory(
            product_id = product_id,
            platform   = "lazada",
            price      = result["price"],
            timestamp  = datetime.now(),
        )
        db.add(history)
        db.commit()

        print(f"[DB] Lazada price saved — RM{result['price']} for product ID {product_id}")

    except Exception as e:
        db.rollback()
        print(f"[DB] ERROR saving price: {e}")
    finally:
        db.close()


# ============================================================
# SEARCH LAZADA
# Lazada search URL: lazada.com.my/catalog/?q=QUERY
# Cards use different selectors than Shopee
# ============================================================
def search_lazada(query, brand="", category=""):
    print(f"\n[Lazada] Searching: {query}")
    driver = create_driver()
    results = []

    try:
        search_url = f"https://www.lazada.com.my/catalog/?q={quote(query)}"
        driver.get(search_url)
        print("[Lazada] Page opened. Waiting for load...")
        time.sleep(10)   # Lazada needs a bit longer to render fully

        # Scroll down to load more results
        for i in range(SCROLL_TIMES):
            driver.run_js("window.scrollBy(0, 800);")
            time.sleep(2)

        # Lazada product card selectors — try each in order
        card_selectors = [
            "div[data-qa-locator='product-item']",
            "div.Bm3ON",     # common Lazada card class
            "div.ooOxS",     # fallback Lazada card class
        ]

        cards = []
        for selector in card_selectors:
            cards = driver.eles(f"css:{selector}")
            if len(cards) > 0:
                print(f"[Lazada] Found {len(cards)} cards using: {selector}")
                break

        if len(cards) == 0:
            print("[Lazada] WARNING: No product cards found")
            driver.get_screenshot(path=r"D:\REVALIO\Backend\lazada_debug.png")
            return results

        for card in cards:
            try:
                # Title
                title_el = card.ele("css:div.RfADt a") or card.ele("css:a.title")
                title    = title_el.text.strip() if title_el else None

                # URL — Lazada gives full href on the anchor
                url_el = card.ele("css:div.RfADt a") or card.ele("css:a.title")
                url    = url_el.attr("href") if url_el else None

                # Ensure URL is absolute
                if url and url.startswith("//"):
                    url = "https:" + url
                elif url and not url.startswith("http"):
                    url = "https://www.lazada.com.my" + url

                # Price — Lazada shows price in a span
                price_el  = card.ele("css:span.ooOxS") or card.ele("css:div.price--NVB62 span")
                price_raw = price_el.text.strip() if price_el else None

                # Fallback: scan full card text for RM amount
                if not price_raw:
                    card_text   = card.text
                    price_match = re.search(r"RM[\s]?[\d,]+\.?\d*", card_text)
                    price_raw   = price_match.group(0) if price_match else None

                price = clean_price(price_raw)

                # Image
                img_el    = card.ele("css:img")
                image_url = img_el.attr("src") if img_el else None

                if not title or not price or not url:
                    continue

                if not is_relevant(title, query, brand, category):
                    print(f"[Lazada] Skipped (irrelevant): {title}")
                    continue

                results.append({
                    "title"    : title,
                    "price"    : price,
                    "url"      : url,
                    "image_url": image_url,
                })

                if len(results) >= MAX_RESULTS:
                    break

            except Exception as e:
                print(f"[Lazada] Skipped one card: {e}")
                continue

    except Exception as e:
        print(f"[Lazada] ERROR during search: {e}")

    finally:
        driver.quit()

    print(f"[Lazada] Collected {len(results)} relevant results")
    return results


# ============================================================
# QUICK TEST
# ============================================================
if __name__ == "__main__":
    test_query      = "Apple MacBook Pro 14 M3"
    test_brand      = "Apple"
    test_category   = "laptops"
    test_product_id = 103   # change to your actual product ID in DB

    results = search_lazada(test_query, test_brand, test_category)

    if results:
        sorted_results = sorted(results, key=lambda x: x["price"])
        best = sorted_results[len(sorted_results) // 2]   # median, same as Shopee

        print(f"\n===== BEST RESULT =====")
        print(f"Title:  {best['title']}")
        print(f"Price:  RM{best['price']}")
        print(f"URL:    {best['url']}")
        print(f"Image:  {best['image_url']}")

        save_lazada_price(test_product_id, best)
    else:
        print("\n[Lazada] No relevant results found — nothing saved")
