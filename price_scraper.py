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
SCROLL_TIMES = 3

# Base junk keywords that apply to ALL categories
BASE_JUNK_KEYWORDS = [
    "sleeve", "bag", "hub", "adapter", "mouse",
    "screen protector", "protector", "skin", "stand",
    "dock", "cable", "charger", "refurbished", "pre-order", "preorder",
    "pre order", "deposit", "booking", "book now",
    "refurb", "second hand", "secondhand", "used", "compatible with",
    "for laptop", "accessories", "accessory",
    "2nd", "pre-loved", "like new", "grade a", "grade b",
    "demo", "demo set", "demo display", "demo unit", "display unit",
]

# Extra junk keywords per category
CATEGORY_JUNK_KEYWORDS = {
    "laptops"  : ["case", "cover", "keyboard", "for macbook", "for laptop"],
    "phones"   : ["case", "cover", "tempered glass", "glass", "for iphone", "for samsung", "demo display", "demo set"],
    "keyboards": ["keycap", "wrist rest", "lube", "v3", "demo display", "case", "cover"],
    "monitors" : ["mount", "arm", "vesa", "wall mount"],
}


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
# ============================================================
def create_driver():
    options = ChromiumOptions()
    options.set_browser_path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
    options.set_argument("--window-size=1366,768")
    driver = ChromiumPage(options)
    return driver


# ============================================================
# CLEAN PRICE
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
# SAVE SHOPEE PRICE TO DB
# Saves best result to Product row + logs to PriceHistory
# ============================================================
def save_shopee_price(product_id, result):
    db = Session()
    try:
        product = db.query(Product).filter(Product.id == product_id).first()
        if not product:
            print(f"[DB] ERROR: Product ID {product_id} not found")
            return

        # Update product row
        product.shopee_price       = result["price"]
        product.shopee_url         = result["url"]
        product.last_price_updated = datetime.now()
        product.price_unavailable  = False

        # Log to price history
        history = PriceHistory(
            product_id = product_id,
            platform   = "shopee",
            price      = result["price"],
            timestamp  = datetime.now(),
        )
        db.add(history)
        db.commit()

        print(f"[DB] Shopee price saved — RM{result['price']} for product ID {product_id}")

    except Exception as e:
        db.rollback()
        print(f"[DB] ERROR saving price: {e}")
    finally:
        db.close()


# ============================================================
# SEARCH SHOPEE
# ============================================================
def search_shopee(query, brand="", category=""):
    print(f"\n[Shopee] Searching: {query}")
    driver = create_driver()
    results = []

    try:
        search_url = f"https://shopee.com.my/search?keyword={quote(query)}"
        driver.get(search_url)
        print("[Shopee] Page opened. Waiting for load...")
        time.sleep(8)

        for i in range(SCROLL_TIMES):
            driver.run_js("window.scrollBy(0, 800);")
            time.sleep(2)

        card_selectors = [
            "li.shopee-search-item-result__item",
            "div[data-sqe='item']",
            "div.col-xs-2-4",
        ]

        cards = []
        for selector in card_selectors:
            cards = driver.eles(f"css:{selector}")
            if len(cards) > 0:
                print(f"[Shopee] Found {len(cards)} cards using: {selector}")
                break

        if len(cards) == 0:
            print("[Shopee] WARNING: No product cards found")
            driver.get_screenshot(path=r"D:\REVALIO\Backend\shopee_debug.png")
            return results

        for card in cards:
            try:
                link_el   = card.ele("css:a.contents")
                url       = link_el.attr("href") if link_el else None

                title_el  = card.ele("css:div.whitespace-normal.line-clamp-2")
                title     = title_el.text.strip() if title_el else None

                card_text   = card.text
                price_match = re.search(r"[\d,]+\.?\d*", card_text[card_text.find("RM"):]) if "RM" in card_text else None
                price_raw   = price_match.group(0) if price_match else None
                price       = clean_price(price_raw)

                img_el    = card.ele("css:div.relative.z-0 img")
                image_url = img_el.attr("src") if img_el else None

                if not title or not price or not url:
                    continue

                if not is_relevant(title, query, brand, category):
                    print(f"[Shopee] Skipped (irrelevant): {title}")
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
                print(f"[Shopee] Skipped one card: {e}")
                continue

    except Exception as e:
        print(f"[Shopee] ERROR during search: {e}")

    finally:
        driver.quit()

    print(f"[Shopee] Collected {len(results)} relevant results")
    return results


# ============================================================
# SCRAPE PRICE FROM SAVED URL
# ============================================================
def scrape_price_from_url(url):
    print(f"\n[Shopee] Scraping price from: {url}")
    driver = create_driver()

    try:
        driver.get(url)
        time.sleep(5)

        page_text   = driver.ele("tag:body").text
        price_match = re.search(r"RM[\s]?[\d,]+\.?\d*", page_text)

        if price_match:
            price = clean_price(price_match.group(0))
            print(f"[Shopee] Price found: RM{price}")
            return price

        print("[Shopee] Price not found on page")
        return None

    except Exception as e:
        print(f"[Shopee] ERROR scraping price: {e}")
        return None

    finally:
        driver.quit()


# ============================================================
# QUICK TEST
# ============================================================
if __name__ == "__main__":
    test_query      = "Apple MacBook Pro 14 M3"
    test_brand      = "Apple"
    test_product_id = 103   # change this to your actual product ID in DB

    results = search_shopee(test_query, test_brand)

    if results:
        # Pick cheapest result
        best = min(results, key=lambda x: x["price"])

        print(f"\n===== BEST RESULT =====")
        print(f"Title:  {best['title']}")
        print(f"Price:  RM{best['price']}")
        print(f"URL:    {best['url']}")
        print(f"Image:  {best['image_url']}")

        # Save to DB
        save_shopee_price(test_product_id, best)
    else:
        print("\n[Shopee] No relevant results found — nothing saved")