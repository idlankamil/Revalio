import sys
sys.path.append(r"D:\REVALIO\Backend")

import re
import time
from datetime import datetime
from DrissionPage import ChromiumPage, ChromiumOptions

from models.database import Session, Product, PriceHistory


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
# SCRAPE SHOPEE PRODUCT PAGE DIRECTLY FROM SAVED URL
# Goes straight to the product URL — no searching needed
# ============================================================
def scrape_shopee_url(product_id, shopee_url):
    print(f"  [Shopee URL] Scraping: {shopee_url}")
    driver = create_driver()

    try:
        driver.get(shopee_url)
        time.sleep(8)  # Wait for Shopee to render

        # Scroll a bit to trigger lazy-loaded content
        driver.run_js("window.scrollBy(0, 400);")
        time.sleep(2)

        # --- TITLE ---
        title = None
        title_selectors = [
            "css:._44qnta",
            "css:div.product-briefing span.D6DfX6",
            "css:h1",
        ]
        for sel in title_selectors:
            el = driver.ele(sel)
            if el:
                title = el.text.strip()
                break

        # --- PRICE ---
        price = None
        price_selectors = [
            "css:div.pqTWkA",           # main price container
            "css:div._3n5NQx",          # alternate
            "css:div.product-price",
        ]
        for sel in price_selectors:
            el = driver.ele(sel)
            if el:
                price = clean_price(el.text)
                if price:
                    break

        # Fallback: scan page text for RM amount
        if not price:
            page_text   = driver.ele("css:body").text
            price_match = re.search(r"RM[\s]?[\d,]+\.?\d*", page_text)
            if price_match:
                price = clean_price(price_match.group(0))

        # --- IMAGE ---
        image_url = None
        img_el = driver.ele("css:div._3chmpu img") or driver.ele("css:div.product-image img")
        if img_el:
            image_url = img_el.attr("src")

        if not price:
            print(f"  [Shopee URL] Could not extract price — marking unavailable")
            return None

        print(f"  [Shopee URL] ✅ Price: RM{price} | Title: {title}")
        return {
            "title"    : title,
            "price"    : price,
            "url"      : shopee_url,
            "image_url": image_url,
        }

    except Exception as e:
        print(f"  [Shopee URL] ERROR: {e}")
        return None

    finally:
        driver.quit()


# ============================================================
# SAVE SHOPEE PRICE
# ============================================================
def save_shopee_price(product_id, result):
    db = Session()
    try:
        product = db.query(Product).filter(Product.id == product_id).first()
        if not product:
            print(f"  [DB] ERROR: Product ID {product_id} not found")
            return

        product.shopee_price       = result["price"]
        product.shopee_url         = result["url"]
        product.last_price_updated = datetime.now()
        product.price_unavailable  = False

        history = PriceHistory(
            product_id = product_id,
            platform   = "shopee",
            price      = result["price"],
            timestamp  = datetime.now(),
        )
        db.add(history)
        db.commit()
        print(f"  [DB] Shopee price saved — RM{result['price']} for product ID {product_id}")

    except Exception as e:
        db.rollback()
        print(f"  [DB] ERROR saving price: {e}")
    finally:
        db.close()


# ============================================================
# MARK PRODUCT AS UNAVAILABLE
# ============================================================
def mark_unavailable(product_id, platform):
    db = Session()
    try:
        product = db.query(Product).filter(Product.id == product_id).first()
        if product:
            product.price_unavailable = True
            db.commit()
            print(f"  [DB] Product ID {product_id} marked as unavailable on {platform}")
    except Exception as e:
        db.rollback()
        print(f"  [DB] ERROR marking unavailable: {e}")
    finally:
        db.close()


# ============================================================
# MAIN ENTRY — scrape one product by saved URL
# Called by scheduler
# ============================================================
def run_shopee_url_scrape(product_id, shopee_url):
    result = scrape_shopee_url(product_id, shopee_url)
    if result:
        save_shopee_price(product_id, result)
        return True
    else:
        mark_unavailable(product_id, "shopee")
        return False
