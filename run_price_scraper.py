import sys
sys.path.append(r"D:\REVALIO\Backend")

import time
from models.database import Session, Product
from services.price_scraper import search_shopee, save_shopee_price


# ============================================================
# SELECTED PRODUCTS TO SCRAPE
# Only is_selected = True products
# ============================================================

SELECTED_PRODUCTS = [
    # Laptops
    {"id": 102, "name": "Apple MacBook Air M3 13-inch",  "brand": "Apple",       "category": "laptops"},
    {"id": 103, "name": "Apple MacBook Pro 14 M3",       "brand": "Apple",       "category": "laptops"},
    {"id": 104, "name": "ASUS ROG Zephyrus G14 2024",    "brand": "ASUS",        "category": "laptops"},
    {"id": 110, "name": "ASUS ZenBook 14 OLED 2024",     "brand": "ASUS",        "category": "laptops"},
    {"id": 111, "name": "Acer Swift X 14 2024",          "brand": "Acer",        "category": "laptops"},

    # Phones
    {"id": 113, "name": "iPhone 15 Pro Max",             "brand": "Apple",       "category": "phones"},
    {"id": 114, "name": "Samsung Galaxy S24 Ultra",      "brand": "Samsung",     "category": "phones"},
    {"id": 118, "name": "Samsung Galaxy S24",            "brand": "Samsung",     "category": "phones"},
    {"id": 148, "name": "iPhone 17 Pro",                 "brand": "Apple",       "category": "phones"},
    {"id": 149, "name": "Google Pixel 10 Pro",           "brand": "Google",      "category": "phones"},

    # Keyboards
    {"id": 125, "name": "Logitech MX Keys S",            "brand": "Logitech",    "category": "keyboards"},
    {"id": 127, "name": "SteelSeries Apex Pro TKL",      "brand": "SteelSeries", "category": "keyboards"},
    {"id": 131, "name": "Ducky One 3",                   "brand": "Ducky",       "category": "keyboards"},
    {"id": 132, "name": "NuPhy Air75",                   "brand": "NuPhy",       "category": "keyboards"},
    {"id": 133, "name": "Glorious GMMK 2",               "brand": "Glorious",    "category": "keyboards"},

    # Monitors
    {"id": 136, "name": "LG UltraGear 27GS60F",          "brand": "LG",          "category": "monitors"},
    {"id": 137, "name": "Samsung Odyssey G7 32",         "brand": "Samsung",     "category": "monitors"},
    {"id": 140, "name": "BenQ PD2725U",                  "brand": "BenQ",        "category": "monitors"},
    {"id": 141, "name": "Gigabyte M27Q",                 "brand": "Gigabyte",    "category": "monitors"},
    {"id": 147, "name": "Dell UltraSharp U2723QE",       "brand": "Dell",        "category": "monitors"},
]


DELAY_BETWEEN_PRODUCTS = 15  # seconds — give Shopee time to breathe between searches


# ============================================================
# MAIN BATCH RUN
# ============================================================
def run_all():
    print(f"\n{'='*60}")
    print(f"REVALIO — Shopee Price Scraper (Batch Run)")
    print(f"Total products: {len(SELECTED_PRODUCTS)}")
    print(f"{'='*60}\n")

    success = 0
    failed  = []

    for i, product in enumerate(SELECTED_PRODUCTS, 1):
        print(f"\n[{i}/{len(SELECTED_PRODUCTS)}] {product['name']}")
        print("-" * 40)

        try:
            results = search_shopee(product["name"], product["brand"], product["category"])

            if results:
                sorted_results = sorted(results, key=lambda x: x["price"])
                best = sorted_results[len(sorted_results) // 2]
                save_shopee_price(product["id"], best)
                success += 1
                print(f"✅ Saved — RM{best['price']}")
            else:
                print(f"⚠️  No results found — marking as unavailable")
                db = Session()
                try:
                    p = db.query(Product).filter(Product.id == product["id"]).first()
                    if p:
                        p.price_unavailable = True
                        db.commit()
                finally:
                    db.close()
                failed.append(product["name"])

        except Exception as e:
            print(f"❌ ERROR: {e}")
            failed.append(product["name"])

        if i < len(SELECTED_PRODUCTS):
            print(f"[Delay] Waiting {DELAY_BETWEEN_PRODUCTS}s before next product...")
            time.sleep(DELAY_BETWEEN_PRODUCTS)

    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE")
    print(f"✅ Success: {success}/{len(SELECTED_PRODUCTS)}")
    if failed:
        print(f"❌ Failed:  {len(failed)}")
        for name in failed:
            print(f"   - {name}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_all()