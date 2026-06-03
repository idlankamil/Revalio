import sys
sys.path.append(r"D:\REVALIO\Backend")

import time
import logging
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from models.database import Session, Product
from services.shopee_url_scraper import run_shopee_url_scrape
from services.lazada_url_scraper  import run_lazada_url_scrape


# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(r"D:\REVALIO\Backend\logs\scheduler.log"),
        logging.StreamHandler(),   # also print to console
    ]
)
log = logging.getLogger(__name__)

DELAY_BETWEEN_PRODUCTS = 15  # seconds between each product to avoid rate limits


# ============================================================
# FETCH ALL PRODUCTS WITH SAVED URLS FROM DB
# Only scrapes products that already have a URL saved
# ============================================================
def get_products_with_urls():
    db = Session()
    try:
        products = db.query(Product).filter(Product.is_selected == True).all()
        result = []
        for p in products:
            result.append({
                "id"         : p.id,
                "name"       : p.name,
                "shopee_url" : p.shopee_url,
                "lazada_url" : p.lazada_url,
            })
        return result
    finally:
        db.close()


# ============================================================
# SCRAPE JOB — runs weekly
# Loops all selected products, hits saved URLs directly
# ============================================================
def weekly_scrape_job():
    log.info("=" * 60)
    log.info(f"REVALIO Weekly Scrape Started — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info("=" * 60)

    products = get_products_with_urls()
    log.info(f"Total products to scrape: {len(products)}")

    shopee_success = 0
    shopee_failed  = []
    lazada_success = 0
    lazada_failed  = []

    for i, product in enumerate(products, 1):
        log.info(f"\n[{i}/{len(products)}] {product['name']}")
        log.info("-" * 40)

        # --- SHOPEE ---
        if product["shopee_url"]:
            ok = run_shopee_url_scrape(product["id"], product["shopee_url"])
            if ok:
                shopee_success += 1
            else:
                shopee_failed.append(product["name"])
        else:
            log.info("  [Shopee] No saved URL — skipping")

        time.sleep(DELAY_BETWEEN_PRODUCTS)

        # --- LAZADA ---
        if product["lazada_url"]:
            ok = run_lazada_url_scrape(product["id"], product["lazada_url"])
            if ok:
                lazada_success += 1
            else:
                lazada_failed.append(product["name"])
        else:
            log.info("  [Lazada] No saved URL — skipping")

        # Delay between products (skip delay after last one)
        if i < len(products):
            log.info(f"  Waiting {DELAY_BETWEEN_PRODUCTS}s before next product...")
            time.sleep(DELAY_BETWEEN_PRODUCTS)

    # --- SUMMARY ---
    log.info(f"\n{'='*60}")
    log.info(f"WEEKLY SCRAPE COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info(f"Shopee ✅ {shopee_success} success  ❌ {len(shopee_failed)} failed")
    log.info(f"Lazada ✅ {lazada_success} success  ❌ {len(lazada_failed)} failed")
    if shopee_failed:
        for name in shopee_failed:
            log.info(f"  Shopee failed: {name}")
    if lazada_failed:
        for name in lazada_failed:
            log.info(f"  Lazada failed: {name}")
    log.info("=" * 60)


# ============================================================
# SCHEDULER SETUP
# Runs every Monday at 3:00 AM
# Change day_of_week and hour to your preference:
#   day_of_week: mon, tue, wed, thu, fri, sat, sun
#   hour: 0-23
# ============================================================
scheduler = BackgroundScheduler()

scheduler.add_job(
    func    = weekly_scrape_job,
    trigger = CronTrigger(day_of_week="mon", hour=3, minute=0),
    id      = "weekly_scrape",
    name    = "Revalio Weekly Price Scrape",
    replace_existing = True,
)


# ============================================================
# START / STOP — called from FastAPI lifespan
# ============================================================
def start_scheduler():
    if not scheduler.running:
        scheduler.start()
        log.info("✅ Revalio scheduler started — runs every Monday at 3:00 AM")


def stop_scheduler():
    if scheduler.running:
        scheduler.shutdown()
        log.info("Revalio scheduler stopped")


# ============================================================
# MANUAL TRIGGER — called from FastAPI endpoint
# POST /admin/run-scraper
# ============================================================
def trigger_now():
    log.info("🔧 Manual scrape triggered via API")
    weekly_scrape_job()
