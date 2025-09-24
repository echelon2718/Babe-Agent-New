import schedule
import time
import threading
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import os
import json

from modules.crud_utility import fetch_all_product_combos, fetch_all_product_item

load_dotenv()

app_id = os.getenv("APP_ID")
secret_key = os.getenv("SECRET_KEY")

print(f"App ID: {app_id}")
print(f"Secret: {secret_key}")

def job():
    print(f"[{datetime.now()}] Starting combo fetch job.")
    with open("token_cache.json", "r") as file:
        token_data = json.load(file)

    access_token = token_data.get("access_token", "")

    combos = fetch_all_product_combos(access_token)
    pd.DataFrame(combos).to_csv("product_combos.csv", index=False)
    print(f"[{datetime.now()}] Fetched {len(combos)} and saved product combos.")

    items = fetch_all_product_item(access_token)
    pd.DataFrame(items).to_csv("product_items.csv", index=False)
    print(f"[{datetime.now()}] Fetched {len(items)} and saved product items.")

schedule.every(5).minutes.do(job)

# Fungsi loop yang terus berjalan
def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(1)

# Jalankan worker-nya
if __name__ == "__main__":
    print("Worker dimulai. Menunggu eksekusi setiap 5 menit.")
    job()  # Run immediately on startup
    threading.Thread(target=run_scheduler).start()