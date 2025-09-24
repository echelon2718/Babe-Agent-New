import schedule
import time
import threading
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import os
import json

from modules.crud_utility import fetch_all_product_combos, get_access_token

load_dotenv()

app_id = os.getenv("APP_ID")
secret_key = os.getenv("SECRET_KEY")

print(f"App ID: {app_id}")
print(f"Secret: {secret_key}")

def job():
    print(f"[{datetime.now()}] Starting token fetch job.")
    access_token = get_access_token(app_id, secret_key)
    print(access_token)
    try:
        with open("token_cache.json", "w") as f:
            json.dump({"access_token": access_token, "timestamp": datetime.now().isoformat()}, f)
        print(f"[{datetime.now()}] Access token saved to token_cache.json.")
    except Exception as e:
        print(f"[{datetime.now()}] Failed to save token: {e}")

schedule.every(240).minutes.do(job)

# Fungsi loop yang terus berjalan
def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(1)

# Jalankan worker-nya
if __name__ == "__main__":
    print("Worker dimulai. Menunggu eksekusi setiap 240 menit.")
    job()  # Run immediately on startup
    threading.Thread(target=run_scheduler).start()