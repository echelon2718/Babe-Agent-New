import requests
import time
import json


def update_status(order_id: str, status: str, access_token: str) -> None:
    url = "https://api-open.olsera.co.id/api/open-api/v1/en/order/openorder/updatestatus"
    params = {"order_id": order_id, "status": status}

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    while True:
        try:
            response = requests.post(url, json=params, headers=headers)
            if response.status_code == 429:
                print(f"[429] Rate limited on order_id={order_id}, waiting 20s...")
                time.sleep(20)
                continue
            response.raise_for_status()
            print(f"[SUCCESS] Voided order_id={order_id}")
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            print(f"[HTTP ERROR] order_id={order_id}: {http_err} - {response.text}")
            break
        except Exception as err:
            print(f"[OTHER ERROR] order_id={order_id}: {err}")
            break


def get_order_ids_from_log(log_path: str) -> list[str]:
    order_ids = []
    with open(log_path, "r") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 2:
                order_ids.append(parts[1])  # ambil kolom ke-2 (id)
    print(f"[INFO] Collected {len(order_ids)} order IDs from log")
    return order_ids


def clear_log_file(log_path: str):
    with open(log_path, "w") as f:
        f.truncate(0)  # kosongkan isi file
    print(f"[SUCCESS] Cleared log file: {log_path}")


def void_orders_from_log(log_path: str, outlet_id: int):
    with open("./storage/app/token_cache.json", "r") as file:
            token_data = json.load(file)
    token = token_data.get("access_token",None)
    order_ids = get_order_ids_from_log(log_path)

    for order_id in order_ids:
        update_status(order_id=order_id, status="X", access_token=token)
        time.sleep(0.7)  # jaga-jaga supaya tidak 429

    # setelah selesai -> clear log
    clear_log_file(log_path)


if __name__ == "__main__":
    outlet_id = 1  # ganti sesuai kebutuhan
    log_file = "log/order.log"  # ganti sesuai nama file log Anda
    void_orders_from_log(log_file, outlet_id)
