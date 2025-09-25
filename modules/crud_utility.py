import requests


def get_access_token(app_id: str, secret_key: str) -> str:
    url = "https://api-open.olsera.co.id/api/open-api/v1/id/token"

    params = {
        "app_id": app_id,
        "secret_key": secret_key,  # Ganti dengan client_id yang sesuai
        "grant_type": "secret_key",  # Ganti dengan client_secret yang sesuai
    }

    try:
        response = requests.post(url, params=params)
        response.raise_for_status()  # Akan memunculkan exception jika status bukan 2xx
        response_data = response.json()
        return response_data["access_token"]
    except requests.exceptions.HTTPError as http_err:
        print(
            f"HTTP error occurred on order detail update: {http_err} - Response: {response.text}"
        )
    except Exception as err:
        print(f"Other error occurred on order detail update: {err}")


def refresh_access_token(refresh_token: str) -> None:
    url = "https://api-open.olsera.co.id/api/open-api/v1/id/token"

    params = {
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",  # Ganti dengan client_secret yang sesuai
    }

    try:
        response = requests.post(url, params=params)
        response.raise_for_status()  # Akan memunculkan exception jika status bukan 2xx
        response_data = response.json()
        return response_data["access_token"]
    except requests.exceptions.HTTPError as http_err:
        print(
            f"HTTP error occurred on order detail update: {http_err} - Response: {response.text}"
        )
    except Exception as err:
        print(f"Other error occurred on order detail update: {err}")


def cek_kastamer(nomor_telepon: str, access_token: str) -> tuple:
    url = "https://api-open.olsera.co.id/api/open-api/v1/en/customersupplier/customer"
    params = {
        "search_column[]": "phone",
        "search_text[]": (
            "+62" + nomor_telepon[1:]
            if nomor_telepon.startswith("0")
            else nomor_telepon
        ),
    }

    headers = {"Authorization": f"Bearer {access_token}"}

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()  # Akan memunculkan exception jika status bukan 2xx
        data = response.json()

        return data["data"][0]["id"], data["data"][0]["name"] if data["data"] else None
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - Response: {response.text}")
        return None
    except Exception as err:
        print(f"Other error occurred: {err}")
        return {}


def create_order(
    order_date: str,
    access_token: str,
    customer_id: str = None,
    nomor_telepon: str = None,
    nama_kastamer: str = None,
    notes: str = "",
) -> tuple:

    url = "https://api-open.olsera.co.id/api/open-api/v1/en/order/openorder"

    if customer_id is not None:
        params = {
            "order_date": order_date,
            "currency_id": "IDR",
            "customer_id": customer_id,
            "notes": notes,
        }
    else:
        params = {
            "order_date": order_date,
            "currency_id": "IDR",
            "customer_phone": nomor_telepon,
            "customer_name": nama_kastamer,
            "customer_type_id": "195972",
            "notes": notes,
        }

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(url, json=params, headers=headers)
        response.raise_for_status()
        json_response = response.json()
        order_id = json_response["data"]["id"]
        order_no = json_response["data"]["order_no"]
        return order_id, order_no
    except requests.exceptions.HTTPError as http_err:
        print(
            f"HTTP error occurred on product inputting: {http_err} - Response: {response.text}"
        )
    except Exception as err:
        print(f"Other error occurred on product inputting: {err}")


def add_prod_to_order(
    order_id: str, product_id: str, quantity: int, access_token: str
) -> None:
    url = "https://api-open.olsera.co.id/api/open-api/v1/en/order/openorder/additem"
    params = {"order_id": order_id, "item_products": product_id, "item_qty": quantity}

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(url, json=params, headers=headers)
        response.raise_for_status()  # Akan memunculkan exception jika status bukan 2xx
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(
            f"HTTP error occurred on product inputting: {http_err} - Response: {response.text}"
        )
        return None
    except Exception as err:
        print(f"Other error occurred on product inputting: {err}")
        return None

def add_combo_to_order(
    order_id: str,
    combo_id: str,
    quantity: int,
    combo_items: list[dict],
    access_token: str
) :
    """
    Tambahkan combo ke order yang sudah ada.

    :param order_id: ID order (dari openorder list)
    :param combo_id: ID combo (dari product combo)
    :param quantity: Jumlah combo yang mau ditambahkan
    :param combo_items: List of dict, contoh:
        [
            {"id": "1137772", "product_id": "82305178", "product_variant_id": "44459921"},
            {"id": "1137770", "product_id": "82305178", "product_variant_id": "44459919"}
        ]
    :param access_token: Bearer token
    :return: dict hasil response jika sukses, None jika gagal
    """
    url = "https://api-open.olsera.co.id/api/open-api/v1/en/order/openorder/additemcombo"

    # Build form-data payload
    payload = {
        "order_id": str(order_id),
        "item_combo_id": str(combo_id),
        "item_combo_qty": str(quantity),
    }

    # Masukkan item combo dalam format array-style form-data
    for i, item in enumerate(combo_items):
        payload[f"item_combo_items[{i}][id]"] = str(item["id"])
        payload[f"item_combo_items[{i}][product_id]"] = str(item["product_id"])
        if item.get("product_variant_id"):
            payload[f"item_combo_items[{i}][product_variant_id]"] = str(item["product_variant_id"])

    headers = {
        "Authorization": f"Bearer {access_token}",
    }

    try:
        response = requests.post(url, data=payload, headers=headers)  # gunakan data=payload agar jadi form-data
        response.raise_for_status()
        return True, response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred on combo inputting: {http_err} - Response: {response.text}")
        return False, None
    except Exception as err:
        print(f"Other error occurred on combo inputting: {err}")
        return False, None

def get_product_item_df(access_token, page=1):
    url = "https://api-open.olsera.co.id/api/open-api/v1/en/product"

    params = {
        "per_page": 100,
        "page": page,
    }

    headers = {"Authorization": f"Bearer {access_token}"}

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()  # Raise error kalau bukan status 200-an
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - Response: {response.text}")
    except Exception as err:
        print(f"Other error occurred: {err}")


def get_product_combo_df(access_token, page=1):
    url = "https://api-open.olsera.co.id/api/open-api/v1/en/productcombo"

    params = {
        "per_page": 100,
        "page": page,
    }

    headers = {"Authorization": f"Bearer {access_token}"}

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()  # Raise error kalau bukan status 200-an
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - Response: {response.text}")
    except Exception as err:
        print(f"Other error occurred: {err}")

def get_product_combo_df_v2(access_token, page=1):
    url = "https://api-open.olsera.co.id/api/open-api/v1/en/productcombo-with-product"

    params = {
        "per_page": 100,
        "page": page,
    }

    headers = {"Authorization": f"Bearer {access_token}"}

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()  # Raise error kalau bukan status 200-an
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - Response: {response.text}")
    except Exception as err:
        print(f"Other error occurred: {err}")

def fetch_all_product_item(access_token):
    all_items = []
    page = 1

    while True:
        sample = get_product_item_df(access_token, page=page)
        if not sample or "data" not in sample or not sample["data"]:
            print("No more data to fetch.")
            break
        all_items.extend(sample["data"])
        print(f"Fetched page {page} with {len(sample['data'])} items.")
        page += 1

    return all_items


def fetch_all_product_combos(access_token):
    all_combos = []
    page = 1

    while True:
        sample = get_product_combo_df(access_token, page=page)
        if not sample or "data" not in sample or not sample["data"]:
            print("No more data to fetch.")
            break
        all_combos.extend(sample["data"])
        print(f"Fetched page {page} with {len(sample['data'])} items.")
        page += 1

    return all_combos

def fetch_all_product_combos_v2(access_token):
    all_combos = []
    page = 1

    while True:
        sample = get_product_combo_df_v2(access_token, page=page)
        if not sample or "data" not in sample or not sample["data"]:
            print("No more data to fetch.")
            break

        all_combos.extend(sample["data"])
        print(f"Fetched page {page} with {len(sample['data'])} items.")
        page += 1

    return all_combos

def fetch_product_item_details(item_id: str, access_token: str):
    url = "https://api-open.olsera.co.id/api/open-api/v1/en/product/detail"
    params = {
        "id": item_id,
    }

    headers = {"Authorization": f"Bearer {access_token}"}
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()  # Raise error kalau bukan status 200-an
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - Response: {response.text}")
        return response.json()
    except Exception as err:
        print(f"Other error occurred: {err}")
        return response.json()


def fetch_product_combo_details(combo_id: str, access_token: str):
    url = "https://api-open.olsera.co.id/api/open-api/v1/en/productcombo/detail"
    params = {
        "id": combo_id,
    }

    headers = {"Authorization": f"Bearer {access_token}"}

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()  # Raise error kalau bukan status 200-an
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - Response: {response.text}")
    except Exception as err:
        print(f"Other error occurred: {err}")


def fetch_open_ord_id_via_resi(resi: str, access_token: str):
    url = "https://api-open.olsera.co.id/api/open-api/v1/en/order/openorder"

    params = {
        "search_column[]": "order_no",
        "search_text[]": resi,
    }

    headers = {"Authorization": f"Bearer {access_token}"}

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()  # Raise error kalau bukan status 200-an
        data = response.json()
        if data and "data" in data:
            return data["data"][0]["id"]
        else:
            print("No order found for the given resi.")
            return None
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - Response: {response.text}")


def fetch_close_ord_id_via_resi(resi: str, access_token: str):
    url = "	https://api-open.olsera.co.id/api/open-api/v1/en/order/closeorder"

    params = {
        "search_column[]": "order_no",
        "search_text[]": resi,
    }

    headers = {"Authorization": f"Bearer {access_token}"}

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()  # Raise error kalau bukan status 200-an
        data = response.json()
        if data and "data" in data:
            return data["data"][0]["id"]
        else:
            print("No order found for the given resi.")
            return None
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - Response: {response.text}")
        return None


def fetch_order_details(order_id: str, access_token: str):
    url = "https://api-open.olsera.co.id/api/open-api/v1/en/order/openorder/detail"
    params = {
        "id": order_id,
    }

    headers = {"Authorization": f"Bearer {access_token}"}

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()  # Raise error kalau bukan status 200-an
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - Response: {response.text}")
    except Exception as err:
        print(f"Other error occurred: {err}")

def fetch_open_order_table(start_date: str, end_date: str, access_token: str):
    url = "https://api-open.olsera.co.id/api/open-api/v1/en/order/openorder"

    params = {
        "start_date": start_date,
        "end_date": end_date,
    }

    headers = {"Authorization": f"Bearer {access_token}"}

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()  # Raise error kalau bukan status 200-an
        data = response.json()
        if data and "data" in data:
            return data["data"]
        else:
            print("No order found for the given resi.")
            return None
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - Response: {response.text}")

def update_order_detail(
    order_id: str,
    id: str,
    disc: int,
    note: str,
    price: str,
    qty: int,
    access_token: str,
) -> None:
    url = (
        "https://api-open.olsera.co.id/api/open-api/v1/en/order/openorder/updatedetail"
    )
    params = {
        "order_id": order_id,
        "id": id,
        "discount": disc,
        "note": note,
        "price": price,
        "qty": qty,
    }

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(url, json=params, headers=headers)
        response.raise_for_status()  # Akan memunculkan exception jika status bukan 2xx
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(
            f"HTTP error occurred on order detail update: {http_err} - Response: {response.text}"
        )
    except Exception as err:
        print(f"Other error occurred on order detail update: {err}")


def update_order_attr(order_id: str, name: str, value: str, access_token: str) -> None:
    url = "https://api-open.olsera.co.id/api/open-api/v1/en/order/openorder/updateattr"
    params = {"order_id": order_id, "name": name, "value": value}

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(url, json=params, headers=headers)
        response.raise_for_status()  # Akan memunculkan exception jika status bukan 2xx
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(
            f"HTTP error occurred on order attribute update: {http_err} - Response: {response.text}"
        )
    except Exception as err:
        print(f"Other error occurred on order attribute update: {err}")


def list_payment_modes(order_id: str, access_token: str) -> list:
    url = "https://api-open.olsera.co.id/api/open-api/v1/en/order/openorder/editpayment"

    params = {"order_id": order_id}

    headers = {"Authorization": f"Bearer {access_token}"}

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()  # Akan memunculkan exception jika status bukan 2xx
        data = response.json()
        payment_modes = data["data"]["payment_modes"]
        return payment_modes

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - Response: {response.text}")
    except Exception as err:
        print(f"Other error occurred: {err}")
        return None


def update_payment(
    order_id: str,
    payment_amount: str,
    payment_date: str,
    payment_mode_id: str,
    access_token: str,
    payment_payee: str = "",
    payment_seq: str = "0",
    payment_currency_id: str = "IDR",
):
    url = (
        "https://api-open.olsera.co.id/api/open-api/v1/en/order/openorder/updatepayment"
    )
    params = {
        "order_id": order_id,
        "payment_amount": payment_amount,
        "payment_date": payment_date,
        "payment_mode_id": payment_mode_id,
        "payment_payee": payment_payee,
        "payment_seq": payment_seq,
        "payment_currency_id": payment_currency_id,
    }

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(url, json=params, headers=headers)
        response.raise_for_status()  # Akan memunculkan exception jika status bukan 2xx
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(
            f"HTTP error occurred on order detail update: {http_err} - Response: {response.text}"
        )
    except Exception as err:
        print(f"Other error occurred on order detail update: {err}")


def update_status(order_id: str, status: str, access_token: str) -> None:
    url = (
        "https://api-open.olsera.co.id/api/open-api/v1/en/order/openorder/updatestatus"
    )
    params = {"order_id": order_id, "status": status}

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(url, json=params, headers=headers)
        response.raise_for_status()  # Akan memunculkan exception jika status bukan 2xx
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(
            f"HTTP error occurred on order status update: {http_err} - Response: {response.text}"
        )
    except Exception as err:
        print(f"Other error occurred on order status update: {err}")


def search_ongkir_related_product(keywords: str, access_token: str) -> tuple:
    url = "https://api-open.olsera.co.id/api/open-api/v1/en/product"
    params = {
        "search_column[]": "name",
        "search_text[]": keywords,
    }

    headers = {"Authorization": f"Bearer {access_token}"}

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()  # Akan memunculkan exception jika status bukan 2xx
        data = response.json()

        return data["data"][0]["id"]
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - Response: {response.text}")
        return None
    except Exception as err:
        print(f"Other error occurred: {err}")
        return {}


def void_order(order_no: str, access_token: str):
    order_id = fetch_open_ord_id_via_resi(order_no, access_token)

    if order_id is None:
        order_id = fetch_close_ord_id_via_resi(order_no, access_token)

    if order_id is None:
        print(f"Order ID not found for order number: {order_no}")
        return False

    update_status(order_id, "X", access_token)
    return True


def cetak_struk(order_no: str, phone: str) -> str:
    url = f"https://invoice.olsera.co.id/pos-receipt?lang=id&store=kulkasbabe&order_no={order_no}"
    print(url)
    return url

def _add_combo_to_order(order_id:str, combo_id:str,quantity:int, combo_items:list,access_token:str) : 
    url = "https://api-open.olsera.co.id/api/open-api/v1/en/order/openorder/additemcombo"

    params = {
        "order_id" : order_id,
        "item_combo_id" : combo_id,
        "item_combo_qty" : quantity,
        "item_combo_items" : combo_items,
    }

    headers = {
        "Authorization" : f"Bearer {access_token}",
        "Content-Type" : "application/json"
    }
    try : 
        response = requests.post(url,json=params,headers=headers)
        response.raise_for_status()
        return True, response.json()
    except requests.exceptions.HTTPError as http_err : 
        print(f"HTTP error occurred on order detail update: {http_err} - Response: {response.text}")
        return False, None
