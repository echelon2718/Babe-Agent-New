from modules.crud_utility import *
from modules.maps_utility import *
from collections import defaultdict
import ast
import json
import requests
import os
import pandas as pd


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


class StrukMaker:
    def __init__(
        self,
        combo_df="./product_combos.csv",
        item_df="./product_items.csv",
        access_token_dir="./token_cache.json",
    ):
        self.combo_df = combo_df
        self.item_df = item_df
        self.access_token_dir = access_token_dir
        self.variant_priority_order = ["C", "P", "L"]

    def receive_item_and_choose_variant(self, idx, access_token):
        item_df = pd.read_csv(self.item_df)
        selected_index = item_df[item_df["id"] == idx].reset_index(drop=True)

        if selected_index.empty:
            return (
                None,
                f"Produk dengan id {idx} tidak ditemukan. Mungkin terjadi perubahan pada database atau item sudah tidak tersedia.",
            )

        # Inisialisasi prodvar_id
        prodvar_id = None

        if selected_index.at[0, "variants"] != "[]":
            try:
                variants = ast.literal_eval(selected_index.at[0, "variants"])
                variants_df = pd.DataFrame(variants)

            except Exception as e:
                return None, "Gagal memproses varian produk."

            variants_ready = variants_df[variants_df["stock_qty"] > 0]

            for prefix in self.variant_priority_order:
                sel = variants_ready[variants_ready["name"].str.startswith(prefix)]
                if not sel.empty:
                    variant_id = sel["id"].iloc[0]
                    prodvar_id = f"{idx}|{variant_id}"
                    break

            if prodvar_id is None:
                return (
                    None,
                    f"Tidak ada varian yang sesuai untuk produk dengan id {idx}.",
                )

            return (
                prodvar_id,
                f"Produk dengan id {idx} memiliki varian {variant_id}, sehingga {prodvar_id} adalah id yang sesuai",
            )

        # Jika tidak ada varian, gunakan id produk itu sendiri
        return idx, f"Produk dengan id {idx} tidak memiliki varian."

    def add_item_to_cart(self, raw_cart, cart, access_token):
        idx, msg = self.receive_item_and_choose_variant(raw_cart["id"], access_token)

        if idx is None:
            return False, msg

        cart.append(
            {
                "prod_id": raw_cart["id"],
                "prodvar_id": idx,
                "name": raw_cart["name"],
                "qty": raw_cart["qty"],
                "price": fetch_product_item_details(
                    raw_cart["id"], access_token=access_token
                ).get("data", {})["sell_price_pos"],
                "disc": 0,
            }
        )

        return True, msg

    def unpack_combo_and_add_to_cart(self, raw_cart, cart, access_token):
        try:
            combo_details = fetch_product_combo_details(raw_cart["id"], access_token)
            combo_items = combo_details["data"]["items"]["data"]
        except Exception as e:
            return False, "Gagal ambil detail paket."

        total_harga_normal = 0.0

        # Tambah tiap item
        for item in combo_items:
            product_id = item.get("product_id")
            var_id = item.get("product_variant_id")
            qty_total = item.get("qty", 0) * raw_cart["qty"]

            # Fetch detail item untuk cek stok & harga
            try:
                item_details = fetch_product_item_details(
                    product_id, access_token=access_token
                )
                data = item_details.get("data", {})
            except Exception as e:
                return False, f"Gagal ambil data produk {product_id}. Detail error: {e}"

            # Cek variant dan harga
            if not data.get("variant"):
                idx = f"{product_id}"
                harga = float(data.get("sell_price_pos", 0))
            else:
                variants = data.get("variant", [])
                item_df = pd.DataFrame(variants)
                harga = (
                    float(item_df["sell_price_pos"].iloc[0])
                    if not item_df.empty
                    else 0.0
                )
                variants_ready = item_df[item_df["stock_qty"] >= qty_total]

                if variants_ready.empty:
                    return (
                        False,
                        f"Maaf, stok tidak cukup untuk {item.get('product_name')}.",
                    )

                # Pilih varian: prioritas P, L, C, X
                chosen_variant_id = None

                if var_id is not None:
                    # Cek jika varian permintaan tersedia
                    row_req = item_df[item_df["id"] == var_id]
                    if not row_req.empty and row_req.iloc[0]["stock_qty"] >= qty_total:
                        chosen_variant_id = var_id

                if chosen_variant_id is None:
                    for prefix in self.variant_priority_order:
                        sel = variants_ready[
                            variants_ready["name"].str.startswith(prefix)
                        ]
                        if not sel.empty:
                            chosen_variant_id = sel["id"].iloc[0]
                            break

                if chosen_variant_id is None:
                    return (
                        False,
                        f"Stok varian tidak mencukupi untuk {item.get('product_name')}.",
                    )

                idx = f"{product_id}|{chosen_variant_id}"

            total_harga_normal += harga * qty_total

            cart.append(
                {
                    "prod_id": product_id,
                    "prodvar_id": idx,
                    "name": item["product_name"],
                    "qty": qty_total,
                    "price": fetch_product_item_details(
                        product_id, access_token=access_token
                    ).get("data", {})["sell_price_pos"],
                    "disc": 0,
                }
            )

        bundle_entry = cart[
            -len(combo_items) :
        ]  # Ambil entry terakhir yang merupakan paket
        harga_paket = float(combo_details["data"]["sell_price_pos"]) * raw_cart["qty"]

        for item in bundle_entry:
            price = float(item["price"])
            item["disc"] = (
                price
                * (total_harga_normal - harga_paket)
                / total_harga_normal
                * float(item["qty"])
                if total_harga_normal > 0
                else 0.0
            )

        return True, "Paket berhasil ditambahkan ke keranjang."

    def aggregate_cart_by_prodvar(self, cart: list) -> list:
        # Aggregation by prodvar_id
        agg_by_prodvar = defaultdict(
            lambda: {"prodvar_id": None, "name": None, "qty": 0, "disc": 0.0}
        )

        for item in cart:
            pvar = str(item["prodvar_id"])
            if agg_by_prodvar[pvar]["prodvar_id"] is None:
                agg_by_prodvar[pvar]["prodvar_id"] = pvar
                agg_by_prodvar[pvar]["name"] = item["name"]
            agg_by_prodvar[pvar]["qty"] += item["qty"]
            agg_by_prodvar[pvar]["disc"] += float(item["disc"])

        # Convert to list and display
        aggregated_by_prodvar = list(agg_by_prodvar.values())
        return aggregated_by_prodvar

    def move_cart_to_order(self, cart: list, order_id: str, access_token: str):
        """
        Pindahkan semua item dalam cart ke order dengan ID order_id.
        Mengembalikan True jika sukses, atau string pesan error jika gagal.
        """
        if not cart:
            return False, "Cart kosong, tidak ada item untuk dipindahkan."

        for item in cart:
            prodvar_id = item["prodvar_id"]
            qty = item["qty"]
            try:
                resp = add_prod_to_order(
                    order_id, prodvar_id, qty, access_token=access_token
                )
                if resp is None:
                    return (
                        False,
                        "Gagal menambahkan produk ke order, produk habis. Struk di-voidkan.",
                    )

            except requests.exceptions.HTTPError as http_err:
                return (
                    False,
                    "Ada kesalahan HTTP saat memasukkan produk ke order. Struk di-voidkan.",
                )

            except Exception as e:
                return (
                    False,
                    f"Ada kesalahan saat memasukkan produk ke order. Error: {e}",
                )

        # Setelah itu, tambahkan diskon untuk tiap item
        try:
            ord_detail = fetch_order_details(
                order_id=order_id, access_token=access_token
            )
            orders = ord_detail["data"]["orderitems"]

        except Exception as e:
            return False, "Gagal update diskon paket."

        # Hitung dan update tiap item
        for idx, item in enumerate(orders):
            item_id = item["id"]
            item_qty = item["qty"]
            item_disc = cart[idx]["disc"] if idx < len(cart) else 0.0
            item_price = int(float(item.get("fprice", 0).replace(".", "")))

            try:
                update_order_detail(
                    order_id=str(order_id),
                    id=str(item_id),
                    disc=str(item_disc),
                    price=str(item_price),
                    qty=str(item_qty),
                    note="Promo Paket",
                    access_token=access_token,
                )
            except Exception as e:
                return False, "Gagal update detail paket di order."
        return (
            True,
            "Semua item berhasil dipindahkan ke order dan diskon paket telah diperbarui.",
        )

    def handle_order(self, raw_cart):
        with open(self.access_token_dir, "r") as f:
            access_token = json.load(f).get("access_token", None)

        # 1. Create order
        customer = cek_kastamer(
            nomor_telepon=raw_cart["telepon"], access_token=access_token
        )

        cust_telp = raw_cart["telepon"]
        cust_name = raw_cart["name"]

        today_str = datetime.now().strftime("%Y-%m-%d")

        try:
            print("Membuat order...")
            order_id, order_no = create_order(
                order_date=today_str,
                customer_id=customer[0] if customer else None,
                nama_kastamer=cust_name,
                nomor_telepon=cust_telp,
                notes="Buat order web",
                access_token=access_token,
            )
            print(f"Order berhasil dibuat dengan ID: {order_id} dan Nomor: {order_no}")
        except Exception as e:
            return None, None, f"Gagal membuat order: {e}"

        ongkir_name = distance_cost_rule(
            raw_cart["jarak"], raw_cart["is_free_ongkir"]
        )  # INI BELUM SELESAI
        id_ongkir = search_ongkir_related_product(ongkir_name, access_token)

        if ongkir_name != "Gratis Ongkir":
            add_prod_to_order(
                order_id=order_id,
                product_id=id_ongkir,
                quantity=1,
                access_token=access_token,
            )

        else:
            pass

        # 2. Masukkan item ke cart
        cart = []
        for item in raw_cart["cells"]:
            if item["type"] == "combo":
                success, msg = self.unpack_combo_and_add_to_cart(
                    item, cart, access_token
                )
                if not success:
                    update_status(order_id, "X", access_token)
                    return (
                        None,
                        None,
                        "Struk di voidkan, order gagal dibuat karena masalah dengan paket: "
                        + msg,
                    )
            elif item["type"] == "item":
                success, msg = self.add_item_to_cart(item, cart, access_token)
                if not success:
                    update_status(order_id, "X", access_token)
                    return (
                        None,
                        None,
                        "Struk divoidkan, order gagal dibuat karena masalah dengan item: "
                        + msg,
                    )

        # 3. Aggregate by prodvar_id
        aggregated_cart = self.aggregate_cart_by_prodvar(cart)

        # 4. Move aggregated cart to order
        success, msg = self.move_cart_to_order(aggregated_cart, order_id, access_token)
        if not success:
            return None, None, msg

        # 5. Update ongkir
        return order_id, order_no, "Order berhasil dibuat dan ongkir telah diperbarui."
