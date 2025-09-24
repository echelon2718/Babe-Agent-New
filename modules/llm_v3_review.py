from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
import google.generativeai as genai
import re
import json
import pandas as pd
import requests
import ast
from modules.crud_utility import (
    add_prod_to_order,
    update_payment,
    fetch_product_combo_details,
    fetch_order_details,
    fetch_open_order_table,
    update_order_detail,
    cek_kastamer,
    create_order,
    list_payment_modes,
    update_status,
    cetak_struk,
    fetch_product_item_details,
    update_order_attr,
    void_order,
)
from modules.maps_utility import (
    resolve_maps_shortlink,
    get_travel_distance,
    get_fastest_route_details,
    address_to_latlng,
    distance_cost_rule,
    is_free_delivery,
    estimasi_tiba,
)
from datetime import datetime, timedelta
import logging
from typing import Optional, Tuple, Dict, Any
import time
from collections import defaultdict

nltk.download("punkt")
nltk.download("punkt_tab")

logger = logging.getLogger(__name__)

payment_dict = {
    "Cash": 0,
    "BRI": 1,
    "Hutang": 2,
    "BCA": 3,
    "QRIS": 4,
}

with open("./modules/prompts/reconfirm_translator_prompt.txt", "r") as file:
    reconfirm_translator_prompt = file.read()

with open("./modules/prompts/item_selection_prompt.txt", "r") as file:
    item_selection_prompt = file.read()

with open("./modules/prompts/combo_selection_prompt.txt", "r") as file:
    combo_selection_prompt = file.read()

with open("./modules/prompts/merch_selection_prompt.txt", "r") as file:
    merch_selection_prompt = file.read()

with open("./modules/prompts/guarantee_selection_prompt.txt", "r") as file:
    garansi_selection_prompt = file.read()

with open("./modules/prompts/coupon_selection_prompt.txt", "r") as file:
    kupon_selection_prompt = file.read()

with open("./modules/prompts/voucher_selection_prompt.txt", "r") as file:
    voucher_selection_prompt = file.read()

with open("./modules/prompts/compliment_selection_prompt.txt", "r") as file:
    komplimen_selection_prompt = file.read()

with open("./modules/prompts/delivery_selection_prompt.txt", "r") as file:
    delivery_selection_prompt = file.read()

with open("./modules/prompts/prize_selection_prompt.txt", "r") as file:
    hadiah_selection_prompt = file.read()

with open("./modules/prompts/notes_prompt.txt", "r") as file:
    notes_prompt = file.read()

task_instructions = {
    "reconfirm_translator_prompt": reconfirm_translator_prompt,
    "item_selection_prompt": item_selection_prompt,
    "combo_selection_prompt": combo_selection_prompt,
    "merch_selection_prompt": merch_selection_prompt,
    "garansi_selection_prompt": garansi_selection_prompt,
    "kupon_selection_prompt": kupon_selection_prompt,
    "voucher_selection_prompt": voucher_selection_prompt,
    "komplimen_selection_prompt": komplimen_selection_prompt,
    "delivery_selection_prompt": delivery_selection_prompt,
    "hadiah_selection_prompt": hadiah_selection_prompt,
    "notes_prompt": notes_prompt,
}

def detect_keywords(text):
    tokens = re.split(r"[\s]+", text.strip().lower())  # pisah hanya spasi
    return tokens

class AgentBabe:
    def __init__(
        self,
        instructions: dict = task_instructions,
        df_product_dir: str = "./product_items.csv",
        df_combo_dir: str = "./product_combos.csv",
        top_k_retrieve: int = 5,
        gmap_api_key: Optional[str] = None,
    ):
        self.instructions = instructions
        self.df_product_dir = df_product_dir
        self.df_combo_dir = df_combo_dir
        self.top_k_retrieve = top_k_retrieve
        self.gmap_api_key = gmap_api_key
        self.model_name = {
            "flash": "gemini-2.5-flash",
            "pro": "gemini-2.5-pro",
        }
        self.longlat_toko = (-7.560745951139057, 110.8493297202405)
        self.free_areas = [
            "Gedongan",
            "Gedangan",
            "Gentan",
            "Kadilangu",
            "Kudu",
            "Kwarasan",
            "Langenharjo",
            "Madegondo",
            "Gonilan",
            "Gumpang",
            "Pabelan",
            "Blulukan",
            "Karangasem",
            "Baturan",
            "Gajahan",
            "Paulan",
        ]
    
    def clean_llm_json_output(self, text: str) -> dict:
        # Hilangkan ```json ... ```
        cleaned = re.sub(
            r"^```json\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE
        )

        # Parsing ke dict
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            print("[ERROR] Gagal parse JSON:", e)
            return None
    
    def select_id_by_agent(
        self,
        query: str,
        df: pd.DataFrame,
        task_instruction: str,
        id_col: str = "id",
        evaluation_col: str = "name",
    ):
        tokenized_corpus = [word_tokenize(prod.lower()) for prod in df[evaluation_col]]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = word_tokenize(query.lower())
        sim_score = bm25.get_scores(tokenized_query)
        df["bm25_score"] = sim_score

        sim_score_table = df.sort_values(by="bm25_score", ascending=False)[
            : self.top_k_retrieve
        ]
        sim_score_table = sim_score_table[[id_col, evaluation_col]].to_dict(
            orient="records"
        )

        LLM = genai.GenerativeModel(
            model_name=self.model_name["flash"],
            system_instruction=task_instruction,
        )
        idx = LLM.generate_content(f"Query: {query}, List: {sim_score_table}")

        try:
            idx = int(idx.text)
            return idx
        except ValueError:
            print("[ERROR] Gagal mendapatkan indeks dari Gemini:", idx.text)
            return None
    
    def reconfirm_translator(self, message):
        model = genai.GenerativeModel(
            model_name=self.model_name["flash"],
            system_instruction=self.instructions["reconfirm_translator_prompt"],
        )

        gemini_ans = model.generate_content(message)
        try:
            sanitized_response = self.clean_llm_json_output(gemini_ans.text)
        except Exception as e:
            logger.error("Gagal membersihkan output JSON dari Gemini: %s", e)
            return {
                "fallback": "Ada kesalahan dalam mem-parsing output dari AI. Silakan coba lagi dan pastikan formatnya sesuai."
            }

        return sanitized_response

    def _process_item(
        self,
        order_id: str,
        nama_produk: str,
        qty: int,
        cart: list,
        access_token: str,
    ):
        ############### 1. Baca database produk satuan ###############
        product_df = pd.read_csv(self.df_product_dir)
        df = product_df[product_df["pos_hidden"] == 0]

        ############### 2. Cari ID menggunakan SLM ###############
        idx = self.select_id_by_agent(
            nama_produk,
            df,
            self.instructions["item_selection_prompt"],
            id_col="id",
            evaluation_col="name",
        )
        if idx is None or idx == -99999:
            logger.error("Gagal menemukan produk item: %s", nama_produk)
            update_status(order_id, "X", access_token=access_token)
            return (
                False,
                f"Gagal menemukan produk item {nama_produk}, tolong masukkan dengan format <Nama produk> (<QTY> Paket/Item), dan hindari penggunaan singkatan (AI tidak tahu konteks dalam singkatan itu). Sebisa mungkin, sertakan juga brand-nya apa agar menghindari kesalahpahaman AI, misal AM bisa dianggap dari Mix Max Anggur Merah, QRO Anggur Merah, atau Kawa Kawa Anggur Merah, tapi kalau ini tidak dianggap masalah, silakan diabaikan. Jika error ini masih berlangsung, cek backoffice Olsera. Struk di-voidkan",
            )

        ############### 3. Ambil data berdasarkan ID yang dipilih ###############
        df_sel = df[df["id"] == idx].reset_index(drop=True)

        if df_sel.empty:
            logger.error("Data produk dengan id %s tidak ditemukan dalam df.", idx)
            print("ERROR ID 0 NIH, df_sel nya empty.")
            print("ID yang di retrieve agent: ", idx)
            print("Nama produk yang berusaha di retrieve agent: ", nama_produk)
            return (
                False,
                f"Produk dengan id {idx} tidak ditemukan. Mungkin terjadi perubahan pada database atau item sudah tidak tersedia.",
            )
        
        product_id, product_name, item_left_to_pick = idx, nama_produk, qty

        try:
            item_details = fetch_product_item_details(
                product_id, access_token=access_token
            )
            time.sleep(1)

            if item_details.get("error", None):
                if item_details["error"]["status_code"] == 429:
                    logger.error(
                        "Rate limit exceeded for item_details ID %s", product_id
                    )
                    return f"Agent mengalami limit dari API Olsera"
                elif item_details["error"]["status_code"] == 404:
                    logger.error(
                        "Item not found for item_details ID %s", product_id
                    )
                    return f"Produk {product_name} tidak ditemukan di Olsera. Silakan cek kembali nama produk atau pastikan produk tersebut masih tersedia."
                else:
                    logger.error(
                        "Error fetching item_details ID %s: %s",
                        product_id,
                        item_details["error"],
                    )
                    return f"Gagal ambil data produk {product_name}. Terjadi error yang tidak diketahui. Errornya: {item_details['error']['message']}"
                    
            data = item_details.get("data", None)
            product_name = data.get("name", product_name)
            if data is None:
                logger.error("Data item_details kosong untuk ID %s", product_id)
                return f"Gagal ambil data produk {product_name}. Berdasarkan API Olsera, produk ini tidak ditemukan. Anda bisa mencobanya sekali lagi, kalau ini tetap terjadi, mohon cek Olsera Backoffice untuk memastikan produk ini masih ada."
        except Exception as e:
            logger.error("Gagal fetch item_details ID %s: %s", product_id, e)
            return f"Gagal ambil data produk {product_name}. Terjadi error yang tidak diketahui. Errornya: {e}"

        if not data.get("variant"):
            idx = f"{product_id}"
            harga = float(data.get("sell_price_pos", 0))

            cart.append(
                {
                    "type": "product",
                    "prod_id": product_id,
                    "prodvar_id": idx,
                    "name": product_name,
                    "qty": item_left_to_pick,
                    "price": harga,
                    "disc": 0,
                }
            )
        
        else:
            for var in data.get("variant", []):
                var_stock = int(float(var.get("stock_qty", 0)))
                var_hold_qty = int(float(var.get("hold_qty", 0)))
                print(
                    f"VAR_STOCK {var.get('name'):<45} {product_id}|{var['id']} {var_stock:<15} {var_hold_qty:<15}"
                )
                if (
                    var_stock <= 0
                    or var.get("name").startswith("X")
                    or var_stock - var_hold_qty <= 0
                ):
                    print(
                        "Stok varian tidak cukup atau sudah di-hold, lanjut ke varian berikutnya."
                    )
                    continue

                pick_qty = min(item_left_to_pick, var_stock)
                cart.append(
                    {
                        "type": "product",
                        "prod_id": product_id,
                        "prodvar_id": f"{product_id}|{var['id']}",
                        "name": product_name,
                        "qty": pick_qty,
                        "price": var.get("sell_price_pos", 0),
                        "disc": 0,
                    }
                )

                item_left_to_pick -= pick_qty
                if item_left_to_pick <= 0:
                    break

            if item_left_to_pick > 0:
                logger.error(
                    "Stok tidak cukup untuk produk %s, masih ada %d yang perlu diambil",
                    product_id,
                    item_left_to_pick,
                )
                return f"Maaf, stok tidak cukup untuk {product_name}. Silakan coba lagi dengan jumlah yang lebih sedikit."
            
            try:
                print(f"ITEM  {nama_produk:<45} {product_id:<10} {qty:<15} {cart[-1]['disc']:<10}")
            except Exception as e:
                pass

            return (
                True,
                f"Item {nama_produk} berhasil ditambahkan ke order dengan ID {order_id}.",
            )
    
    def _process_combo(
        self,
        order_id: str,
        nama_combo: str,
        qty: int,
        cart: list,
        access_token: str,
    ) -> Tuple[bool, str]:
        ############### 1. Baca database produk combo ###############
        combo_df = pd.read_csv(self.df_combo_dir)
        df = combo_df[combo_df["pos_hidden"] == 0]
        task_instruction = self.instructions["combo_selection_prompt"]
        lower = nama_combo.strip().lower()

        ############### 2. Sederhanakan pencarian dengan filtering ###############
        if lower.startswith(("merch", "mer")):
            mask = df["name"].str.lower().str.contains("merch|merh")
            df = df[mask]
            task_instruction = self.instructions["merch_selection_prompt"]
        elif lower.startswith(("babe garansiin", "garansi", "garan")):
            df = df[df["name"] == "Babe Garansi-in !!!"]
            task_instruction = self.instructions["guarantee_selection_prompt"]
        elif "kupon" in lower:
            df = df[df["name"].str.lower().str.contains("kupon")]
            task_instruction = self.instructions["coupon_selection_prompt"]
        elif "voucher" in lower:
            df = df[df["name"].str.lower().str.contains("voucher")]
            task_instruction = self.instructions["voucher_selection_prompt"]
        elif lower.startswith(("komplimen", "komp")):
            df = df[df["name"].str.lower().str.startswith(("komplimen", "komp"))]
            task_instruction = self.instructions["compliment_selection_prompt"]
        elif "delivery" in lower:
            df = df[df["name"].str.lower().str.contains("delivery")]
            task_instruction = self.instructions["delivery_selection_prompt"]
        elif lower.startswith("hadiah"):
            df = df[df["name"].str.lower().str.startswith("hadiah")]
            task_instruction = self.instructions["prize_selection_prompt"]

        ############### 3. Cari ID menggunakan SLM ###############

        if df.empty:
            logger.error("Tidak ada paket matching untuk: %s", nama_combo)
            return f"Gagal menemukan paket: {nama_combo}"
        if len(df) == 1:
            combo_id = df["id"].iloc[0]
        else:
            combo_id = self.select_id_by_agent(
                nama_combo, df, task_instruction, id_col="id", evaluation_col="name"
            )
        
        if combo_id is None:
            logger.error("select_id_by_agent gagal untuk paket: %s", combo_id)
            return f"Gagal menemukan paket: {combo_id}"
        
        if combo_id == -99999:
            logger.error("Tidak ada paket yang sesuai dengan nama: %s", nama_combo)
            update_status(order_id, "X", access_token=access_token)
            return f"AI gagal menemukan paket yang sesuai: {nama_combo}, ini disebabkan karena AI tidak yakin dengan kecocokan antara nama yang dimasukkan dengan hasil pencarian yang ditemukan (untuk menghindari pengambilan asal). Untuk itu, mohon masukkan dengan format <Nama produk> (<QTY> Paket/Item), dan hindari penggunaan singkatan (AI tidak tahu konteks dalam singkatan itu). Sebisa mungkin, sertakan juga brand-nya apa agar menghindari kesalahpahaman AI, misal AM bisa dianggap dari Mix Max Anggur Merah, QRO Anggur Merah, atau Kawa Kawa Anggur Merah, tapi kalau ini tidak dianggap masalah, silakan diabaikan. CONTOH: 2 Atlas Lychee + 2 Beer (1 Paket). Jika error ini masih berlangsung, cek backoffice Olsera. Struk di-voidkan."



        ############### 4. Masukkan ke keranjang ###############
        price = int(float(combo_df[combo_df["id"] == combo_id]["sell_price_pos"].values))
        cart.append(
            {
                "type": "combo_new",
                "prod_id": combo_id,
                "prodvar_id": "UNIQUE",
                "name": nama_combo,
                "qty": qty,
                "price": price,
                "disc": 0,
            }
        )

        try:
            print(f"PAKET  {nama_combo:<45} {combo_id:<10} {qty:<15} {cart[-1]['disc']:<10}")
        except Exception as e:
            pass

        return (
            True,
            f"Paket {nama_combo} berhasil ditambahkan ke order dengan ID {order_id}.",
        )

    def _process_combo_old(
        self,
        order_id: str,
        nama_combo: str,
        qty_pesan: int,
        cart: list,
        access_token: str,     
    ):
        ############### 1. Baca database produk combo ###############
        combo_df = pd.read_csv(self.df_combo_dir)
        df = combo_df[combo_df["pos_hidden"] == 0]
        task_instruction = self.instructions["combo_selection_prompt"]
        lower = nama_combo.strip().lower()

        ############### 2. Sederhanakan pencarian dengan filtering ###############
        if lower.startswith(("merch", "mer")):
            mask = df["name"].str.lower().str.contains("merch|merh")
            df = df[mask]
            task_instruction = self.instructions["merch_selection_prompt"]
        elif lower.startswith(("babe garansiin", "garansi", "garan")):
            df = df[df["name"] == "Babe Garansi-in !!!"]
            task_instruction = self.instructions["guarantee_selection_prompt"]
        elif "kupon" in lower:
            df = df[df["name"].str.lower().str.contains("kupon")]
            task_instruction = self.instructions["coupon_selection_prompt"]
        elif "voucher" in lower:
            df = df[df["name"].str.lower().str.contains("voucher")]
            task_instruction = self.instructions["voucher_selection_prompt"]
        elif lower.startswith(("komplimen", "komp")):
            df = df[df["name"].str.lower().str.startswith(("komplimen", "komp"))]
            task_instruction = self.instructions["compliment_selection_prompt"]
        elif "delivery" in lower:
            df = df[df["name"].str.lower().str.contains("delivery")]
            task_instruction = self.instructions["delivery_selection_prompt"]
        elif lower.startswith("hadiah"):
            df = df[df["name"].str.lower().str.startswith("hadiah")]
            task_instruction = self.instructions["prize_selection_prompt"]

        ############### 3. Cari ID menggunakan SLM ############### 

        if df.empty:
            logger.error("Tidak ada paket matching untuk: %s", nama_combo)
            return f"Gagal menemukan paket: {nama_combo}"
        if len(df) == 1:
            combo_id = df["id"].iloc[0]
        else:
            combo_id = self.select_id_by_agent(
                nama_combo, df, task_instruction, id_col="id", evaluation_col="name"
            )
        
        if combo_id is None:
            logger.error("select_id_by_agent gagal untuk paket: %s", combo_id)
            return f"Gagal menemukan paket: {combo_id}"
        
        if combo_id == -99999:
            logger.error("Tidak ada paket yang sesuai dengan nama: %s", nama_combo)
            update_status(order_id, "X", access_token=access_token)
            return f"AI gagal menemukan paket yang sesuai: {nama_combo}, ini disebabkan karena AI tidak yakin dengan kecocokan antara nama yang dimasukkan dengan hasil pencarian yang ditemukan (untuk menghindari pengambilan asal). Untuk itu, mohon masukkan dengan format <Nama produk> (<QTY> Paket/Item), dan hindari penggunaan singkatan (AI tidak tahu konteks dalam singkatan itu). Sebisa mungkin, sertakan juga brand-nya apa agar menghindari kesalahpahaman AI, misal AM bisa dianggap dari Mix Max Anggur Merah, QRO Anggur Merah, atau Kawa Kawa Anggur Merah, tapi kalau ini tidak dianggap masalah, silakan diabaikan. CONTOH: 2 Atlas Lychee + 2 Beer (1 Paket). Jika error ini masih berlangsung, cek backoffice Olsera. Struk di-voidkan."
        
        ############### 4. Ambil detail combo dari API Olsera ###############
        try:
            combo_details = fetch_product_combo_details(combo_id, access_token)
            combo_items = combo_details["data"]["items"]["data"]
        except Exception as e:
            logger.error("Gagal fetch combo details untuk %s: %s", combo_id, e)
            return "Gagal ambil detail paket."
        try:
            print(f"PAKET  {combo_details['data']['name']:<45} {combo_details['data']['id']:<10} {qty_pesan:<15}")
        except Exception as e:
            pass

        ############### 5. Masukkan ke keranjang ###############
        # Hitung total harga normal untuk diskon
        total_harga_normal = 0.0

        count_item = 0
        # Tambah tiap item
        for item in combo_items:
            # 1. Deklarasi product ID, variant ID, dan qty_total
            product_id = item.get("product_id")
            var_id = item.get("product_variant_id")
            item_left_to_pick = item.get("qty", 0) * qty_pesan

            # 2. Fetch detail item untuk cek stok per varian dan harga
            try:
                item_details = fetch_product_item_details(
                    product_id, access_token=access_token
                )
                time.sleep(1)
                if item_details.get("error", None):
                    if item_details["error"]["status_code"] == 429:
                        logger.error(
                            "Rate limit exceeded for item_details ID %s", product_id
                        )
                        return f"Agent mengalami limit dari API Olsera"
                    elif item_details["error"]["status_code"] == 404:
                        logger.error(
                            "Item not found for item_details ID %s", product_id
                        )
                        return f"Produk {item.get('product_name')} tidak ditemukan di Olsera. Silakan cek kembali nama produk atau pastikan produk tersebut masih tersedia."
                    else:
                        logger.error(
                            "Error fetching item_details ID %s: %s",
                            product_id,
                            item_details["error"],
                        )
                        return f"Gagal ambil data produk {item.get('product_name')}. Terjadi error yang tidak diketahui. Errornya: {item_details['error']['message']}"

                data = item_details.get("data", None)
                if data is None:
                    logger.error("Data item_details kosong untuk ID %s", product_id)
                    return f"Gagal ambil data produk {item.get('product_name')} dari paket {combo_details['data']['name']}. Berdasarkan API Olsera, produk ini tidak ditemukan. Anda bisa mencobanya sekali lagi, kalau ini tetap terjadi, mohon cek Olsera Backoffice untuk memastikan produk ini masih ada."
            except Exception as e:
                logger.error("Gagal fetch item_details ID %s: %s", product_id, e)
                return f"Gagal ambil data produk {item.get('product_name')}. Terjadi error yang tidak diketahui. Errornya: {e}"

            # Cek variant dan harga
            if not data.get("variant"):
                idx = f"{product_id}"
                harga = float(data.get("sell_price_pos", 0))

                cart.append(
                    {
                        "type": "combo_old",
                        "prod_id": product_id,
                        "prodvar_id": idx,
                        "name": item.get("product_name"),
                        "qty": item_left_to_pick,
                        "price": harga,
                        "disc": 0,
                    }
                )
                total_harga_normal += harga * item_left_to_pick

                count_item += 1

            else:
                for var in data.get("variant", []):
                    var_stock = int(float(var.get("stock_qty", 0)))
                    var_hold_qty = int(float(var.get("hold_qty", 0)))
                    print(
                        f"VAR_STOCK {var.get('name'):<45} {product_id}|{var['id']} {var_stock:<15} {var_hold_qty:<15}"
                    )
                    if (
                        var_stock <= 0
                        or var.get("name").startswith("X")
                        or var_stock - var_hold_qty <= 0
                    ):
                        print(
                            "Stok varian tidak cukup atau sudah di-hold, lanjut ke varian berikutnya."
                        )
                        continue

                    pick_qty = min(item_left_to_pick, var_stock)
                    cart.append(
                        {
                            "prod_id": product_id,
                            "prodvar_id": f"{product_id}|{var['id']}",
                            "name": item.get("product_name"),
                            "qty": pick_qty,
                            "price": var.get("sell_price_pos", 0),
                            "disc": 0,
                        }
                    )
                    print(
                        f"PAKET {item['product_name']:<45} {product_id}|{var['id']} {pick_qty:<15}"
                    )
                    count_item += 1

                    item_left_to_pick -= pick_qty
                    total_harga_normal += float(var.get("sell_price_pos", 0)) * pick_qty
                    if item_left_to_pick <= 0:
                        break

                if item_left_to_pick > 0:
                    logger.error(
                        "Stok tidak cukup untuk produk %s, masih ada %d yang perlu diambil",
                        product_id,
                        item_left_to_pick,
                    )
                    return f"Maaf, stok tidak cukup untuk {item.get('product_name')}. Silakan coba lagi dengan jumlah yang lebih sedikit."
            
        ############## PATCH: Diskon untuk paket lewat Cart ##############
        bundle_entry = cart[-count_item:]  # Ambil entry terakhir yang merupakan paket
        harga_paket = float(combo_details["data"]["sell_price_pos"]) * qty_pesan

        # print("DEBUG: bundle_entry:", bundle_entry)
        if int(total_harga_normal) == int(harga_paket):
            pass
        else:
            for item in bundle_entry:
                price = float(item["price"])
                # print(total_harga_normal)
                item["disc"] = (
                    price
                    * (total_harga_normal - harga_paket)
                    / total_harga_normal
                    * float(item["qty"])
                    if total_harga_normal > 0
                    else price
                )
                # print(f"Diskon untuk {item['name']:<45} {item['prodvar_id']} {item['qty']:<15} {item['disc']:.2f}")
                logger.debug("Paket-item %s diskon: %s", item["name"], item["disc"])

        return True
