from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
import google.generativeai as genai
import re
import json
import pandas as pd
import os
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
    add_combo_to_order,
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
        df_combo_dir: str = "./product_combos_v2.csv",
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
        print("Mencari id berdasarkan SLM...")
        idx = self.select_id_by_agent(
            nama_produk,
            df,
            self.instructions["item_selection_prompt"],
            id_col="id",
            evaluation_col="name",
        )
        print("ID yang ditemukan:", idx)
        if idx is None or idx == -99999:    
            print("Gagal menemukan produk item:", nama_produk)
            logger.error("Gagal menemukan produk item: %s", nama_produk)
            update_status(order_id, "X", access_token=access_token)
            return {
                "success" : False,
                "msg" : f"Gagal menemukan produk item {nama_produk}, tolong masukkan dengan format <Nama produk> (<QTY> Paket/Item), dan hindari penggunaan singkatan (AI tidak tahu konteks dalam singkatan itu). Sebisa mungkin, sertakan juga brand-nya apa agar menghindari kesalahpahaman AI, misal AM bisa dianggap dari Mix Max Anggur Merah, QRO Anggur Merah, atau Kawa Kawa Anggur Merah, tapi kalau ini tidak dianggap masalah, silakan diabaikan. Jika error ini masih berlangsung, cek backoffice Olsera. Struk di-voidkan",
            }

        ############### 3. Ambil data berdasarkan ID yang dipilih ###############
        df_sel = df[df["id"] == idx].reset_index(drop=True)

        if df_sel.empty:
            logger.error("Data produk dengan id %s tidak ditemukan dalam df.", idx)
            print("ERROR ID 0 NIH, df_sel nya empty.")
            print("ID yang di retrieve agent: ", idx)
            print("Nama produk yang berusaha di retrieve agent: ", nama_produk)
            return { 
                "success" : False,
                "msg": f"Produk dengan id {idx} tidak ditemukan. Mungkin terjadi perubahan pada database atau item sudah tidak tersedia.",
            }
        
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
                    return { 
                        "success" : False,
                        "msg" : f"Produk {product_name} tidak dapat diambil karena mengalami limit dari API Olsera. Coba lagi nanti.",
                    }
                elif item_details["error"]["status_code"] == 404:
                    logger.error(
                        "Item not found for item_details ID %s", product_id
                    )
                    return {
                        "success" : False,
                        "msg" : f"Produk {product_name} tidak ditemukan di Olsera. Silakan cek kembali nama produk atau pastikan produk tersebut masih tersedia.",
                    }
                else:
                    logger.error(
                        "Error fetching item_details ID %s: %s",
                        product_id,
                        item_details["error"],
                    )
                    return {
                        "success" : False,
                        "msg" : f"Gagal ambil data produk {product_name}. Terjadi error yang tidak diketahui. Errornya: {item_details['error']['message']}",
                    }
                    
            data = item_details.get("data", None)
            product_name = data.get("name", product_name)
            if data is None:
                logger.error("Data item_details kosong untuk ID %s", product_id)
                return {
                    "success" : False,
                    "msg" : f"Gagal ambil data produk {product_name}. Terjadi error yang tidak diketahui. Data yang diterima kosong.",
                }
        except Exception as e:
            logger.error("Gagal fetch item_details ID %s: %s", product_id, e)
            return {
                "success" : False,
                "msg" : f"Gagal ambil data produk {product_name}. Terjadi error yang tidak diketahui. Errornya: {e}",
            }

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
            return { 
                "success": True,
                "msg": f"Item {nama_produk} berhasil ditambahkan ke order dengan ID {order_id}.",
            }
        
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
                return {
                    "success" : False,
                    "msg" : f"Maaf, stok tidak cukup untuk {product_name}. Silakan coba lagi dengan jumlah yang lebih sedikit.",
                }
            
            try:
                print(f"ITEM  {nama_produk:<45} {product_id:<10} {qty:<15} {cart[-1]['disc']:<10}")
            except Exception as e:
                return {
                    "success" : False,
                    "msg" : f"Terjadi Kesalahan dalam menghandle item: {e}",
                }
            
            return { 
                "success": True,
                "msg": f"Item {nama_produk} berhasil ditambahkan ke order dengan ID {order_id}.",
            }
    
    def _process_combo(
        self,
        order_id: str,
        nama_combo: str,
        qty: int,
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
            task_instruction = self.instructions["garansi_selection_prompt"]
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
        # if len(df) == 1:
        #     combo_id = df["id"].iloc[0]
        # else:
        #     combo_id = self.select_id_by_agent(
        #         nama_combo, df, task_instruction, id_col="id", evaluation_col="name"
        #     )

        combo_id = df["id"].iloc[0] if len(df) == 1 else self.select_id_by_agent(
            nama_combo,df,task_instruction,id_col="id",evaluation_col="name"
        )
        
        if combo_id is None:
            logger.error("select_id_by_agent gagal untuk paket: %s", combo_id)
            return {
                "success" : False,
                "msg" : f"Gagal menemukan paket: {combo_id}, mohon coba lagi.",
            }
        
        if combo_id == -99999:
            logger.error("Tidak ada paket yang sesuai dengan nama: %s", nama_combo)
            update_status(order_id, "X", access_token=access_token)
            return {
                "success" : False,
                "msg" : f"AI gagal menemukan paket yang sesuai: {nama_combo}, ini disebabkan karena AI tidak yakin dengan kecocokan antara nama yang dimasukkan dengan hasil pencarian yang ditemukan (untuk menghindari pengambilan asal). Untuk itu, mohon masukkan dengan format <Nama produk> (<QTY> Paket/Item), dan hindari penggunaan singkatan (AI tidak tahu konteks dalam singkatan itu). Sebisa mungkin, sertakan juga brand-nya apa agar menghindari kesalahpahaman AI, misal AM bisa dianggap dari Mix Max Anggur Merah, QRO Anggur Merah, atau Kawa Kawa Anggur Merah, tapi kalau ini tidak dianggap masalah, silakan diabaikan. CONTOH: 2 Atlas Lychee + 2 Beer (1 Paket). Jika error ini masih berlangsung, cek backoffice Olsera. Struk di-voidkan."
            }



        ############### 4. Masukkan ke keranjang ###############
        price = int(float(combo_df[combo_df["id"] == combo_id]["sell_price_pos"].values))
        items_raw = combo_df.loc[combo_df["id"] == combo_id, "items"].iloc[0]
        if isinstance(items_raw, str):
            try:
                items = ast.literal_eval(items_raw) if isinstance(items_raw, str) else items_raw
            except json.JSONDecodeError:
                print(f"[ERROR] Gagal parse items: {e}")
                items = []
        else:
            items = items_raw
        cart.append(
            {
                "type": "combo_new",
                "prod_id": combo_id,
                "prodvar_id": "UNIQUE",
                "name": nama_combo,
                "qty": qty,
                "price": price,
                "disc": 0,
                "items" : items,
            }
        )

        try:
            print(f"PAKET  {nama_combo:<45} {combo_id:<10} {qty:<15} {cart[-1]['disc']:<10}")
        except Exception as e:
            print(msg:=f"Terjadi Kesalahan dalam menghandle paket:{e}")
            # return False,msg
            return {
                "success" : False,
                "msg" : msg,
            }

        return { 
            "success": True,
            "msg": f"Paket {nama_combo} berhasil ditambahkan ke order dengan ID {order_id}.",
        }

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

    def move_cart_to_order(self, cart: list, order_id: str, access_token: str,type_combos:bool = False):
        """
        Pindahkan semua item dalam cart ke order dengan ID order_id.
        Mengembalikan True jika sukses, atau string pesan error jika gagal.
        """
        if not cart:
            logger.error("Cart kosong, tidak ada item untuk dipindahkan.")
            return False, "Cart kosong, tidak ada item untuk dipindahkan."
        
        if type_combos : 
            for combo in cart : 
                combo_id = combo["prod_id"]
                qty_combo = combo["qty"]
                price = combo["price"]
                combo_items = combo["items"]
                try : 
                    success,response = add_combo_to_order(
                        order_id=str(order_id),
                        combo_id=str(combo_id),
                        quantity=qty_combo,
                        access_token=access_token,
                        combo_items=combo_items
                    )
                    if not success : 
                        print(f"Memasukkan paket {combo['name']} ke dalam keranjang...")
                        return False,f"Gagal menambahkan paket {combo['name']} ke order. Error: {response}"
                except Exception as e : 
                    print(f"Memasukkan paket {combo['name']} ke dalam keranjang...")
                    return False,f"Gagal menambahkan paket {combo['name']} ke order. Error: {e}"
        else : 
            for item in cart:
                prodvar_id = item["prodvar_id"]
                qty = item["qty"]
                try:
                    print(f"Memasukkan {item['name']} ke dalam keranjang...")
                    resp = add_prod_to_order(
                        order_id, prodvar_id, qty, access_token=access_token
                    )
                    if resp is None:
                        # update_status(order_id, "X", access_token=access_token)
                        continue
                        return (
                            False,
                            f"Gagal menambahkan produk ke order karena produk {item['name']} habis atau tersisa sedikit. Mohon restock ulang, sementara struk di voidkan.",
                        )
                    logger.debug(
                        "Response add_prod_to_order: %s", getattr(resp, "text", "")
                    )
                except requests.exceptions.HTTPError as http_err:
                    logger.error("HTTPError saat menambahkan produk ke order: %s", http_err)
                    # update_status(order_id, "X", access_token=access_token)
                    return False, "Ada kesalahan HTTP saat memasukkan produk ke order."
                except Exception as e:
                    logger.error("Error lain saat menambahkan produk ke order: %s", e)
                    print("Error lain saat menambahkan produk ke order: %s", e)
                    # update_status(order_id, "X", access_token=access_token)
                    return (
                        False,
                        f"Ada kesalahan saat memasukkan produk ke order. Error: {e}",
                    )

        # Setelah itu, tambahkan diskon untuk tiap item
        # try:
        #     ord_detail = fetch_order_details(
        #         order_id=order_id, access_token=access_token
        #     )
        #     paket_items = ord_detail["data"]["orderitems"]
        #     # paket_items = pd.DataFrame(paket_items)
        # except Exception as e:
        #     logger.error("Gagal fetch detail order untuk update diskon: %s", e)
        #     return False, "Gagal update diskon paket."

        # # Hitung dan update tiap item
        # for idx, item in enumerate(paket_items):
        #     item_id = item["id"]
        #     item_qty = item["qty"]
        #     item_disc = cart[idx]["disc"] if idx < len(cart) else 0.0
        #     item_price = int(float(item.get("fprice", 0).replace(".", "")))

        #     try:
        #         update_order_detail(
        #             order_id=str(order_id),
        #             id=str(item_id),
        #             disc=str(item_disc),
        #             price=str(item_price),
        #             qty=str(item_qty),
        #             note="Promo Paket",
        #             access_token=access_token,
        #         )

        #     except Exception as e:
        #         logger.error(
        #             "Gagal update_order_detail untuk item_id %s: %s", item_id, e
        #         )
        #         return False, "Gagal update detail paket di order."

        return True, "Semua item berhasil ditambahkan ke order."

    def handle_order(
        self, query: str, access_token_dir: str, sudah_bayar: bool = False
    ):
        with open(access_token_dir, "r") as file:
            token_data = json.load(file)

        access_token = token_data.get("access_token", "")
        if not access_token:
            logger.error("Access token terkena limit %s", access_token_dir)
            return "Access token terkena limit."

        print("Query diterima: %s", query)
        reconfirm_json = self.reconfirm_translator(query)
        logger.debug("Hasil reconfirm: %s", reconfirm_json)
        print("Hasil reconfirm:", reconfirm_json)

        if reconfirm_json.get("fallback"):
            print(f"Error, format pesan tidak sesuai:", reconfirm_json["fallback"])
            return reconfirm_json["fallback"]

        if reconfirm_json.get("pembatalan"):
            print("[DEBUG] Pembatalan order dengan ID:", reconfirm_json["pembatalan"])

            # Kalau masih bukan list setelah semua itu, bungkus jadi list
            if not isinstance(reconfirm_json["pembatalan"], list):
                reconfirm_json["pembatalan"] = [str(reconfirm_json["pembatalan"])]

                try:
                    joined = ",".join(reconfirm_json["pembatalan"])
                    reconfirm_json["pembatalan"] = [
                        x.strip() for x in joined.split(",") if x.strip()
                    ]
                except ValueError:
                    print("Split sudah dicoba")
                    pass

            # Cek apakah list-nya kosong
            if not reconfirm_json["pembatalan"]:
                print("[ERROR] Tidak ada ID yang diberikan untuk pembatalan.")
                return "[ERROR Pada Orderan {reconfirm_json.get('cust_name')}({reconfirm_json.get('phone_num')})] Tidak ada ID yang diberikan untuk pembatalan."

            # Batalkan order
            fail_messages = []
            for order_id in reconfirm_json["pembatalan"]:
                try:
                    #   update_status(order_id, "X", access_token)
                    stat_void = void_order(order_id, access_token)
                    if stat_void:
                        print(f"Order {order_id} berhasil di-void.")
                    else:
                        raise ValueError(
                            f"[ERROR] Order {order_id} tidak ditemukan, kemungkinan order ini sudah di-void."
                        )
                except requests.exceptions.HTTPError as http_err:
                    fail_messages.append(
                        f"[ERROR] Gagal menghubungi API Olsera untuk membatalkan order {order_id}."
                    )
                    continue
                except Exception as err:
                    fail_messages.append(
                        f"[ERROR] Ada kesalahan dalam membatalkan order {order_id}: {err}"
                    )
                    continue

            if len(fail_messages) > 0:
                return "[ERROR Pada Orderan {reconfirm_json.get('cust_name')}({reconfirm_json.get('phone_num')})] Maaf be, ada beberapa order yang gagal aku batalin:\n{} Selain itu berhasil.".format(
                    "\n".join(fail_messages)
                )
            else:
                return f"Order dengan ID {', '.join(reconfirm_json['pembatalan'])} udah dibatalin be."

        if reconfirm_json.get("kosongkan_keranjang"):
            yesterday = (datetime.today().date() - timedelta(days=1)).strftime("%Y-%m-%d")
            today = datetime.today().date().strftime("%Y-%m-%d")
            ret = fetch_open_order_table(
                start_date=yesterday,
                end_date=today,
                access_token=access_token
            )
            if ret is not None:
                ret = pd.DataFrame(ret)['id'].tolist()

                for order_id in ret:
                    void_order(order_id, access_token)
                    time.sleep(1) # Cegah rate limit

                return f"Semua order yang masih open di tanggal {yesterday} sampai {today} sudah aku batalin be."

        # Ubah alamat
        try:
            if reconfirm_json["address"][:4] == "http":
                result = resolve_maps_shortlink(
                reconfirm_json["address"], api_key=self.gmap_api_key
                )
                print("DEBUG resolve_maps_shortlink result:", result)
                alamat_cust, longlat_cust, kelurahan, kecamatan, kota, provinsi = (
                    resolve_maps_shortlink(
                        reconfirm_json["address"], api_key=self.gmap_api_key
                    )
                )
                distance_and_time = get_travel_distance(
                    self.longlat_toko, longlat_cust, api_key=self.gmap_api_key
                )
                # distance_and_time = get_fastest_route_details(self.longlat_toko, longlat_cust, api_key=self.gmap_api_key)
                distance = distance_and_time["distance_meters"] / 1000
                reconfirm_json["distance"] = distance
            else:
                alamat_cust = reconfirm_json["address"]
                kelurahan, kecamatan, kota, provinsi = (
                    None,
                    None,
                    None,
                )  # Temporarily set to None
                longlat_cust = address_to_latlng(
                    reconfirm_json["address"], api_key=self.gmap_api_key
                )
                distance_and_time = get_travel_distance(
                    self.longlat_toko, longlat_cust, api_key=self.gmap_api_key
                )
                # distance_and_time = get_fastest_route_details(self.longlat_toko, longlat_cust, api_key=self.gmap_api_key)
                distance = distance_and_time["distance_meters"] / 1000
                reconfirm_json["distance"] = distance

            if reconfirm_json["distance"] > 45:
                logger.error("Jarak terlalu jauh: %s km", reconfirm_json["distance"])
                return "Maaf, jarak pengiriman terlalu jauh. Silakan hubungi telemarketer untuk bantuan lebih lanjut."
        except Exception as e:
            logger.error("Gagal resolve alamat: %s", reconfirm_json["address"])
            print("Error : ",e)
            return f"[ERROR Pada Orderan {reconfirm_json.get('cust_name')}({reconfirm_json.get('phone_num')})] Maaf be, aku gagal buka alamatnya 😅. Pastiin format alamatnya dalam bentuk link gini yaa: https://maps.app.goo.gl/XXX. Detail error: {e}"

        # Buat notes
        try:
            gemini_model = genai.GenerativeModel(
                model_name=self.model_name["flash"],
                system_instruction=task_instructions["notes_prompt"],
            )
            notes = gemini_model.generate_content(query)
            notes_text = getattr(notes, "text", "") or ""
            logger.debug("Notes dihasilkan: %s", notes_text)
        except Exception as e:
            logger.error("Gagal generate notes: %s", e)
            notes_text = ""

        try:
            kastamer = cek_kastamer(
                nomor_telepon=reconfirm_json["phone_num"], access_token=access_token
            )

            cust_telp = reconfirm_json["phone_num"]

            if kastamer is None:
                cust_id = None
                cust_name = reconfirm_json["cust_name"]
            else:
                cust_id = kastamer[0]
                cust_name = kastamer[1]

        except Exception as e:
            logger.error("Gagal cek atau buat customer: %s", e)
            return f"[ERROR Pada Orderan {reconfirm_json.get('cust_name')}({reconfirm_json.get('phone_num')})] Maaf be, aku gagal memproses data pelanggannya. Ini detail errornya: {e}"

        # Create order
        today_str = datetime.now().strftime("%Y-%m-%d")
        try:
            print("Membuat order baru...")
            print(cust_id, cust_name, cust_telp)
            order_id, order_no = create_order(
                order_date=today_str,
                customer_id=cust_id,
                nama_kastamer=cust_name,
                nomor_telepon=cust_telp,
                notes=notes_text,
                access_token=access_token,
            )
            logger.debug("Order dibuat: ID=%s, No=%s", order_id, order_no)
            print(f"Order ID: {order_id}, Order No: {order_no}")
            log_dir = "log"
            log_file = os.path.join(log_dir,"order.log")
            os.makedirs(log_dir,exist_ok=True)
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open(log_file,"a",encoding="utf-8") as f : 
                f.write(f"{order_no}|{order_id}|{now_str}\n")
        except Exception as e:
            logger.error("Gagal membuat order: %s", e)
            # return f"Terjadi kesalahan, gagal membuat order baru. Mohon coba lagi. Error: {e}"
            return f"[ERROR Pada Orderan {reconfirm_json.get('cust_name')}({reconfirm_json.get('phone_num')})] Keknya masalah jaringan be, aku gagal membuat order baru. Coba lagi abis ini yaa, kalau masih error coba tanyakan ke developer. Errornya: {e}"

        subsidi_ongkir = is_free_delivery(alamat_cust, self.free_areas)
        ongkir = distance_cost_rule(reconfirm_json["distance"], subsidi_ongkir[0])

        # Add Ongkir
        if ongkir != "Gratis Ongkir" and ongkir != "Subsidi Ongkir 10K":
            reconfirm_json["ordered_products"].append(
                {
                    "tipe": "Item",
                    "produk": distance_cost_rule(reconfirm_json["distance"]),
                    "quantity": 1,
                }
            )

        elif ongkir == "Subsidi Ongkir 10K":
            reconfirm_json["ordered_products"].append(
                {
                    "tipe": "Item",
                    "produk": "Subsidi Ongkir 10K",
                    "quantity": 1,
                }
            )

        else:
            pass

        # print(reconfirm_json)
        print("PESANAN RECONFIRM")
        print("Nama Pelanggan:", reconfirm_json["cust_name"])
        print("Nomor Telepon:", reconfirm_json["phone_num"])
        print("Alamat:", reconfirm_json["address"])
        print("-" * 75)
        print(f"{'JENIS':<5} {'BARANG':<45} {'ID':<20} {'QTY':<10}")
        print("-" * 75)

        ############## SIAPKAN KERANJANG SEMENTARA #####################
        cart_temp = []
        cart_temp_product = []
        cart_temp_combo = []

        # Add products to order
        ordered_products = reconfirm_json.get("ordered_products", [])
        print("DEBUG Ordered Products:", ordered_products)
        for product in ordered_products:
            # print(f"\n[DEBUG] Memproses produk: {product['produk']} | Tipe: {product['tipe']}")
            tipe = product.get("tipe", "").lower()
            print("DEBUG TIPE : ",tipe)
            nama_produk = product.get("produk", "")
            qty = product.get("quantity", 0)

            if not nama_produk or qty <= 0:
                logger.warning(
                    "Produk diabaikan karena nama/qty tidak valid: %s", product
                )
                continue

            if tipe == "item":
                print("Mulai memproses item...")
                response = self._process_item(
                    order_id, nama_produk, qty, cart_temp_product, access_token
                )
                time.sleep(1)  # Delay untuk menghindari rate limit API Olsera

                if not response["success"]:
                    # Jika error di dalam, void entire order dan return
                    update_status(order_id, "X", access_token)
                    print("Response error:", response["msg"])
                    return f"[ERROR Pada Orderan {reconfirm_json.get('cust_name')}({reconfirm_json.get('phone_num')})] Ada kesalahan saat menambahkan item, struk di-void. Error: {response['msg']}"

            elif tipe == "paket":
                response = self._process_combo(
                    order_id, nama_produk, qty, cart_temp_combo, access_token
                )
                time.sleep(1)
                if not response["success"] :
                    # Jika mengembalikan string pesan error, batalkan order
                    update_status(order_id, "X", access_token)
                    print("Response error:", response["msg"])
                    return response["msg"]

            else:
                print(
                    f"[ERROR Pada Orderan {reconfirm_json.get('cust_name')}({reconfirm_json.get('phone_num')})] Jenis tidak dikenali. Pastikan untuk memasukkan produk dengan kurung () yang menjelaskan jenis produk, apakah item atau paket. Misal: Hennesey 650 mL (item). Anda memasukkan: {product['tipe']}"
                )
                continue

        # Auto add merch
        # total_amount = int(float(order_details['data']['total_amount']))
        # print("Keranjang Sementara:", json.dumps(cart_temp, indent=4))
        # print(f"Cart sementara : {cart_temp}")
        cart_product_df = pd.DataFrame(cart_temp_product)
        total_amount = (
            (
                cart_product_df["price"].astype(float) * cart_product_df["qty"].astype(int)
                - cart_product_df["disc"].astype(float)
            ).sum()
            if not cart_product_df.empty
            else 0
        )
        # (df_new['price'].astype(float) * df_new['qty'].astype(int) - df_new['disc'].astype(float)).sum()
        # print("Harga Total: ", total_amount)

        cart_combo_df = pd.DataFrame(cart_temp_combo)
        if not cart_combo_df.empty:
            total_amount += (
                (cart_combo_df["price"].astype(float) * cart_combo_df["qty"].astype(int)).sum()
            )

        if total_amount < 100000:
            self._process_item(order_id, "Cup Babe", 1, cart_temp, access_token)
            time.sleep(1)  # Delay untuk menghindari rate limit API Olsera

        else:
            self._process_item(order_id, "Cup Babe", 2, cart_temp, access_token)
            time.sleep(1)  # Delay untuk menghindari rate limit API Olsera

        if total_amount > 150000 and total_amount < 250000:
            self._process_combo(order_id, "Merch Babe 1", 1, cart_temp, access_token)
            time.sleep(1)  # Delay untuk menghindari rate limit API Olsera

        elif total_amount >= 250000:
            self._process_combo(order_id, "Merch Babe 2", 1, cart_temp, access_token)
            time.sleep(1)  # Delay untuk menghindari rate limit API Olsera

        else:
            pass

        # agg_cart = self.aggregate_cart_by_prodvar(cart_temp)
        print("Memproses data product ke keranjang...")
        cart_stat, msg = self.move_cart_to_order(cart_temp_product, order_id, access_token)
        if not cart_stat:
            logger.error("Gagal memindahkan cart ke order: %s", msg)
            update_status(order_id, "X", access_token)
            return f"[ERROR Pada Orderan {reconfirm_json.get('cust_name')}({reconfirm_json.get('phone_num')})] Ada error, detailnya: {msg}"
        

        print("Memproses data paket ke keranjang...")
        cart_stat, msg = self.move_cart_to_order(cart_temp_combo, order_id, access_token,type_combos=True)
        if not cart_stat:
            logger.error("Gagal memindahkan cart combo ke order: %s", msg)
            update_status(order_id, "X", access_token)
            return f"[ERROR Pada Orderan {reconfirm_json.get('cust_name')}({reconfirm_json.get('phone_num')})] Ada error, detailnya: {msg}"

        # Tambahkan diskon --- DIMATIKAN SEMENTARA UNTUK DEBUGGING
        try:
            disc_resp = self.add_discount(
                order_id,
                mode=reconfirm_json["mode_diskon"],
                access_token=access_token,
                discount=reconfirm_json["disc"],
                notes="",
            )
            if not disc_resp["success"]:
                logger.error("Gagal menambahkan diskon: %s", disc_resp["msg"])
                update_status(order_id, "X", access_token)
                return f"[ERROR Pada Orderan {reconfirm_json.get('cust_name')}({reconfirm_json.get('phone_num')})] Terdapat kesalahan dalam pengecekan diskon: {disc_resp['msg']}"
        except Exception as e:
            logger.error("Gagal menambahkan diskon: %s", e)
            update_status(order_id, "X", access_token)
            # return f"Ada kesalahan saat mengecek diskon. Mohon coba kirim ulang, sementara struk di voidkan. Error: {e}"
            return f"[ERROR Pada Orderan {reconfirm_json.get('cust_name')}({reconfirm_json.get('phone_num')})] Maaf be, biasanya aku perlu ngecek dulu nominal diskon yang babe masukin (walaupun gak ada diskon sama sekali), tapi keknya ada yang error deh. Coba kirim ulang aja yaa, kalau masih error coba tanyakan ke developer. Errornya: {e}"
        # Retrieve order details after adding products --- DIMATIKAN SEMENTARA UNTUK DEBUGGING
        try:
            order_details = fetch_order_details(order_id, access_token)

        except Exception as e:
            logger.error(
                "Struk di voidkan karena kegagalan fetch detail order setelah tambah produk: %s",
                e,
            )
            update_status(order_id, "X", access_token)
            # return f"Gagal mengambil detail order. Mohon coba lagi. Error {e}"
            return f"[ERROR Pada Orderan {reconfirm_json.get('cust_name')}({reconfirm_json.get('phone_num')})] Maaf be, aku gagal mengambil detail order setelah memasukkan produk. Coba kirim ulang aja yaa, kalau masih error coba tanyakan ke developer. Errornya: {e}"

            
        status = reconfirm_json.get("status", "").lower()
        if sudah_bayar or status == "lunas":
            try:
                # Ambil ID metode pembayaran
                payment_modes = list_payment_modes(order_id, access_token)
                # payment_dict: mapping nama pembayaran ke indeks
                jenis = reconfirm_json.get("payment_type", "")
                idx = payment_dict.get(jenis)
                if idx is None or idx >= len(payment_modes):
                    logger.warning("Metode pembayaran '%s' tidak dikenal", jenis)
                    # lanjutkan tanpa bayar atau return error?
                else:
                    payment_id = payment_modes[idx]["id"]
                    total_amount = int(float(order_details["data"]["total_amount"]))
                    update_payment(
                        order_id=order_id,
                        payment_amount=str(total_amount),
                        payment_date=today_str,
                        payment_mode_id=str(payment_id),
                        access_token=access_token,
                        payment_payee="Kevin Tes API Agent AI",
                        payment_seq="0",
                        payment_currency_id="IDR",
                    )
                    update_status(order_id, "Z", access_token)
                    logger.debug(
                        "Order %s ditandai lunas dan status diupdate.", order_id
                    )
            except Exception as e:
                logger.error("[ERROR Pada Orderan {reconfirm_json.get('cust_name')}({reconfirm_json.get('phone_num')})] Gagal proses pembayaran: %s", e)
                # Tidak membatalkan order karena produk sudah masuk; tergantung kebijakan.

        # 8. Cetak struk --- DIMATIKAN SEMENTARA UNTUK DEBUGGING
        print("Mencetak struk...")
        try:
            struk_url = cetak_struk(order_no, cust_telp)
        except Exception as e:
            logger.error("Gagal cetak struk: %s", e)
            struk_url = None

        # Catatan pending atau lunas
        pending_line = "*[PENDING ORDER]*\n" if "pending" in query.lower() else ""
        update_line = "*[UPDATE STRUK]*\n" if "update-struk" in detect_keywords(query) else ""
        req_update_line = "*[REQUEST UPDATE STRUK]*\n" if "req-update" in detect_keywords(query) else ""

        # Buat estimasi tiba
        try:
            max_luncur_str = estimasi_tiba(
                reconfirm_json["distance"],
                reconfirm_json["jenis_pengiriman"],
                datetime.now(),
            )
            max_luncur_dt = datetime.combine(datetime.today(), datetime.strptime(max_luncur_str, "%H:%M").time())
            if reconfirm_json["jenis_pengiriman"] != "FD":
                max_luncur_dt += timedelta(minutes=int(float(reconfirm_json["tambahan_waktu"]) + 3))

            max_luncur = max_luncur_dt.strftime("%H:%M")
        except Exception as e:
            max_luncur_menit = int(distance_and_time["duration_seconds"] / 60) + 20
            max_luncur = (
                datetime.now() + timedelta(minutes=max_luncur_menit)
            ).strftime("%H:%M")

        max_luncur_line = (
            f"MAKSIMAL DILUNCURKAN DARI GUDANG: {max_luncur}"
            if reconfirm_json["jenis_pengiriman"] == "FD"
            else f"ESTIMASI SAMPAI: {max_luncur}"
        )

        total_ftotal = order_details["data"].get("ftotal_amount", "")

        status_lines = [pending_line.strip(), update_line.strip(), req_update_line.strip()]
        status_lines = [line for line in status_lines if line]
        distance_val = (
            int(reconfirm_json["distance"])
            if reconfirm_json["distance"] > 14
            else round(reconfirm_json["distance"], 1)
        )
        lokasi = f"{kelurahan}, {kecamatan.replace('Kecamatan ', '').replace('Kec. ', '').replace('kecamatan', '').replace('kec.', '')}"


        invoice_lines = [
            *status_lines,
            "",
            f"Nama: {reconfirm_json.get('cust_name', '')}",
            f"Nomor Telepon: {reconfirm_json.get('phone_num', '')}",
            f"Alamat: {reconfirm_json.get('address', '')}",
            "",
            "",
            max_luncur_line.strip(),
            f"Jarak: {distance_val} km (*{lokasi}*)",
            "",
            "",
            "Makasih yaa Cah udah Jajan di Babe!",
            f"Total Jajan: {total_ftotal} (*{reconfirm_json.get('payment_type', '').upper()}*)",
            f"Cek Jajanmu di sini: {struk_url or 'Gagal mencetak struk. Tolong ulangi.'}",
            f"Jam Order: *{datetime.now().strftime('%H:%M')}*",
            "",
            "",
            f"Jenis Pengiriman: {reconfirm_json.get('jenis_pengiriman', '')}",
            f"*NOTES: {reconfirm_json.get('notes') or 'Tidak ada catatan tambahan.'}*",
        ]

        invoice = "\n".join([line for line in invoice_lines if line is not None])
        print("Mengembalikan invoice: ")
        print(invoice)
        return invoice

    def add_discount(self, order_id, mode, access_token, discount=0, notes=""):
        ord_dtl = fetch_order_details(order_id, access_token)
        id_order = ord_dtl["data"].get("id", 0)
        total_price = float(ord_dtl["data"].get("total_amount", 0))
        order_list = pd.DataFrame(ord_dtl["data"].get("orderitems", []))
        if order_list.empty:
            print("Tidak ada item dalam order, tidak bisa menambahkan diskon.")
            return { 
                "success" : False,
                "msg" : "Tidak ada item dalam order ini, proses pengecekan diskon dilewatkan. Jika anda merasa sudah memasukkan item namun error ini muncul, kemungkinan besar item Anda gagal dimasukkan ke keranjang. Silakan coba lagi."
            }

        order_list = order_list[order_list["amount"].astype(float) > 0].reset_index(
            drop=True
        )

        success_count = 0
        error_messages = []

        for index, row in order_list.iterrows():
            item_id = row["id"]
            item_qty = row["qty"]
            try:
                # print("COBA UPDATE DISKON")
                # print("HARGA TOTAL", total_price)
                # print("DISKON", discount)
                # print("MODE DISKON", mode)
                item_price = float(row["amount"])
                # Hindari ZeroDivisionError
                if total_price > 0 and mode == "number":
                    item_disc = float(row["discount"]) + item_price * (
                        discount / total_price
                    )

                elif total_price > 0 and mode == "percentage":
                    item_disc = float(row["discount"]) + item_price * discount

                else:
                    raise ValueError("Mode diskon tidak dikenali atau total_price nol.")

                # print(f"[DEBUG] item_id={item_id}, item_price={item_price}, row_discount={row['discount']}, discount={discount}, item_disc={item_disc}")
                # print("BERHASIL UPDATE DISKON: ", item_disc)
            except Exception as e:
                print(f"Gagal menghitung diskon untuk {row['product_name']}: {e}")
                item_disc = 0.0
            # Ambil fprice bersih angka
            try:
                fprice_str = row.get("fprice", "").replace(".", "")
                price_int = int(float(fprice_str)) if fprice_str else 0
            except Exception:
                price_int = 0

            try:
                update_order_detail(
                    order_id=str(id_order),
                    id=str(item_id),
                    disc=str(item_disc),
                    price=str(price_int),
                    qty=str(item_qty),
                    note=notes,
                    access_token=access_token,
                )
                success_count += 1
                # return True, f"Berhasil update diskon untuk item {row['product_name']} dengan diskon {item_disc}."

            except Exception as e:
                msg = f"Gagal update detail order untuk item {row['product_name']}: {e}"
                print(msg)
                error_messages.append(msg)

        # Return di luar loop
        if success_count == len(order_list):
            return {
                "success": True,
                "msg" : "Berhasil update diskon untuk semua item."
            }
        elif success_count > 0:
            return {
                "success": False,
                "msg" : f"Sebagian berhasil. Gagal untuk: {'; '.join(error_messages)}"
            }
        else:
            return {
                "success": False,
                "msg" : "Gagal update diskon untuk semua item."
            }
