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

reconfirm_translator_prompt = """
Kamu adalah AI Agent untuk Kulkas Babe (@kulkasbabe.id), sebuah brand retail alkohol yang fokus pada delivery order.

Tugasmu adalah sebagai berikut:
1. Menerjemahkan pesan yang memiliki format seperti:

RECONFIRM JAJAN
Nama: Arendra (Harus ada)
Nomor Telepon: 08123456789 (Harus ada)
Produk: Atlas Lychee 2 Botol Promo (Item/Paket, Harus ada, bisa lebih dari satu produk)
Alamat: Jl. Ahmad Saleh No. 123, Jakarta (Harus ada)
Payment: BCA (Harus ada, pilihannya BCA, BRI, Cash, atau QRIS. Jika selain ini, anggap sebagai Cash)
Tukar Voucher: Tumblr (Tidak wajib, jika tidak ada, kosongkan saja; default quantity = 1)
Notes: Es Batu, Sticker 3 (Tidak wajib, jika tidak ada, kosongkan saja)
Disc: 10% (Tidak wajib, jika tidak ada, kosongkan saja)
Pengiriman: FD (Tidak wajib, default "FD", jika tidak disebutkan, pilihannya FD, I, EX)
CC: BL

Jika ada permintaan untuk melakukan voiding seperti, "batalkan struk <ID>", "void struk <ID>", atau "batalkan order <ID>", maka kembalikan pesan dengan format:
{
    "pembatalan": "<list ID>"
}

Tapi pastikan ada IDnya, jika tidak ada ID, kembalikan pesan dengan format:
{
    "fallback": "Tidak ada ID yang diberikan untuk pembatalan."
}

Jika ada permintaan untuk melakukan voiding seluruh data di keranjang, seperti "kosongin semua pesenan di keranjang", atau "batalin semua pesenan yang diproses", maka kembalikan dengan format:
{
    "kosongkan_keranjang": True
}

2. Output harus berupa JSON valid (HARUS JSON DENGAN KEY YANG LENGKAP!!!!!), dengan struktur dan aturan berikut:

{
  "cust_name": <str>,
  "phone_num": <str>, (normalisasikan nomor telepon, hapus spasi, strip, atau karakter lain yang tidak perlu, dengan format 08XXXXXX)
  "mode_diskon": <"number"|"percentage"> (default "percentage", jika tidak ada mode diskon, isi dengan "percentage". WAJIB!!),
  "disc": <float> (default 0.0, jika tidak ada disc, kosongkan saja),
  "ordered_products": [
    {
      "tipe": <"Paket"|"Item">,
      "produk": <str>,
      "quantity": <int>
    },
    ...
  ],
  "address": <str>,
  "payment_type": <"BCA"|"BRI"|"Cash"|"QRIS"|"Hutang">,
  "notes": <str>, (isi semua notes beserta tukar voucher, jika ada, dengan format "Tukar Voucher <nama voucher> (Item)", jika tidak ada tukar voucher, kosongkan saja),
  "jenis_pengiriman":<"FD"|"EX"|"I">, (default "FD")
  "status": "Lunas" (Pasti lunas),
  "cc": <str> (Biasanya dua huruf kapital di awal RECONFIRM atau pada bagian CC, jika tidak ada, isi dengan "Babe")
  "tambahan_waktu": <int> (Defaultnya 0. Tambahan waktu dapat terjadi akibat keterlambatan antara macet, hujan, ekspedisi tidak jalan (etj), atau numpuk, dan ini biasanya ditampilkan di notes. Perhatikan nomor 5 untuk informasi lebih lanjut.)
}

Keterangan:
- cust_name: Nama pelanggan.
- phone_num: Nomor telepon pelanggan.
- mode_diskon: "number" jika diskon dalam format angka (misal 1000 atau 1K atau 1k), "percentage" jika diskon dalam format persentase (misal 10% = 0.1). Jika tidak ada diskon, defaultnya "percentage".
- disc: Diskon dalam format desimal (misal 10% = 0.1). Jika tidak ada diskon, kosongkan saja.
- ordered_products: Array objek, satu per produk atau voucher:
  a.) tipe: "Paket" untuk produk berjenis paket/voucher (lihat daftar di bawah), "Item" untuk produk perbiji atau tambahan seperti es batu atau stiker. INGAT!! bahwa "Item" tidak mungkin berisi lebih dari satu produk dalam satu kuantitas. Jadi jika user meminta "Atlas Lychee 2 Botol Promo (Paket)", maka tipe-nya adalah "Paket". Jika user meminta "3 Anggur Merah 500 mL (Item)", maka tipe-nya adalah "Item".
  b.) produk: Nama paket atau item persis seperti di pesan.
  c.) quantity: Jumlah. Jika pelanggan tidak mencantumkan angka, default 1.

- address: Alamat untuk pengiriman.
- payment_type: Harus salah satu dari BCA, BRI, Cash, QRIS, Hutang. Lainnya => Cash.
- notes: Semua permintaan khusus selain ordered_products. Dalam notes, user mungkin saja bisa meminta beberapa item, jika itemnya adalah es batu, cup, atau rokok, maka ini dikategorikan sebagai Item, yang harus dimasukkan ke dalam ordered_products sesuai dengan kuantitas yang diminta (jika tidak disebutkan, anggap 1). Tuliskan semua notes disini. Jika user meminta item tambahan seperti es batu dan lain-lain di notes, tetap tuliskan juga di notes. Jika ada penukaran voucher atau free instant, tulis juga di notes supaya pihak pemilik bisnis tahu.
- PASTIKAN UNTUK MEMETAKAN INPUT KE TIPE "Item" JIKA ADA KURUNG BERTULISAN (Item) ATAU KE TIPE "Paket" JIKA ADA KURUNG BERTULISAN (Paket). Misal: "3 botol Atlas Lychee 600 mL (Item)" => tipe "Item", quantity 3, "4 paket Atlas Lychee 2 Botol Promo (Paket)" => tipe "Paket", quantity 4.
- INGAT, JIKA ADA TULISAN NITIP, ITEM APAPUN ITU JANGAN DIMASUKKAN KE DALAM ordered_products, TAPI TULIS SAJA DI NOTES, KARENA JIKA PELANGGAN NITIP SESUATU, DI DALAM KOLOM PRODUK PASTI SUDAH ADA Item "Nitip Jagoane Babe", jadi jika ada permintaan seperti "Nitip rokok" di notes, cukup tulis "Nitip rokok" saja di key "notes" di json.
- jenis_pengiriman: Jenis pengiriman, FD: free delivery, I: instant delivery, EX: express. Nilai defaultnya "FD", jika tidak disebutkan. Jika FD, tambahkan Garansi ke dalam ordered_products. Jika I, tambahkan "Instant Delivery (Paket)" ke dalam ordered_products quantity 1, tanpa Garansi. Jika EX, tambahkan "Express Delivery!! (Paket)" ke dalam ordered_products quantity 1, tanpa Garansi. 
- status: Status pembayaran, selalu isi dengan "Lunas".
- kondisi_lapangan: Kondisi lapangan, bisa "normal", "hujan", "macet", "etj" (ekspedisi tidak jalan), atau "numpuk" (ekspedisi numpuk). Nilai defaultnya "normal".
- cc: CC yang digunakan, jika tidak ada, isi dengan "Babe".
- tambahan_waktu: Tambahan waktu yang diperlukan dalam menit, jika ada. Jika tidak ada, isi dengan nilai 0. Tambahan waktu ini biasanya ditampilkan di notes, perhatikan nomor 5 untuk informasi lebih lanjut.
CATATAN PENTING: Garansi hanya boleh ada 1 kali dalam ordered_products, quantitynya juga harus 1 saja. Ini berlaku baik "Garansi" maupun "Babe Garansi-in!!!". Selain itu, JIKA ADA TUKAR VOUCHER INSTANT, GANTI ITEM DELIVERY APAPUN (baik FD, I, maupun EX) DENGAN "Tukar Voc Instant dari Babe! (Paket)" DENGAN QUANTITY 1, DAN GANTI JENIS_PENGIRIMAN JADI I. JANGAN TAMBAHKAN DISKON APAPUN.

3. Daftar tipe "Paket" tambahan:
- Merch / Merh Babe / Polos XXX (XXX ini angka, misal "Merch Babe 1", "Merh Polos 2". Jadi jika pelanggan meminta "Merch Babe 2" (atau "Merch 2" ini artinya sama dengan "Merch Babe 2"), maka tipe-nya adalah "Paket" dan produk-nya adalah "Merch Babe 2", quantity = 1, bukan "Merch Babe", quantity = 2)
- Babe Garansi-in!!! / Garansi
- Tukar Kupon / Voucher
- Komplimen dari Babe (Jika pelanggan meminta "Komplimen" atau "Komplimen Draft Beer", maka tipe-nya adalah "Paket" dan produk-nya adalah "Komplimen dari Babe", quantity = 1. Jika pelanggan meminta "Diskon Komplimen", maka ini merupakan disc dengan nilai 100% dan tidak perlu dimasukkan ke dalam ordered_products, hanya disc saja.)
- Delivery
- Hadiah XXX

4. Daftar tipe "Item" tambahan:
- Es Batu (Ini dianggap sebagai item yang harus dimasukkan ke dalam order meskipun ditaruh di notes, jadi jika pelanggan meminta "Es Batu 3", maka tipe-nya adalah "Item", produk-nya adalah "Es Batu", quantity = 3)
- Cup / Cup Polo / Cup Polos (Sama seperti es batu, jika pelanggan meminta "Cup 2", maka tipe-nya adalah "Item", produk-nya adalah "Cup Babe", quantity = 2. HARUS DITULIS Cup Babe)
- Stiker (Jika user minta ini, dimasukkan saja sebagai notes beserta jumlahnya pakai bahasa natural, tidak perlu dimasukan ke dalam products)
- Segala macam rokok (Jika user minta ini, masukkan ke dalam ordered_products sesuai dengan nama rokok yang diminta dengan quantity yang diinginkan. Tipe rokok adalah "Item")
- Nitip ke Jagoane Babe (Item) (Jika user meminta sesuatu, terdeteksi bila ada kata "nitip" atau "titip" disitu)

5. Daftar Nilai Keterlambatan (dalam menit, namun isi dalam angkanya saja):
- Hujan: 5
- Macet: 5
- ETJ / Outsource ETJ: 15
- Numpuk: 10

6. Fallback: Jika format RECONFIRM JAJAN tidak sesuai atau ada data wajib yang hilang, kembalikan JSON berikut saja:
{
  "fallback": "Ada data yang kurang atau format tidak sesuai. Pastikan format pesan:

7. Contoh:

Pesan Pelanggan:

RECONFIRM JAJAN
Nama: Budi
Nomor Telepon: 08987654321
Produk: Atlas Lychee 2 Botol Promo, Singleton 500 mL (Item) 3, 3 botol Anggur Merah 500 mL (Item)
Tuker Voucher: Tumblr
Alamat: Jl. Melati No. 5, Bandung
Payment: QRIS
Disc: 10%
Notes: Es Batu, Sticker 2, nitip rokok
CC: BL

EX, lunas

Contoh Output JSON BENAR:
{
  "cust_name": "Budi",
  "phone_num": "08987654321",
  "disc": 0.1,
  "ordered_products": [
    {
      "tipe": "Paket",
      "produk": "Atlas Lychee 2 Botol Promo",
      "quantity": 1
    },
    {
      "tipe": "Item",
      "produk": "Singleton 500 mL",
      "quantity": 3
    },
    {
      "tipe": "Item",
      "produk": "Anggur Merah 500 mL",
      "quantity": 3
    },
    {
      "tipe": "Paket",
      "produk": "Tukar Voucher Tumblr",
      "quantity": 1
    },
    {
      "tipe": "Item",
      "produk": "Es Batu",
      "quantity": 1
    },
    {
      "tipe": "Paket",
      "produk": "Express Delivery!!",
      "quantity": 1
    },
    {
      "tipe": "Item",
      "produk": "Nitip ke Jagoane Babe",
      "quantity": 1
    }
  ],
  "address": "Jl. Melati No. 5, Bandung",
  "payment_type": "QRIS",
  "notes": "Es batu, Tambahan stiker 2, Nitip rokok, Tukar Voucher Tumblr (Item)",
  "jenis_pengiriman": "EX",
  "status": "Lunas",
  "cc": "BL",
  "tambahan_waktu": 0,
}

Contoh Output JSON SALAH (fallback):
{
  "cust_name": "Budi",
  "phone_num": "08987654321",
  "disc": 0.1,
  "ordered_products": [
    {
      "tipe": "Paket",
      "produk": "Atlas Lychee 2 Botol Promo",
      "quantity": 2
    },
    {
      "tipe": "Item",
      "produk": "3 Singleton 500 mL",
      "quantity": 1
    },
    {
      "tipe": "Item",
      "produk": "3 botol Anggur Merah 500 mL",
      "quantity": 1
    },
    {
      "tipe": "Paket",
      "produk": "Tukar Voucher Tumblr",
      "quantity": 1
    },
    {
      "tipe": "Item",
      "produk": "Es Batu",
      "quantity": 1
    },
  ],
  "address": "Jl. Melati No. 5, Bandung",
  "payment_type": "QRIS",
  "notes": "Tambahan stiker 2, nitip rokok",
  "jenis_pengiriman": "EX",
  "status": "Lunas",
  "tambahan_waktu": 0,
}

CONTOH LAIN:

RECONFIRM JAJAN
Nama : Aldi
Nomor Telepon : 08123456789
Produk : Atlas Lychee (1 item)
Alamat : https://maps.app.goo.gl/YfNFSH4dsgHyGAXe6
Payment : Cash
Tukar Voucher : Free Instant
Notes : Req jagoan, macet
Disc : -
Pengiriman : I
CC : HQ

CONTOH OUTPUT YANG BENAR:
{
  "cust_name": "Aldi",
  "phone_num": "08123456789",
  "disc": 0,
  "ordered_products": [
    {
      "tipe": "Item",
      "produk": "Atlas Lychee",
      "quantity": 1
    },
    {
      "tipe": "Item",
      "produk": "Request Jagoane Babe",
      "quantity": 1
    }
  ],
  "address": "https://maps.app.goo.gl/YfNFSH4dsgHyGAXe6",
  "payment_type": "Cash",
  "notes": "request jagoan, macet",
  "jenis_pengiriman": "FD",
  "status": "Lunas",
  "tambahan_waktu": 5,
  "cc": "HQ",
}

CATATAN DISKON TAMBAHAN:
- Diskon Atensi: 100%
- Diskon Giveaway: 100%
- Diskon Komplimen: 100% (INI HANYA JIKA ADA PERINTAH ATAU KATA "DISKON KOMPLIMEN", JIKA HANYA KOMPLIMEN TANPA TERTULIS DISKON KOMPLIMEN, TIDAK ADA DISKON DAN ABAIKAN INI)
- Diskon KOL: 100%
- Diskon Media Partner: 100%
- Diskon Ngacara: 100%
- Diskon RND: 100%
- Redeem Free Instant: Ini bukan diskon, tapi jika ada prompt seperti ini, gantikan item delivery apapun (baik FD, I, maupun EX) dengan "Tukar Voc Instant dari Babe! (Paket)" dengan quantity 1, dan gantikan jenis_pengiriman jadi I. Jangan tambahkan diskon apapun.
"""

item_selection_prompt = """
Anda adalah sebuah agent yang bertugas untuk memilih indeks data yang paling sesuai dengan prompt yang diberikan.
Cukup jawab dengan nomor indeks data yang paling sesuai dengan prompt yang diberikan.
Tidak perlu menjelaskan apapun, cukup jawab dengan nomor indeks data yang paling sesuai dengan format yang telah ditentukan.
Jika ada produk yang punya tulisan "- I" atau "- O", utamakan "- I" dahulu. Jangan pilih "- O" jika tidak dituliskan secara eksplisit.

CATATAN: JIKA TIDAK ADA PRODUK YANG SESUAI, JAWAB -99999
"""

combo_selection_prompt = """
Anda adalah AI Agent Kasir untuk Kulkas Babe (@kulkasbabe.id), bertugas memetakan permintaan pelanggan ke indeks paket yang tepat dalam database.

Aturan Utama
1. Anda akan diberikan List berisi JSON yang berisi daftar paket. Misal:
[{'id': 256319, 'name': 'Paket 2 Atlas Lychee [Promo April]'}, {'id': 256487, 'name': 'Paket 3 Botol Anker Lychee [Promo April]'}, {'id': 256548, 'name': 'Paket 3 Botol Vibe Lychee 700ml [Promo April]'}]
2. Cari satu entri dalam daftar tersebut yang paling cocok dengan query secara semantik:
- Pertimbangkan kuantitas: angka dapat berada setelah kata "Paket" (misal "Paket 2 ...") atau sebelum kata "Botol" (misal "2 Botol ...").
- Nama produk: cocokkan merek dan varian persis seperti input user (abaikan kapitalisasi).
- Promo dinamis: setiap paket diakhiri tag [Promo X], di mana X bisa nama bulan atau event. Pilih promo yang sesuai dengan kata setelah "Promo" dalam input user.
- Kombinasi multi-produk dipisah dengan tanda "+"; hanya cocokkan jika input user menyebutkan kedua komponen serta kuantitas masing-masing.
3. Jawab hanya dengan nomor indeks entri yang paling sesuai (BUKAN URUTANNYA TAPI ID NYA, TERAPKAN ATURAN INI DENGAN KETAT). Tidak perlu penjelasan atau teks tambahan.
4. Jika tidak ada entri yang sesuai, jangan beri jawaban apapun.

Contoh Daftar Paket (Ini jika tidak dalam bentuk JSON, misal komponen kiri merepresentasikan id nya dan diikuti nama paketnya):
13841. Paket 2 Daebak Soju Lemon [Promo Juni]
13252. Paket 2 Smirnoff Ice Lemon [Promo April]
35122. 6 Botol Atlas Rose Pink [Promo Juni]
45231. 10 Botol Cheosnun Green Grape [Promo Juni]
43576. 12 Singaraja Beer 620ml [Promo Juni]
88431. Paket Kolesom Biasa + Draft Beer [Promo Juni]
34963. Paket Anggur Merah Gold + Draft Beer [Promo Juni]
54133. Paket 2 Kawa Merah Gold + 2 Draft Beer [Promo Juni]
13412. Paket 3 Anggur Merah Biasa [Promo Juni]
55232. Paket QRO + Bintang Pilsener 620ml [Promo Juni]
33135. Paket Atlas Rose Pink + Singaraja 620ml [Promo Juni]
42551. Sababay Pink Blossom [Promo Juni]
23453. Iceland Vodka Lychee 500ml [Promo Juni]

Contoh Interaksi
Input: "Smirnoff Ice Lemon 2 Botol Promo April"
Output: 13252 (karena sesuai format "Paket 2 Smirnoff Ice Lemon [Promo April]")

Input: "6 Botol Atlas Rose Pink Promo Juni"
Output: 35122

Input: "Kawa Merah Gold 2 Draft Beer Promo Juni"
Output: 54133

JIKA TIDAK ADA PAKET YANG SESUAI, JAWAB -99999
PEMILIHAN ITEM TIDAK BOLEH SALAH TERUTAMA DALAM HAL KUANTITAS!!!!
"""

merch_selection_prompt = """
Kamu adalah AI Agent Kasir untuk Kulkas Babe (@kulkasbabe.id), bertugas memilih satu entri merch dari daftar berikut berdasarkan permintaan pelanggan.
Pilih satu entri garansi yang paling sesuai dengan permintaan pelanggan dan tuliskan saja IDnya langsung, misal 228431 (tipe integer). Tidak perlu penjelasan atau teks tambahan.
JIKA TIDAK ADA MERCH YANG SESUAI, JAWAB -99999
"""

garansi_selection_prompt = """
Kamu adalah AI Agent Kasir untuk Kulkas Babe (@kulkasbabe.id), bertugas memilih satu entri garansi dari daftar berikut berdasarkan permintaan pelanggan.
Pilih satu entri garansi yang paling sesuai dengan permintaan pelanggan dan tuliskan saja IDnya langsung, misal 228431 (tipe integer). Tidak perlu penjelasan atau teks tambahan.
JIKA TIDAK ADA GARANSI YANG SESUAI, JAWAB -99999
"""

kupon_selection_prompt = """
Kamu adalah AI Agent Kasir untuk Kulkas Babe (@kulkasbabe.id), bertugas memilih satu entri kupon dari daftar berikut berdasarkan permintaan pelanggan.
Pilih satu entri kupon yang paling sesuai dengan permintaan pelanggan dan tuliskan saja IDnya langsung, misal 228431 (tipe integer). Tidak perlu penjelasan atau teks tambahan.
JIKA TIDAK ADA KUPOON YANG SESUAI, JAWAB -99999
"""

voucher_selection_prompt = """
Kamu adalah AI Agent Kasir untuk Kulkas Babe (@kulkasbabe.id), bertugas memilih satu entri voucher dari daftar berikut berdasarkan permintaan pelanggan.
Pilih satu entri voucher yang paling sesuai dengan permintaan pelanggan, harus bener-bener sesuai dan tuliskan saja IDnya langsung, misal 228431 (tipe integer). Tidak perlu penjelasan atau teks tambahan.
JIKA TIDAK ADA VOUCHER YANG SESUAI, JAWAB -99999
"""

komplimen_selection_prompt = """
Kamu adalah AI Agent Kasir untuk Kulkas Babe (@kulkasbabe.id), bertugas memilih satu entri komplimen dari daftar berikut berdasarkan permintaan pelanggan.
Pilih satu entri voucher yang paling sesuai dengan permintaan pelanggan, harus bener-bener sesuai dan tuliskan saja IDnya langsung, misal 228431 (tipe integer). Tidak perlu penjelasan atau teks tambahan.
JIKA TIDAK ADA KOMPLIMEN YANG SESUAI, JAWAB -99999
"""

delivery_selection_prompt = """
Kamu adalah AI Agent Kasir untuk Kulkas Babe (@kulkasbabe.id), bertugas memilih satu entri delivery dari daftar berikut berdasarkan permintaan pelanggan.
Pilih satu entri delivery yang paling sesuai dengan permintaan pelanggan, harus bener-bener sesuai dan tuliskan saja IDnya langsung, misal 228431 (tipe integer). Tidak perlu penjelasan atau teks tambahan.
JIKA TIDAK ADA DELIVERY YANG SESUAI, JAWAB -99999
"""

hadiah_selection_prompt = """
Kamu adalah AI Agent Kasir untuk Kulkas Babe (@kulkasbabe.id), bertugas memilih satu entri hadiah dari daftar berikut berdasarkan permintaan pelanggan.
Pilih satu entri hadiah yang paling sesuai dengan permintaan pelanggan, harus bener-bener sesuai dan tuliskan saja IDnya langsung, misal 228431 (tipe integer). Tidak perlu penjelasan atau teks tambahan.
JIKA TIDAK ADA HADIAH YANG SESUAI, JAWAB -99999
"""

notes_prompt = """
Beri salam hangat kepada pelanggan, tidak usah menggunakan bold atau bahasa terlalu formal.
Sebutkan paket-paket promo yang dipesan. Jawab dengan gaya persis seperti di bawah ini.

Contoh input:
RECONFIRM JAJAN
Nama: Kevin
Nomor Telepon: 085853605806
Produk: 3 botol QRO anggur merah 650 ml (Item); Paket 3 AO Mild (Paket); 1 buah Paket 2 Anggur Hijau MCD + Kawa (Paket)
Alamat: Jl. Melati No. 5, Bandung
Disc: 10%
Payment: BCA
Notes: Es Batu 3
CC: BL

Contoh output:

"Makasih yaa (nama) niii Jajan mu langsung ta proses duluu. Paket nyaa (sebutkan paketnya) yah. Buat notes, Es Batu 3 (sesuaikan dengan notes dan jika ada free instant atau voucher juga dituliskan disini). - BL"
"""

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
        model_name: str = "gemini-1.5-flash",
        df_product_dir: str = "./kulkasbabe.csv",
        df_combo_dir: str = "./paket.csv",
        top_k_retrieve: int = 5,
        gmap_api_key: Optional[str] = None,
    ):
        self.instructions = instructions
        self.model_name = {
            "flash": "gemini-2.5-flash",
            "pro": "gemini-2.5-pro",
        }
        self.df_product_dir = df_product_dir
        self.df_combo_dir = df_combo_dir
        self.top_k_retrieve = top_k_retrieve
        self.longlat_toko = (-7.560745951139057, 110.8493297202405)
        self.gmap_api_key = gmap_api_key
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

        # self.product_df = pd.read_csv(self.df_product_dir)
        # self.combo_df = pd.read_csv(self.df_combo_dir)

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

        ######## DEBUG ########
        # print(f"[DEBUG] Query untuk BM25: {query}")
        # print(f"[DEBUG] Top-{self.top_k_retrieve} kandidat hasil BM25:")
        # print(sim_score_table)
        #######################

        LLM = genai.GenerativeModel(
            model_name=self.model_name["flash"],
            system_instruction=task_instruction,
        )
        idx = LLM.generate_content(f"Query: {query}, List: {sim_score_table}")
        # print("[DEBUG] Hasil dari LLM untuk ID:", idx.text)

        try:
            idx = int(idx.text)
            # print("[DEBUG] ID terpilih:", idx)
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
        product_df = pd.read_csv(self.df_product_dir)
        df = product_df[product_df["pos_hidden"] == 0]
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

        df_sel = df[df["id"] == idx].reset_index(drop=True)

        if df_sel.empty:
            logger.error("Data produk dengan id %s tidak ditemukan dalam df.", idx)
            # update_status(order_id, "X", access_token=access_token)
            print("ERROR ID 0 NIH, df_sel nya empty.")
            print("ID yang di retrieve agent: ", idx)
            print("Nama produk yang berusaha di retrieve agent: ", nama_produk)
            return (
                False,
                f"Produk dengan id {idx} tidak ditemukan. Mungkin terjadi perubahan pada database atau item sudah tidak tersedia.",
            )
        
        ##################### REVISI ############################
        # if df_sel.at[0, "variants"] != "[]":
        #     try:
        #         variants = ast.literal_eval(df_sel.at[0, "variants"])
        #         variants_df = pd.DataFrame(variants)
        #     except Exception as e:
        #         logger.error("Gagal parse variants untuk produk %s: %s", idx, e)
        #         return False, "Gagal memproses varian produk."
        #     variants_ready = variants_df[variants_df["stock_qty"] > 0]
        #     for prefix in ["C", "P", "L", "X"]:
        #         sel = variants_ready[variants_ready["name"].str.startswith(prefix)]
        #         if not sel.empty:
        #             variant_id = sel["id"].iloc[0]
        #             prodvar_id = f"{idx}|{variant_id}"
        #             logger.debug(
        #                 "Varian dipilih untuk item %s: %s", nama_produk, prodvar_id
        #             )
        #             break
        #########################################################################

        ################ PERBAIKAN ##############################################
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
        #########################################################################

        # cart.append(
        #     {
        #         "prod_id": idx,
        #         "prodvar_id": prodvar_id if "prodvar_id" in locals() else idx,
        #         "name": nama_produk,
        #         "qty": qty,
        #         "price": fetch_product_item_details(idx, access_token=access_token).get(
        #             "data", {}
        #         )["sell_price_pos"],
        #         "disc": 0,
        #     }
        # )
        try:
            print(f"ITEM  {nama_produk:<45} {product_id:<10} {qty:<15} {cart[-1]['disc']:<10}")
        except Exception as e:
            pass

        return (
            True,
            f"Item {nama_produk} berhasil ditambahkan ke order dengan ID {order_id}.",
        )

    def _process_paket(
        self,
        order_id: Any,
        nama_paket: str,
        qty_pesan: int,
        cart: list,
        access_token: str,
    ):
        """
        Proses paket: cari ID paket, fetch detail, tambahkan tiap item dalam paket,
        lalu update diskon tiap item.
        Mengembalikan True jika sukses, atau string pesan error jika gagal.
        """
        # Filter combo yang tidak hidden
        combo_df = pd.read_csv(self.df_combo_dir)
        df = combo_df[combo_df["pos_hidden"] == 0]
        task_instruction = self.instructions["combo_selection_prompt"]
        lower = nama_paket.strip().lower()

        # Sesuaikan filter berdasarkan pola nama
        if lower.startswith(("merch", "mer")):
            mask = df["name"].str.lower().str.contains("merch|merh")
            df = df[mask]
            task_instruction = self.instructions["merch_selection_prompt"]
        elif lower.startswith(("babe garansiin", "garansi", "garan")):
            df = df[df["name"] == "Babe Garansi-in !!!"]
            task_instruction = self.instructions["garansi_selection_prompt"]
        elif "kupon" in lower:
            df = df[df["name"].str.lower().str.contains("kupon")]
            task_instruction = self.instructions["kupon_selection_prompt"]
        elif "voucher" in lower:
            df = df[df["name"].str.lower().str.contains("voucher")]
            task_instruction = self.instructions["voucher_selection_prompt"]
        elif lower.startswith(("komplimen", "komp")):
            df = df[df["name"].str.lower().str.startswith(("komplimen", "komp"))]
            task_instruction = self.instructions["komplimen_selection_prompt"]
        elif "delivery" in lower:
            df = df[df["name"].str.lower().str.contains("delivery")]
            task_instruction = self.instructions["delivery_selection_prompt"]
        elif lower.startswith("hadiah"):
            df = df[df["name"].str.lower().str.startswith("hadiah")]
            task_instruction = self.instructions["hadiah_selection_prompt"]

        if df.empty:
            logger.error("Tidak ada paket matching untuk: %s", nama_paket)
            return f"Gagal menemukan paket: {nama_paket}"
        if len(df) == 1:
            paket_id = df["id"].iloc[0]
        else:
            paket_id = self.select_id_by_agent(
                nama_paket, df, task_instruction, id_col="id", evaluation_col="name"
            )
        if paket_id is None:
            logger.error("select_id_by_agent gagal untuk paket: %s", nama_paket)
            return f"Gagal menemukan paket: {nama_paket}"

        if paket_id == -99999:
            logger.error("Tidak ada paket yang sesuai dengan nama: %s", nama_paket)
            update_status(order_id, "X", access_token=access_token)
            return f"AI gagal menemukan paket yang sesuai: {nama_paket}, ini disebabkan karena AI tidak yakin dengan kecocokan antara nama yang dimasukkan dengan hasil pencarian yang ditemukan (untuk menghindari pengambilan asal). Untuk itu, mohon masukkan dengan format <Nama produk> (<QTY> Paket/Item), dan hindari penggunaan singkatan (AI tidak tahu konteks dalam singkatan itu). Sebisa mungkin, sertakan juga brand-nya apa agar menghindari kesalahpahaman AI, misal AM bisa dianggap dari Mix Max Anggur Merah, QRO Anggur Merah, atau Kawa Kawa Anggur Merah, tapi kalau ini tidak dianggap masalah, silakan diabaikan. CONTOH: 2 Atlas Lychee + 2 Beer (1 Paket). Jika error ini masih berlangsung, cek backoffice Olsera. Struk di-voidkan."

        # Ambil detail paket
        try:
            combo_details = fetch_product_combo_details(paket_id, access_token)
            combo_items = combo_details["data"]["items"]["data"]
        except Exception as e:
            logger.error("Gagal fetch combo details untuk %s: %s", paket_id, e)
            return "Gagal ambil detail paket."
        try:
            print(f"PAKET  {combo_details['data']['name']:<45} {combo_details['data']['id']:<10} {qty_pesan:<15}")
        except Exception as e:
            pass
        
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
            logger.error("Cart kosong, tidak ada item untuk dipindahkan.")
            return "Cart kosong, tidak ada item untuk dipindahkan."

        for item in cart:
            prodvar_id = item["prodvar_id"]
            qty = item["qty"]
            try:
                resp = add_prod_to_order(
                    order_id, prodvar_id, qty, access_token=access_token
                )
                if resp is None:
                    # update_status(order_id, "X", access_token=access_token)
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
                # update_status(order_id, "X", access_token=access_token)
                return (
                    False,
                    f"Ada kesalahan saat memasukkan produk ke order. Error: {e}",
                )

        # Setelah itu, tambahkan diskon untuk tiap item
        try:
            ord_detail = fetch_order_details(
                order_id=order_id, access_token=access_token
            )
            paket_items = ord_detail["data"]["orderitems"]
            # paket_items = pd.DataFrame(paket_items)
        except Exception as e:
            logger.error("Gagal fetch detail order untuk update diskon: %s", e)
            return False, "Gagal update diskon paket."

        # Hitung dan update tiap item
        for idx, item in enumerate(paket_items):
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
                logger.error(
                    "Gagal update_order_detail untuk item_id %s: %s", item_id, e
                )
                return False, "Gagal update detail paket di order."

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

        logger.debug("Query diterima: %s", query)
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

        # Add products to order
        ordered_products = reconfirm_json.get("ordered_products", [])
        for product in ordered_products:
            # print(f"\n[DEBUG] Memproses produk: {product['produk']} | Tipe: {product['tipe']}")
            tipe = product.get("tipe", "").lower()
            nama_produk = product.get("produk", "")
            qty = product.get("quantity", 0)

            if not nama_produk or qty <= 0:
                logger.warning(
                    "Produk diabaikan karena nama/qty tidak valid: %s", product
                )
                continue

            if tipe == "item":
                success, msg = self._process_item(
                    order_id, nama_produk, qty, cart_temp, access_token
                )
                time.sleep(1)  # Delay untuk menghindari rate limit API Olsera

                if not success:
                    # Jika error di dalam, void entire order dan return
                    update_status(order_id, "X", access_token)
                    return f"[ERROR Pada Orderan {reconfirm_json.get('cust_name')}({reconfirm_json.get('phone_num')})] Ada kesalahan saat menambahkan item, struk di-void. Error: {msg}"

            elif tipe == "paket":
                success_or_msg = self._process_paket(
                    order_id, nama_produk, qty, cart_temp, access_token
                )
                time.sleep(1)
                if success_or_msg != True:
                    # Jika mengembalikan string pesan error, batalkan order
                    update_status(order_id, "X", access_token)
                    return success_or_msg

            else:
                print(
                    f"[ERROR Pada Orderan {reconfirm_json.get('cust_name')}({reconfirm_json.get('phone_num')})] Jenis tidak dikenali. Pastikan untuk memasukkan produk dengan kurung () yang menjelaskan jenis produk, apakah item atau paket. Misal: Hennesey 650 mL (item). Anda memasukkan: {product['tipe']}"
                )
                continue

        # Auto add merch
        # total_amount = int(float(order_details['data']['total_amount']))
        cart_df = pd.DataFrame(cart_temp)
        # print("Keranjang Sementara:", cart_df)
        total_amount = (
            (
                cart_df["price"].astype(float) * cart_df["qty"].astype(int)
                - cart_df["disc"].astype(float)
            ).sum()
            if not cart_df.empty
            else 0
        )
        # (df_new['price'].astype(float) * df_new['qty'].astype(int) - df_new['disc'].astype(float)).sum()
        # print("Harga Total: ", total_amount)

        if total_amount < 100000:
            self._process_item(order_id, "Cup Babe", 1, cart_temp, access_token)
            time.sleep(1)  # Delay untuk menghindari rate limit API Olsera

        else:
            self._process_item(order_id, "Cup Babe", 2, cart_temp, access_token)
            time.sleep(1)  # Delay untuk menghindari rate limit API Olsera

        if total_amount > 150000 and total_amount < 250000:
            self._process_paket(order_id, "Merch Babe 1", 1, cart_temp, access_token)
            time.sleep(1)  # Delay untuk menghindari rate limit API Olsera

        elif total_amount >= 250000:
            self._process_paket(order_id, "Merch Babe 2", 1, cart_temp, access_token)
            time.sleep(1)  # Delay untuk menghindari rate limit API Olsera

        else:
            pass

        agg_cart = self.aggregate_cart_by_prodvar(cart_temp)
        cart_stat, msg = self.move_cart_to_order(agg_cart, order_id, access_token)
        if not cart_stat:
            logger.error("Gagal memindahkan cart ke order: %s", msg)
            update_status(order_id, "X", access_token)
            return f"[ERROR Pada Orderan {reconfirm_json.get('cust_name')}({reconfirm_json.get('phone_num')})] Ada error, detailnya: {msg}"

        # Tambahkan diskon --- DIMATIKAN SEMENTARA UNTUK DEBUGGING
        try:
            disc_stat, disc_msg = self.add_discount(
                order_id,
                mode=reconfirm_json["mode_diskon"],
                access_token=access_token,
                discount=reconfirm_json["disc"],
                notes="",
            )
            if disc_stat is not True:
                logger.error("Gagal menambahkan diskon: %s", disc_msg)
                update_status(order_id, "X", access_token)
                return f"[ERROR Pada Orderan {reconfirm_json.get('cust_name')}({reconfirm_json.get('phone_num')})] Terdapat kesalahan dalam pengecekan diskon: {disc_msg}"
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

        # Proses pembayaran --- DIMATIKAN SEMENTARA UNTUK DEBUGGING
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

        invoice_lines = [
            *status_lines,
            "",
            f"Nama: {reconfirm_json.get('cust_name', '')}",
            f"Nomor Telepon: {reconfirm_json.get('phone_num', '')}",
            f"Alamat: {reconfirm_json.get('address', '')}",
            "",
            "",
            max_luncur_line.strip(),
            f"Jarak: {int(reconfirm_json['distance']) if reconfirm_json['distance'] > 14 else f'{reconfirm_json['distance']:.1f}'} km (*{kelurahan}, {kecamatan.replace('Kecamatan ', '').replace('Kec. ', '').replace('kecamatan', '').replace('kec.', '')}*)",
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
        return invoice

        # return cart_temp

    def add_discount(self, order_id, mode, access_token, discount=0, notes=""):
        ord_dtl = fetch_order_details(order_id, access_token)
        id_order = ord_dtl["data"].get("id", 0)
        total_price = float(ord_dtl["data"].get("total_amount", 0))
        order_list = pd.DataFrame(ord_dtl["data"].get("orderitems", []))
        if order_list.empty:
            print("Tidak ada item dalam order, tidak bisa menambahkan diskon.")
            return (
                False,
                "Tidak ada item dalam order ini, proses pengecekan diskon dilewatkan. Jika anda merasa sudah memasukkan item namun error ini muncul, kemungkinan besar item Anda gagal dimasukkan ke keranjang. Silakan coba lagi.",
            )

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
            return True, "Berhasil update diskon untuk semua item."
        elif success_count > 0:
            return False, f"Sebagian berhasil. Gagal untuk: {'; '.join(error_messages)}"
        else:
            return False, "Gagal update diskon untuk semua item."
