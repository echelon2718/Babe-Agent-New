from modules.llm_v3_review import AgentBabe
# from modules.llm_call_new_v2 import AgentBabe
from dotenv import load_dotenv
import os
import google.generativeai as genai
load_dotenv()

gmap_api_key = os.getenv("GMAP_API_KEY")
genai_api_key = os.getenv("GOOGLE_GENAI_API_KEY")
gmap_api_key = os.getenv("GMAP_API_KEY")
genai.configure(api_key=genai_api_key)

agent = AgentBabe(df_combo_dir='./product_combos_v2.csv', df_product_dir='./product_items.csv', top_k_retrieve=100, gmap_api_key=gmap_api_key)

agent.handle_order(
    '''
!babe

RECONFIRM JAJAN

Nama : INI TES AI, JANGAN DIPROSES (Evaluasi error forwarding)
Nomor Telepon : +62 813-9288-5302
Produk : singaraja (1 item), draft beer (3 item),Paket Anggur Merah Gold + Draft Beer [Promo September] (1 Paket)
Alamat : https://maps.app.goo.gl/zD2n21vXfbNvyQR67
Payment : cash
Tukar Voucher : -
Notes : JANGAN DIPROSES, INI TES AI
Disc : -
Pengiriman: FD
CC : CC
''',access_token_dir="./storage/app/token_cache.json",
    sudah_bayar=True,
)