from typing import Optional
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import pandas as pd
import uvicorn
import json

pd.set_option("display.max_columns", None)

# --- constants ---
DROP_COLS = [
    "photo_md", "collections", "photo_xs", "category_name", "brand_id", "brand_name",
    "published", "pos_hidden", "description", "notes", "published_date",
    "store_name", "store_url", "length_dimension", "width_dimension", "high_dimension", "weight_dimension"
]

RENAME_MAP = {
    "name": "Nama Produk",
    "variant_name": "Varian",
    "klasifikasi": "Kategori",
    "low_stock_alert": "Low Stock Alert",
    "buy_price_final": "Harga Beli",
    "sell_price_pos_final": "Harga Jual di POS",
    "stock_qty_final": "Stock Qty",
}


# --- pipeline functions ---
def get_product_page(session: requests.Session, access_token: str, page: int = 1):
    url = "https://api-open.olsera.co.id/api/open-api/v1/en/product"
    params = {"per_page": 100, "page": page}
    headers = {"Authorization": f"Bearer {access_token}"}
    resp = session.get(url, params=params, headers=headers)
    resp.raise_for_status()
    return resp.json()


def fetch_all_products(access_token: str):
    items = []
    page = 1
    with requests.Session() as s:
        while True:
            sample = get_product_page(s, access_token, page=page)
            if not sample or "data" not in sample or not sample["data"]:
                break
            items.extend(sample["data"])
            page += 1
    return items


def transform_products_to_clean_json(items: list):
    item_df = pd.DataFrame(items).drop(columns=DROP_COLS)

    exploded = item_df.explode("variants").reset_index(drop=True)

    variants_expanded = pd.json_normalize(exploded["variants"])
    variants_expanded = variants_expanded.rename(
        columns={"id": "variant_id", "product_id": "id", "name": "variant_name"}
    )

    merged = pd.concat([exploded.drop(columns=["variants"]), variants_expanded], axis=1)

    merged_filt = merged[
        ["name", "variant_name", "klasifikasi", "stock_qty", "buy_price", "sell_price_pos", "low_stock_alert"]
    ]

    df = merged_filt.copy()

    # --- 1. Ambil buy_price kolom kedua ---
    buy_price_idx = [i for i, col in enumerate(df.columns) if col == "buy_price"][1]
    df["buy_price_final"] = df.iloc[:, buy_price_idx]

    # --- 2. Ambil sell_price_pos kolom pertama ---
    sell_price_idx = [i for i, col in enumerate(df.columns) if col == "sell_price_pos"][0]
    df["sell_price_pos_final"] = df.iloc[:, sell_price_idx]

    # --- 3. Ambil stock_qty = max dari semua kolom stock_qty ---
    stock_qty_idx = [i for i, col in enumerate(df.columns) if col == "stock_qty"]
    stock_qty_values = df.iloc[:, stock_qty_idx].apply(pd.to_numeric, errors="coerce")
    df["stock_qty_final"] = stock_qty_values.max(axis=1, skipna=True)

    df_cleaned = df.drop(columns=["buy_price", "sell_price_pos", "stock_qty"])
    df_cleaned = df_cleaned.replace([float("inf"), float("-inf")], None)
    df_cleaned = df_cleaned.where(pd.notnull(df_cleaned), None)
    df_cleaned_json = df_cleaned.rename(columns=RENAME_MAP).to_dict(orient="records")
    return df_cleaned_json


def build_product_pipeline(access_token: str):
    items = fetch_all_products(access_token)
    return transform_products_to_clean_json(items)


# --- FastAPI app ---
app = FastAPI(title="Product Pipeline API (Olsera)")

# CORS supaya bisa dipanggil dari Apps Script / frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TokenBody(BaseModel):
    access_token: Optional[str] = None


# @app.post("/products", response_model=list)
# def products_endpoint(body: TokenBody, authorization: Optional[str] = Header(None)):
#     """
#     Endpoint utama:
#     - Bisa pakai Authorization header: Bearer <token>
#     - Atau body JSON: {"access_token": "<token>"}
#     """
#     with open("./token_cache.json") as f:
#         token_cache = json.load(f)

#     access_token = token_cache.get("access_token", "")

#     if authorization:
#         if authorization.lower().startswith("bearer "):
#             access_token = authorization.split(" ", 1)[1].strip()
#         else:
#             access_token = authorization.strip()
#     elif body.access_token:
#         access_token = body.access_token

#     if not access_token:
#         raise HTTPException(status_code=400, detail="Missing access_token")

#     try:
#         df_cleaned_json = build_product_pipeline(access_token)
#         return JSONResponse(content=df_cleaned_json)
#     except requests.exceptions.HTTPError as e:
#         raise HTTPException(status_code=502, detail=f"Upstream API error: {str(e)}")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/products", response_model=list)
def products_endpoint(authorization: Optional[str] = Header(None)):
    """
    Endpoint utama (GET):
    - Bisa pakai Authorization header: Bearer <token>
    - Jika tidak ada, fallback ke token_cache.json
    """
    with open("./token_cache.json") as f:
        token_cache = json.load(f)

    access_token = token_cache.get("access_token", "")

    if authorization:
        if authorization.lower().startswith("bearer "):
            access_token = authorization.split(" ", 1)[1].strip()
        else:
            access_token = authorization.strip()

    if not access_token:
        raise HTTPException(status_code=400, detail="Missing access_token")

    try:
        # df_cleaned_json = build_product_pipeline(access_token)
        # 1. Setelah selesai pipeline
        df = build_product_pipeline(access_token)
        print("Berhasil")

        return JSONResponse(content=df)
    except requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Upstream API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "ok"}


# --- entrypoint supaya bisa jalan langsung pakai python ---
if __name__ == "__main__":
    uvicorn.run("server_GAS:app", host="0.0.0.0", port=7231, reload=False)