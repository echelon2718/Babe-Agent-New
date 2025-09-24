from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import time
import requests
import re
from datetime import datetime, timedelta
import urllib.parse
from math import ceil

instan_siang =  [25, 25, 25, 30, 30, 35, 35, 40, 40, 45, 50, 55, 55, 55, 55, 55, 55, 65, 65, 75]
express_siang = [20, 20, 20, 25, 25, 25, 30, 30, 30, 35, 45, 45, 45, 45, 50, 50, 50, 50, 50, 60]

instan_malam =  [20, 20, 25, 30, 30, 30, 35, 35, 35, 45, 45, 45, 50, 50, 50, 55, 55, 60, 60, 70]
express_malam = [15, 15, 15, 25, 25, 25, 25, 25, 25, 40, 40, 40, 40, 45, 50, 50, 50, 50, 50, 60]

def address_to_latlng(address, api_key):
    """
    Mengubah alamat menjadi koordinat latitude dan longitude menggunakan Google Maps Geocoding API.
    
    Params:
    - address: str, alamat seperti "Jl. Sudirman, Jakarta"
    - api_key: str, API key Google Maps

    Returns:
    - (lat, lng): tuple of float, atau (None, None) jika gagal
    """
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": address,
        "key": api_key
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    if data["status"] != "OK":
        return None, None

    location = data["results"][0]["geometry"]["location"]
    lat = location["lat"]
    lng = location["lng"]

    return lat, lng


# def resolve_maps_shortlink(shortlink, api_key):
#     # Step 1: Buka browser headless dan resolve shortlink
#     options = Options()
#     options.add_argument("--headless=new")  # Headless modern
#     options.add_argument("--no-sandbox")  # Wajib untuk VPS
#     options.add_argument("--disable-dev-shm-usage")  # Hindari crash karena RAM VPS kecil
#     options.add_argument("--disable-gpu")
#     options.add_argument("--disable-software-rasterizer")
#     options.add_argument("--remote-debugging-port=9222")  # ← Fix penting!
#     options.add_argument("--user-data-dir=/tmp/selenium")  # Biar tidak konflik profile

#     driver = webdriver.Chrome(options=options)

#     driver.get(shortlink)
#     time.sleep(5)  # Tunggu JS redirect
#     final_url = driver.current_url
#     driver.quit()

#     # Step 2: Ekstrak koordinat dari URL hasil redirect
#     match = re.search(r'/@(-?\d+\.\d+),(-?\d+\.\d+)', final_url)
#     if not match:
#         return None, None
#     lat, lng = map(float, match.groups())

#     # Step 3: Ambil alamat dari koordinat via Geocoding API
#     endpoint = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&key={api_key}"
#     response = requests.get(endpoint)
#     data = response.json()

#     if data["status"] != "OK":
#         return None, (lat, lng)
    
#     address = data["results"][0]["formatted_address"]
#     return address, (lat, lng)


# def resolve_maps_shortlink(shortlink, api_key):
#     """
#     Resolve a Google Maps shortlink or share link to full address and components.

#     Returns:
#         address (str) or None,
#         (lat, lng) tuple or (None, None),
#         kelurahan, kecamatan, kota, provinsi or None each
#     """
#     # Try simple HTTP redirect resolution first
#     try:
#         resp = requests.get(shortlink, allow_redirects=True, timeout=10)
#         final_url = resp.url
#     except requests.RequestException:
#         final_url = None

#     # Fallback to headless browser if necessary
#     if not final_url or ('maps' not in final_url):
#         options = Options()
#         options.add_argument("--headless=new")
#         options.add_argument("--no-sandbox")
#         options.add_argument("--disable-dev-shm-usage")
#         options.add_argument("--disable-gpu")
#         options.add_argument("--disable-software-rasterizer")
#         options.add_argument("--remote-debugging-port=9222")
#         options.add_argument("--user-data-dir=/tmp/selenium")

#         driver = webdriver.Chrome(options=options)
#         driver.get(shortlink)
#         # Wait for URL to contain coordinates or map data
#         WebDriverWait(driver, 10).until(
#             lambda d: '/@' in d.current_url or '!3d' in d.current_url
#         )
#         final_url = driver.current_url
#         driver.quit()

#     # Patterns for coordinates in URL
#     lat = lng = None
#     # Pattern 1: /@lat,lng
#     m = re.search(r'/@(-?\d+\.\d+),(-?\d+\.\d+)', final_url or '')
#     if m:
#         lat, lng = map(float, m.groups())
#     else:
#         # Pattern 2: !3dlat!4dlng
#         m = re.search(r'!3d(-?\d+\.\d+)!4d(-?\d+\.\d+)', final_url or '')
#         if m:
#             lat, lng = map(float, m.groups())

#     if lat is None or lng is None:
#         return None, (None, None), None, None, None, None

#     # Reverse geocode
#     endpoint = (
#         f"https://maps.googleapis.com/maps/api/geocode/json"
#         f"?latlng={lat},{lng}&key={api_key}"
#     )
#     try:
#         geo = requests.get(endpoint).json()
#         if geo.get("status") != "OK":
#             return None, (lat, lng), None, None, None, None
#         result = geo["results"][0]
#     except requests.RequestException:
#         return None, (lat, lng), None, None, None, None

#     address = result.get("formatted_address")
#     comps = result.get("address_components", [])
#     kelurahan = kecamatan = kota = provinsi = None
#     for comp in comps:
#         types = comp.get("types", [])
#         if "administrative_area_level_4" in types:
#             kelurahan = comp.get("long_name")
#         elif "administrative_area_level_3" in types:
#             kecamatan = comp.get("long_name")
#         elif "administrative_area_level_2" in types:
#             kota = comp.get("long_name")
#         elif "administrative_area_level_1" in types:
#             provinsi = comp.get("long_name")

def resolve_maps_shortlink(shortlink, api_key, timeout=10):
    # --- Step 1: Open headless browser and resolve shortlink ---
    options = Options()
    options.page_load_strategy = 'none'
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-software-rasterizer")
    options.add_argument("--remote-debugging-port=9222")
    options.add_argument("--user-data-dir=/tmp/selenium")

    driver = webdriver.Chrome(options=options)
    driver.get(shortlink)
    start_time = time.time()
    final_url = driver.current_url
    # Tunggu hingga URL berubah atau timeout
    while time.time() - start_time < timeout:
        url = driver.current_url
        # Matikan saat url mengandung /maps/place/
        if '/maps/place/' in url:
            final_url = url
            break
        time.sleep(0.1)
    driver.quit()

    # --- Step 2: Extract place name ---
    place_match = re.search(r'/maps/place/([^/]+)', final_url)
    place_name = urllib.parse.unquote_plus(place_match.group(1)) if place_match else None

    # --- Step 3: Extract coordinates ---
    lat = lng = None
    # Coba format @lat,lng
    match_at = re.search(r'/@(-?\d+\.\d+),(-?\d+\.\d+)', final_url)
    if match_at:
        lat, lng = map(float, match_at.groups())
    else:
        # Coba format !3dlat!4dlng
        match_34 = re.search(r'!3d(-?\d+\.\d+)!4d(-?\d+\.\d+)', final_url)
        if match_34:
            lat, lng = map(float, match_34.groups())

    # --- Step 4: Geocoding API ---
    if lat is not None and lng is not None:
        endpoint = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&key={api_key}"
    elif place_name:
        addr = urllib.parse.quote_plus(place_name)
        endpoint = f"https://maps.googleapis.com/maps/api/geocode/json?address={addr}&key={api_key}"
    else:
        return None, None, None, None, None, None, None

    res = requests.get(endpoint).json()
    if res.get('status') != 'OK' or not res.get('results'):
        return place_name, (lat, lng), None, None, None, None, place_name

    info = res['results'][0]
    formatted_address = info.get('formatted_address')

    # --- Step 5: Extract components ---
    kelurahan = kecamatan = kota = provinsi = None
    for comp in info.get('address_components', []):
        types = comp.get('types', [])
        if any(t in types for t in ['administrative_area_level_4','sublocality_level_1','locality']):
            kelurahan = kelurahan or comp.get('long_name')
        if 'administrative_area_level_3' in types:
            kecamatan = comp.get('long_name')
        if 'administrative_area_level_2' in types:
            kota = comp.get('long_name')
        if 'administrative_area_level_1' in types:
            provinsi = comp.get('long_name')

    return formatted_address, (lat, lng), kelurahan, kecamatan, kota, provinsi

def get_travel_distance(origin, destination, api_key, mode="driving"):
    """
    origin: tuple (lat1, lng1)
    destination: tuple (lat2, lng2)
    mode: driving, walking, bicycling, transit
    """
    base_url = "https://maps.googleapis.com/maps/api/distancematrix/json"

    params = {
        "origins": f"{origin[0]},{origin[1]}",
        "destinations": f"{destination[0]},{destination[1]}",
        "mode": mode,
        "key": api_key
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    if data["status"] != "OK":
        return None

    row = data["rows"][0]["elements"][0]
    if row["status"] != "OK":
        return None

    distance_text = row["distance"]["text"]
    distance_meters = row["distance"]["value"]
    duration_text = row["duration"]["text"]
    duration_seconds = row["duration"]["value"]

    return {
        "distance_text": distance_text,
        "distance_meters": distance_meters,
        "duration_text": duration_text,
        "duration_seconds": duration_seconds
    }

def get_fastest_route_details(origin, destination, api_key, mode="driving"):
    """
    Menemukan rute tercepat antara origin dan destination menggunakan Google Directions API,
    dengan mempertimbangkan lalu lintas saat ini.

    Args:
        origin (tuple): Tuple koordinat (latitude, longitude) untuk titik awal.
        destination (tuple): Tuple koordinat (latitude, longitude) untuk titik tujuan.
        api_key (str): Kunci API Google Maps Anda.
        mode (str, optional): Mode perjalanan. Pilihan: "driving", "walking", "bicycling", "transit". Defaultnya adalah "driving".

    Returns:
        dict: Sebuah dictionary berisi detail rute, atau None jika terjadi kesalahan.
              Contoh:
              {
                  "distance_text": "20.3 km",
                  "distance_meters": 20299,
                  "duration_text": "35 mins",        // Durasi tanpa lalu lintas berat
                  "duration_seconds": 2103,
                  "duration_in_traffic_text": "45 mins", // Durasi dengan lalu lintas saat ini
                  "duration_in_traffic_seconds": 2700
              }
    """
    # Gunakan endpoint Directions API
    base_url = "https://maps.googleapis.com/maps/api/directions/json"

    params = {
        "origin": f"{origin[0]},{origin[1]}",
        "destination": f"{destination[0]},{destination[1]}",
        "mode": mode,
        "key": api_key,
        # Kunci utama untuk rute tercepat: minta data berdasarkan waktu sekarang
        # Ini akan mengaktifkan perhitungan berdasarkan lalu lintas real-time.
        "departure_time": "now"
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Akan memunculkan error jika status code bukan 2xx
        data = response.json()

        if data["status"] != "OK" or not data.get("routes"):
            print(f"Error dari API: {data.get('status')}. Pesan: {data.get('error_message', 'Tidak ada rute ditemukan.')}")
            return None

        # Directions API mengembalikan daftar rute, rute pertama biasanya yang terbaik/direkomendasikan.
        # Rute terdiri dari beberapa "leg" (segmen), untuk A->B hanya ada 1 leg.
        leg = data["routes"][0]["legs"][0]

        distance_text = leg["distance"]["text"]
        distance_meters = leg["distance"]["value"]
        
        # Durasi standar (tanpa lalu lintas padat)
        duration_text = leg["duration"]["text"]
        duration_seconds = leg["duration"]["value"]
        
        # Durasi dengan lalu lintas saat ini (jika tersedia)
        duration_in_traffic_text = leg.get("duration_in_traffic", {}).get("text", duration_text)
        duration_in_traffic_seconds = leg.get("duration_in_traffic", {}).get("value", duration_seconds)


        return {
            "distance_text": distance_text,
            "distance_meters": distance_meters,
            "duration_text": duration_text,
            "duration_seconds": duration_seconds,
            "duration_in_traffic_text": duration_in_traffic_text,
            "duration_in_traffic_seconds": duration_in_traffic_seconds
        }

    except requests.exceptions.RequestException as e:
        print(f"Terjadi kesalahan saat melakukan request: {e}")
        return None
    except KeyError as e:
        print(f"Gagal mem-parsing respons JSON, key tidak ditemukan: {e}")
        return None

def distance_cost_rule(dist: float, is_free: bool = False) -> str:
    if is_free and dist >= 9.5 and dist < 14:
        return "Subsidi Ongkir 10K"

    if dist < 9.5:
        return "Gratis Ongkir"
        
    if dist >= 9.5 and dist <= 13.9:
        return "Ongkir 10K"
    
    elif dist > 14 and dist <= 18.9:
        return "Ongkir 15K"
    
    elif dist > 18.9 and dist <= 23.9:
        return "Ongkir 20K"
    
    elif dist > 23.9 and dist <= 28.9:
        return "Ongkir 25K"
    
    elif dist > 28.9 and dist <= 33.9:
        return "Ongkir 30K"
    
    elif dist > 33.9 and dist <= 38.9:
        return "Ongkir 35K"
    
    elif dist > 38.9 and dist <= 43.9:
        return "Ongkir 40K"
    
    elif dist > 43.9 and dist <= 48.9:
        return "Ongkir 45K"
    
    else:
        return "Ongkir 45K"

def is_free_delivery(address, free_areas):
    address = address.lower()
    free_areas = [area.lower() for area in free_areas]
    """
    Mengecek apakah alamat mengandung salah satu kata dari daftar area gratis ongkir.

    Params:
    - address: str, alamat lengkap
    - free_areas: list of str, daftar area dengan ongkir gratis

    Returns:
    - bool, True jika alamat mengandung salah satu area, False jika tidak
    - area_matched: str atau None
    """
    address_lower = address.lower()
    for area in free_areas:
        if area.lower() in address_lower:
            return True, area
    return False, None

def waktu_siang(dt: datetime):
    return dt.hour >= 11 and dt.hour < 20

def waktu_malam(dt: datetime):
    return dt.hour >= 20 or dt.hour < 4

def estimasi_tiba(jarak_km: float, tipe: str, waktu_mulai: datetime) -> datetime:
    tipe = tipe.upper()
    km_index = ceil(jarak_km) - 1  # index 0 = km 1

    # Tentukan mode waktu
    if waktu_siang(waktu_mulai):
        instan = instan_siang
        express = express_siang
    elif waktu_malam(waktu_mulai):
        instan = instan_malam
        express = express_malam
    else:
        raise ValueError("Jam operasional hanya 11:00-19:00 (siang) atau 20:00-04:00 (malam)")

    # Hitung waktu tambahan
    if tipe == "FD":
        waktu_tambah = timedelta(minutes=35)
    elif tipe == "I":
        if km_index >= len(instan):
            raise ValueError("Jarak terlalu jauh untuk pengiriman Instan (maks 20 km)")
        waktu_tambah = timedelta(minutes=instan[km_index])
    elif tipe == "EX":
        if km_index >= len(express):
            raise ValueError("Jarak terlalu jauh untuk pengiriman Express (maks 20 km)")
        waktu_tambah = timedelta(minutes=express[km_index])
    else:
        raise ValueError("Tipe harus 'FD', 'I', atau 'EX'")

    return (waktu_mulai + waktu_tambah).strftime('%H:%M')