import json
import pika
import time
import requests
from datetime import datetime, timedelta
from modules.maps_utility import estimasi_tiba

credentials = pika.PlainCredentials('guest', 'guest')
parameters = pika.ConnectionParameters(
    host='31.97.106.30',
    port=5679,
    credentials=credentials
)

connection = pika.BlockingConnection(parameters)
channel = connection.channel()


channel.queue_declare(queue='whatsapp_hook_queue', durable=True)
channel.queue_declare(queue='whatsapp_message_queue', durable=True)

def forward_struk(payload: dict):
    # Dapatkan list grup yang akan di forward
    group_ids = requests.get("http://31.97.106.30:3000/api/groups/active").json().get("data", None)

    # Buat estimasi tiba
    max_luncur_str = estimasi_tiba(
        payload.get("distance", 0),
        payload.get("jenis_pengiriman", 0),
        datetime.now(),
    )

    max_luncur_dt = datetime.combine(datetime.today(), datetime.strptime(max_luncur_str, "%H:%M").time())
    max_luncur_dt += timedelta(minutes=int(float(payload.get("tambahan_waktu", 0))))

    max_luncur = max_luncur_dt.strftime("%H:%M")

    max_luncur_line = (
        f"MAKSIMAL DILUNCURKAN DARI GUDANG: {max_luncur}"
        if payload.get("jenis_pengiriman") == "FD"
        else f"ESTIMASI SAMPAI: {max_luncur}"
    )

    invoice_lines = [
        f"Nama: {payload.get('cust_name', 'Unknown')}",
        f"Nomor Telepon: {payload.get('phone_number', 'Tidak diketahui')}",
        f"Alamat: {payload.get('address', 'Tidak diketahui')}",
        "",
        "",
        max_luncur_line.strip(),
        f"Jarak: {int(payload.get('distance', 0)) if payload.get('distance', 0) > 14 else f'{payload.get('distance', 0):.1f}'} km (*{payload.get('kelurahan', 'Unk. Kelurahan')}, {payload.get('kecamatan', '').replace('Kecamatan ', '').replace('Kec. ', '').replace('kecamatan', '').replace('kec.', '')}*)",
        "",
        "",
        "Makasih yaa Cah udah Jajan di Babe!",
        f"Total Jajan: {payload.get('total_amount', 0)} (*{payload.get('payment_type', '').upper()}*)",
        f"Cek Jajanmu di sini: {payload.get('struk_url', '')}"
        f"Jam Order: *{datetime.now().strftime('%H:%M')}*",
        "",
        "",
        f"Jenis Pengiriman: {payload.get('jenis_pengiriman', '')}",
        f"*NOTES: {payload.get('notes') or 'Tidak ada catatan tambahan.'}*",

    ]

    invoice = "\n".join([line for line in invoice_lines if line is not None])

    try:
        for group_id in group_ids:
                group_payload = {
                    "command": "send_message",
                    "number": payload.get("from_number"),
                    "number_recipient": group_id.get("groupId"),
                    "message": invoice
                }
                channel.basic_publish(
                    exchange='',
                    routing_key='whatsapp_message_queue',
                    body=json.dumps(group_payload),
                    properties=pika.BasicProperties(
                        delivery_mode=2
                    )
                )
                time.sleep(3)

        return {
            "status": 200,
            "message": "Invoice forwarded successfully",
            "content": invoice
        }

    except Exception as e:
        return {
            "status": 500,
            "message": f"Ada error: {str(e)}",
            "content": invoice
        }