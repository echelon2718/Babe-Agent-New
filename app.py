import pandas as pd
import json
import os
import pika
from dotenv import load_dotenv
# from modules.llm_call_new_v2 import AgentBabe
from modules.llm_v3_review import AgentBabe
import google.generativeai as genai
pd.options.mode.chained_assignment = None  
import threading
import time
import requests

load_dotenv()
genai_api_key = os.getenv("GOOGLE_GENAI_API_KEY")
gmap_api_key = os.getenv("GMAP_API_KEY")
genai.configure(api_key=genai_api_key)


agent = AgentBabe(df_combo_dir='./product_combos_v2.csv', df_product_dir='./product_items.csv', top_k_retrieve=100, gmap_api_key=gmap_api_key)
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

def send_reply(from_number, to_number, order_body, mode="launch"):
    response_message_container = {"result": None}
    group_ids = requests.get("http://31.97.106.30:3000/api/groups/active").json().get("data", None)

    def run_agent():
        try:
            result = agent.handle_order(order_body, access_token_dir="./storage/app/token_cache.json")
            response_message_container["result"] = result
        except Exception as e:
            response_message_container["result"] = f"❌ Terjadi kesalahan saat memproses pesanan. Error {e}"

    thread = threading.Thread(target=run_agent)
    thread.start()

    thread.join(timeout=30)

    if response_message_container["result"] is None:
        # Kirim fallback terlebih dahulu
        fallback_message = (
            f"Bentar be, proses akan memakan waktu sedikit lebih lama karena pemrosesan link maps atau produknya lumayan banyak, sebentar yaa 😊"
        )
        fallback_payload = {
            "command": "send_message",
            "number": from_number,
            "number_recipient": to_number,
            "message": fallback_message
        }
        channel.basic_publish(
            exchange='',
            routing_key='whatsapp_message_queue',
            body=json.dumps(fallback_payload),
            properties=pika.BasicProperties(delivery_mode=2)
        )
        print(f"⚠️ Balasan fallback dikirim ke {to_number}")

        # Tunggu thread selesai dan kirim jawaban asli
        thread.join()
        final_message = response_message_container["result"]
        final_payload = {
            "command": "send_message",
            "number": from_number,
            "number_recipient": to_number,
            "message": final_message
        }

        channel.basic_publish(
            exchange='',
            routing_key='whatsapp_message_queue',
            body=json.dumps(final_payload),
            properties=pika.BasicProperties(delivery_mode=2)
        )

        if "Makasih yaa" in final_message and "*[REQUEST UPDATE STRUK]*" not in final_message: 
            for group_id in group_ids:
                group_payload = {
                    "command": "send_message",
                    "number": from_number,
                    "number_recipient": group_id.get("groupId"),
                    "message": final_message
                }
                channel.basic_publish(
                    exchange='',
                    routing_key='whatsapp_message_queue',
                    body=json.dumps(group_payload),
                    properties=pika.BasicProperties(delivery_mode=2)
                )
                time.sleep(1)

        print(f"📤 Balasan akhir dikirim ke {to_number}")
    else:
        # Jika selesai kurang dari 45 detik, langsung kirim
        final_message = response_message_container["result"]
        final_payload = {
            "command": "send_message",
            "number": from_number,
            "number_recipient": to_number,
            "message": final_message
        }

        channel.basic_publish(
            exchange='',
            routing_key='whatsapp_message_queue',
            body=json.dumps(final_payload),
            properties=pika.BasicProperties(delivery_mode=2)
        )

        if "Makasih yaa" in final_message and "*[REQUEST UPDATE STRUK]*" not in final_message: 
            for group_id in group_ids:
                group_payload = {
                    "command": "send_message",
                    "number": from_number,
                    "number_recipient": group_id.get("groupId"),
                    "message": final_message
                }
                channel.basic_publish(
                    exchange='',
                    routing_key='whatsapp_message_queue',
                    body=json.dumps(group_payload),
                    properties=pika.BasicProperties(delivery_mode=2)
                )
                time.sleep(1)
                
        print(f"📤 Balasan langsung dikirim ke {to_number}")


def callback(ch, method, properties, body):
    try:
        payload = json.loads(body)
        print("✅ Pesan diterima:")
        print(json.dumps(payload, indent=4))

        if payload.get("type") == "order":
            from_number = payload.get("sessionId")  
            to_number = payload.get("from")
            order_body = payload.get("body", "Pesanan tidak lengkap")

            
            send_reply(from_number, to_number, order_body)

        
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print("✅ Pesan berhasil diproses dan di-acknowledge.")

    except Exception as e:
        print(f"❌ Error saat proses pesan: {e}")
        
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

if __name__ == "__main__":
    try:
        
        
        channel.basic_qos(prefetch_count=1)

        
        channel.basic_consume(
            queue='whatsapp_hook_queue',
            on_message_callback=callback,
            auto_ack=False 
        )

        print("🔄 Menunggu pesan dari 'whatsapp_hook_queue'...")
        channel.start_consuming()
    except KeyboardInterrupt:
        print("🔴 Proses dihentikan oleh pengguna.")
    finally:
        connection.close()
        print("🔌 Koneksi RabbitMQ ditutup.")
