import asyncio
import json
import os
import random
from datetime import datetime, timezone

import websockets
from dotenv import load_dotenv
from websockets.exceptions import ConnectionClosed, ConnectionClosedError


load_dotenv()

API_KEY = os.getenv("AISSTREAM_API_KEY")
AISSTREAM_URL = "wss://stream.aisstream.io/v0/stream"

# 처음 테스트는 부산항보다 조금 넓게 잡는 것을 추천
BUSAN_BBOX = [[[34.70, 128.40], [35.60, 129.70]]]

SUBSCRIBE_MESSAGE = {
    "APIKey": API_KEY,
    "BoundingBoxes": BUSAN_BBOX,
    "FilterMessageTypes": ["PositionReport"],
}


def parse_message(message_json):
    # AISstream은 binary frame으로 UTF-8 JSON을 보낼 수 있음
    if isinstance(message_json, bytes):
        message_json = message_json.decode("utf-8")

    message = json.loads(message_json)
    message_type = message.get("MessageType")

    if message_type == "SubscriptionConfirmation":
        compression = message.get("Message", {}).get("CompressionEnabled")
        print(f"[OK] Subscription confirmed. CompressionEnabled={compression}")
        return

    metadata = message.get("MetaData") or {}
    body = message.get("Message", {}).get(message_type, {}) or {}

    print("=" * 80)
    print("received_at:", datetime.now(timezone.utc).isoformat())
    print("message_type:", message_type)
    print("mmsi:", metadata.get("MMSI") or body.get("UserID"))
    print("ship_name:", metadata.get("ShipName"))
    print(
        "lat:",
        metadata.get("Latitude") or metadata.get("latitude") or body.get("Latitude"),
    )
    print(
        "lon:",
        metadata.get("Longitude") or metadata.get("longitude") or body.get("Longitude"),
    )

    if message_type == "PositionReport":
        print("sog:", body.get("Sog"))
        print("cog:", body.get("Cog"))
        print("heading:", body.get("TrueHeading"))
        print("nav_status:", body.get("NavigationalStatus"))


async def stream_once():
    if not API_KEY:
        raise ValueError(".env 파일에 AISSTREAM_API_KEY가 없습니다.")

    async with websockets.connect(
        AISSTREAM_URL,
        ping_interval=20,
        ping_timeout=20,
        compression="deflate",
        max_size=None,
        close_timeout=5,
    ) as websocket:
        # 연결 후 3초 안에 subscription을 보내야 함
        await websocket.send(json.dumps(SUBSCRIBE_MESSAGE))
        print("[INFO] Connected and subscription sent.")

        async for message_json in websocket:
            parse_message(message_json)


async def stream_forever():
    retry = 0

    while True:
        try:
            await stream_once()
            retry = 0

        except KeyboardInterrupt:
            print("\n[STOP] 사용자에 의해 종료되었습니다.")
            break

        except (ConnectionClosed, ConnectionClosedError) as e:
            retry += 1
            wait = min(60, 2**retry) + random.uniform(0, 1)
            print(f"[WARN] WebSocket disconnected: {e}")
            print(f"[INFO] Reconnecting in {wait:.1f} seconds...")
            await asyncio.sleep(wait)

        except Exception as e:
            retry += 1
            wait = min(60, 2**retry) + random.uniform(0, 1)
            print(f"[ERROR] Unexpected error: {type(e).__name__}: {e}")
            print(f"[INFO] Reconnecting in {wait:.1f} seconds...")
            await asyncio.sleep(wait)


if __name__ == "__main__":
    asyncio.run(stream_forever())
