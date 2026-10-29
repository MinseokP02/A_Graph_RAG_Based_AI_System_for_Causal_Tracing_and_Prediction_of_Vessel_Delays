import argparse
import asyncio
import csv
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import websockets
from dotenv import load_dotenv
from websockets.exceptions import ConnectionClosed, ConnectionClosedError

load_dotenv()

AISSTREAM_API_KEY = os.getenv("AISSTREAM_API_KEY")
AISSTREAM_URL = "wss://stream.aisstream.io/v0/stream"

# 부산항 근처 bbox
# 데이터가 너무 안 들어오면 범위를 조금 넓히세요.
REGION_BOUNDING_BOXES = {
    "korea": [[[39.00, 124.00], [33.00, 132.00]]],
    "busan": [[[35.60, 128.40], [34.70, 129.70]]],
    "incheon": [[[38.00, 126.00], [36.80, 127.00]]],
}
BUSAN_PORT_BBOX = REGION_BOUNDING_BOXES["busan"]

FILTER_MESSAGE_TYPES = [
    "PositionReport",
    "StandardClassBPositionReport",
    "ExtendedClassBPositionReport",
    "ShipStaticData",
    "StaticDataReport",
]

OUTPUT_CSV = "data/ais/korea_ais_stream.csv"

# 몇 개마다 로그를 찍을지 설정
# 1이면 데이터가 들어올 때마다 매번 출력
LOG_EVERY = 100

# 메시지가 없을 때 대기 로그 출력 주기, 초 단위
WAIT_TIMEOUT_SECONDS = 30

CSV_FIELDS = [
    "time_utc",
    "received_at_utc",
    "message_type",
    "mmsi",
    "ship_name",
    "call_sign",
    "imo_number",
    "latitude",
    "longitude",
    "sog",
    "cog",
    "true_heading",
    "navigational_status",
    "destination",
    "eta",
    "ship_type",
]


def safe_get(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def get_message_body(message: dict[str, Any]) -> dict[str, Any]:
    message_type = message.get("MessageType")
    body = safe_get(message, "Message", message_type, default={})
    return body if isinstance(body, dict) else {}


def normalize_ais_message(message: dict[str, Any]) -> dict[str, Any] | None:
    message_type = message.get("MessageType")

    # 구독 확인 메시지는 AIS 데이터가 아니므로 CSV 저장하지 않음
    if message_type == "SubscriptionConfirmation":
        compression = safe_get(message, "Message", "CompressionEnabled")
        print(f"[OK] Subscription confirmed. CompressionEnabled={compression}")
        return None

    metadata = message.get("MetaData") or {}
    body = get_message_body(message)

    report_b = body.get("ReportB", {}) if isinstance(body.get("ReportB"), dict) else {}

    dimension = body.get("Dimension", {})
    if not isinstance(dimension, dict):
        dimension = {}

    row = {
        "time_utc": metadata.get("time_utc"),
        "received_at_utc": datetime.now(timezone.utc).isoformat(),
        "message_type": message_type,
        "mmsi": metadata.get("MMSI") or body.get("UserID") or body.get("Mmsi"),
        "ship_name": metadata.get("ShipName")
        or body.get("Name")
        or body.get("ShipName"),
        "call_sign": body.get("CallSign") or report_b.get("CallSign"),
        "imo_number": body.get("ImoNumber")
        or body.get("IMONumber")
        or report_b.get("ImoNumber"),
        "latitude": body.get("Latitude")
        or metadata.get("latitude")
        or metadata.get("Latitude"),
        "longitude": body.get("Longitude")
        or metadata.get("longitude")
        or metadata.get("Longitude"),
        "sog": body.get("Sog"),
        "cog": body.get("Cog"),
        "true_heading": body.get("TrueHeading"),
        "navigational_status": body.get("NavigationalStatus"),
        "destination": body.get("Destination") or report_b.get("Destination"),
        "eta": body.get("Eta")
        or body.get("ETA")
        or report_b.get("Eta")
        or report_b.get("ETA"),
        "ship_type": body.get("ShipType") or report_b.get("ShipType"),
    }
    return row


def print_row_log(row: dict[str, Any], saved_count: int) -> None:
    print(
        f"[SAVE {saved_count:06d}] "
        f"type={row.get('message_type')} | "
        f"mmsi={row.get('mmsi')} | "
        f"ship={row.get('ship_name') or row.get('name')} | "
        f"lat={row.get('latitude')} | "
        f"lon={row.get('longitude')} | "
        f"sog={row.get('sog')} | "
        f"cog={row.get('cog')} | "
        f"time={row.get('time_utc') or row.get('received_at_utc')}"
    )


async def collect_once(
    output_csv: str,
    max_messages: int | None,
    bounding_boxes: list[list[list[float]]] | None = None,
    saved_start_count: int = 0,
) -> int:
    if not AISSTREAM_API_KEY:
        raise ValueError(
            ".env 파일에 AISSTREAM_API_KEY가 없습니다. "
            "AISSTREAM_API_KEY=발급받은키 형태로 설정하세요."
        )

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_has_content = output_path.exists() and output_path.stat().st_size > 0

    subscription_message = {
        "APIKey": AISSTREAM_API_KEY,
        "BoundingBoxes": bounding_boxes or REGION_BOUNDING_BOXES["korea"],
        "FilterMessageTypes": FILTER_MESSAGE_TYPES,
    }

    saved_count = saved_start_count

    async with websockets.connect(
        AISSTREAM_URL,
        ping_interval=20,
        ping_timeout=20,
        max_size=None,
        close_timeout=5,
    ) as websocket:
        await websocket.send(json.dumps(subscription_message))

        print("[INFO] AISStream connected.")
        print(f"[INFO] Saving to: {output_path.resolve()}")
        print(f"[INFO] BoundingBoxes: {subscription_message['BoundingBoxes']}")
        print(
            f"[INFO] FilterMessageTypes: {subscription_message['FilterMessageTypes']}"
        )

        # Keep one ordered stream open; each small write is flushed immediately.
        with output_path.open(  # noqa: ASYNC230
            "a", newline="", encoding="utf-8-sig"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)

            if not file_has_content:
                writer.writeheader()
                f.flush()

            while True:
                try:
                    message_json = await asyncio.wait_for(
                        websocket.recv(),
                        timeout=WAIT_TIMEOUT_SECONDS,
                    )

                except asyncio.TimeoutError:
                    now = datetime.now(timezone.utc).isoformat()
                    print(
                        f"[WAIT] {now} - {WAIT_TIMEOUT_SECONDS}초 동안 수신된 AIS 메시지가 없습니다."
                    )
                    continue

                if isinstance(message_json, bytes):
                    message_json = message_json.decode("utf-8")

                try:
                    message = json.loads(message_json)
                    row = normalize_ais_message(message)

                    if row is None:
                        continue

                    writer.writerow(row)
                    f.flush()

                    saved_count += 1

                    if saved_count % LOG_EVERY == 0:
                        print_row_log(row, saved_count)

                    if max_messages is not None and saved_count >= max_messages:
                        print(f"[DONE] Finished. Saved {saved_count} AIS messages.")
                        return saved_count

                except Exception as e:  # noqa: BLE001 -- skip one malformed upstream payload
                    print(f"[ERROR] Failed to process message: {type(e).__name__}: {e}")
                    print(f"[ERROR] Raw preview: {str(message_json)[:500]}")


async def collect_forever(
    output_csv: str = OUTPUT_CSV,
    max_messages: int | None = 1000,
    bounding_boxes: list[list[list[float]]] | None = None,
) -> None:
    retry = 0
    saved_count = 0

    while True:
        try:
            saved_count = await collect_once(
                output_csv=output_csv,
                max_messages=max_messages,
                bounding_boxes=bounding_boxes,
                saved_start_count=saved_count,
            )

            if max_messages is not None and saved_count >= max_messages:
                break

            retry = 0

        except KeyboardInterrupt:
            print("\n[STOP] 사용자에 의해 수집이 중단되었습니다.")
            print(f"[INFO] 현재까지 저장된 AIS 메시지 수: {saved_count}")
            break

        except (ConnectionClosed, ConnectionClosedError) as e:
            retry += 1
            wait = min(60, 2**retry) + random.uniform(0, 1)
            print(f"[WARN] WebSocket disconnected: {e}")
            print(f"[INFO] Reconnecting in {wait:.1f} seconds...")
            await asyncio.sleep(wait)

        except Exception as e:  # noqa: BLE001 -- reconnect after unexpected runtime failures
            retry += 1
            wait = min(60, 2**retry) + random.uniform(0, 1)
            print(f"[ERROR] Unexpected error: {type(e).__name__}: {e}")
            print(f"[INFO] Reconnecting in {wait:.1f} seconds...")
            await asyncio.sleep(wait)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuously collect AISStream messages around Korea."
    )
    parser.add_argument(
        "--region",
        choices=tuple(REGION_BOUNDING_BOXES),
        default="korea",
        help="Collection area (default: korea)",
    )
    parser.add_argument("--output", help="CSV output path; defaults to the region name")
    parser.add_argument(
        "--max-messages",
        type=int,
        default=0,
        help="Stop after this many messages; 0 means collect continuously",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=0,
        help="Stop after this many wall-clock seconds; 0 means no time limit",
    )
    args = parser.parse_args()
    if args.max_messages < 0:
        parser.error("--max-messages must be zero or positive")
    if args.duration_seconds < 0:
        parser.error("--duration-seconds must be zero or positive")
    if not args.output:
        args.output = f"data/ais/{args.region}_ais_stream.csv"
    return args


if __name__ == "__main__":
    args = parse_args()
    try:
        collector = collect_forever(
            output_csv=args.output,
            max_messages=args.max_messages or None,
            bounding_boxes=REGION_BOUNDING_BOXES[args.region],
        )
        if args.duration_seconds:
            try:
                asyncio.run(asyncio.wait_for(collector, timeout=args.duration_seconds))
            except asyncio.TimeoutError:
                print(
                    f"[DONE] Reached {args.duration_seconds:g}-second collection limit."
                )
        else:
            asyncio.run(collector)
    except KeyboardInterrupt:
        print("\n[STOP] 사용자에 의해 종료되었습니다.")
