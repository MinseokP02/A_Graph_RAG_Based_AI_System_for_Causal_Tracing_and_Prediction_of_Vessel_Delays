from collect_ais_to_csv import (
    BUSAN_PORT_BBOX,
    CSV_FIELDS,
    FILTER_MESSAGE_TYPES,
    REGION_BOUNDING_BOXES,
    normalize_ais_message,
)


def test_collector_keeps_timestamp_columns_separate() -> None:
    assert CSV_FIELDS[:2] == ["time_utc", "received_at_utc"]
    assert len(CSV_FIELDS) == len(set(CSV_FIELDS))


def test_collector_subscribes_to_static_identity_messages() -> None:
    assert "PositionReport" in FILTER_MESSAGE_TYPES
    assert "ShipStaticData" in FILTER_MESSAGE_TYPES

    row = normalize_ais_message(
        {
            "MessageType": "ShipStaticData",
            "MetaData": {"MMSI": 440000001, "ShipName": "TEST SHIP"},
            "Message": {
                "ShipStaticData": {
                    "CallSign": "D7TEST",
                    "ImoNumber": 9876543,
                }
            },
        }
    )

    assert row is not None
    assert row["mmsi"] == 440000001
    assert row["call_sign"] == "D7TEST"
    assert row["imo_number"] == 9876543


def test_collector_targets_busan_and_includes_class_b() -> None:
    assert BUSAN_PORT_BBOX == [[[35.6, 128.4], [34.7, 129.7]]]
    assert REGION_BOUNDING_BOXES["korea"] == [[[39.0, 124.0], [33.0, 132.0]]]
    assert REGION_BOUNDING_BOXES["incheon"] == [[[38.0, 126.0], [36.8, 127.0]]]
    assert "StandardClassBPositionReport" in FILTER_MESSAGE_TYPES
    assert "ExtendedClassBPositionReport" in FILTER_MESSAGE_TYPES
    assert "StaticDataReport" in FILTER_MESSAGE_TYPES
