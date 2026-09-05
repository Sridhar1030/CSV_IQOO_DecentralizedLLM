"""OTLP/HTTP codec helpers.

The proxy has to decode what the app's OTel SDK sent, modify it, and re-encode
it in the *same* format the SDK used, so the exporter on the other side is
none the wiser. OTLP/HTTP allows protobuf or JSON, with optional gzip.
"""

from __future__ import annotations

import gzip
from typing import Any

from google.protobuf import json_format
from google.protobuf.message import Message
from opentelemetry.proto.collector.logs.v1 import logs_service_pb2
from opentelemetry.proto.collector.metrics.v1 import metrics_service_pb2
from opentelemetry.proto.collector.trace.v1 import trace_service_pb2
from opentelemetry.proto.common.v1 import common_pb2

PROTOBUF = "application/x-protobuf"
JSON = "application/json"

TRACES_REQUEST = trace_service_pb2.ExportTraceServiceRequest
TRACES_RESPONSE = trace_service_pb2.ExportTraceServiceResponse
METRICS_REQUEST = metrics_service_pb2.ExportMetricsServiceRequest
METRICS_RESPONSE = metrics_service_pb2.ExportMetricsServiceResponse
LOGS_REQUEST = logs_service_pb2.ExportLogsServiceRequest
LOGS_RESPONSE = logs_service_pb2.ExportLogsServiceResponse


class DecodeError(ValueError):
    """The body was not valid OTLP in the declared content type."""


def is_json(content_type: str | None) -> bool:
    return bool(content_type) and content_type.split(";")[0].strip().lower() == JSON


def decode(body: bytes, content_type: str | None, message_type, encoding: str | None = None):
    """Bytes -> protobuf message, honouring content-type and gzip."""
    if encoding and "gzip" in encoding.lower():
        try:
            body = gzip.decompress(body)
        except OSError as exc:
            raise DecodeError(f"body declared gzip but failed to decompress: {exc}") from exc

    message = message_type()
    try:
        if is_json(content_type):
            json_format.Parse(body.decode("utf-8"), message)
        else:
            message.ParseFromString(body)
    except Exception as exc:
        raise DecodeError(f"could not decode OTLP payload: {exc}") from exc
    return message


def encode(message: Message, content_type: str | None) -> bytes:
    """Protobuf message -> bytes in the same content type it arrived as."""
    if is_json(content_type):
        return json_format.MessageToJson(message).encode("utf-8")
    return message.SerializeToString()


def empty_response(response_type, content_type: str | None) -> bytes:
    return encode(response_type(), content_type)


# ----------------------------------------------------------------------
# attribute helpers
# ----------------------------------------------------------------------
def any_value(value: Any) -> common_pb2.AnyValue:
    """Python value -> OTLP AnyValue. bool must be checked before int."""
    if isinstance(value, bool):
        return common_pb2.AnyValue(bool_value=value)
    if isinstance(value, int):
        return common_pb2.AnyValue(int_value=value)
    if isinstance(value, float):
        return common_pb2.AnyValue(double_value=value)
    return common_pb2.AnyValue(string_value=str(value))


def unwrap(value: common_pb2.AnyValue) -> Any:
    """OTLP AnyValue -> Python value."""
    which = value.WhichOneof("value")
    if which is None:
        return None
    if which == "array_value":
        return [unwrap(v) for v in value.array_value.values]
    if which == "kvlist_value":
        return {kv.key: unwrap(kv.value) for kv in value.kvlist_value.values}
    if which == "bytes_value":
        return value.bytes_value
    return getattr(value, which)


def to_dict(attributes) -> dict[str, Any]:
    """Repeated KeyValue -> plain dict."""
    return {kv.key: unwrap(kv.value) for kv in attributes}


def get(attributes, key: str, default: Any = None) -> Any:
    for kv in attributes:
        if kv.key == key:
            return unwrap(kv.value)
    return default


def set_attribute(attributes, key: str, value: Any, overwrite: bool = False) -> bool:
    """Add an attribute. Returns False if it was already present and kept.

    Defaults to *not* overwriting: whatever the application reported about
    itself is more authoritative than anything the proxy infers.
    """
    for kv in attributes:
        if kv.key == key:
            if not overwrite:
                return False
            kv.value.CopyFrom(any_value(value))
            return True
    attributes.add(key=key, value=any_value(value))
    return True


def as_int(value: Any) -> int | None:
    """Token counts arrive as int, float, or a numeric string depending on SDK."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return None
    return None
