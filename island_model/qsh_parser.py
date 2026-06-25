# парсер QSH (QScalp) бинарного формата, v4, gzip + deals stream

import gzip
import io
import struct
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd


# Streaming support roadmap is tracked as GitHub issues, not inline
# comments in source. See:
#   - Quotes stream (type 0x20) — issue #8
#   - OwnOrders stream          — issue #9
# The previous version of this file had two dangling roadmap comments
# that shipped in releases without ever being actioned. Moving them
# to issues makes them searchable, assignable, and impossible to lose.

def read_byte(stream: io.BytesIO) -> int:
    b = stream.read(1)
    if not b:
        raise EOFError("Unexpected end of QSH stream")
    return b[0]


def read_leb128(stream: io.BytesIO) -> int:
    """Signed LEB128."""
    value = 0
    shift = 0
    while True:
        b = read_byte(stream)
        value |= (b & 0x7F) << shift
        shift += 7
        if (b & 0x80) == 0:
            # Sign extension
            if shift < 64 and (b & 0x40):
                value |= -(1 << shift)
            return value


def read_uleb128(stream: io.BytesIO) -> int:
    """Unsigned LEB128 (max 4 bytes = 268435455)."""
    value = 0
    shift = 0
    while True:
        b = read_byte(stream)
        value |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            return value
        shift += 7


ULEB128_MAX4 = 0x0FFFFFFF  # sentinel для Growing

def read_growing(stream: io.BytesIO, last_value: int) -> int:
    """Growing encoding — для монотонно растущих значений (timestamp, id)."""
    offset = read_uleb128(stream)
    if offset == ULEB128_MAX4:
        return last_value + read_leb128(stream)
    return last_value + offset


def read_dotnet_string(stream: io.BytesIO) -> str:
    """Строка в формате .NET BinaryReader."""
    length = read_uleb128(stream)
    if length == 0:
        return ""
    data = stream.read(length)
    return data.decode("utf-8")


def read_int64_le(stream: io.BytesIO) -> int:
    data = stream.read(8)
    return struct.unpack("<q", data)[0]


def ticks_to_datetime(ticks: int) -> datetime:
    """.NET DateTime.Ticks -> Python datetime (UTC)."""
    # .NET ticks = 100-nanosecond intervals since 0001-01-01
    return datetime(1, 1, 1) + timedelta(microseconds=ticks // 10)


def ms_to_datetime(ms: int) -> datetime:
    return datetime(1, 1, 1) + timedelta(milliseconds=ms)


STREAM_DEALS = 0x20

DEAL_TYPE = {0: "Unknown", 1: "Buy", 2: "Sell"}


@dataclass
class QshHeader:
    version: int
    application: str
    comment: str
    recording_time: datetime
    stream_count: int
    streams: list


@dataclass
class StreamHeader:
    stream_type: int
    instrument: str
    price_step: float


@dataclass
class Deal:
    timestamp: datetime
    price: float
    volume: int
    side: str  # "Buy", "Sell", "Unknown"
    oi: int    # Open Interest


class QshParser:
    """Парсер QSH файлов (v4, gzip, Deals stream)."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.header = None
        self.deals = []

    def parse(self) -> list[Deal]:
        """Парсим файл, возвращаем список сделок."""
        with gzip.open(self.filepath, "rb") as f:
            data = f.read()

        stream = io.BytesIO(data)

        self.header = self._parse_header(stream)

        if self.header.streams and self.header.streams[0].stream_type == STREAM_DEALS:
            self.deals = self._parse_deals(stream, self.header)
        else:
            raise ValueError(
                f"Expected Deals stream (0x20), got: "
                f"{self.header.streams[0].stream_type if self.header.streams else 'none'}"
            )

        return self.deals

    def _parse_header(self, stream: io.BytesIO) -> QshHeader:
        # Signature: "QScalp History Data" (19 bytes, no prefix)
        sig = stream.read(19).decode("ascii")
        if sig != "QScalp History Data":
            raise ValueError(f"Invalid QSH signature: {sig!r}")

        version = read_byte(stream)
        if version != 4:
            raise ValueError(f"Unsupported QSH version: {version} (expected 4)")

        application = read_dotnet_string(stream)
        comment = read_dotnet_string(stream)

        rec_ticks = read_int64_le(stream)
        recording_time = ticks_to_datetime(rec_ticks)

        stream_count = read_byte(stream)

        streams = []
        for _ in range(stream_count):
            stype = read_byte(stream)
            instrument = read_dotnet_string(stream)

            # price step из инструмента типа "TRANSAQ:RIH6:FUT:1:10"
            parts = instrument.split(":")
            price_step = float(parts[-1]) if len(parts) >= 5 else 1.0

            streams.append(StreamHeader(
                stream_type=stype,
                instrument=instrument,
                price_step=price_step,
            ))

        return QshHeader(
            version=version,
            application=application,
            comment=comment,
            recording_time=recording_time,
            stream_count=stream_count,
            streams=streams,
        )

    def _parse_deals(self, stream: io.BytesIO, header: QshHeader) -> list[Deal]:
        price_step = header.streams[0].price_step
        single_stream = header.stream_count == 1

        # recording time в миллисекундах
        rec_ticks = int(
            (header.recording_time - datetime(1, 1, 1)).total_seconds() * 1000
        )

        # running state (delta-encoded поля)
        frame_ms = rec_ticks
        deal_ms = 0
        deal_id = 0
        order_id = 0
        price_raw = 0
        volume = 0
        oi = 0

        deals = []

        while True:
            try:
                frame_ms = read_growing(stream, frame_ms)

                if not single_stream:
                    _stream_idx = read_byte(stream)

                # deal flags
                flags = read_byte(stream)
                deal_type = flags & 0x03
                side = DEAL_TYPE.get(deal_type, "Unknown")

                if flags & 0x04:  # DateTime
                    deal_ms = read_growing(stream, deal_ms)

                if flags & 0x08:  # Id
                    deal_id = read_growing(stream, deal_id)

                if flags & 0x10:  # OrderId
                    order_id += read_leb128(stream)

                if flags & 0x20:  # Price (delta)
                    price_raw += read_leb128(stream)

                if flags & 0x40:  # Volume (absolute)
                    volume = read_leb128(stream)

                if flags & 0x80:  # OI (delta)
                    oi += read_leb128(stream)

                actual_price = price_raw * price_step

                if deal_ms > 0:
                    ts = ms_to_datetime(deal_ms)
                else:
                    ts = ms_to_datetime(frame_ms)

                deals.append(Deal(
                    timestamp=ts,
                    price=actual_price,
                    volume=abs(volume),
                    side=side,
                    oi=oi,
                ))

            except EOFError:
                break

        return deals

    def to_dataframe(self) -> pd.DataFrame:
        """Сделки -> pandas DataFrame."""
        if not self.deals:
            self.parse()

        records = [
            {
                "timestamp": d.timestamp,
                "price": d.price,
                "volume": d.volume,
                "side": d.side,
                "oi": d.oi,
            }
            for d in self.deals
        ]
        df = pd.DataFrame(records)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp").sort_index()
        return df

    # Weighted-avg-price (VWAP) column tracked as GitHub issue #10.
    # Inline roadmap comment removed so the only roadmap-tracking
    # surface in this codebase is the issue tracker.
    def to_ohlcv(self, freq: str = "1min") -> pd.DataFrame:
        """Агрегация тиков в OHLCV свечи (freq: '1min', '5min', '1h', '1D')."""
        df = self.to_dataframe()
        if df.empty:
            return df

        ohlcv = df["price"].resample(freq).ohlc()
        ohlcv.columns = ["open", "high", "low", "close"]
        ohlcv["volume"] = df["volume"].resample(freq).sum()
        ohlcv["trades"] = df["price"].resample(freq).count()

        buy_mask = df["side"] == "Buy"
        sell_mask = df["side"] == "Sell"
        ohlcv["buy_volume"] = df.loc[buy_mask, "volume"].resample(freq).sum()
        ohlcv["sell_volume"] = df.loc[sell_mask, "volume"].resample(freq).sum()

        # OI — последнее значение за свечу
        ohlcv["oi"] = df["oi"].resample(freq).last()

        return ohlcv.dropna(subset=["open"])

    def summary(self) -> str:
        """Текстовая сводка по файлу."""
        if not self.deals:
            self.parse()

        h = self.header
        lines = [
            f"QSH File: {self.filepath}",
            f"Version: {h.version}",
            f"Application: {h.application}",
            f"Comment: {h.comment}",
            f"Recording Time: {h.recording_time}",
            f"Streams: {h.stream_count}",
        ]
        for i, s in enumerate(h.streams):
            lines.append(f"  Stream {i}: type=0x{s.stream_type:02x} "
                         f"({['?','?','Quotes','Deals'][s.stream_type // 16] if s.stream_type <= 0x30 else '?'}), "
                         f"instrument={s.instrument}, step={s.price_step}")

        n = len(self.deals)
        if n > 0:
            lines.append(f"Total deals: {n:,}")
            lines.append(f"Time range: {self.deals[0].timestamp} — {self.deals[-1].timestamp}")
            prices = [d.price for d in self.deals]
            volumes = [d.volume for d in self.deals]
            buys = sum(1 for d in self.deals if d.side == "Buy")
            sells = sum(1 for d in self.deals if d.side == "Sell")
            total_vol = sum(volumes)
            lines.append(f"Price range: {min(prices):,.0f} — {max(prices):,.0f}")
            lines.append(f"Total volume: {total_vol:,} contracts")
            lines.append(f"Buys: {buys:,} ({buys/n:.1%}), Sells: {sells:,} ({sells/n:.1%})")
            lines.append(f"Avg trade size: {total_vol/n:.1f} contracts")
            if self.deals[-1].oi > 0:
                lines.append(f"Open Interest: {self.deals[0].oi:,} → {self.deals[-1].oi:,}")

        return "\n".join(lines)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python qsh_parser.py <file.qsh> [freq]")
        print("  freq: 1min, 5min, 1h, 1D (default: 1min)")
        sys.exit(1)

    filepath = sys.argv[1]
    freq = sys.argv[2] if len(sys.argv) > 2 else "1min"

    parser = QshParser(filepath)
    parser.parse()
    print(parser.summary())
    print()

    ohlcv = parser.to_ohlcv(freq)
    print(f"OHLCV candles ({freq}):")
    print(ohlcv.to_string())
