#!/usr/bin/env python3
"""Remove identifying metadata from PNG and JPEG files without decoding pixels."""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_PRIVATE_CHUNKS = {b"eXIf", b"iCCP", b"iTXt", b"tEXt", b"tIME", b"zTXt"}
JPEG_PRIVATE_MARKERS = set(range(0xE1, 0xEE)) | {0xFE}
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}


def sanitize_png(data: bytes) -> bytes:
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("invalid PNG signature")

    output = bytearray(PNG_SIGNATURE)
    offset = len(PNG_SIGNATURE)
    saw_end = False
    while offset < len(data):
        if offset + 12 > len(data):
            raise ValueError("truncated PNG chunk")
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_end = offset + 12 + length
        if chunk_end > len(data):
            raise ValueError("truncated PNG chunk data")
        chunk_type = data[offset + 4 : offset + 8]
        if chunk_type not in PNG_PRIVATE_CHUNKS:
            output.extend(data[offset:chunk_end])
        offset = chunk_end
        if chunk_type == b"IEND":
            saw_end = True
            break

    if not saw_end or offset != len(data):
        raise ValueError("invalid data after PNG end chunk")
    return bytes(output)


def sanitize_jpeg(data: bytes) -> bytes:
    if not data.startswith(b"\xff\xd8"):
        raise ValueError("invalid JPEG signature")

    output = bytearray(data[:2])
    offset = 2
    while offset < len(data):
        marker_start = offset
        if data[offset] != 0xFF:
            raise ValueError("invalid JPEG marker")
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            raise ValueError("truncated JPEG marker")
        marker = data[offset]
        offset += 1

        if marker == 0xDA:  # Start of scan: compressed pixel data follows.
            output.extend(data[marker_start:])
            return bytes(output)
        if marker == 0xD9:
            output.extend(data[marker_start:offset])
            if offset != len(data):
                raise ValueError("invalid data after JPEG end marker")
            return bytes(output)
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            output.extend(data[marker_start:offset])
            continue
        if offset + 2 > len(data):
            raise ValueError("truncated JPEG segment")
        segment_length = struct.unpack(">H", data[offset : offset + 2])[0]
        if segment_length < 2:
            raise ValueError("invalid JPEG segment length")
        segment_end = offset + segment_length
        if segment_end > len(data):
            raise ValueError("truncated JPEG segment data")
        if marker not in JPEG_PRIVATE_MARKERS:
            output.extend(data[marker_start:segment_end])
        offset = segment_end

    raise ValueError("JPEG has no image scan")


def sanitized_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    if data.startswith(PNG_SIGNATURE):
        return sanitize_png(data)
    if data.startswith(b"\xff\xd8"):
        return sanitize_jpeg(data)
    raise ValueError(f"unsupported image encoding: {path}")


def iter_images(paths: list[Path]):
    for path in paths:
        if path.is_dir():
            yield from (
                candidate
                for candidate in sorted(path.rglob("*"))
                if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES
            )
        elif path.suffix.lower() in IMAGE_SUFFIXES:
            yield path
        else:
            raise ValueError(f"not a supported image or directory: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report files containing removable metadata without changing them",
    )
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    changed = []
    for path in iter_images(args.paths):
        sanitized = sanitized_bytes(path)
        if sanitized == path.read_bytes():
            continue
        changed.append(path)
        if not args.check:
            path.write_bytes(sanitized)

    if changed:
        action = "contains removable metadata" if args.check else "sanitized"
        for path in changed:
            print(f"{path}: {action}")
    return 1 if args.check and changed else 0


if __name__ == "__main__":
    sys.exit(main())
