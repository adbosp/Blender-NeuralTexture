import os
import struct
import sys
import tempfile
import zlib

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from neural_texture.core import pngio


def test_roundtrip_bits_and_channels():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "t.png")
        for bits in (8, 16):
            q = 255 if bits == 8 else 65535
            for c in (1, 3, 4):
                a = rng.random((37, 53, c)).astype(np.float32)
                pngio.write_png(p, a, bits)
                b = pngio.read_png(p)
                assert np.allclose(np.rint(a * q) / q, b, atol=1e-6), (bits, c)


def _encode(img, ft):
    h, w, c = img.shape
    rows, prev = [], np.zeros(w * c, np.int32)
    for y in range(h):
        line = img[y].reshape(-1).astype(np.int32)
        out = np.zeros_like(line)
        for i in range(len(line)):
            a = line[i - c] if i >= c else 0
            b = prev[i]
            cc = prev[i - c] if i >= c else 0
            if ft == 0:
                pr = 0
            elif ft == 1:
                pr = a
            elif ft == 2:
                pr = b
            elif ft == 3:
                pr = (a + b) >> 1
            else:
                p = a + b - cc
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - cc)
                pr = a if pa <= pb and pa <= pc else (b if pb <= pc else cc)
            out[i] = (line[i] - pr) & 255
        rows.append(bytes([ft]) + out.astype(np.uint8).tobytes())
        prev = line

    def ch(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2 if c == 3 else 6, 0, 0, 0))
            + ch(b"IDAT", zlib.compress(b"".join(rows))) + ch(b"IEND", b""))


def test_all_png_filters_decode():
    rng = np.random.default_rng(1)
    for c in (3, 4):
        img = rng.integers(0, 256, (12, 17, c))
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "f.png")
            for ft in range(5):
                open(p, "wb").write(_encode(img, ft))
                r = pngio.read_png(p)
                assert np.array_equal(np.rint(r[::-1] * 255).astype(int), img), (c, ft)
