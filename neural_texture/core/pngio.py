"""Tiny dependency-free PNG reader/writer (8/16-bit, gray/RGB/RGBA).

Used instead of bpy image saving so control images and textures are written
byte-exactly, without Blender's view transform touching them.

Arrays are float32 in [0, 1] with BOTTOM-UP rows (Blender convention); the
functions flip to/from PNG's top-down order.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np

_SIG = b"\x89PNG\r\n\x1a\n"
_COLOR_TYPES = {1: 0, 2: 4, 3: 2, 4: 6}  # channels -> PNG colour type
_CHANNELS = {0: 1, 4: 2, 2: 3, 6: 4}


def _chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def write_png(path: str, img: np.ndarray, bits: int = 8) -> None:
    if img.ndim == 2:
        img = img[..., None]
    h, w, c = img.shape
    if c not in _COLOR_TYPES:
        raise ValueError(f"unsupported channel count {c}")
    top_down = np.clip(img[::-1], 0.0, 1.0)
    if bits == 8:
        data = np.rint(top_down * 255.0).astype(">u1")
    elif bits == 16:
        data = np.rint(top_down * 65535.0).astype(">u2")
    else:
        raise ValueError("bits must be 8 or 16")
    raw = data.reshape(h, w * c).tobytes() if bits == 8 else data.reshape(h, -1).tobytes()
    stride = w * c * (bits // 8)
    rows = b"".join(b"\x00" + raw[i * stride:(i + 1) * stride] for i in range(h))
    ihdr = struct.pack(">IIBBBBB", w, h, bits, _COLOR_TYPES[c], 0, 0, 0)
    with open(path, "wb") as f:
        f.write(_SIG + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(rows, 6)) + _chunk(b"IEND", b""))


def _sub(a, b, c):
    return a


def _avg(a, b, c):
    return (a + b) >> 1


def _paeth_vec(a, b, c):
    p = a + b - c
    pa, pb, pc = np.abs(p - a), np.abs(p - b), np.abs(p - c)
    return np.where((pa <= pb) & (pa <= pc), a, np.where(pb <= pc, b, c))


def _sequential(line, prev, bpp, pred):
    """Undo a left-dependent filter one pixel (bpp bytes) at a time."""
    cur = line.copy()
    zero = np.zeros(bpp, dtype=np.int32)
    left, upleft = zero, zero
    for i in range(0, len(line), bpp):
        up = prev[i:i + bpp]
        n = len(up)
        v = (cur[i:i + n] + pred(left[:n], up, upleft[:n])) & 0xFF
        cur[i:i + n] = v
        left, upleft = v, up
    return cur


def read_png(path: str) -> np.ndarray:
    """Read a non-interlaced 8/16-bit gray/GA/RGB/RGBA/palette PNG -> float32 bottom-up."""
    with open(path, "rb") as f:
        buf = f.read()
    if buf[:8] != _SIG:
        raise ValueError(f"{path}: not a PNG")
    pos, idat, plte, trns = 8, [], None, None
    w = h = bits = ctype = interlace = None
    while pos < len(buf):
        (n,) = struct.unpack(">I", buf[pos:pos + 4])
        tag = buf[pos + 4:pos + 8]
        data = buf[pos + 8:pos + 8 + n]
        pos += 12 + n
        if tag == b"IHDR":
            w, h, bits, ctype, _, _, interlace = struct.unpack(">IIBBBBB", data)
        elif tag == b"IDAT":
            idat.append(data)
        elif tag == b"PLTE":
            plte = np.frombuffer(data, np.uint8).reshape(-1, 3)
        elif tag == b"tRNS":
            trns = np.frombuffer(data, np.uint8)
        elif tag == b"IEND":
            break
    if interlace:
        raise ValueError("interlaced PNG not supported")
    if bits not in (8, 16) and ctype != 3:
        raise ValueError(f"{bits}-bit PNG not supported")
    ch = 1 if ctype == 3 else _CHANNELS[ctype]
    bpp = max(1, ch * bits // 8)
    stride = (w * ch * bits + 7) // 8
    raw = zlib.decompress(b"".join(idat))
    out = np.zeros((h, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.int32)
    for y in range(h):
        ft = raw[y * (stride + 1)]
        line = np.frombuffer(raw, np.uint8, stride, y * (stride + 1) + 1).astype(np.int32)
        if ft == 0:
            cur = line
        elif ft == 1:  # Sub: running sum per channel -> vectorised cumsum
            cur = (np.cumsum(line.reshape(-1, bpp), axis=0) & 0xFF).reshape(-1) if stride % bpp == 0 \
                else _sequential(line, prev, bpp, _sub)
        elif ft == 2:
            cur = (line + prev) & 0xFF
        elif ft == 3:
            cur = _sequential(line, prev, bpp, _avg)
        elif ft == 4:
            cur = _sequential(line, prev, bpp, _paeth_vec)
        else:
            raise ValueError(f"bad filter {ft}")
        out[y] = cur
        prev = cur
    if ctype == 3:
        if bits != 8:
            raise ValueError("only 8-bit palette PNG supported")
        idx = out[:, :w]
        rgb = plte[idx].astype(np.float32) / 255.0
        if trns is not None:
            alpha = np.ones(len(plte), np.float32)
            alpha[:len(trns)] = trns / 255.0
            rgb = np.concatenate([rgb, alpha[idx][..., None]], axis=-1)
        return np.ascontiguousarray(rgb[::-1])
    if bits == 8:
        arr = out.reshape(h, w, ch).astype(np.float32) / 255.0
    else:
        arr = out.view(">u2").reshape(h, w, ch).astype(np.float32) / 65535.0
    return np.ascontiguousarray(arr[::-1])
