"""Material classes and physically-plausible PBR priors (V3).

Used for: material-aware prompts, roughness/metallic synthesis, detail scale.
Class detection works on English and Vietnamese text and on material names.
Values follow common PBR guidance (e.g. dielectrics metallic=0, raw metals
metallic=1 with bright albedo; roughness ranges per surface type).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple


@dataclass
class MaterialClass:
    name: str
    roughness: Tuple[float, float]         # (smooth parts, rough parts)
    metallic: float = 0.0
    detail_gain: float = 1.0               # scales generated detail normal
    micro_sigma: float = 0.75              # texels at 2K
    cavity_rough: float = 0.15             # extra roughness in cavities (low AO / dark detail)
    albedo_range: Tuple[float, float] = (0.03, 0.9)  # plausible sRGB luminance range
    prompt: str = ""
    negative: str = ""
    keywords: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


CLASSES: Dict[str, MaterialClass] = {c.name: c for c in [
    MaterialClass("skin", (0.38, 0.62), 0.0, 0.7, 0.6, 0.08, (0.08, 0.75),
                  "realistic human skin, natural pores, subtle subsurface tones",
                  "plastic, wax, makeup shine",
                  ["skin", "face", "head", "body", "human", "character", "da", "mat", "khuon mat", "nguoi"]),
    MaterialClass("stone", (0.62, 0.95), 0.0, 1.3, 0.9, 0.2, (0.05, 0.8),
                  "weathered stone surface, mineral grain, chips and cracks",
                  "glossy, plastic",
                  ["stone", "rock", "granite", "marble", "concrete", "masonry", "statue", "ancient", "ruin",
                   "da", "da co", "tuong da", "cam thach", "be tong", "co dai", "cu ki"]),
    MaterialClass("metal", (0.18, 0.55), 1.0, 0.8, 0.6, 0.1, (0.45, 0.98),
                  "bare metal surface, fine machining marks, subtle scratches",
                  "rust, paint, plastic",
                  ["metal", "metallic", "steel", "iron", "chrome", "aluminum", "aluminium", "silver",
                   "gold", "copper", "brass", "bronze", "kim loai", "thep", "sat", "vang", "bac", "dong", "nhom"]),
    MaterialClass("rusted_metal", (0.55, 0.92), 0.35, 1.2, 0.8, 0.2, (0.05, 0.7),
                  "corroded rusty metal, flaking oxide, pitted surface", "clean, shiny",
                  ["rust", "rusty", "rusted", "corroded", "oxidized", "gi", "gi set", "han gi"]),
    MaterialClass("painted_metal", (0.3, 0.6), 0.0, 0.7, 0.6, 0.12, (0.04, 0.85),
                  "painted metal, chipped paint edges, fine wear", "",
                  ["painted metal", "paint", "son", "son kim loai"]),
    MaterialClass("wood", (0.45, 0.85), 0.0, 1.0, 0.8, 0.15, (0.05, 0.7),
                  "natural wood, visible grain and fibers", "plastic",
                  ["wood", "wooden", "timber", "plank", "bark", "go", "van go"]),
    MaterialClass("fabric", (0.75, 0.97), 0.0, 0.9, 0.5, 0.05, (0.03, 0.85),
                  "woven fabric, visible threads and fibers", "shiny, plastic",
                  ["fabric", "cloth", "cotton", "wool", "denim", "silk", "velvet", "vai", "len", "lua", "quan ao"]),
    MaterialClass("leather", (0.4, 0.75), 0.0, 0.9, 0.6, 0.12, (0.03, 0.6),
                  "leather with natural grain and creases", "",
                  ["leather", "hide", "da thuoc", "do da"]),
    MaterialClass("plastic", (0.25, 0.55), 0.0, 0.5, 0.6, 0.05, (0.03, 0.9),
                  "molded plastic surface", "",
                  ["plastic", "rubber", "vinyl", "nhua", "cao su"]),
    MaterialClass("ceramic", (0.05, 0.35), 0.0, 0.5, 0.6, 0.05, (0.1, 0.92),
                  "glazed ceramic surface", "",
                  ["ceramic", "porcelain", "tile", "glazed", "gom", "su"]),
    MaterialClass("organic", (0.45, 0.85), 0.0, 1.1, 0.8, 0.15, (0.04, 0.75),
                  "organic natural surface", "",
                  ["organic", "moss", "leaf", "plant", "bark", "scale", "reptile", "creature", "reu", "la cay", "vay"]),
    MaterialClass("generic", (0.4, 0.8), 0.0, 1.0, 0.75, 0.12, (0.03, 0.9), "", "", []),
]}


def strip_accents(text: str) -> str:
    t = unicodedata.normalize("NFD", text)
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    return t.replace("đ", "d").replace("Đ", "D").lower()


def _score(text: str, cls: MaterialClass) -> int:
    t = f" {strip_accents(text)} "
    score = 0
    for kw in cls.keywords:
        if re.search(r"(?<![a-z])" + re.escape(kw) + r"(?![a-z])", t):
            score += 2 if " " in kw else 1
    return score


def classify(text: str, default: str = "generic") -> str:
    """Best matching class for free text (prompt, material or object name)."""
    if not text:
        return default
    # Multi-word / more specific classes first on ties.
    order = ["rusted_metal", "painted_metal", "stone", "metal", "skin", "wood", "fabric", "leather",
             "ceramic", "plastic", "organic"]
    best, best_s = default, 0
    for name in order:
        s = _score(text, CLASSES[name])
        if s > best_s:
            best, best_s = name, s
    # "da" is ambiguous in Vietnamese (skin / stone): resolve with context words.
    t = strip_accents(text)
    if best in ("skin", "stone") and re.search(r"(?<![a-z])da(?![a-z])", t):
        if re.search(r"\b(co|tuong|khoi|phien|cam thach|stone|rock|ancient|reu)\b", t):
            best = "stone"
        elif re.search(r"\b(mat|nguoi|mặt|face|skin|lan da|lỗ chân lông|lo chan long)\b", t):
            best = "skin"
    return best


def classify_slots(prompt: str, material_names: List[Optional[str]], object_name: str = "",
                   overrides: Optional[Dict[str, str]] = None) -> List[str]:
    """Per-slot class: explicit override > prompt > material name > object name."""
    overrides = overrides or {}
    prompt_cls = classify(prompt, default="")
    out = []
    for i, mname in enumerate(material_names):
        key = str(i)
        if key in overrides or (mname and mname in overrides):
            out.append(overrides.get(key) or overrides.get(mname))
            continue
        if prompt_cls:
            out.append(prompt_cls)
            continue
        out.append(classify(f"{mname or ''} {object_name}"))
    return out


def get(name: str) -> MaterialClass:
    return CLASSES.get(name, CLASSES["generic"])
