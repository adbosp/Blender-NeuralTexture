"""Text-prompt -> structured editing intent (spec section 11).

The orchestrating agent (Claude) SHOULD produce the structured ``Intent``
JSON itself and pass it to the pipeline. This rule-based parser is the
deterministic fallback used by the Blender UI and for validation: it never
invents intent, it only recognises explicit phrases.

Most important job: detect requests that textures cannot honestly
represent ("make the nose twice as large") and return a structured
geometry warning instead of faking them with an extreme normal map.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List

NEGATION = re.compile(r"^\s*(do not|don't|dont|never|keep|preserve|without|no\b|avoid|retain|maintain)", re.I)

SIZE_WORDS = r"(bigger|larger|smaller|longer|shorter|wider|narrower|taller|thicker|thinner|fatter|slimmer|rounder|sharper|pointier|flatter|deeper|higher|lower)"
GEOMETRY_PATTERNS = [
    (re.compile(r"\b(twice|two times|three times|\d+(\.\d+)?\s*x|\d+\s*%)\s+(as\s+)?(big|large|long|wide|tall|small)", re.I),
     "scale change"),
    (re.compile(r"\b(make|making|made|turn)\b[^.;,]*\b" + SIZE_WORDS + r"\b", re.I), "shape/size change"),
    (re.compile(r"\b(reshape|re-shape|resculpt|sculpt|extrude|deform|elongate|shrink|enlarge|inflate)\b", re.I),
     "sculpting operation"),
    (re.compile(r"\b(change|alter|modify)\b[^.;,]*\b(silhouette|shape|proportions?|topology|pose)\b", re.I),
     "silhouette/proportion change"),
    (re.compile(r"\badd\b[^.;,]*\b(horns?|ears?|tail|wings?|spikes?|antlers?|fingers?|limbs?|arms?|legs?)\b", re.I),
     "new geometry"),
    (re.compile(r"\b(move|rotate|open|close)\b[^.;,]*\b(mouth|jaw|eyes?|eyelids?|arm|hand|head|bones?)\b", re.I),
     "pose/rig change"),
]

MESO_WORDS = re.compile(r"\b(wrinkles?|lip|lips|nostrils?|eyelids?|scars?|seams?|folds?|creases?|cracks?|stitch(es|ing)?|dents?)\b", re.I)
MICRO_WORDS = re.compile(r"\b(pores?|micro|high[- ]frequency|fine|skin grain|grain|fib(er|re)s?|scratch(es)?|noise|speckles?)\b", re.I)
MICRO_ONLY = re.compile(r"\b(only|just)\b[^.;]*\b(high[- ]frequency|micro|fine)\b|\b(high[- ]frequency|micro)[^.;]*\bonly\b", re.I)

SMOOTH = re.compile(r"\b(smoother|smooth|softer|polished|glossier|glossy|shinier|shiny|wet)\b", re.I)
ROUGH = re.compile(r"\b(rougher|rough|coarser|matte|dry|drier|weathered|worn)\b", re.I)
METAL = re.compile(r"\b(metal|metallic|chrome|steel|iron|gold|silver|copper|brass|bronze)\b", re.I)
WEAK = re.compile(r"\b(slightly|subtle|subtly|a bit|a little|gently|light(ly)?|minor)\b", re.I)
STRONG = re.compile(r"\b(very|strongly|heavily|much|extremely|a lot|significantly|dramatically)\b", re.I)
RESEMBLE = re.compile(r"\b(resemble|look like|likeness|match(es)?|similar to|based on)\b", re.I)


@dataclass
class GeometryWarning:
    clause: str
    kind: str
    message: str

    def to_dict(self):
        return asdict(self)


@dataclass
class Intent:
    # Structural preservation (spec section 5). All True by default.
    preserve_silhouette: bool = True
    preserve_proportions: bool = True
    preserve_macro_normal: bool = True
    preserve_uvs: bool = True
    # Detail bands to generate (spec section 4).
    generate_micro: bool = True
    generate_meso: bool = False
    detail_strength: float = 0.6      # 0..1 multiplier for generated detail normal
    meso_strength: float = 0.35
    # Surface response.
    roughness_offset: float = 0.0     # added to existing roughness
    metallic_hint: bool = False
    # Appearance.
    reference_role: str = "appearance"  # appearance | identity | material
    identity_strength: float = 0.8
    appearance_prompt: str = ""        # cleaned text forwarded to the image model
    negative_prompt: str = ""
    target_maps: List[str] = field(default_factory=lambda: ["basecolor", "detail_normal", "normal", "roughness"])
    # V3: material awareness. material_class applies to all slots unless
    # material_classes maps slot index / material name -> class.
    material_class: str = ""
    material_classes: Dict[str, str] = field(default_factory=dict)
    mode: str = "enhance"              # enhance (keep look, add detail) | restyle (new appearance)
    language: str = "en"
    geometry_warnings: List[GeometryWarning] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["geometry_warnings"] = [w.to_dict() if isinstance(w, GeometryWarning) else w
                                  for w in self.geometry_warnings]
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "Intent":
        d = dict(d)
        warns = [GeometryWarning(**w) if isinstance(w, dict) else w for w in d.pop("geometry_warnings", [])]
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        it = cls(**known)
        it.geometry_warnings = warns
        return it

    @property
    def requires_geometry_mode(self) -> bool:
        return bool(self.geometry_warnings)


def _clauses(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?;\n])\s+|\n+|,?\s+\b(?:but|however|nhưng|nhung|song)\b\s+|,\s+(?=(?:và\s+)?(?:giữ|keep|do not|don't|không|đừng)\b)",
                     text, flags=re.I)
    out = []
    for p in parts:
        p = (p or "").strip()
        if p:
            out.append(p)
    return out


def _intensity(clause: str) -> float:
    if WEAK.search(clause):
        return 0.5
    if STRONG.search(clause):
        return 1.5
    return 1.0


# Vietnamese -> English phrase table, applied to accent-stripped lowercase text.
# Longest phrases first. Lets artists write e.g.
#   "làm model này thành đá cổ nhưng giữ normal gốc"
VI_PHRASES = [
    ("khong duoc", "do not"), ("dung lam", "do not make"), ("dung", "do not"), ("khong", "do not"),
    ("giu nguyen", "keep"), ("giu lai", "keep"), ("bao toan", "preserve"), ("giu", "keep"),
    ("normal map goc", "original large normal"), ("normal goc", "original large normal"),
    ("ban do phap tuyen goc", "original large normal"),
    ("hinh dang", "shape"), ("hinh khoi", "shape"), ("duong vien", "silhouette"), ("ti le", "proportions"),
    ("thay doi", "change"), ("bien doi", "change"),
    ("gap doi", "twice as large"), ("gap ba", "three times as large"),
    ("to hon", "bigger"), ("lon hon", "larger"), ("nho hon", "smaller"), ("dai hon", "longer"),
    ("ngan hon", "shorter"), ("rong hon", "wider"), ("hep hon", "narrower"), ("cao hon", "taller"),
    ("day hon", "thicker"), ("mong hon", "thinner"), ("beo hon", "fatter"), ("nhon hon", "sharper"),
    ("them sung", "add horns"), ("them tai", "add ears"), ("them duoi", "add tail"), ("them canh", "add wings"),
    ("mo mieng", "open the mouth"), ("ha ham", "open the jaw"), ("nham mat", "close the eyes"),
    ("nan lai", "resculpt"), ("keo dai", "elongate"),
    ("lo chan long", "pores"), ("vi chi tiet", "micro detail"), ("chi tiet tan so cao", "high-frequency detail"),
    ("chi tiet nho", "fine detail"), ("chi tiet", "details"), ("nep nhan", "wrinkles"), ("vet nut", "cracks"),
    ("vet seo", "scars"), ("seo", "scars"), ("duong may", "seams"), ("vet xuoc", "scratches"),
    ("min hon", "smoother"), ("muot hon", "smoother"), ("min", "smooth"), ("muot", "smooth"),
    ("bong hon", "glossier"), ("bong", "glossy"), ("tho hon", "rougher"), ("nham hon", "rougher"),
    ("tho", "rough"), ("nham", "rough"), ("san sui", "rough"), ("phong hoa", "weathered"),
    ("bao mon", "worn"), ("cu ki", "old worn"), ("cu", "worn"),
    ("mot chut", "slightly"), ("hoi", "slightly"), ("nhe", "subtle"), ("tinh te", "subtle"),
    ("rat", "very"), ("manh", "strongly"),
    ("trong giong", "look like"), ("giong", "resemble"),
    ("chi them", "only add"), ("thoi", "only"), ("chi", "only"),
    ("them", "add"), ("lam cho", "make"), ("lam", "make"), ("thanh", "into"),
    ("da co", "ancient stone"), ("tuong da", "stone statue"), ("cam thach", "marble"), ("be tong", "concrete"),
    ("kim loai", "metal"), ("gi set", "rusty"), ("go", "wood"), ("vai", "fabric"), ("do da", "leather"),
    ("lan da", "skin"), ("da mat", "face skin"), ("khuon mat", "face"), ("mat", "face"),
    ("co dai", "ancient"), ("co", "ancient"), ("vang", "gold"), ("bac", "silver"), ("dong", "bronze"),
    ("sat", "iron"), ("thep", "steel"), ("reu", "moss"), ("model nay", "the model"), ("nhan vat", "character"),
    ("va", "and"), ("cua", "of"), ("cho", "for"), ("nay", "this"),
]


def is_vietnamese(text: str) -> bool:
    from .materials import strip_accents
    if re.search(r"[ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]", text.lower()):
        return True
    t = f" {strip_accents(text)} "
    return bool(re.search(r" (khong|giu|lam|thanh|cho|nay|nhung|hon|them) ", t))


def vi_to_en(text: str) -> str:
    from .materials import strip_accents
    t = strip_accents(text)
    for vi, en in VI_PHRASES:
        t = re.sub(r"(?<![a-z])" + re.escape(vi) + r"(?![a-z])", en, t)
    # Vietnamese places "da" (skin/stone) contextually; default to the material classifier.
    return re.sub(r"\s+", " ", t).strip()


def parse_prompt(text: str, has_reference: bool = False) -> Intent:
    from . import materials as _mats
    it = Intent()
    original = text or ""
    if is_vietnamese(original):
        it.language = "vi"
        text = "\n".join(vi_to_en(c) for c in _clauses(original))
    it.material_class = _mats.classify(original, default="")
    if it.material_class and re.search(r"\b(into|turn|make it|as|restyle|convert|thanh)\b",
                                       _mats.strip_accents(original) + " " + (text or "")):
        it.mode = "restyle"
    appearance: List[str] = []
    detail_scale = 1.0
    micro_only = False
    meso_requested = False

    for clause in _clauses(text or ""):
        negated = bool(NEGATION.match(clause))
        low = clause.lower()

        if negated:
            # Preservation statements reinforce defaults; never trigger edits.
            if "normal" in low and ("large" in low or "macro" in low or "original" in low or "existing" in low):
                it.preserve_macro_normal = True
            if "uv" in low:
                it.preserve_uvs = True
            continue

        geo_hit = False
        for pat, kind in GEOMETRY_PATTERNS:
            m = pat.search(clause)
            if m:
                # "make the skin smoother" is surface, not shape.
                if kind == "shape/size change" and re.search(r"\b(skin|surface|texture|finish|material)\b", low) \
                        and not re.search(SIZE_WORDS, low):
                    continue
                it.geometry_warnings.append(GeometryWarning(
                    clause=clause, kind=kind,
                    message=("This request changes geometry. Texture/normal generation will not "
                             "implement it; use geometry or displacement editing mode instead.")))
                geo_hit = True
                break
        if geo_hit:
            continue

        k = _intensity(clause)
        if MICRO_ONLY.search(clause):
            micro_only = True
        else:
            wants_detail = re.search(r"\b(add|more|define|enhance|sharpen|stronger)\b", low)
            if wants_detail and (MESO_WORDS.search(clause) or
                                 (re.search(r"\bdetails?\b", low) and not MICRO_WORDS.search(clause))):
                meso_requested = True
                it.meso_strength = round(min(1.0, 0.35 * k), 3)
            if MICRO_WORDS.search(clause):
                it.generate_micro = True
        if SMOOTH.search(clause):
            it.roughness_offset -= 0.08 * k
            detail_scale *= 1.0 - 0.3 * k
        if ROUGH.search(clause):
            it.roughness_offset += 0.08 * k
            detail_scale *= 1.0 + 0.25 * k
        if METAL.search(clause):
            it.metallic_hint = True
        if RESEMBLE.search(clause) and has_reference:
            it.reference_role = "identity"
            it.identity_strength = round(min(1.0, 0.8 * k), 3)
        appearance.append(clause)

    if micro_only:
        it.generate_micro = True
        it.notes.append("Generated normal detail restricted to the high-frequency band.")
    if meso_requested:
        it.generate_meso = True
        if micro_only:
            it.meso_strength = min(it.meso_strength, 0.2)
            it.notes.append("Explicitly named features (e.g. lips/nose) kept as subtle meso detail "
                            "despite the high-frequency-only request.")
    it.detail_strength = round(float(min(1.0, max(0.05, it.detail_strength * detail_scale))), 3)
    it.roughness_offset = round(float(max(-0.5, min(0.5, it.roughness_offset))), 3)
    if it.metallic_hint and "metallic" not in it.target_maps:
        it.target_maps.append("metallic")
    appearance_text = " ".join(appearance)
    if it.material_class:
        mc = _mats.get(it.material_class)
        appearance_text = ", ".join(x for x in (mc.prompt, appearance_text) if x)
        if mc.negative:
            it.negative_prompt = mc.negative
        if mc.metallic > 0:
            it.metallic_hint = True
    it.appearance_prompt = appearance_text
    it.negative_prompt = ((it.negative_prompt + ", ") if it.negative_prompt else "") + ("shadows, cast shadows, directional lighting, specular highlights, "
                          "reflections, rim light, vignette, background, text, watermark")
    return it


def resolve_intent(cfg) -> Intent:
    """Structured intent from the job (agent-provided) or parsed from its prompt."""
    if getattr(cfg, "intent", None):
        it = Intent.from_dict(cfg.intent)
    else:
        it = parse_prompt(cfg.prompt, has_reference=bool(cfg.reference_images))
    for k, v in (getattr(cfg, "intent_overrides", None) or {}).items():
        if k in Intent.__dataclass_fields__ and k != "geometry_warnings":
            setattr(it, k, v)
    if it.material_class and getattr(cfg, "intent_overrides", {}).get("material_class"):
        from . import materials as _m
        mc = _m.get(it.material_class)
        if mc.prompt and mc.prompt not in it.appearance_prompt:
            it.appearance_prompt = ", ".join(x for x in (mc.prompt, it.appearance_prompt) if x)
        if mc.metallic > 0:
            it.metallic_hint = True
    return it
