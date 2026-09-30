"""Prompt construction for the image model (pure, shared by synth and bridge)."""

from __future__ import annotations

from typing import Dict

from .intent import Intent

VIEW_PHRASES = {
    "back_left": "three-quarter view from the back left",
    "back_right": "three-quarter view from the back right",
    "top_front": "high angle view from the front",
    "top_back": "high angle view from behind",
    "top_left": "high angle view from the left",
    "top_right": "high angle view from the right",
    "bottom_front": "low angle view from the front",
    "bottom_back": "low angle view from behind",
    "bottom_left": "low angle view from the left",
    "bottom_right": "low angle view from the right",
    "front": "front view",
    "back": "back view, seen from behind",
    "left": "side view from the left",
    "right": "side view from the right",
    "top": "view from directly above",
    "bottom": "view from directly below",
    "front_left": "three-quarter view from the front left",
    "front_right": "three-quarter view from the front right",
}

ALBEDO_STYLE = ("flat even diffuse lighting, unlit albedo texture, no shadows, "
                "neutral color, physically based base color, highly detailed surface")


def build_prompts(intent: Intent, view: str, extra_positive: str = "") -> Dict[str, str]:
    parts = [p for p in (intent.appearance_prompt.strip(), extra_positive.strip()) if p]
    parts.append(VIEW_PHRASES.get(view, view.replace("_", " ")))
    parts.append(ALBEDO_STYLE)
    neg = intent.negative_prompt or ""
    neg += ", deformed, extra limbs, changed silhouette, different proportions, blurry, lowres"
    return {"positive": ", ".join(parts), "negative": neg.strip(", ")}
