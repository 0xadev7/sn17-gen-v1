from __future__ import annotations
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Set, Tuple

# --- Configurable rule sets --------------------------------------------------

BASE_PREFIX = (
    "studio product photo, single centered object, 3/4 view, isolated on light gray "
    "gradient, neutral three-point lighting, physically-plausible materials, high detail"
)

BASE_NEGATIVE = (
    "no humans, no hands, no text, no labels, no watermarks, no background scene, "
    "no environment, no cropping, no blur, no depth of field, no duplicates, no multiple objects"
)

# Buckets that trigger special handling
SPECULAR_KEYS = {
    "glass",
    "crystal",
    "transparent",
    "translucent",
    "acrylic",
    "plastic (transparent)",
    "clear",
    "chandelier",
    "vase",
    "cup",
    "goblet",
    "goblets",
}
GEM_KEYS = {
    "diamond",
    "topaz",
    "gem",
    "gemstone",
    "ruby",
    "emerald",
    "sapphire",
    "opal",
    "jewel",
    "pearl",
}
METAL_KEYS = {
    "metal",
    "steel",
    "stainless",
    "titanium",
    "brass",
    "bronze",
    "copper",
    "gold",
    "silver",
    "chrome",
    "polished",
    "anodized",
    "chisel",
    "wrench",
    "ring",
    "band",
    "candlestick",
}
THIN_FILIGREE_KEYS = {
    "filigree",
    "lace",
    "woven",
    "bamboo",
    "basket",
    "mesh",
    "wire",
    "chandelier",
    "droplets",
    "plant",
    "leaf",
    "leaves",
    "branch",
    "branches",
    "yarn",
    "thread",
    "strand",
    "strands",
    "hair",
    "fringe",
}
TEXTILE_KEYS = {
    "scarf",
    "wool",
    "leather",
    "fabric",
    "cloth",
    "textile",
    "silk",
    "cotton",
    "velvet",
    "robe",
    "garment",
    "glove",
}
ORGANIC_TINY_KEYS = {"bee", "insect", "bumblebee", "flower", "iris", "rosemary"}
ANATOMY_PEOPLE_KEYS = {
    "princess",
    "mermaid",
    "doll",
    "noblewoman",
    "woman",
    "man",
    "person",
    "portrait",
    "costume",
    "robe",
}
SCENEY_VERBS = {
    "leaving",
    "riding",
    "charging",
    "floating",
    "dangling",
    "casting",
    "spiraling",
    "lying",
    "lying down",
    "standing upright",
    "aloft",
    "hovering",
    "soaring",
}
MULTI_COUNT_PATTERNS = [
    r"\bin various sizes\b",
    r"\bset of\b",
    r"\bcollection of\b",
    r"\bmultiple\b",
    r"\bseveral\b",
]

# Material prompt add-ons (positive)
MATERIAL_ADDONS: Dict[str, str] = {
    "specular": "specular highlights, low roughness, visible rim lighting, preserve refraction cues",
    "gems": "facet cuts visible, dispersion sparkle hints, crisp edges, controlled reflections",
    "metal": "anisotropic metal BRDF, subtle micro-scratches, brushed finish, varied roughness",
    "textile": "fine weave/stitch detail, natural folds, fiber microtexture, soft subsurface scattering",
    "filigree": "thin geometry preserved, avoid strand merging, high-frequency detail retained",
    "organic_tiny": "macro detail preserved, crisp silhouette, no motion blur",
}

# Render/material preset suggestions (you can feed these into your renderer)
MATERIAL_PRESETS: Dict[str, Dict[str, float]] = {
    "glass": {
        "ior": 1.52,
        "roughness_min": 0.02,
        "roughness_max": 0.08,
        "anisotropy": 0.0,
    },
    "gem": {
        "ior": 1.55,
        "roughness_min": 0.01,
        "roughness_max": 0.05,
        "anisotropy": 0.0,
    },
    "metal_brushed": {
        "ior": 0.0,
        "roughness_min": 0.08,
        "roughness_max": 0.25,
        "anisotropy": 0.6,
    },
    "textile": {
        "ior": 0.0,
        "roughness_min": 0.35,
        "roughness_max": 0.65,
        "anisotropy": 0.0,
    },
}

# --- Output schema -----------------------------------------------------------


@dataclass
class PromptRefactor:
    original: str
    rewritten_prompt: str  # Final positive prompt for T2I
    positive_prompt: str  # (BASE_PREFIX + material add-ons + rewritten noun phrase)
    negative_prompt: str  # BASE_NEGATIVE (possibly extended)
    hard_tags: List[str]  # Categories that triggered rules
    mv_count: int  # Recommend #views before Trellis (8 or 12)
    bg_removal_stage: str  # "after-mv" or "before-mv"
    candidate_beam: int  # 1..3: generate K candidates then pick by validator
    material_presets: Dict[str, Dict[str, float]]  # Optional renderer presets
    notes: List[str]  # Human-readable rationale

    def asdict(self) -> Dict:
        return asdict(self)


# --- Core function -----------------------------------------------------------


def refactor_prompt(raw: str) -> PromptRefactor:
    """
    Refactor a free-form prompt into an object-centric, material-aware prompt with routing hints.

    Heuristics implemented:
    - De-scene/verb: prefer noun-centric phrasing
    - Collapse multi-object phrasing to a single exemplar
    - Add material-specific positive tokens
    - Provide negative prompt to avoid scenes/people/duplicates
    - Route to harder MV settings for thin/specular/gems/textiles/organic tiny
    - Suggest renderer presets for materials
    """
    original = raw.strip()
    low = original.lower().strip()
    notes: List[str] = []

    # (1) Collapse multi-object phrasing → single item
    collapsed = low
    for pat in MULTI_COUNT_PATTERNS:
        if re.search(pat, collapsed):
            collapsed = re.sub(pat, "a single", collapsed)
            notes.append("Rewrote multi-object phrasing to single exemplar.")

    # (2) Remove overt scene/action hints but keep descriptors
    # Simple pass: drop leading/trailing verb phrases we know about
    for v in sorted(SCENEY_VERBS, key=len, reverse=True):
        collapsed = re.sub(rf"\b{re.escape(v)}\b", "", collapsed)

    # Normalize whitespace and articles
    collapsed = re.sub(r"\s+", " ", collapsed).strip(",. ").strip()
    if not collapsed.startswith(("a ", "an ", "the ")):
        collapsed = f"a {collapsed}"

    # (3) If human/anatomy present → figurine/prop rewrite
    hard_tags: Set[str] = set()
    if any(k in collapsed for k in ANATOMY_PEOPLE_KEYS):
        collapsed += ", figurine/prop, no human anatomy"
        hard_tags.add("anatomy_to_prop")
        notes.append("Converted human/character concept into a figurine/prop.")

    # (4) Detect material/shape categories
    has_specular = any(k in collapsed for k in SPECULAR_KEYS)
    has_gem = any(k in collapsed for k in GEM_KEYS)
    has_metal = any(k in collapsed for k in METAL_KEYS)
    has_thin = any(k in collapsed for k in THIN_FILIGREE_KEYS)
    has_textile = any(k in collapsed for k in TEXTILE_KEYS)
    has_organic_tiny = any(k in collapsed for k in ORGANIC_TINY_KEYS)

    # Record hard tags
    if has_specular:
        hard_tags.add("specular")
    if has_gem:
        hard_tags.add("gems")
    if has_metal:
        hard_tags.add("metal")
    if has_thin:
        hard_tags.add("thin_filigree")
    if has_textile:
        hard_tags.add("textile")
    if has_organic_tiny:
        hard_tags.add("organic_tiny")

    # (5) Build positive prompt
    positive_parts = [BASE_PREFIX, collapsed]

    if has_specular:
        positive_parts.append(MATERIAL_ADDONS["specular"])
        notes.append("Added specular/transparency cues (rim light, low roughness).")
    if has_gem:
        positive_parts.append(MATERIAL_ADDONS["gems"])
        notes.append("Added gem facet/dispersion cues.")
    if has_metal:
        positive_parts.append(MATERIAL_ADDONS["metal"])
        notes.append("Added anisotropic metal/brushed finish cues.")
    if has_textile:
        positive_parts.append(MATERIAL_ADDONS["textile"])
        notes.append("Added textile weave/fiber and soft-SSS cues.")
    if has_thin:
        positive_parts.append(MATERIAL_ADDONS["filigree"])
        notes.append("Emphasized preservation of thin/filigree geometry.")
    if has_organic_tiny:
        positive_parts.append(MATERIAL_ADDONS["organic_tiny"])
        notes.append("Requested macro-level crispness for small organic subject.")

    positive_prompt = ", ".join(p for p in positive_parts if p)

    # (6) Negative prompt (extend for tricky cases)
    negative_parts = [BASE_NEGATIVE]
    if has_thin:
        negative_parts.append("no alpha fringing, no strand fusion")
    if has_specular or has_gem:
        negative_parts.append("no fake caustics, no blown highlights")
    if has_metal:
        negative_parts.append("no plastic look, no painted metal")
    negative_prompt = ", ".join(negative_parts)

    # (7) Routing hints
    # Default: generate MV after t2i and do BG removal AFTER MV
    mv_count = (
        12
        if (has_thin or has_specular or has_gem or has_textile or has_organic_tiny)
        else 8
    )
    bg_stage = "after-mv"  # robust default; thin/specular especially need context
    candidate_beam = 3 if (has_thin or has_specular or has_gem) else 2

    # (8) Material presets for renderer
    presets: Dict[str, Dict[str, float]] = {}
    if has_specular:
        presets["glass"] = MATERIAL_PRESETS["glass"]
    if has_gem:
        presets["gem"] = MATERIAL_PRESETS["gem"]
    if has_metal:
        presets["metal_brushed"] = MATERIAL_PRESETS["metal_brushed"]
    if has_textile:
        presets["textile"] = MATERIAL_PRESETS["textile"]

    # (9) Final composed rewritten prompt (what you feed to SD3.5)
    rewritten_prompt = positive_prompt

    return PromptRefactor(
        original=original,
        rewritten_prompt=rewritten_prompt,
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        hard_tags=sorted(hard_tags),
        mv_count=mv_count,
        bg_removal_stage=bg_stage,
        candidate_beam=candidate_beam,
        material_presets=presets,
        notes=notes,
    )


# --- Example -----------------------------------------------------------------
if __name__ == "__main__":
    samples = [
        "polished brass candlestick standing upright",
        "a dazzling diamond chandelier with sparkling crystals and golden accents",
        "white bus leaving depot",
        "woven bamboo basket filled with colorful fruits",
        "a small, fuzzy bumblebee with a bright yellow and black striped body",
    ]
    for s in samples:
        out = refactor_prompt(s)
        print("\n---")
        print(out.rewritten_prompt)
        print("NEG:", out.negative_prompt)
        print(
            "tags:",
            out.hard_tags,
            "| mv:",
            out.mv_count,
            "| beam:",
            out.candidate_beam,
            "| bg:",
            out.bg_removal_stage,
        )
        print("presets:", out.material_presets)
        print("notes:", out.notes)
