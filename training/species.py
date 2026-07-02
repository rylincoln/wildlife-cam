"""Canonical wildlife class taxonomy for the SW-Colorado detector.

This is the **single source of truth** for the model's class names. Three things
must agree, and they all derive from here:

1. the ``names`` block in the training ``data.yaml`` (via :func:`data_yaml_names`),
2. what the fine-tuned model emits as ``model.names`` after training, and
3. the ``detection.animal_classes`` list in ``config.yaml`` (via
   :func:`config_animal_classes`) — which is an *exact string* filter in
   ``wildlife.gate.select_keepers``.

Important design note: a fine-tuned single-model detector **replaces** the stock
COCO model — it does *not* fall back to COCO for classes it wasn't trained on. So
anything you still want detected (people, black bear, generic birds) must be a
*trained* class here. That is what :data:`SUPPORT` is for; COCO gives you none of
them for free once you swap ``model_path`` to your custom weights.

Class names are lowercase ``snake_case`` (YOLO-friendly, stable dict keys).
"""

from __future__ import annotations

__all__ = [
    "TIER1",
    "TIER2",
    "TIER3",
    "SUPPORT",
    "ALL",
    "MANIFEST",
    "training_classes",
    "data_yaml_names",
    "config_animal_classes",
    "write_manifest",
    "read_manifest",
]

# A label pool records the exact ordered class list it was written against here,
# so downstream tools use the SAME class ids (a label's integer id is meaningless
# without the list it indexes into).
MANIFEST = "classes.txt"

# Tier 1 — highest trail-cam frequency in the San Juans / Durango / Four Corners.
# Build the first model on these.
TIER1: list[str] = [
    "mule_deer",
    "elk",
    "coyote",
    "red_fox",
    "gray_fox",
    "mountain_lion",  # cougar / puma
    "wild_turkey",
]

# Tier 2 — common / regionally important; add once Tier 1 works.
TIER2: list[str] = [
    "bobcat",
    "raccoon",
    "striped_skunk",
    "yellow_bellied_marmot",
    "tree_squirrel",
    "cottontail",
    "snowshoe_hare",
    "porcupine",
    "bighorn_sheep",  # NOT COCO 'sheep' (domestic)
    "pronghorn",
    "moose",
]

# Tier 3 — uncommon / high-country / data-starved. Add only with real imagery;
# expect low recall. (canada_lynx: the San Juans were the epicenter of Colorado's
# lynx reintroduction, so it's a marquee-but-rare local target.)
TIER3: list[str] = [
    "canada_lynx",
    "mountain_goat",
    "badger",
    "ringtail",
    "pine_marten",
    "dusky_grouse",
]

# Support classes: a custom model replaces COCO, so re-train these if you still
# want them. person = human/hiker/security events; black_bear = abundant here and
# COCO 'bear' is unreliable on night-IR; bird = generic presence (species-level
# birds beyond wild_turkey usually aren't worth a class on a trail cam).
SUPPORT: list[str] = [
    "person",
    "black_bear",
    "bird",
]

ALL: list[str] = [*SUPPORT, *TIER1, *TIER2, *TIER3]

_TIER_MAP = {"1": TIER1, "2": TIER2, "3": TIER3}


def training_classes(
    tiers: tuple[str, ...] = ("1",), *, include_support: bool = True
) -> list[str]:
    """Return the ordered class list to train, deduplicated, support-classes first.

    Args:
        tiers: Which wildlife tiers to include, any of ``"1"``, ``"2"``, ``"3"``.
        include_support: Prepend :data:`SUPPORT` (person/black_bear/bird). Keep
            this on unless you truly don't care about detecting them, because the
            custom model won't fall back to COCO for them.

    Returns:
        Ordered, de-duplicated class names. The index of each name is its YOLO
        class id, so keep this order stable once you start labeling.
    """
    names: list[str] = []
    if include_support:
        names.extend(SUPPORT)
    for tier in tiers:
        names.extend(_TIER_MAP[tier])
    # De-dupe while preserving first-seen order.
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def data_yaml_names(classes: list[str]) -> dict[int, str]:
    """Return the zero-indexed ``{id: name}`` mapping for a ``data.yaml`` ``names`` block."""
    return {i: name for i, name in enumerate(classes)}


def write_manifest(directory, classes: list[str]) -> None:
    """Record the ordered class list for a label pool (``<directory>/classes.txt``)."""
    from pathlib import Path

    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    (d / MANIFEST).write_text("\n".join(classes) + "\n", encoding="utf-8")


def read_manifest(directory) -> list[str] | None:
    """Return the recorded class list for a label pool, or ``None`` if absent."""
    from pathlib import Path

    p = Path(directory) / MANIFEST
    if not p.is_file():
        return None
    return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def config_animal_classes(classes: list[str]) -> str:
    """Render a ready-to-paste ``detection.animal_classes`` YAML snippet.

    The trained model emits every class in ``classes``; which ones you actually
    *keep* is up to the gate. This lists them all so nothing is silently dropped.
    """
    lines = ["animal_classes:"]
    lines.extend(f"  - {name}" for name in classes)
    return "\n".join(lines)
