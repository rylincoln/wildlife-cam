#!/usr/bin/env python3
"""Download a target-species SUBSET of a LILA camera-trap dataset.

The public LILA image archives are enormous (Caltech 105GB, Idaho ~1.45TB) and
mostly empty frames. This downloads only the images whose category maps to an
active class (see training/species.py) -- and only those a downstream
convert_lila run will actually keep (no unmapped non-empty animal in the frame)
-- from the dataset's per-image HTTPS base URL. Concurrent, and resumable (skips
files already on disk).

Then point convert_lila.py at the same metadata JSON + the downloaded images.

Examples::

    # Caltech: boxed images of your target species (~tens of thousands)
    python training/download_lila.py \
        --metadata ~/wildlife/data/caltech/caltech_bboxes.json \
        --base-url https://storage.googleapis.com/public-datasets-lila/caltech-unzipped/cct_images \
        --out ~/wildlife/data/caltech/images --tiers 12

    # cap per class to keep the download small / balanced
    python training/download_lila.py ... --max-per-class 4000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

sys.path.insert(0, str(Path(__file__).resolve().parent))
import convert_lila  # noqa: E402  (reuse the category->class alias map)
import species  # noqa: E402

_EMPTY = convert_lila._EMPTY


def md_animal_files(md: dict, conf_thresh: float) -> set[str]:
    """Return the set of file_names MegaDetector found an animal (or human) in.

    For datasets with no native boxes (e.g. Idaho), this lets us download only
    frames that will actually yield a box downstream -- skipping the many empty
    sequence frames a sequence-level species label would otherwise pull in.
    """
    keep: set[str] = set()
    for im in md.get("images", []):
        for d in im.get("detections", []):
            # categories: "1"=animal, "2"=human, "3"=vehicle
            if str(d.get("category")) in ("1", "2") and float(d.get("conf", 0)) >= conf_thresh:
                keep.add(im["file"])
                break
    return keep


def target_file_names(meta: dict, active: set[str], *, boxed_only: bool,
                      max_per_class: int | None, md_files: set[str] | None = None) -> list[str]:
    """Return the file_names to download: images convert_lila will keep with a target.

    Only images where every non-empty annotation maps to an active class (so the
    converter won't skip the frame) and at least one maps to a target species.
    When ``md_files`` is given, also require the image to be one MegaDetector
    boxed (for datasets whose boxes come from MD, not native annotations).
    ``max_per_class`` caps how many images we take per dominant class.
    """
    cat_name = {c["id"]: c["name"] for c in meta.get("categories", [])}
    images = {im["id"]: im for im in meta.get("images", [])}
    anns_by_img: dict[str, list[dict]] = defaultdict(list)
    for a in meta.get("annotations", []):
        anns_by_img[a["image_id"]].append(a)

    per_class = Counter()
    wanted: list[str] = []
    for img_id, image in images.items():
        anns = anns_by_img.get(img_id, [])
        mapped = [convert_lila._map_category(cat_name.get(a["category_id"], ""), active) for a in anns]
        species_here = [m for m in mapped if m not in (None, _EMPTY)]
        if not species_here or any(m is None for m in mapped):
            continue  # no target, or an unmapped non-empty animal -> convert would skip it
        if boxed_only and not any("bbox" in a or "bbox_relative" in a for a in anns):
            continue
        if md_files is not None and image["file_name"] not in md_files:
            continue  # no MegaDetector box -> convert can't produce a label
        if max_per_class is not None:
            dom = species_here[0]
            if per_class[dom] >= max_per_class:
                continue
            per_class[dom] += 1
        wanted.append(image["file_name"])
    return sorted(set(wanted))


def _download_one(base_url: str, file_name: str, out: Path) -> str:
    """Download one image (skip if present). Return 'ok' | 'skip' | 'err:...'."""
    dst = out / file_name
    if dst.is_file() and dst.stat().st_size > 0:
        return "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = f"{base_url.rstrip('/')}/{file_name}"
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        urlretrieve(url, tmp)
        tmp.replace(dst)
        return "ok"
    except (HTTPError, URLError, OSError) as exc:
        if tmp.exists():
            tmp.unlink()
        return f"err:{exc}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata", required=True, help="COCO-Camera-Traps JSON.")
    p.add_argument("--base-url", required=True, help="Per-image HTTPS base URL (file_name appended).")
    p.add_argument("--out", required=True, help="Image output dir (file_name paths preserved).")
    p.add_argument("--tiers", default="12")
    p.add_argument("--no-support", action="store_true")
    p.add_argument("--boxed-only", action="store_true", help="Only images that carry a bbox annotation.")
    p.add_argument("--md-results", default=None, help="LILA MegaDetector-results JSON: only fetch "
                   "images MD boxed (for datasets with no native boxes, e.g. Idaho).")
    p.add_argument("--md-conf", type=float, default=0.2, help="Min MD confidence for --md-results.")
    p.add_argument("--max-per-class", type=int, default=None, help="Cap images per dominant class.")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=None, help="Only download the first N (testing).")
    args = p.parse_args()

    out = Path(args.out).expanduser()
    active = set(species.training_classes(tuple(args.tiers), include_support=not args.no_support))
    meta = json.loads(Path(args.metadata).expanduser().read_text())

    md_files = None
    if args.md_results:
        md = json.loads(Path(args.md_results).expanduser().read_text())
        md_files = md_animal_files(md, args.md_conf)
        print(f"MegaDetector boxed {len(md_files)} image(s) (conf>={args.md_conf})")

    files = target_file_names(meta, active, boxed_only=args.boxed_only,
                              max_per_class=args.max_per_class, md_files=md_files)
    if args.limit:
        files = files[: args.limit]
    print(f"{len(files)} target image(s) to fetch -> {out}")

    counts = Counter()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_download_one, args.base_url, fn, out): fn for fn in files}
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            counts["ok" if res == "ok" else "skip" if res == "skip" else "err"] += 1
            if res.startswith("err") and counts["err"] <= 10:
                print(f"  {res}  ({futs[fut]})", file=sys.stderr)
            if i % 1000 == 0 or i == len(files):
                rate = i / max(1e-6, time.time() - t0)
                print(f"  {i}/{len(files)}  ok={counts['ok']} skip={counts['skip']} "
                      f"err={counts['err']}  ({rate:.0f}/s)")
    print(f"Done: ok={counts['ok']} skip={counts['skip']} err={counts['err']}")
    print(f"Next: python training/convert_lila.py --images {out} --cct {args.metadata} "
          f"--out <pool> --tiers {args.tiers}")
    return 1 if counts["err"] and not counts["ok"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
