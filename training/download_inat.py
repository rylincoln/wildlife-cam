#!/usr/bin/env python3
"""Download research-grade, CC-licensed iNaturalist photos for a target species.

Fills the classes LILA camera-trap sets don't cover (yellow_bellied_marmot,
snowshoe_hare, porcupine) and supplements thin ones with clean exemplars
(bighorn_sheep, and pronghorn to dilute the noisy Idaho MegaDetector boxes).

iNaturalist photos are *iconic* (the animal large and centered) and carry NO
bounding boxes -- so this only fetches the imagery. Boxes come next from
``label_boxes.py`` (MegaDetector v6 + the species we already know from the taxon
query). GBIF is intentionally NOT used: for these species in Colorado ~95-99% of
GBIF media is re-ingested iNaturalist, and the rest is museum-specimen photos
(dead skins/skulls) that would hurt a trail-cam detector.

Layout (one subdir per OBSERVATION so ``prepare_dataset`` keeps an observation's
photos on one side of the train/val split -- the same burst-safety convert_lila
gives sequences)::

    <out>/<observation_id>/<photo_id>.jpg
    <out>/attribution.csv            (photo_id, obs_id, license, attribution, url)

Example::

    python training/download_inat.py --class yellow_bellied_marmot \
        --place-id 34 --per-species 1200 --out ~/wildlife/data/inat/yellow_bellied_marmot

Then:  python training/label_boxes.py --images <out> --assume-species yellow_bellied_marmot ...
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

API = "https://api.inaturalist.org/v1"
# Be a good API citizen: descriptive UA with contact per iNat guidance.
UA = "wildlife-cam/1.0 (https://github.com/rylincoln/wildlife-cam; rylincoln@gmail.com)"

# Verified research-grade taxon ids (Colorado counts checked live). taxon_id
# includes descendant taxa, so subspecies (e.g. both bighorn subspecies) are covered.
CLASS_TAXON: dict[str, int] = {
    "yellow_bellied_marmot": 46086,  # Marmota flaviventris
    "snowshoe_hare": 43132,          # Lepus americanus
    "porcupine": 44026,              # Erethizon dorsatum
    "bighorn_sheep": 42391,          # Ovis canadensis
    "pronghorn": 42429,              # Antilocapra americana
    "person": 43584,                 # Homo sapiens (fallback source only; prefer LILA idaho human)
}
# iNat photo size tokens (…/<id>/<size>.jpg): square75 small240 medium500 large1024 original.
_SIZES = ("square", "small", "medium", "large", "original")


def _get_json(url: str, retries: int = 4) -> dict:
    """GET JSON with retry + backoff. iNat returns 429/503 (and sometimes an HTML
    error page that breaks json.loads) under load -- a single transient failure
    must not abort a multi-hundred-record fetch. Honors Retry-After on throttle."""
    import json
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            last = exc
            if exc.code in (429, 500, 502, 503, 504):
                wait = float(exc.headers.get("Retry-After", 0) or 0) or (2 ** attempt)
                time.sleep(min(wait, 30))
                continue
            raise  # 4xx (bad query) should fail loudly, not retry
        except (URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"iNat request failed after {retries} attempts: {url} ({last})")


def resolve_taxon(name: str) -> int | None:
    """Look up a species' iNaturalist taxon id by scientific/common name."""
    from urllib.parse import quote
    data = _get_json(f"{API}/taxa?q={quote(name)}&rank=species&per_page=5")
    results = data.get("results", [])
    return int(results[0]["id"]) if results else None


def photo_url(photo: dict, size: str) -> str | None:
    """Build a sized photo URL from a photo record's ``url`` (a square.jpg link)."""
    base = photo.get("url") or ""
    for s in _SIZES:
        if f"/{s}." in base:
            return base.replace(f"/{s}.", f"/{size}.", 1)
    return None


def fetch_photo_records(taxon_id: int, place_id: str, licenses: str, cap: int,
                        one_per_obs: bool = True) -> list[dict]:
    """Page /v1/observations (id_above cursor) -> [{obs_id, photo_id, url, license, attribution}].

    Uses the id_above sliding window (mandatory past 10k results, leak-free
    always) rather than page increment. Caps to one photo per observation by
    default so near-duplicate photos of the same individual can't split across
    train/val. ``place_id`` is passed through verbatim ('' = full range;
    comma-separated ids = union, e.g. broaden a scarce class to neighbor states).
    """
    from urllib.parse import urlencode
    out: list[dict] = []
    seen_photos: set[int] = set()
    last_id = 0
    per_page = 200
    while len(out) < cap:
        params = {
            "taxon_id": taxon_id, "quality_grade": "research", "photos": "true",
            "photo_license": licenses, "per_page": per_page,
            "order_by": "id", "order": "asc", "id_above": last_id,
            "captive": "false",  # drop captive/cultivated (zoo/mounted) records
        }
        if place_id:
            params["place_id"] = place_id
        try:
            data = _get_json(f"{API}/observations?{urlencode(params)}")
        except RuntimeError as exc:  # retries exhausted -- keep what we have
            print(f"  warning: stopping paging early ({exc}); keeping {len(out)} record(s).", file=sys.stderr)
            break
        results = data.get("results", [])
        if not results:
            break
        for obs in results:
            oid = obs.get("id")
            if oid is None:
                continue
            last_id = max(last_id, int(oid))
            for ph in (obs.get("photos") or []):
                pid = ph.get("id")
                if pid is None:
                    continue
                pid = int(pid)
                if pid in seen_photos:
                    continue
                url = photo_url(ph, "large")
                if not url:
                    continue  # try the next photo of this obs, not skip the obs
                seen_photos.add(pid)
                out.append({"obs_id": int(oid), "photo_id": pid, "url": url,
                            "license": ph.get("license_code") or "", "attribution": ph.get("attribution") or ""})
                if one_per_obs:
                    break  # took the first USABLE photo of this observation
            if len(out) >= cap:
                break
        if len(results) < per_page:
            break  # last window
        time.sleep(1.0)  # <=60 req/min courtesy throttle on the API
    return out[:cap]


def _download_one(rec: dict, out: Path) -> str:
    """Download one photo into <out>/<obs_id>/<photo_id>.jpg (skip if present)."""
    dst = out / str(rec["obs_id"]) / f"{rec['photo_id']}.jpg"
    if dst.is_file() and dst.stat().st_size > 0:
        return "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")
    try:
        req = Request(rec["url"], headers={"User-Agent": UA})
        with urlopen(req, timeout=60) as resp, open(tmp, "wb") as fh:
            fh.write(resp.read())
        if tmp.stat().st_size == 0:
            tmp.unlink()
            return "err:empty"
        tmp.replace(dst)
        return "ok"
    except (HTTPError, URLError, OSError) as exc:
        if tmp.exists():
            tmp.unlink()
        return f"err:{exc}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--class", dest="klass", help="Target class (looks up taxon id): "
                   + ", ".join(CLASS_TAXON))
    p.add_argument("--taxon-id", type=int, default=None, help="Explicit iNat taxon id (overrides --class).")
    p.add_argument("--taxon", default=None, help="Resolve a taxon id by name instead.")
    p.add_argument("--out", required=True, help="Image output dir (per-observation subdirs).")
    p.add_argument("--place-id", default="34", help="iNat place_id ('' = full range; 34 = Colorado; "
                   "comma-separated = union, to broaden a scarce class to neighbor states).")
    p.add_argument("--license", default="cc0,cc-by,cc-by-nc",
                   help="photo_license filter (the IMAGE license, not the observation's).")
    p.add_argument("--per-species", type=int, default=1200, help="Max photos to fetch (one per observation).")
    p.add_argument("--all-photos-per-obs", action="store_true",
                   help="Keep every photo of an observation (default: 1, to avoid near-dup val leakage).")
    p.add_argument("--workers", type=int, default=6, help="Concurrent photo downloads (S3, be polite).")
    args = p.parse_args()

    if args.taxon_id:
        taxon_id = args.taxon_id
    elif args.taxon:
        taxon_id = resolve_taxon(args.taxon)
    elif args.klass:
        taxon_id = CLASS_TAXON.get(args.klass) or resolve_taxon(args.klass.replace("_", " "))
    else:
        print("Provide --class, --taxon-id, or --taxon.", file=sys.stderr)
        return 1
    if not taxon_id:
        print("Could not resolve a taxon id.", file=sys.stderr)
        return 1

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    print(f"taxon_id={taxon_id} place_id={args.place_id or '(all)'} license={args.license} "
          f"cap={args.per_species}")
    recs = fetch_photo_records(taxon_id, args.place_id, args.license, args.per_species,
                               one_per_obs=not args.all_photos_per_obs)
    print(f"{len(recs)} photo(s) across {len({r['obs_id'] for r in recs})} observation(s) -> {out}")
    if not recs:
        print("No photos found. Broaden --place-id (or set it empty for full range).", file=sys.stderr)
        return 1

    counts = Counter()
    kept: list[dict] = []  # records whose image is actually on disk (ok or already present)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_download_one, r, out): r for r in recs}
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            counts["ok" if res == "ok" else "skip" if res == "skip" else "err"] += 1
            if res in ("ok", "skip"):
                kept.append(futs[fut])
            if res.startswith("err") and counts["err"] <= 10:
                print(f"  {res}", file=sys.stderr)
            if i % 200 == 0 or i == len(recs):
                rate = i / max(1e-6, time.time() - t0)
                print(f"  {i}/{len(recs)}  ok={counts['ok']} skip={counts['skip']} err={counts['err']} ({rate:.0f}/s)")
    print(f"Done: ok={counts['ok']} skip={counts['skip']} err={counts['err']}")

    # Attribution manifest AFTER download, for images actually on disk (CC-BY /
    # CC-BY-NC require crediting the photographer). Union with any existing rows,
    # keyed by photo_id, so a resumed run doesn't drop prior images' credits.
    cols = ["photo_id", "obs_id", "license", "attribution", "url"]
    attr_path = out / "attribution.csv"
    rows: dict[str, dict] = {}
    if attr_path.is_file():
        with open(attr_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                rows[str(row.get("photo_id"))] = {c: row.get(c, "") for c in cols}
    for r in kept:
        rows[str(r["photo_id"])] = {c: r[c] for c in cols}
    with open(attr_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows.values())
    print(f"Next: python training/label_boxes.py --images {out} --assume-species {args.klass or '<class>'} "
          f"--out ~/wildlife/labeled/inat_{args.klass or '<class>'}")
    return 1 if counts["err"] and not counts["ok"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
