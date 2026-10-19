"""Import the Google Form photo collection into data/raw/own_collection/<code>/ + metadata CSVs.

Inputs (both exported from Google, nothing to rearrange by hand):
  --csv    the responses spreadsheet as CSV (Form -> Responses -> spreadsheet -> File -> Download -> CSV)
  --files  the form's "File responses" folder: on Colab the mounted Drive folder, on Windows the folder
           extracted from Drive's zip download. Google keeps one sub-folder per upload question and renames
           every file "<original name> - <uploader's Google name>", which is how photos are matched to rows.

What it does:
  * keeps only rows whose consent box is ticked (18+ and agreed); everyone else is skipped entirely
  * gives every person a stable code P0001, P0002, ... (same email -> same code, also across re-runs, so
    newly arrived responses never reshuffle existing people between train/val/test)
  * converts every photo to an EXIF-rotated JPEG (also HEIC from iPhones, if pillow-heif is installed),
    which strips GPS/location metadata, and caps the long side at --max-side
  * writes the upload question of each photo as conditions for the per-group breakdown:
    photo_type, pose, glasses, headwear, mask, lighting, device
  * names/emails go ONLY to data/private/form_code_map.csv (keep it with one team member, delete after
    the competition); identities.csv / images.csv contain codes only

    python scripts/import_google_form.py --csv responses.csv --files "Face Collection (File responses)"
If some uploader names cannot be matched to a row, the script lists them; add them to a CSV with columns
uploader,email (or uploader,full_name) and pass it as --name-map.
"""
import argparse
import difflib
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

# (header prefix, photo_type, conditions) - matched on the start of each upload question's text
CATEGORIES = [
    ("front face", "front", {}),
    ("side pose", "side", {"pose": "profile"}),
    ("glasses or sunglasses", "glasses", {"glasses": "glasses"}),
    ("head covering", "head_covering", {"headwear": "head_covering"}),
    ("face mask", "mask", {"mask": "mask"}),
    ("difficult lighting", "difficult_light", {"lighting": "difficult"}),
    ("low quality photos", "low_quality", {"device": "low_quality"}),
    ("different look", "different_look", {}),
]
DEFAULTS = {"pose": "frontal", "glasses": "none", "headwear": "none", "mask": "none", "lighting": "normal",
            "device": "phone"}
FIELDS = {"timestamp": "timestamp", "email address": "email", "full name": "full_name", "age group": "age_group",
          "gender": "gender", "i am 18": "consent"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif", ".heic", ".heif"}


def norm(s) -> str:
    """Lowercase letters/digits only: survives ':' -> '_' on Windows zips, double spaces, punctuation."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def words(s) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def map_columns(columns) -> tuple[dict, dict]:
    """-> ({column: field}, {column: (photo_type, conditions)})"""
    fields, uploads = {}, {}
    for c in columns:
        w = words(c)
        for prefix, field in FIELDS.items():
            if w.startswith(words(prefix)):
                fields[c] = field
        for prefix, ptype, cond in CATEGORIES:
            if w.startswith(prefix):
                uploads[c] = (ptype, cond)
    missing = {"consent"} - set(fields.values())
    if missing or not uploads:
        raise SystemExit(f"could not find the consent column and upload questions in the CSV header: {list(columns)}")
    return fields, uploads


def person_key(row) -> str:
    return str(row.get("email") or "").strip().lower() or "name:" + norm(row.get("full_name", ""))


def index_files(files_dir: Path, upload_cols: list) -> list[dict]:
    """Every uploaded file with its question (from the sub-folder name) and uploader name (from the filename)."""
    keys = {c: norm(c) for c in upload_cols}
    out = []
    for p in files_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMG_EXT:
            continue
        folder = norm(re.sub(r"\(file responses\)", "", p.parent.name, flags=re.I))
        col = max(keys, key=lambda c: len(_common_prefix(keys[c], folder)))
        # 12 matching characters, or the whole name for short questions like "Face mask" (folders may be truncated)
        if len(_common_prefix(keys[col], folder)) < min(12, len(keys[col])):
            col = None
        stem, sep, uploader = p.stem.rpartition(" - ")
        out.append({"path": p, "column": col, "uploader": uploader.strip() if sep else ""})
    return out


def _common_prefix(a: str, b: str) -> str:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return a[:n]


def to_jpeg(src: Path, dst: Path, max_side: int) -> bool:
    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            if max(im.size) > max_side:
                im.thumbnail((max_side, max_side), Image.LANCZOS)
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, "JPEG", quality=95)  # no exif= -> location / device metadata is dropped
        return True
    except Exception as e:
        hint = " (pip install pillow-heif for iPhone HEIC photos)" if src.suffix.lower() in {".heic", ".heif"} else ""
        print(f"  ! could not read {src.name}: {type(e).__name__}{hint}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--files", required=True, help='the "File responses" folder (or any folder above it)')
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--meta-dir", default="data")
    ap.add_argument("--source", default="own_collection")
    ap.add_argument("--code-map", default="data/private/form_code_map.csv")
    ap.add_argument("--name-map", default="", help="CSV uploader,email (or uploader,full_name) for unmatched uploaders")
    ap.add_argument("--max-side", type=int, default=2048)
    ap.add_argument("--min-photos", type=int, default=2)
    args = ap.parse_args()

    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except ImportError:
        pass

    raw = pd.read_csv(args.csv, dtype=str).fillna("")
    fields, uploads = map_columns(raw.columns)
    df = raw.rename(columns=fields)
    if "timestamp" in df:
        df = df.assign(_t=pd.to_datetime(df["timestamp"], errors="coerce", format="mixed")).sort_values("_t", kind="stable")
    consented = df[df["consent"].str.strip() != ""]
    print(f"responses: {len(df)}, with consent: {len(consented)} (others skipped entirely)")

    # stable codes: reuse the existing map, append new people in submission order
    cmap_path = Path(args.code_map)
    cmap = pd.read_csv(cmap_path, dtype=str) if cmap_path.exists() else pd.DataFrame(columns=["code", "key", "full_name", "email"])
    code_of = dict(zip(cmap["key"], cmap["code"]))
    next_n = 1 + max([int(c[1:]) for c in cmap["code"]] or [0])
    people = {}
    for _, r in consented.iterrows():
        k = person_key(r)
        if k not in code_of:
            code_of[k] = f"P{next_n:04d}"
            next_n += 1
        people.setdefault(k, {"code": code_of[k], "full_name": r.get("full_name", ""), "email": r.get("email", ""),
                              "gender": r.get("gender", "").strip().lower(), "age_group": r.get("age_group", "").strip(),
                              "expected": Counter()})
        for col in uploads:
            people[k]["expected"][col] += str(r[col]).count("http")

    # uploader name (Google account) -> person
    by_name = defaultdict(set)
    for k, p in people.items():
        by_name[norm(p["full_name"])].add(k)
    manual = {}
    if args.name_map:
        nm = pd.read_csv(args.name_map, dtype=str).fillna("")
        for _, r in nm.iterrows():
            target = (r.get("email", "").strip().lower() if "email" in nm else "") or "name:" + norm(r.get("full_name", ""))
            manual[norm(r["uploader"])] = target

    def match(uploader: str):
        u = norm(uploader)
        if u in manual:
            return manual[u] if manual[u] in people else None
        if len(by_name.get(u, ())) == 1:
            return next(iter(by_name[u]))
        close = difflib.get_close_matches(u, [n for n, ks in by_name.items() if len(ks) == 1], n=1, cutoff=0.85)
        return next(iter(by_name[close[0]])) if close else None

    files = index_files(Path(args.files), list(uploads))
    print(f"uploaded files found: {len(files)}")
    # uploads by people who did not consent are expected to stay unmatched - they are never copied
    no_consent = {norm(n) for n in df[df["consent"].str.strip() == ""].get("full_name", [])}
    out_root = Path(args.raw) / args.source
    staged, counters = defaultdict(list), defaultdict(Counter)
    unmatched, no_question = Counter(), 0
    for f in sorted(files, key=lambda f: str(f["path"])):
        if f["column"] is None:
            no_question += 1
            continue
        k = match(f["uploader"])
        if k is None:
            if norm(f["uploader"]) not in no_consent:
                unmatched[f["uploader"] or "(no name in filename)"] += 1
            continue
        ptype, cond = uploads[f["column"]]
        counters[k][ptype] += 1
        staged[k].append((f, f"{people[k]['code']}_{ptype}_{counters[k][ptype]}.jpg", ptype, cond))

    id_rows, img_rows = [], []
    for k, p in people.items():
        if len(staged[k]) < args.min_photos:
            print(f"  ! {p['code']}: only {len(staged[k])} photo(s) found (min {args.min_photos}) - not imported")
            continue
        got = Counter()
        for f, name, ptype, cond in staged[k]:
            if to_jpeg(f["path"], out_root / p["code"] / name, args.max_side):
                img_rows.append({"filename": name, "photo_type": ptype, **DEFAULTS, **cond})
                got[f["column"]] += 1
        short = [f"{uploads[c][0]} {got[c]}/{n}" for c, n in p["expected"].items() if got[c] < n]
        if short:
            print(f"  ! {p['code']}: fewer photos than the form lists for " + ", ".join(short))
        id_rows.append({"identity": p["code"], "gender": p["gender"], "age_group": p["age_group"]})

    # withdrawals: people imported earlier but no longer consented / present lose their photos and metadata
    kept = {r["identity"] for r in id_rows}
    is_ours = re.compile(r"^P\d{4}$")
    for d in sorted(out_root.iterdir()) if out_root.exists() else []:
        if d.is_dir() and is_ours.match(d.name) and d.name not in kept:
            for f in d.iterdir():
                f.unlink()
            d.rmdir()
            print(f"  - removed {d.name}: no longer among the consented responses")
    meta = Path(args.meta_dir)
    meta.mkdir(parents=True, exist_ok=True)
    for path, rows, key in [(meta / "identities.csv", id_rows, "identity"), (meta / "images.csv", img_rows, "filename")]:
        old = pd.read_csv(path, dtype=str) if path.exists() else pd.DataFrame(columns=[key])
        old = old[~old[key].astype(str).str.match(r"^P\d{4}(_|$)")]  # our previous rows are fully replaced
        pd.concat([old, pd.DataFrame(rows)]).to_csv(path, index=False)
    cmap_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"code": p["code"], "key": k, "full_name": p["full_name"], "email": p["email"]}
                  for k, p in people.items()] +
                 [r for r in cmap.to_dict("records") if r["key"] not in people]).sort_values("code").to_csv(cmap_path, index=False)

    print(f"\nimported {len(img_rows)} photos of {len(kept)} people into {out_root}")
    print("photos per type:", dict(Counter(r["photo_type"] for r in img_rows)))
    if no_question:
        print(f"  ! {no_question} files were not inside a recognisable question folder - point --files at the "
              "\"File responses\" folder")
    if unmatched:
        print(f"  ! files from {len(unmatched)} uploader name(s) could not be matched to a consented response: "
              + ", ".join(f"{u} ({n})" for u, n in sorted(unmatched.items())))
        print("    -> add them to a CSV with columns uploader,email and re-run with --name-map")
    print(f"code <-> name map: {cmap_path}  (private: never share or commit; delete after the competition)")


if __name__ == "__main__":
    sys.exit(main())
