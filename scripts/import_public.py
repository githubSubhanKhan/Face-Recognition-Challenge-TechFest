"""Import a downloaded public dataset into the unified layout  data/raw/<source>/<identity>/<image>.

  python scripts/import_public.py --format generic --src /kaggle/input/bollywood-celeb-faces --name bollywood
  python scripts/import_public.py --format imfdb   --src /kaggle/input/imfdb/IMFDB_final  --name imfdb
  python scripts/import_public.py --format halfface --src downloads/halfFace/Dataset-1 --name halfface

generic  : <src>/<identity>/**/<images>      (identity = first folder level under src)
imfdb    : <src>/<Actor>/<Movie>/images/*.jpg (identity = actor; same rule as generic)
halfface : <src>/P_XXX/pXXX_<masked><idx><m|f>.jpg -> also writes gender + mask metadata

Images are symlinked (fast, no disk use); pass --copy if the source will disappear.
Metadata rows are appended to data/identities.csv and data/images.csv.
"""
import argparse
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from pakface.data import IMG_EXT, normalize_identity  # noqa: E402

HALF_RE = re.compile(r"p(\d+)_([01])(\d)([mf])", re.I)


def place(src: Path, dst: Path, copy: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        try:
            os.symlink(src.resolve(), dst)
        except OSError:
            shutil.copy2(src, dst)


def append_csv(path: Path, rows: list, key: str):
    if not rows:
        return
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old, new]).drop_duplicates(subset=[key], keep="last")
    new.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=["generic", "imfdb", "halfface"], default="generic")
    ap.add_argument("--src", required=True)
    ap.add_argument("--name", required=True, help="source name, e.g. imfdb / halfface / bollywood")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--meta-dir", default="data")
    ap.add_argument("--max-per-id", type=int, default=60, help="cap images per identity (keeps sources balanced)")
    ap.add_argument("--copy", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.raw) / args.name
    id_rows, img_rows, n = [], [], 0
    for ident_dir in sorted(d for d in src.iterdir() if d.is_dir()):
        imgs = sorted(p for p in ident_dir.rglob("*") if p.suffix.lower() in IMG_EXT)[: args.max_per_id]
        if len(imgs) < 2:
            continue
        ident = f"{args.name}_{normalize_identity(ident_dir.name)}" if args.format == "halfface" else ident_dir.name
        for p in imgs:
            dst_name = f"{p.parent.name}_{p.name}" if p.parent != ident_dir else p.name
            place(p, out / ident / dst_name, args.copy)
            n += 1
            if args.format == "halfface":
                m = HALF_RE.search(p.stem)
                if m:
                    img_rows.append({"filename": dst_name, "mask": "mask" if m.group(2) == "1" else "none"})
                    id_rows.append({"identity": ident, "gender": "male" if m.group(4).lower() == "m" else "female"})
    append_csv(Path(args.meta_dir) / "identities.csv", id_rows, "identity")
    append_csv(Path(args.meta_dir) / "images.csv", img_rows, "filename")
    print(f"imported {n} images from {src} into {out} ({len(list(out.iterdir())) if out.exists() else 0} identities)")


if __name__ == "__main__":
    main()
