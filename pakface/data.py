"""Dataset handling: scanning, merging sources, de-duplication, identity-disjoint splits, pair generation.

Expected raw layout (one folder per person, any depth of source folders above it):

    data/raw/<source>/<identity>/*.jpg
    e.g. data/raw/own_collection/P0007/IMG_2231.jpg
         data/raw/halfface/P_012/p012_00m.jpg
         data/raw/imfdb/AamirKhan/....jpg

Optional metadata CSVs (used for the per-group fairness breakdown):
    identities.csv : identity, gender, age_group, skin_tone, region, ...
    images.csv     : path_or_filename, headwear, glasses, mask, lighting, device, ...
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif"}


def normalize_identity(name: str) -> str:
    """'Aamir_Khan', 'aamir khan', 'AamirKhan' -> 'aamirkhan' so the same person never ends up
    under two labels when sources overlap (IMFDB and the Kaggle Bollywood sets share actors)."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def scan(root: str, merge_identities_across_sources: bool = True) -> pd.DataFrame:
    rows = []
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in IMG_EXT and p.is_file():
            rel = p.relative_to(root).parts
            source = rel[0] if len(rel) >= 3 else "default"
            ident_raw = p.parent.name
            key = normalize_identity(ident_raw)
            # only merge real names across sources; codes like "p007", "s12", "0042" are dataset-local IDs
            is_code = re.fullmatch(r"[a-z]{0,3}_?\d+", key) is not None or len(key) < 4
            identity = key if (merge_identities_across_sources and not is_code) else f"{source}__{key}"
            rows.append({"path": str(p), "source": source, "identity_raw": ident_raw, "identity": identity})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"no images found under {root}")
    return df


# ---------------------------------------------------------------- de-duplication
def dhash(img_gray_small: np.ndarray) -> int:
    """64-bit difference hash of a 9x8 grayscale thumbnail."""
    diff = img_gray_small[:, 1:] > img_gray_small[:, :-1]
    return int(sum(1 << i for i, v in enumerate(diff.flatten()) if v))


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def find_duplicates(df: pd.DataFrame, hashes: list[int], max_dist: int = 4):
    """Returns (within-identity duplicates to drop, cross-identity near-duplicates = probable label noise)."""
    hs = np.array(hashes, dtype=np.uint64)
    ids = df["identity"].to_numpy()
    drop, conflicts = set(), []
    # bucket by top 16 bits to keep it ~linear; exact near-dups almost always share them
    buckets: dict[int, list[int]] = {}
    for i, h in enumerate(hs):
        buckets.setdefault(int(h) >> 48, []).append(i)
    for idx in buckets.values():
        for a_pos, a in enumerate(idx):
            if a in drop:
                continue
            for b in idx[a_pos + 1 :]:
                if b in drop:
                    continue
                if hamming(int(hs[a]), int(hs[b])) <= max_dist:
                    if ids[a] == ids[b]:
                        drop.add(b)
                    else:
                        conflicts.append((df.iloc[a]["path"], df.iloc[b]["path"]))
    return drop, conflicts


# ---------------------------------------------------------------- splitting
def identity_disjoint_split(df: pd.DataFrame, val_frac=0.15, test_frac=0.2, seed=42,
                            min_images_eval=2, test_sources: list | None = None,
                            eval_source_train_frac: float = 0.0) -> pd.DataFrame:
    """Assign every IDENTITY (not image) to train/val/test so no person appears in two splits.

    test_sources: if given, only identities from these sources go to val/test (e.g. ["own_collection"]
    so the benchmark you report on is Pakistani, consented and never seen by the pretrained model).
    eval_source_train_frac: with test_sources, this fraction of their identities stays in train (the
    problem statement requires fine-tuning on your own collected data); the rest is split val/test.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    counts = df.groupby("identity").size()
    eligible = counts[counts >= min_images_eval].index
    if test_sources:
        src_ok = df[df["source"].isin(test_sources)]["identity"].unique()
        eligible = [i for i in eligible if i in set(src_ok)]
    eligible = np.array(sorted(eligible))
    rng.shuffle(eligible)
    if test_sources and eval_source_train_frac > 0:
        eligible = eligible[int(round(len(eligible) * eval_source_train_frac)):]  # the rest stays in train
    n = len(eligible)
    if test_sources:  # every eligible identity comes from the eval sources -> split them val/test only
        n_test = int(round(n * test_frac / (val_frac + test_frac)))
        n_val = n - n_test
    else:
        n_test, n_val = int(round(n * test_frac)), int(round(n * val_frac))
    test_ids, val_ids = set(eligible[:n_test]), set(eligible[n_test : n_test + n_val])
    df["split"] = "train"
    df.loc[df["identity"].isin(val_ids), "split"] = "val"
    df.loc[df["identity"].isin(test_ids), "split"] = "test"
    assert_disjoint(df)
    return df


def assert_disjoint(df: pd.DataFrame) -> None:
    per_id = df.groupby("identity")["split"].nunique()
    bad = per_id[per_id > 1]
    if len(bad):
        raise AssertionError(f"{len(bad)} identities appear in more than one split: {list(bad.index[:5])}")


# ---------------------------------------------------------------- pairs
def make_pairs(df: pd.DataFrame, n_genuine_per_id: int = 20, impostor_ratio: float = 1.0, seed: int = 0,
               hard_negative_cols: tuple = ("gender",)) -> pd.DataFrame:
    """Balanced verification pairs. Half of the impostors are 'hard': same gender (and any other
    hard_negative_cols present), because the hidden benchmark will certainly contain look-alike pairs."""
    rng = np.random.default_rng(seed)
    groups = {k: v["path"].tolist() for k, v in df.groupby("identity")}
    ids = [k for k, v in groups.items() if len(v) >= 2]
    gen = []
    for i in ids:
        imgs = groups[i]
        all_pairs = [(a, b) for k, a in enumerate(imgs) for b in imgs[k + 1 :]]
        take = min(n_genuine_per_id, len(all_pairs))
        for j in rng.choice(len(all_pairs), take, replace=False):
            gen.append((*all_pairs[j], 1, i, i))
    n_imp = int(len(gen) * impostor_ratio)
    all_ids = list(groups)
    attrs = df.drop_duplicates("identity").set_index("identity")
    hard_cols = [c for c in hard_negative_cols if c in attrs.columns]
    imp, seen = [], set()
    tries = 0
    while len(imp) < n_imp and tries < n_imp * 20:
        tries += 1
        a_id = all_ids[int(rng.integers(len(all_ids)))]
        pool = []
        if hard_cols and rng.random() < 0.5:
            same = attrs.index[(attrs[hard_cols] == attrs.loc[a_id, hard_cols]).all(axis=1)]
            pool = [x for x in same if x != a_id]
        if pool:
            b_id = pool[int(rng.integers(len(pool)))]
        else:
            b_id = all_ids[int(rng.integers(len(all_ids)))]
            if b_id == a_id:
                continue
        a = groups[a_id][int(rng.integers(len(groups[a_id])))]
        b = groups[b_id][int(rng.integers(len(groups[b_id])))]
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        imp.append((a, b, 0, a_id, b_id))
    pairs = pd.DataFrame(gen + imp, columns=["path_a", "path_b", "label", "id_a", "id_b"])
    return pairs.sample(frac=1, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------- metadata for fairness breakdown
def attach_metadata(df: pd.DataFrame, identities_csv: str | None, images_csv: str | None) -> pd.DataFrame:
    if identities_csv and Path(identities_csv).exists():
        # matched on the folder name as written (normalised); add a `source` column to the CSV
        # if two sources reuse the same folder names (e.g. both have P001)
        meta = pd.read_csv(identities_csv)
        meta["_key"] = meta["identity"].astype(str).map(normalize_identity)
        meta = meta.drop(columns=["identity"])
        df["_key"] = df["identity_raw"].astype(str).map(normalize_identity)
        on = ["_key", "source"] if "source" in meta.columns else ["_key"]
        meta = meta.drop_duplicates(subset=on, keep="last")
        df = df.merge(meta, on=on, how="left").drop(columns=["_key"])
    if images_csv and Path(images_csv).exists():
        im = pd.read_csv(images_csv)
        key = im.columns[0]
        im["_fname"] = im[key].astype(str).map(lambda s: Path(s).name)
        df["_fname"] = df["path"].map(lambda s: Path(s).name)
        df = df.merge(im.drop(columns=[key]), on="_fname", how="left").drop(columns=["_fname"])
    return df
