"""Offline tests: run the real scripts end-to-end on a synthetic dataset with a fake detector/embedder.

No model downloads or GPU needed. Validates metrics maths, splitting/leakage guards, caching,
calibration, fine-tuning loop, ablation and the predict CLI.

    python tests/test_end_to_end.py
"""
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pakface import augment, detect, embedders  # noqa: E402
from pakface.align import ARCFACE_TEMPLATE, align_face  # noqa: E402
from pakface.detect import Detection  # noqa: E402
from pakface.matching import Calibration, best_accuracy_threshold, best_f1_threshold  # noqa: E402
from pakface.metrics import compute_eer, confusion_metrics, full_report  # noqa: E402


# ---------------------------------------------------------------- fakes
class FakeDetector:
    def __init__(self, *a, **k):
        pass

    def detect(self, img):
        h, w = img.shape[:2]
        s = w / 112.0
        return Detection(np.array([0, 0, w, h], np.float32), ARCFACE_TEMPLATE * s, 0.9, "plain")

    def detect_all(self, img, min_rel_size=0.25):
        """Synthetic group photo = N square faces side by side (width = N x height)."""
        h, w = img.shape[:2]
        if w < 2 * h or w % h:
            return [self.detect(img)]
        s = h / 112.0
        return [Detection(np.array([i * h, 0, (i + 1) * h, h], np.float32),
                          ARCFACE_TEMPLATE * s + np.array([i * h, 0], np.float32), 0.9, "multi") for i in range(w // h)]


class FakeEmbedder(embedders.BaseEmbedder):
    """Embedding = 8x8 downsampled grayscale, centred. Identities are 8x8 patterns, so this separates them."""

    def __init__(self, name):
        self.name = name

    def embed_batch(self, crops):
        f = np.stack([cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), (8, 8), interpolation=cv2.INTER_AREA)
                      .astype(np.float32).ravel() for c in crops])
        f -= f.mean(1, keepdims=True)
        n = np.linalg.norm(f, axis=1, keepdims=True) + 1e-9
        return f / n, n[:, 0]


def fake_build_embedders(cfg):
    return [FakeEmbedder(m.name) for m in cfg.models]


def fake_load_adaface_hf(path, repo_id="", token=None):
    import torch.nn as nn

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(nn.Conv2d(3, 8, 5, stride=4, padding=2), nn.BatchNorm2d(8), nn.ReLU(),
                                      nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(128, 32))

        def forward(self, x):
            return self.body(x)

    import torch

    torch.manual_seed(0)
    return Tiny()


def load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), ROOT / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(mod, argv):
    old = sys.argv
    sys.argv = ["x"] + argv
    try:
        mod.main()
    finally:
        sys.argv = old


# ---------------------------------------------------------------- unit tests
def test_metrics():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(5000), np.zeros(5000)]
    s = np.r_[rng.normal(1, 1, 5000), rng.normal(-1, 1, 5000)]
    eer = compute_eer(s, y)
    # two unit-variance Gaussians 2 apart -> EER = Phi(-1) = 0.1587
    assert abs(eer - 0.1587) < 0.01, eer
    t, acc = best_accuracy_threshold(s, y)
    brute = max(((s > tt).astype(int) == y).mean() for tt in np.linspace(-3, 3, 2001))
    assert acc >= brute - 1e-3 and abs(t) < 0.15, (t, acc, brute)
    r = confusion_metrics(s, y, t)
    assert r["tp"] + r["fn"] == 5000 and abs(r["accuracy"] - acc) < 1e-9
    rep = full_report(s, y, t)
    assert 0.9 < rep["auc"] < 0.95 and rep["far"] == r["far"]
    tf, f1 = best_f1_threshold(s, y)
    brute_f1 = max(confusion_metrics(s, y, tt)["f1"] for tt in np.linspace(-3, 3, 2001))
    assert f1 >= brute_f1 - 1e-3 and abs(confusion_metrics(s, y, tf)["f1"] - f1) < 1e-9, (tf, f1, brute_f1)
    print("ok metrics: EER %.4f  best-acc %.4f @ %.3f" % (eer, acc, t))


def test_calibration_bins():
    rng = np.random.default_rng(1)
    q = rng.uniform(0, 1, 6000)
    y = rng.integers(0, 2, 6000)
    # low-quality pairs have compressed scores -> they need a lower threshold
    s = np.where(y == 1, 0.6, 0.0) * (0.3 + 0.7 * q) + rng.normal(0, 0.08, 6000)
    c = Calibration(use_bins=True).fit(s, y, q, n_bins=3)
    assert c.bin_thresholds[0] < c.bin_thresholds[-1], c.bin_thresholds
    acc_bins = (c.decide(s, q) == y).mean()
    acc_glob = ((s > c.global_threshold).astype(int) == y).mean()
    assert acc_bins >= acc_glob, (acc_bins, acc_glob)
    print(f"ok quality-binned thresholds {np.round(c.bin_thresholds, 3)}: acc {acc_glob:.4f} -> {acc_bins:.4f}")


def test_align_and_augment():
    rng = np.random.default_rng(2)
    img = rng.integers(0, 255, (112, 112, 3), dtype=np.uint8)
    out = align_face(img, ARCFACE_TEMPLATE)
    assert np.abs(out.astype(int) - img.astype(int)).mean() < 1.0  # identity transform
    big = rng.integers(0, 255, (480, 360, 3), dtype=np.uint8)
    for f in augment.PHOTOMETRIC + augment.OCCLUSIONS:
        o = f(img, rng) if f is not augment.grayscale else f(img)
        assert o.shape == img.shape and o.dtype == np.uint8, f.__name__
    for name, f in augment.STRESS_TESTS.items():
        o = f(big)
        assert o.shape == big.shape and o.dtype == np.uint8, name
    for _ in range(50):
        o = augment.train_augment(img, rng)
        assert o.shape == img.shape
    print("ok alignment + all augmentations / stress tests")


def make_dataset(root: Path, n_ids=40, per_id=8, sources=("own_collection", "public_a")):
    rng = np.random.default_rng(3)
    k = 0
    rows = []
    for si, src in enumerate(sources):
        for i in range(n_ids // len(sources)):
            pattern = rng.integers(0, 255, (8, 8)).astype(np.uint8)
            ident = f"P{k:04d}" if src == "own_collection" else f"Person Name {k}"
            gender = ["male", "female"][k % 2]
            rows.append({"identity": ident, "gender": gender, "region": ["sindh", "punjab", "kpk", "balochistan"][k % 4]})
            k += 1
            d = root / "raw" / src / ident
            d.mkdir(parents=True)
            for j in range(per_id):
                face = cv2.resize(pattern, (112, 112), interpolation=cv2.INTER_NEAREST)
                face = np.clip(face.astype(np.float32) + rng.normal(0, 25, face.shape), 0, 255).astype(np.uint8)
                if j % 4 == 3:
                    face = (face * 0.25).astype(np.uint8)  # dark images -> exercise enhancement
                cv2.imwrite(str(d / f"img_{j}.jpg"), cv2.cvtColor(face, cv2.COLOR_GRAY2BGR))
    import pandas as pd

    pd.DataFrame(rows).to_csv(root / "identities.csv", index=False)


def test_end_to_end():
    tmp = Path(tempfile.mkdtemp())
    cwd = os.getcwd()
    try:
        make_dataset(tmp)
        os.chdir(tmp)
        detect.FaceDetector = FakeDetector
        embedders.build_embedders = fake_build_embedders

        from pakface.config import default_config

        cfg = default_config()
        cfg.ctx_id = -1
        os.makedirs("configs")
        cfg.to_json("configs/pipeline.json")

        run(load_script("01_prepare_data.py"), ["--raw", "raw", "--out", ".", "--identities", "identities.csv",
                                                 "--images", "none.csv", "--ctx", "-1", "--dedupe-dist", "0"])
        run(load_script("02_split.py"), ["--manifest", "manifest.csv", "--out", ".", "--genuine-per-id", "10"])
        import pandas as pd

        sp = pd.read_csv("splits.csv")
        assert set(sp[sp["split"] != "train"]["source"]) == {"own_collection"}
        assert sp.groupby("identity")["split"].nunique().max() == 1
        own = sp[sp["source"] == "own_collection"].groupby("identity")["split"].first()
        assert (own == "train").any() and (own != "train").any(), "own data must feed both fine-tuning and eval"

        try:
            import torch  # noqa: F401
            have_torch = True
        except Exception:
            have_torch = False
            print("   (torch not installed: skipping fine-tuning step)")
        if have_torch:
            ft = load_script("03_finetune.py")
            ft.load_adaface_hf = fake_load_adaface_hf
            run(ft, ["--splits", "splits.csv", "--pairs-val", "pairs_val.csv", "--epochs", "2", "--batch", "16",
                     "--workers", "0", "--min-images", "2"])
            assert Path("checkpoints/finetune_history.json").exists()
            assert Path("checkpoints/adaface_ir101_finetuned.pt").exists(), "fine-tuned model must always be saved"

        run(load_script("04_evaluate.py"), ["--pairs-val", "pairs_val.csv", "--pairs-test", "pairs_test.csv",
                                             "--splits", "splits.csv", "--ctx", "-1"])
        m = json.loads(Path("results/full_pipeline/metrics.json").read_text())
        print(f"   synthetic test accuracy {m['accuracy']:.4f}, EER {m['eer']:.4f}")
        assert m["accuracy"] > 0.9
        for f in ["test_roc.png", "test_det.png", "test_hist.png", "per_group.csv", "summary.md"]:
            assert Path("results/full_pipeline", f).exists(), f
        assert "gender" in Path("results/full_pipeline/summary.md").read_text()
        idm = m["identification"]
        print(f"   synthetic rank-1 {idm['rank1']:.4f}, DIR {idm['dir']:.4f}, FPIR {idm['fpir']:.4f}")
        assert idm["rank1"] > 0.9 and idm["rank5"] >= idm["rank1"]

        run(load_script("05_ablation.py"), ["--pairs-val", "pairs_val.csv", "--pairs-test", "pairs_test.csv",
                                             "--ctx", "-1"])
        ab = pd.read_csv("results/ablation/ablation.csv")
        assert len(ab) >= 9 and Path("results/ablation/stress_tests.csv").exists()

        pt = pd.read_csv("pairs_test.csv")
        pd.DataFrame({"img1": pt["path_a"], "img2": pt["path_b"], "is_same": pt["label"]}).to_csv("bench.csv", index=False)
        pr = load_script("predict.py")
        run(pr, ["--ctx", "-1", "pairs", "--csv", "bench.csv", "--out", "submission.csv"])
        sub = pd.read_csv("submission.csv")
        assert {"score", "prediction"} <= set(sub.columns) and len(sub) == len(pt)
        # identification mode: gallery = first 4 imgs per own id, probes = rest
        gal, prb = Path("gallery"), Path("probes")
        for d in Path("raw/own_collection").iterdir():
            imgs = sorted(d.iterdir())
            for i, p in enumerate(imgs):
                dst = (gal if i < 4 else prb) / d.name / p.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(p, dst)
        run(pr, ["--ctx", "-1", "identify", "--gallery", "gallery", "--probes", "probes", "--out", "ids.csv"])
        ids = pd.read_csv("ids.csv")
        acc = (ids["true_identity"] == ids["predicted_identity"]).mean()
        assert acc > 0.9, acc
        run(pr, ["--ctx", "-1", "one", "--img1", str(pt["path_a"][0]), "--img2", str(pt["path_b"][0])])

        # group photos: 3 different people side by side
        grp = Path("groups")
        grp.mkdir()
        own_ids = sorted(d.name for d in prb.iterdir())
        for g in range(4):
            row = [cv2.imread(str(sorted((prb / m).iterdir())[0])) for m in own_ids[g * 3:(g + 1) * 3]]
            cv2.imwrite(str(grp / f"group_{g}.png"), np.hstack(row))
        target = own_ids[4]  # middle face of group_1
        run(pr, ["--ctx", "-1", "search", "--query", str(gal / target), "--photos", "groups", "--out", "matches.csv",
                 "--draw", "found"])
        mt = pd.read_csv("matches.csv")
        assert mt.iloc[0]["photo"].endswith("group_1.png") and mt.iloc[0]["face"] == 1, mt.head()
        assert mt[mt["match"] == 1]["photo"].nunique() == 1 and Path("found/group_1_match.jpg").exists()
        gp = Path("probes_groups") / target
        gp.mkdir(parents=True)
        shutil.copy(grp / "group_1.png", gp)
        run(pr, ["--ctx", "-1", "identify", "--gallery", "gallery", "--probes", "probes_groups", "--out", "ids_g.csv"])
        ig = pd.read_csv("ids_g.csv")
        assert len(ig) == 3 and list(ig["predicted_identity"]) == own_ids[3:6], ig
        print("ok group photos: search finds the person, identify labels every face")
        print("ok end-to-end: prepare -> split -> finetune -> evaluate -> ablation -> predict")
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_metrics()
    test_calibration_bins()
    test_align_and_augment()
    test_end_to_end()
    print("\nALL TESTS PASSED")
