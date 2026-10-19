"""Build PakFace_Colab.ipynb: the whole codebase + a runnable pipeline in ONE self-contained notebook.

Every source file is embedded as a `%%writefile` cell (so the notebook recreates the project and the code
stays readable for judges), followed by cells that run each stage on Colab (or Kaggle).
The repo files are the source of truth: re-run this after ANY code change, then commit both.

    python tools/build_notebook.py            # writes PakFace_Colab.ipynb at the repo root
"""
import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "PakFace_Colab.ipynb"

# dependency order first; any file not listed is appended so new modules are never silently left out
ORDER = [
    "requirements.txt", "configs/pipeline.json",
    "pakface/__init__.py", "pakface/config.py", "pakface/data.py", "pakface/detect.py", "pakface/align.py",
    "pakface/quality.py", "pakface/enhance.py", "pakface/embedders.py", "pakface/pipeline.py",
    "pakface/matching.py", "pakface/metrics.py", "pakface/augment.py", "pakface/heads.py",
    "pakface/antispoof.py", "pakface/evaluation.py",
    "scripts/00_download_models.py", "scripts/00_sanity_check.py", "scripts/import_public.py",
    "scripts/01_prepare_data.py", "scripts/02_split.py", "scripts/03_finetune.py", "scripts/04_evaluate.py",
    "scripts/05_ablation.py", "scripts/predict.py",
    "tests/test_end_to_end.py", "docs/consent_form.md", "docs/data_collection_protocol.md",
]
GLOBS = ["pakface/*.py", "scripts/*.py", "tests/*.py", "docs/*.md", "configs/*.json"]

_id = 0


def _cell(kind: str, text: str, **meta) -> dict:
    global _id
    _id += 1
    lines = text.strip("\n").split("\n")
    c = {"cell_type": kind, "id": f"c{_id:03d}", "metadata": meta,
         "source": [ln + "\n" for ln in lines[:-1]] + [lines[-1]]}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def md(text: str) -> dict:
    return _cell("markdown", text)


def code(text: str) -> dict:
    return _cell("code", text)


def describe(rel: str) -> str:
    """First paragraph of a module docstring, as the markdown caption of its cell."""
    if not rel.endswith(".py"):
        return ""
    doc = ast.get_docstring(ast.parse((ROOT / rel).read_text(encoding="utf-8"))) or ""
    return doc.split("\n\n")[0].strip()


def readme_section(title_prefix: str) -> str:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    m = re.search(rf"^## {re.escape(title_prefix)}.*?(?=^## )", text, flags=re.S | re.M)
    return m.group(0).strip() if m else ""


def source_files() -> list[str]:
    files = [f for f in ORDER if (ROOT / f).exists()]
    for g in GLOBS:
        for p in sorted(ROOT.glob(g)):
            rel = p.relative_to(ROOT).as_posix()
            if rel not in files:
                files.append(rel)
    return files


def build() -> dict:
    files = source_files()
    dirs = sorted({str(Path(f).parent.as_posix()) for f in files if "/" in f})
    cells = [
        md("""# PakFace — Face Recognition for the Pakistani Context
**Nemo's Deep Dive · AI Odyssey (TechFest '26) · Round 2**

End-to-end pipeline: **SCRFD detection (retry cascade) → 5-point alignment → quality assessment → adaptive
enhancement → AdaFace IR-101 + ArcFace R100 ensemble with flip TTA → cosine score fusion → F1-calibrated,
quality-binned thresholds**, plus fine-tuning on our own consented Pakistani data, full metrics, per-group
fairness, ablation, stress tests, 1:N identification and group-photo search.

**How to run (Google Colab):** `Runtime → Change runtime type → T4 GPU`, then run the cells top to bottom.
Part 1 recreates the code, Part 2 installs and checks it, Part 3 loads data, Part 4 trains and evaluates,
Part 5 runs the hidden-benchmark predictions. After a runtime restart, re-run the **Setup** cell first.

> This notebook is generated from the project repository by `tools/build_notebook.py`, so the code here is
> identical to the code in the repo."""),
        md(readme_section("1.")),
        md("## Setup"),
        code(f"""import os, sys
IN_COLAB = "google.colab" in sys.modules
IN_KAGGLE = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))
PROJECT = "/content/pakface" if IN_COLAB else ("/kaggle/working/pakface" if IN_KAGGLE else os.path.abspath("pakface_nb"))
for d in {dirs!r}:
    os.makedirs(os.path.join(PROJECT, d), exist_ok=True)
os.chdir(PROJECT)
print("environment:", "Colab" if IN_COLAB else "Kaggle" if IN_KAGGLE else "local", "| project dir:", PROJECT)
!nvidia-smi --query-gpu=name,memory.total --format=csv || echo "No GPU: Runtime -> Change runtime type -> T4 GPU\""""),
        md("# Part 1 — Source code\nEach cell below writes one file of the project. Run them all once per session."),
    ]
    section = None
    for rel in files:
        top = rel.split("/")[0] if "/" in rel else "root"
        if top != section:
            section = top
            cells.append(md({"root": "## Project files", "configs": "## Configuration",
                             "pakface": "## `pakface/` — the pipeline library",
                             "scripts": "## `scripts/` — runnable stages",
                             "tests": "## `tests/` — offline end-to-end test (synthetic data, fake models)",
                             "docs": "## `docs/` — consent form and data-collection protocol"}.get(top, f"## {top}")))
        cap = describe(rel)
        cells.append(md(f"### `{rel}`" + (f"\n{cap}" if cap else "")))
        cells.append(code(f"%%writefile {rel}\n" + (ROOT / rel).read_text(encoding="utf-8")))

    cells += [
        md("# Part 2 — Install, test, download models"),
        md("Installs dependencies (building `insightface` takes ~2-3 min). If Colab asks to restart the session, "
           "do it, then re-run **Setup** and continue from here."),
        code("!pip install -q -r requirements.txt"),
        md("Offline end-to-end test of all the plumbing (synthetic faces, fake models — no downloads needed)."),
        code("!python tests/test_end_to_end.py"),
        code("!python scripts/00_download_models.py"),

        md("""# Part 3 — Data
Each source below can be **any** of:
* a **Kaggle dataset** id `"owner/dataset-name"` (the part after `kaggle.com/datasets/` in its URL) —
  downloaded straight into this runtime, **no Google Drive needed**;
* a **folder**, e.g. a dataset attached on Kaggle with *Add Input* (`/kaggle/input/...`) or a Drive folder;
* a **.zip** file.

| Source | Inside | Used for |
|---|---|---|
| `OWN_SRC` | `P0001/*.jpg …` (one folder per person) + optional `identities.csv`, `images.csv` | train + **val/test** (only consented Pakistani data is evaluated on) |
| `IMFDB_SRC` | `<Actor>/*.jpg` (any nesting) | train |
| `HALFFACE_SRC` | halfFace `P_XXX/*.jpg` | train (masks, phones) |

**Kaggle downloads on Colab** need your Kaggle API key once: kaggle.com → Settings → API → *Create New Token*,
then in Colab click the 🔑 *Secrets* icon and add `KAGGLE_USERNAME` and `KAGGLE_KEY` (values from the downloaded
`kaggle.json`). On Kaggle itself nothing is needed.

**Own photos are personal data**: keep them in your private Google Drive (Google Form uploads already are) or a
**private** Kaggle dataset — never public. Empty sources are skipped."""),
        code("""IMFDB_SRC    = ""   # e.g. "owner/imfdb-dataset" (Kaggle), "/kaggle/input/imfdb", or ".../imfdb.zip"
HALFFACE_SRC = ""
OWN_SRC      = ""   # e.g. "/content/drive/MyDrive/<form> (File responses)/..." or "/kaggle/input/<private-dataset>"

if IN_COLAB and any(str(s).startswith("/content/drive") for s in [IMFDB_SRC, HALFFACE_SRC, OWN_SRC]):
    from google.colab import drive
    drive.mount("/content/drive")"""),
        code("""import glob, shutil, subprocess, zipfile
import pandas as pd

IMG = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif")

def unzip(zp, name):
    dst = f"/content/datasets/{name}" if IN_COLAB else os.path.join(PROJECT, "datasets", name)
    if not os.path.exists(dst):
        with zipfile.ZipFile(zp) as z:
            z.extractall(dst)
    return dst

def kaggle_download(slug):
    try:
        import kagglehub
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kagglehub"], check=True)
        import kagglehub
    if IN_COLAB and not os.environ.get("KAGGLE_KEY"):
        try:
            from google.colab import userdata
            os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_USERNAME"), userdata.get("KAGGLE_KEY")
        except Exception as e:
            print("no Kaggle secrets found (🔑 Secrets -> KAGGLE_USERNAME, KAGGLE_KEY):", type(e).__name__)
    return kagglehub.dataset_download(slug)

def fetch(src, name):
    \"\"\"Kaggle id / folder / zip -> local folder, or None if src is empty.\"\"\"
    if not src:
        return None
    if str(src).lower().endswith(".zip"):
        return unzip(src, name)
    if os.path.isdir(src):
        return src
    if src.count("/") == 1 and not src.startswith(("/", ".")):
        path = kaggle_download(src)
        zips = glob.glob(os.path.join(path, "*.zip"))
        return unzip(zips[0], name) if zips and not any(os.path.isdir(p) for p in glob.glob(os.path.join(path, "*"))) else path
    raise SystemExit(f"{name}: {src} is not a Kaggle id 'owner/dataset', a folder, or a .zip")

def identity_root(d):
    \"\"\"Deepest folder whose sub-folders hold the images (one sub-folder per person).\"\"\"
    def is_root(p):
        subs = [s for s in glob.glob(os.path.join(p, "*")) if os.path.isdir(s)]
        return len(subs) >= 2 and all(any(f.lower().endswith(IMG) for f in os.listdir(s)) for s in subs[:5])
    for p, _, _ in os.walk(d):
        if is_root(p):
            return p
    raise SystemExit(f"no <person>/<images> folders found under {d}")

def merge_csv(src, dst, key):
    new = pd.read_csv(src)
    if os.path.exists(dst):
        new = pd.concat([pd.read_csv(dst), new]).drop_duplicates(subset=[key], keep="last")
    new.to_csv(dst, index=False)

src = fetch(IMFDB_SRC, "imfdb")
if src:
    root = identity_root(src)
    !python scripts/import_public.py --format imfdb --src "{root}" --name imfdb --copy
else:
    print("skip IMFDB: IMFDB_SRC is empty")

src = fetch(HALFFACE_SRC, "halfface")
if src:
    root = identity_root(src)
    !python scripts/import_public.py --format halfface --src "{root}" --name halfface --copy
else:
    print("skip halfFace: HALFFACE_SRC is empty")

src = fetch(OWN_SRC, "own_collection")
if src:
    root = identity_root(src)
    shutil.copytree(root, "data/raw/own_collection", dirs_exist_ok=True)
    for name, key in [("identities.csv", "identity"), ("images.csv", None)]:
        hits = glob.glob(os.path.join(src, "**", name), recursive=True)
        if hits:
            merge_csv(hits[0], f"data/{name}", key or pd.read_csv(hits[0]).columns[0])
    print("own_collection:", len(os.listdir("data/raw/own_collection")), "people")
else:
    print("skip own collection: OWN_SRC is empty  <- needed for real val/test numbers "
          "(without it, 02_split falls back to a practice split of the public data)")

for s in sorted(glob.glob("data/raw/*")):
    print(f"{os.path.basename(s):16s} {len(os.listdir(s)):5d} identities")"""),

        md("# Part 4 — Train and evaluate\n### Sanity check\nGPU use, AdaFace channel order (RGB vs BGR, written to the config), quick EER per backbone."),
        code("!python scripts/00_sanity_check.py --data data/raw --ids 40"),
        md("### Prepare: de-duplicate, detect, align, quality-score every image"),
        code("!python scripts/01_prepare_data.py\nprint(open('data/report_prepare.txt').read())"),
        md("### Identity-disjoint split\n40% of own-collection people go to training (fine-tuning must use our "
           "own data); the rest are split val/test. Public datasets are training-only. No person is in two splits."),
        code("!python scripts/02_split.py --eval-sources own_collection"),
        md("### Baseline: pretrained models, full pipeline\nThresholds and fusion weights are fitted on validation "
           "people only (maximising F1, the ranking metric); every metric is reported on unseen test people."),
        code("""from IPython.display import Image, Markdown, display

def show(name):
    display(Markdown(open(f"results/{name}/summary.md").read()))
    for f in ["test_roc.png", "test_det.png", "test_hist.png"]:
        display(Image(f"results/{name}/{f}", width=480))

!python scripts/04_evaluate.py --name pretrained
show("pretrained")"""),
        md("### Ablation and stress tests\nHow much each pipeline stage contributes, and robustness to blur, "
           "low resolution, low light, JPEG, CCTV-style and grayscale degradations."),
        code("!python scripts/05_ablation.py\ndisplay(Markdown(open('results/ablation/ablation.md').read()))"),
        md("### Fine-tuning on our Pakistani training identities\nPartial unfreezing, frozen BatchNorm, "
           "distillation to the original model, heavy degradation + occlusion augmentation (masks, hijab/dupatta, "
           "caps, sunglasses). The best epoch is always saved; we ship it only if it beats the pretrained model."),
        code("""import json
!python scripts/03_finetune.py --epochs 6 --batch 64
hist = json.load(open("checkpoints/finetune_history.json"))
print({k: v for k, v in hist.items() if k != "epochs"})"""),
        code("""CKPT = "checkpoints/adaface_ir101_finetuned.pt"
!python scripts/05_ablation.py --finetuned {CKPT}
display(Markdown(open("results/ablation/ablation.md").read()))"""),
        md("If the fine-tuned model wins on the **test** split, switch the pipeline to it and re-evaluate. "
           "This also rewrites `results/calibration.json`, which the prediction step uses."),
        code("""USE_FINETUNED = hist["beat_pretrained"]   # or set True/False by hand after reading the ablation
cfg = json.load(open("configs/pipeline.json"))
for m in cfg["models"]:
    if m["name"] == "adaface_ir101":
        m["finetuned_ckpt"] = CKPT if USE_FINETUNED else ""
json.dump(cfg, open("configs/pipeline.json", "w"), indent=2)
NAME = "finetuned" if USE_FINETUNED else "final_pretrained"
!python scripts/04_evaluate.py --name {NAME}
show(NAME)"""),
        md("### Per-group fairness and hardest errors"),
        code("""display(pd.read_csv(f"results/{NAME}/per_group.csv"))
display(pd.read_csv(f"results/{NAME}/worst_errors.csv").head(10))"""),
        md("### Save the outputs\nColab deletes the runtime's files when the session ends. This bundles results, "
           "the fine-tuned checkpoint and the config into `pakface_outputs.zip`: downloaded on Colab (also copied "
           "to Drive if it is mounted), kept as notebook output on Kaggle."),
        code("""OUT = os.path.join(PROJECT, "outputs")
for d in ["results", "checkpoints", "configs"]:
    if os.path.exists(d):
        shutil.copytree(d, f"{OUT}/{d}", dirs_exist_ok=True)
for f in ["data/report_prepare.txt", "data/splits.csv"]:
    if os.path.exists(f):
        os.makedirs(f"{OUT}/data", exist_ok=True)
        shutil.copy(f, f"{OUT}/data/")
bundle = shutil.make_archive(os.path.join("/kaggle/working" if IN_KAGGLE else PROJECT, "pakface_outputs"), "zip", OUT)
print("bundle:", bundle)
if IN_COLAB:
    if os.path.isdir("/content/drive/MyDrive"):
        shutil.copy(bundle, "/content/drive/MyDrive/")
        print("copied to Drive: MyDrive/pakface_outputs.zip")
    from google.colab import files
    files.download(bundle)"""),

        md("""# Part 5 — Hidden benchmark and demos
Group photos are handled automatically: an image with 2+ faces is split into one entry per face.
Add `--multi-face off` (before the mode) only if every image is known to be single-person.

**Verification pairs** — the CSV needs two image columns (img1/img2, path_a/path_b, … or the first two columns)."""),
        code("""BENCH_CSV, BENCH_ROOT = "/content/bench/pairs.csv", "/content/bench/images"
!python scripts/predict.py pairs --csv {BENCH_CSV} --root {BENCH_ROOT} --out submission.csv
display(pd.read_csv("submission.csv").head())"""),
        md("**Identification** — `gallery/<person>/*.jpg` and a folder of probe images."),
        code("""!python scripts/predict.py identify --gallery /content/bench/gallery --probes /content/bench/probes --out identification.csv --open-set
display(pd.read_csv("identification.csv").head())"""),
        md("**Search** — find one person across group / classroom / CCTV photos; matches are drawn in green."),
        code("""!python scripts/predict.py search --query /content/demo/person --photos /content/demo/photos --out matches.csv --draw found
for f in sorted(glob.glob("found/*.jpg"))[:4]:
    display(Image(f, width=640))"""),
        md("**Try two of your own images** (Colab: an upload button appears)."),
        code("""if IN_COLAB:
    from google.colab import files
    up = list(files.upload())
    if len(up) >= 2:
        !python scripts/predict.py one --img1 "{up[0]}" --img2 "{up[1]}"
else:
    print("upload widget is Colab-only; use: !python scripts/predict.py one --img1 a.jpg --img2 b.jpg")"""),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    nb = build()
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    n_files = sum(1 for c in nb["cells"] if c["cell_type"] == "code" and c["source"][0].startswith("%%writefile"))
    print(f"wrote {OUT.name}: {len(nb['cells'])} cells, {n_files} source files embedded")
