# PakFace — Round 2 game plan (Nemo's Deep Dive, TechFest '26)

End-to-end face recognition pipeline for the Pakistani context:
**SCRFD detection (retry cascade) → 5-point alignment → quality assessment → adaptive enhancement →
AdaFace IR-101 + ArcFace-R100 (Glint360K) ensemble with flip TTA → cosine score fusion → quality-binned
calibrated thresholds**, plus careful fine-tuning, full metrics, per-group fairness breakdown, ablation
and stress tests.

Report due **22 Oct** · hidden benchmark **23 Oct**.

---

## 1. Strategy in one paragraph

On a hidden benchmark, the winning moves are (1) the strongest *pretrained* backbones for low-quality
faces, (2) a pipeline that never fails on ugly inputs, and (3) a threshold calibrated on data that looks
like the benchmark. Fine-tuning on a few thousand images is the *riskiest* step: it can easily make the
model worse on new people. Fine-tuning on our own collected data is a **required deliverable**, so we always
produce and report a fine-tuned model, but we only *ship* it if it beats the pretrained model on unseen
identities, and we always keep one backbone frozen in the ensemble as an anchor. Everything is
measured on **identity-disjoint** Pakistani test data from our own consented collection.

**Hidden benchmark format:** the rulebook's tagline ("pick a single face out of an entire school", "a sea of
faces, one to find") hints at 1:N identification, while the metric list (FAR/FRR/EER) is verification. We
cover both: `04_evaluate` reports verification metrics *and* rank-1/rank-5 + open-set identification, and
`predict.py` has `pairs`, `identify`, `search` and `one` modes. **Group photos** are handled automatically
(`--multi-face auto`): an image with 2+ faces is split into one entry per face, while single-person images
keep the normal single-face pipeline. `search` finds one person across group / classroom / CCTV photos and
can draw boxes on the matches (`--draw`), which also makes a good live demo.

**Ranking metric is F1** (organisers), but they also look at the other metrics. So thresholds are fitted to
**maximise F1 on validation pairs** (`--method max_f1`, the default), and every other metric is still reported.

| Stage | Choice | Why it helps (say this in the presentation) |
|---|---|---|
| Detection | SCRFD-10G (InsightFace `buffalo_l`), threshold 0.35 | RetinaFace-family detector with 5 landmarks, strong on small/occluded faces |
| Retry cascade | pad → 2× upscale → brighten → whole-image fallback | tight crops, tiny CCTV faces and dark images otherwise produce *no face* = guaranteed error |
| Alignment | similarity transform to ArcFace 112×112 template | backbones were trained on exactly this geometry |
| Quality | sharpness, brightness, contrast, face size, det score, yaw → score in [0,1] | routes enhancement, weights templates, picks threshold bin; never rejects a benchmark pair |
| Enhancement | adaptive gamma + CLAHE (+ mild denoise), **fused** with original embedding | recovers dark/backlit faces without throwing away the original signal; no GAN restoration (it hallucinates identity) |
| Backbone 1 | **AdaFace IR-101, WebFace12M** | quality-adaptive margin — built for blurry, low-res faces (best on IJB-S/TinyFace) |
| Backbone 2 | **ArcFace R100, Glint360K** (`antelopev2`) | different training data + loss → complementary errors |
| TTA | horizontal flip, norm-weighted | cheap, consistent +gain |
| Metric | cosine similarity, weighted score fusion | the angular-margin training objective optimises cosine; weights tuned on val |
| Threshold | max-accuracy threshold fitted on val, **per quality bin** (shrunk to global) | low-quality pairs have compressed genuine scores and need a lower threshold |
| Fine-tuning | AdaFace head, last 25 % unfrozen, frozen BN, distillation to original, heavy degradation + occlusion aug | adapts to Pakistani faces/conditions without forgetting |
| Anti-spoof (optional) | MiniFASNet ONNX wrapper | deployment story only — never applied to benchmark scoring |

**On "100 % accuracy":** aim for it, but don't promise it. Benchmarks with CCTV-quality, masked and
profile faces have a few genuinely ambiguous pairs; strong systems typically land in the high 90s on
mixed-quality Pakistani data. What separates teams is the last 1–3 %, which is exactly what the retry
cascade, ensemble and calibrated thresholds target.

## 2. Data sources

There is **no verified, public, Pakistani-specific face-recognition benchmark** (worth saying in the
report — it is *why* the challenge asks you to collect data). Use:

| Source | What it gives | Use for | Access |
|---|---|---|---|
| **Own consented collection** (`docs/`) | real Pakistani faces, hijab/dupatta/turban/caps, phones, low light | **val + test** (+ train) | collect now — protocol + consent form in `docs/` |
| **halfFace** (Mendeley Data, DOI 10.17632/pk44mkx9vm) | 166 South Asian participants, smartphone photos, masked + unmasked | train (masks, phones) | free download; check licence on the page |
| **IMFDB** (IIIT Hyderabad) | ~34.5k images of 100 South Asian actors; pose, lighting, age, occlusion, low-res | train | free for research; also mirrored on Kaggle/DagsHub |
| Kaggle "Bollywood celeb localized faces" | 100 celebs, 64×64 faces in wild conditions | train (low-res) | Kaggle; overlaps IMFDB actors → names are merged automatically |
| IMFW — Indian Masked Faces in the Wild | masks, gamcha, stoles, scarves, low res | train/eval | request from IAB lab, IIT Jodhpur (may not arrive in time) |
| RFW — Indian subset | ~3k identities, 6k verification pairs | extra eval reference | request from whdeng.cn (likely too slow for 22 Oct) |

Rules baked into the code:
- **Val/test identities come only from `own_collection`** (`02_split.py --eval-sources`). Celebrity sets may
  already be inside WebFace/Glint360K, so testing on them would inflate numbers.
- **40 % of `own_collection` people go to TRAIN** (`--own-train-frac 0.4`), because the problem statement
  requires fine-tuning on your own collected data; the other 60 % are split val/test. With ~50 people that is
  ~20 train / ~12 val / ~18 test. Never the same person in two splits.
- Identity-disjoint split is asserted in code; celebrity names are normalised and merged across sources so
  the same actor can't leak between train and test.
- Near-duplicate images under *different* identities are written to `data/label_conflicts.csv` — fix them.

## 3. Four-day schedule

| Day | Do |
|---|---|
| **Sun 18 / Mon 19** | Kaggle notebook up (§4). Download models, import halfFace + IMFDB. **Start consent collection immediately** — it is the bottleneck. Run `00_sanity_check.py` on the first 20 people. Email organisers the questions in §6. |
| **Tue 20** | Finish collection (≥ 50 people). `01_prepare` → `02_split` → `04_evaluate` (pretrained baseline) → `05_ablation`. Inspect `worst_errors.csv`. |
| **Wed 21** | `03_finetune` (always saved; ship only if it beats pretrained) → re-run 04/05 with `--finetuned`. Freeze config + `calibration.json`. Dry-run `predict.py` on a folder you have never touched. Write report. |
| **Thu 22** | Submit report. Build slides from `results/` (plots, ablation, per-group table, worst errors). |
| **Fri 23** | Hidden benchmark: `predict.py`. |

## 4. Run it

**Easiest: `PakFace_Colab.ipynb`**, one self-contained notebook (also the file to submit if a notebook is
required). It embeds every source file as a `%%writefile` cell, then runs install → tests → models → data
(zips from Google Drive) → prepare → split → evaluate → ablation → fine-tune → predict, and saves outputs
to Drive. Open it in Colab with a T4 GPU and run top to bottom. It also works on Kaggle.

The repo files are the source of truth. **After any code change, regenerate the notebook and commit both:**
```bash
python tools/build_notebook.py
```

### Or step by step (Kaggle, GPU T4/P100, Internet ON)

```bash
!git clone <your-repo> pakface || true   # or upload the zip as a Kaggle dataset
%cd pakface
!pip install -q -r requirements.txt
!python scripts/00_download_models.py

# data -> data/raw/<source>/<identity>/*.jpg
!python scripts/import_public.py --format halfface --src /kaggle/input/<halfface>/Dataset-1 --name halfface
!python scripts/import_public.py --format imfdb    --src /kaggle/input/<imfdb>/IMFDB_final --name imfdb
# own photos straight from the Google Form export (responses CSV + the form's "File responses" folder):
!python scripts/import_google_form.py --csv responses.csv --files "<Form name> (File responses)"
#   -> consented rows only, codes P0001.., JPEGs without location metadata, per-photo conditions from the
#      question it was uploaded to; names/emails only in data/private/form_code_map.csv (never share it).
#      Re-run whenever new responses arrive: codes stay stable, withdrawn people are removed.

!python scripts/00_sanity_check.py --data data/raw/own_collection      # GPU + AdaFace RGB/BGR check
!python scripts/01_prepare_data.py                                    # dedupe, detect, align, quality
!python scripts/02_split.py --eval-sources own_collection             # identity-disjoint split + pairs
!python scripts/04_evaluate.py --name pretrained                      # baseline with full metrics
!python scripts/05_ablation.py                                        # ablation + stress tests
!python scripts/03_finetune.py --epochs 6 --batch 64                  # always saves best epoch; prints if it beat pretrained
# if it saved a checkpoint: set models[adaface_ir101].finetuned_ckpt in configs/pipeline.json, then
!python scripts/04_evaluate.py --name finetuned
!python scripts/05_ablation.py --finetuned checkpoints/adaface_ir101_finetuned.pt

# hidden benchmark day
!python scripts/predict.py pairs --csv /kaggle/input/bench/pairs.csv --root /kaggle/input/bench/images --out submission.csv
!python scripts/predict.py identify --gallery gallery_dir --probes probe_dir --out ids.csv   # if it's identification
!python scripts/predict.py search --query person_dir --photos photos_dir --out matches.csv --draw found/   # find one person
# group photos are split into faces automatically; add --multi-face off (before the mode) only if every image is single-person
```
Offline test of all the plumbing (no models/GPU): `python tests/test_end_to_end.py`.

### Face Embedding Challenge (teacher's student pack / lab day)
Train images (one per person, named by person) are the enrolled gallery; each test image must be closest (cosine)
to the same person's train image. Score = top-1 accuracy, top-5 breaks ties.
```bash
# practice: metrics + submission file
python scripts/eval_student_pack.py --data <pack>/practice --key <pack>/practice_key.csv --group group05
# lab day (no key): writes group05.npz -> check it -> upload it
python scripts/eval_student_pack.py --data <pack>/lab_pack --group group05
python scripts/check_submission.py group05.npz --data <pack>/lab_pack
```
`eval_student_pack.py` is the submission route: in group photos it keeps the face that matches an enrolled
person, which the per-image `embed()` in the teacher's `make_submission.py` cannot do (it has been filled in with
the same pipeline, primary face only, so the official command also works).

### Demo app (Nemo's Deep Dive theme)
```bash
python app.py            # http://localhost:7860: GPU if available, else CPU
python app.py --share    # public https link (Colab / Kaggle, or another device); last cell of the notebook
```
* **Face Match**: capture a live picture or upload a photo for Face 1 and Face 2, press **🐠 Every Fish Has a
  Face?** → same person / different people, with the aligned crops, similarity vs threshold, each backbone's
  score, quality flags and boxes on group photos.
* **Sea of Faces**: one person's face + a group photo → that person is boxed in the crowd.
* **Dive Log**: the session's results, in memory only.

Same pipeline and threshold as `predict.py` (`results/calibration.json`, else 0.30). Images are processed in
memory, never written to the feature caches.

## 5. What the code produces for the report

| File | Report section |
|---|---|
| `results/<name>/summary.md`, `metrics.json` | accuracy, precision, recall, F1, FAR, FRR, EER, AUC, TAR@FAR, per backbone |
| `results/<name>/test_roc.png`, `test_det.png`, `test_hist.png` | ROC, DET, genuine/impostor distributions + threshold |
| `results/<name>/per_group.csv` | fairness: gender, age, skin tone, region, headwear, glasses, mask, lighting, quality flags |
| `results/ablation/ablation.md` | contribution of each pipeline stage |
| `results/ablation/stress_tests.csv` | blur / low-res / low-light / JPEG / CCTV / grayscale robustness, naive vs full |
| `metrics.json` → `identification`, `summary.md` | 1:N identification: rank-1, rank-5, open-set DIR / FPIR |
| `results/<name>/worst_errors.csv` | "challenges faced" slide — show 3–4 real failure pairs (with consent) |
| `checkpoints/finetune_history.json` | fine-tuning curve vs pretrained |
| `data/report_prepare.txt` | dataset statistics, detection-stage breakdown |

Suggested report outline: 1 Problem & context · 2 Data (sources, consent, demographics table, splits,
leakage guards) · 3 Pipeline (diagram + the table in §1) · 4 Fine-tuning (forgetting guards, curve,
honest outcome) · 5 Results (main table, ROC/DET, threshold choice) · 6 Ablation & stress tests · 7 Fairness
across groups · 8 Failure analysis & limitations · 9 Ethics.

## 6. Ask the organisers today

1. Benchmark format: **verification pairs** (same/different) or **identification** (gallery + probes)? Output file format?
2. Do we run our code on their data, or do they run our model? (If they run it: package `predict.py` + models.)
3. ~~Ranking metric?~~ **Answered: F1** (but not the only thing they look at) → thresholds maximise F1.
   Still worth asking: roughly what share of benchmark pairs are same-person? F1 at a fixed threshold depends on it.
4. Are benchmark images full photos or pre-cropped faces? (Both are handled, but it affects calibration.)
5. Can a benchmark image contain **several people**? Not answered yet; both cases are handled
   (`--multi-face auto`, `search` mode).
6. ~~Is a model or dataset provided?~~ **Answered: no, use your own dataset.**

## Layout
```
pakface/        config, detect, align, quality, enhance, embedders, pipeline, matching, metrics,
                augment, heads, antispoof, data, evaluation
scripts/        00_download_models, 00_sanity_check, import_public, 01_prepare_data, 02_split,
                03_finetune, 04_evaluate, 05_ablation, predict
configs/        pipeline.json (all toggles; written by 00_sanity_check)
docs/           consent_form.md, data_collection_protocol.md
tests/          test_end_to_end.py (synthetic data, fake models)
```
