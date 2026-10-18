"""Download every pretrained model the pipeline needs (run once per Kaggle/Colab session).

  * buffalo_l   : SCRFD-10G detector (+ w600k_r50 ArcFace, used as the naive baseline)
  * antelopev2  : glintr100.onnx (ArcFace R100 trained on Glint360K)
  * AdaFace IR-101 WebFace12M from Hugging Face (CVLFace release)

    python scripts/00_download_models.py
"""
import glob
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pakface.config import default_config  # noqa: E402


def fix_nested(pack: str) -> None:
    """antelopev2.zip unpacks to models/antelopev2/antelopev2/*.onnx on some versions."""
    root = os.path.expanduser(f"~/.insightface/models/{pack}")
    for f in glob.glob(os.path.join(root, "*", "*.onnx")):
        dst = os.path.join(root, os.path.basename(f))
        if not os.path.exists(dst):
            shutil.move(f, dst)


def main():
    from insightface.app import FaceAnalysis

    for pack in ["buffalo_l", "antelopev2"]:
        print(f"== insightface pack {pack}")
        try:
            FaceAnalysis(name=pack, providers=["CPUExecutionProvider"])
        except Exception as e:  # nested-folder assertion on antelopev2 is expected the first time
            print(f"   first load raised {type(e).__name__}: {e} -> fixing folder layout")
        fix_nested(pack)
        print("   files:", sorted(os.listdir(os.path.expanduser(f"~/.insightface/models/{pack}"))))

    cfg = default_config()
    from pakface.embedders import download_hf_repo

    for spec in cfg.models:
        if spec.kind == "adaface_hf":
            print(f"== Hugging Face {spec.repo_id} -> {spec.path}")
            download_hf_repo(spec.repo_id, spec.path, os.environ.get("HF_TOKEN"))
    print("done. Next: python scripts/00_sanity_check.py --data data/raw")


if __name__ == "__main__":
    main()
