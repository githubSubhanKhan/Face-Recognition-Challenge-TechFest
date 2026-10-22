"""Nemo's Deep Dive face-matching demo (Gradio web app).

    python app.py                 # http://localhost:7860 (webcam works on localhost)
    python app.py --share         # also a public https link: Colab / Kaggle, or a phone on another network
    python app.py --ctx -1        # force CPU (default: GPU 0 if one is available)

Loads the same frozen pipeline as scripts/predict.py (configs/pipeline.json + results/calibration.json).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from webapp.engine import FaceEngine  # noqa: E402
from webapp.ui import CSS, HEAD, build, theme  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pipeline.json")
    ap.add_argument("--calibration", default="results/calibration.json")
    ap.add_argument("--ctx", type=int, default=None, help="GPU id, -1 for CPU (default: GPU 0 if available)")
    ap.add_argument("--share", action="store_true", help="create a public https link")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to allow other devices on the same network")
    ap.add_argument("--port", type=int, default=None, help="default: first free port from 7860")
    args = ap.parse_args()

    if args.ctx is None:
        import torch

        args.ctx = 0 if torch.cuda.is_available() else -1
    print(f"loading face models on {'GPU' if args.ctx >= 0 else 'CPU'} (first launch can take a minute)...")
    engine = FaceEngine(args.config, args.calibration, args.ctx)
    if not engine.calibrated:
        print(f"WARNING: {args.calibration} not found -> using the uncalibrated default threshold of 0.30")
    build(engine).launch(share=args.share, server_name=args.host, server_port=args.port, theme=theme(), css=CSS,
                         head=HEAD, inbrowser=not args.share, footer_links=[])


if __name__ == "__main__":
    main()
