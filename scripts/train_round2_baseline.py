"""Manually started training entry point for the round-two three-class baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solid_waste_model.core.dinov2_proto import train  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the formal round-two three-class baseline only when invoked manually.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "manifests" / "round2_baseline_manifest.csv")
    parser.add_argument("--model-version", default="solid_waste_dinov2_proto_round2_baseline_k3")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "solid_waste_dinov2_proto_round2_baseline_k3")
    parser.add_argument("--labels", default="煤矸石,矿渣,钢渣")
    parser.add_argument("--local-repository", type=Path, default=PROJECT_ROOT / "runtime" / "torch_hub" / "facebookresearch_dinov2_main")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "runtime" / "torch_hub")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--prototypes-per-class", type=int, default=3)
    parser.add_argument("--kmeans-iterations", type=int, default=50)
    parser.add_argument("--crops", default="full,center,top_left,top_right,bottom_left,bottom_right")
    parser.add_argument("--crop-ratio", type=float, default=0.78)
    parser.add_argument("--class-temperature", type=float, default=0.07)
    parser.add_argument("--vote-temperature", type=float, default=0.03)
    parser.add_argument("--rejection-quantile", type=float, default=0.10)
    parser.add_argument("--include-augmented-train", action="store_true")
    parser.add_argument("--loader", choices=["auto", "torchhub", "huggingface"], default="auto")
    parser.add_argument("--dino-model")
    parser.add_argument("--repository", default="facebookresearch/dinov2")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
