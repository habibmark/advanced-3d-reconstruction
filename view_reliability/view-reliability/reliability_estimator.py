"""AI23 View Reliability Estimator.

Scores each reference view (front / left / back ...) of a character before
3D reconstruction.

For every view it returns:
    q  quality       1 - strongest predicted damage (1 = clean)
    v  consistency   average match score with the other views (1 = consistent)
    u  uncertainty   how unsure the quality prediction is (0 = sure)
    r  reliability   q * v * (1 - u)
    decision         keep (r >= 0.6), enhance (0.3 <= r < 0.6), replace (r < 0.3)
    weight           r normalised over all views (sums to 1)

Usage from Python:
    from reliability_estimator import ReliabilityEstimator
    est = ReliabilityEstimator("reliability_heads.pt")
    results = est.score({"front": "front.png", "left": "left.png", "back": "back.png"})

Usage from the command line:
    python reliability_estimator.py front.png left.png back.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Union

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

DEFAULT_DAMAGES = ["blur", "noise", "exposure", "jpeg", "lowres", "occlusion"]
KEEP_THRESHOLD = 0.6
REPLACE_THRESHOLD = 0.3
PROBLEM_THRESHOLD = 0.15   # report a damage type as the main problem above this strength


# --- networks (must match the training notebook exactly) ---------------------
class QualityUncertainty(nn.Module):
    """Quality head + uncertainty head sharing one hidden layer."""

    def __init__(self, dim: int = 384, n: int = 6):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(0.1))
        self.quality_head = nn.Linear(256, n)        # damage strengths (0..1)
        self.uncertainty_head = nn.Linear(256, n)    # log-variance

    def forward(self, x):
        h = self.shared(x)
        return torch.sigmoid(self.quality_head(h)), self.uncertainty_head(h).clamp(-8, 3)


class Consistency(nn.Module):
    """Predicts whether two views show the same character at the same scale."""

    def __init__(self, dim: int = 384):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4 * dim, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 1))

    def forward(self, a, b):
        return self.net(torch.cat([a, b, (a - b).abs(), a * b], dim=1)).squeeze(1)


# --- helpers ------------------------------------------------------------------
ImageLike = Union[str, Path, Image.Image]

PREP = T.Compose([
    T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def to_rgb_on_white(img: ImageLike) -> Image.Image:
    """Open an image and put it on a white background (training images used white)."""
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        out = Image.new("RGB", rgba.size, "white")
        out.paste(rgba, mask=rgba.split()[3])
        return out
    return img.convert("RGB")


def decide(r: float) -> str:
    if r >= KEEP_THRESHOLD:
        return "keep"
    if r >= REPLACE_THRESHOLD:
        return "enhance"
    return "replace"


# --- main class ---------------------------------------------------------------
class ReliabilityEstimator:
    def __init__(self, heads_path: Union[str, Path] = "reliability_heads.pt",
                 device: str | None = None, backbone: nn.Module | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(heads_path, map_location="cpu")
        self.damages: List[str] = list(ckpt.get("damages", DEFAULT_DAMAGES))

        self.qu = QualityUncertainty(n=len(self.damages))
        self.qu.load_state_dict(ckpt["quality_uncertainty"])
        self.cons = Consistency()
        self.cons.load_state_dict(ckpt["consistency"])
        self.qu.to(self.device).eval()
        self.cons.to(self.device).eval()

        # frozen DINOv2 ViT-S/14 (downloaded from Meta on first use, then cached)
        self.backbone = backbone or torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.backbone.to(self.device).eval()

    @torch.no_grad()
    def features(self, images: List[ImageLike]) -> torch.Tensor:
        batch = torch.stack([PREP(to_rgb_on_white(im)) for im in images]).to(self.device)
        with torch.autocast(device_type="cuda", enabled=(self.device == "cuda")):
            return self.backbone(batch).float()

    @torch.no_grad()
    def score(self, views: Dict[str, ImageLike]) -> List[dict]:
        """views: {"front": path_or_image, "left": ..., "back": ...} (2 or more views)."""
        if len(views) < 2:
            raise ValueError("Give at least 2 views; consistency needs pairs.")
        names = list(views)
        f = self.features(list(views.values()))

        mu, logvar = self.qu(f)
        q = 1 - mu.max(1).values
        u = (2 * torch.exp(0.5 * logvar).mean(1)).clamp(0, 1)

        v = []
        for i in range(len(names)):
            others = [j for j in range(len(names)) if j != i]
            v.append(torch.sigmoid(self.cons(f[i].expand(len(others), -1), f[others])).mean())
        v = torch.stack(v)

        r = q * v * (1 - u)
        total = float(r.sum()) or 1.0
        results = []
        for i, name in enumerate(names):
            strengths = {d: round(float(mu[i, k]), 3) for k, d in enumerate(self.damages)}
            worst = max(strengths, key=strengths.get)
            results.append({
                "view": name,
                "q": round(float(q[i]), 3),
                "v": round(float(v[i]), 3),
                "u": round(float(u[i]), 3),
                "r": round(float(r[i]), 3),
                "decision": decide(float(r[i])),
                "weight": round(float(r[i]) / total, 3),
                "main_problem": worst if strengths[worst] > PROBLEM_THRESHOLD else None,
                "damage_strengths": strengths,
            })
        return results


def main():
    parser = argparse.ArgumentParser(description="Score reference views for reliability.")
    parser.add_argument("images", nargs="+", help="2 or more view images (e.g. front.png left.png back.png)")
    parser.add_argument("--heads", default="reliability_heads.pt", help="path to the trained heads file")
    parser.add_argument("--names", nargs="*", help="view names (default: file names)")
    args = parser.parse_args()

    names = args.names or [Path(p).stem for p in args.images]
    if len(names) != len(args.images):
        parser.error("--names must have one name per image")
    est = ReliabilityEstimator(args.heads)
    print(json.dumps(est.score(dict(zip(names, args.images))), indent=2))


if __name__ == "__main__":
    main()
