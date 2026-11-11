# AI23 View Reliability Estimator

Scores each reference view of a character (front, left, back, …) **before** it is
sent to the 3D reconstruction model (Hunyuan3D-2mv), and says whether to
**keep**, **enhance** or **replace** it.

## How it works

```
view image ──► DINOv2 ViT-S/14 (frozen, pretrained by Meta) ──► 384 features
                                                       │
                     ┌─────────────────────────────────┴──────────────┐
          Network 1 (one view)                          Network 2 (two views)
   shared layer → quality head  → q                    consistency head → v
               → uncertainty head → u
                                     r = q × v × (1 − u)
```

| Output | Meaning |
|---|---|
| `q` | quality: 1 − strongest predicted damage (1 = clean) |
| `v` | consistency: average match with the other views (same character, same scale) |
| `u` | uncertainty of the quality prediction (0 = sure) |
| `r` | reliability = q × v × (1 − u) |
| `decision` | `keep` (r ≥ 0.6), `enhance` (0.3–0.6), `replace` (< 0.3) — starting thresholds, to be tuned |
| `weight` | r normalised over all views |
| `main_problem` | strongest damage type if clearly present (`blur`, `noise`, `exposure`, `jpeg`, `lowres`, `occlusion`) |

## Install

```bash
pip install -r requirements.txt
```

Put the trained heads file `reliability_heads.pt` in this folder.
DINOv2 is downloaded automatically from Meta the first time (internet needed once).

## Use from Python

```python
from reliability_estimator import ReliabilityEstimator

est = ReliabilityEstimator("reliability_heads.pt")   # uses GPU if available
results = est.score({
    "front": "front.png",
    "left":  "left.png",
    "back":  "back.png",
})
for r in results:
    print(r["view"], r["r"], r["decision"], r["main_problem"])

keep = [r["view"] for r in results if r["decision"] == "keep"]
```

Inputs can be file paths or PIL images. Transparent backgrounds are placed on white,
as in training. At least 2 views are required.

## Use from the command line

```bash
python reliability_estimator.py front.png left.png back.png --names front left back
```

Prints the results as JSON.

## Integration notes

* **Where it sits:** after image preprocessing, before Hunyuan3D-2mv.
* **Suggested use:** pass only `keep` views to the reconstruction; send `enhance`
  views to a deblur / upscale step; ask the user to replace `replace` views.
* Hunyuan3D-2mv takes discrete views, so the weights are not used inside it yet;
  they are available for the refinement step.

## Training summary

* **Data:** ~500 cartoon/anime character models from Objaverse, selected with the
  Objaverse++ quality and style labels (CC-BY / CC0 only), rendered from front,
  left and back, then damaged with known strengths (the targets).
* **Split:** 80/10/10 by model (no character appears in both training and test).
* **Preliminary test results (unseen models):**

| Head | Metric | Result |
|---|---|---|
| Quality | Spearman per damage type | 0.32 – 0.66 (mean 0.48) |
| Quality | MAE | 0.08 (mean) |
| Uncertainty | Spearman with real error | 0.73 |
| Consistency | AUROC / accuracy | 0.94 / ≈ 86 % |

**Status:** prototype. Not yet validated on real artist reference sheets or on
reconstruction quality. Known weak points: exposure and low-resolution detection,
wrong-scale detection.

## Files

| File | What it is |
|---|---|
| `reliability_estimator.py` | the estimator (networks, loading, scoring, CLI) |
| `reliability_heads.pt` | trained weights of the heads (PyTorch) |
| `requirements.txt` | Python packages |
