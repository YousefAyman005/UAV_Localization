import argparse
import cv2
import numpy as np
import pandas as pd
import torch

from visloc_utils import (
    MIN_INL, RANSAC_THRESH, SAT_TIF, SAT_CSV, DRONE_CSV, DRONE_DIR,
    load_satellite, run_pipeline, save_dense_viz,
)

OUT_CSV = "visloc_jamma_results.csv"
VIZ_DIR = "visloc_jamma_visualizations"

JAMMA_URL    = "https://github.com/leoluxxx/JamMa/releases/download/v0.1/jamma.ckpt"
BACKBONE_URL = "https://github.com/leoluxxx/JamMa/releases/download/v0.1/convnextv2_nano_pretrain.ckpt"

# Config is tied to jamma.ckpt; lifted verbatim from JamMa/demo/utlis.py.
JAMMA_CFG = {
    "coarse": {"d_model": 256},
    "fine":   {"d_model": 64, "thr": 0.1},
    "match_coarse": {
        "thr": 0.2, "border_rm": 2, "dsmax_temperature": 0.1,
        "inference": True, "train_coarse_percent": 0.4,
        "train_pad_num_gt_min": 200,
    },
    "resolution": [8, 2],
    "fine_window_size": 5,
}

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def jamma_preprocess(bgr, resize, df, device):
    """BGR ndarray -> (ImageNet-normalised RGB tensor [1,3,H',W'], coarse mask [H'/8,W'/8],
    (sx, sy) = native/resized factors so kp can be mapped back to the drone/patch frame)."""
    h0, w0 = bgr.shape[:2]
    scale = resize / max(h0, w0)
    w_new, h_new = int(round(w0 * scale)), int(round(h0 * scale))
    rgb = cv2.cvtColor(cv2.resize(bgr, (w_new, h_new)), cv2.COLOR_BGR2RGB)
    ph = ((h_new + df - 1) // df) * df
    pw = ((w_new + df - 1) // df) * df
    padded = np.zeros((ph, pw, 3), dtype=np.uint8)
    padded[:h_new, :w_new] = rgb
    t = torch.from_numpy(padded).float().div(255.).permute(2, 0, 1).unsqueeze(0)
    t = ((t - _IMAGENET_MEAN) / _IMAGENET_STD).to(device)
    m = torch.zeros((ph // 8, pw // 8), dtype=torch.bool, device=device)
    m[:h_new // 8, :w_new // 8] = True
    return t, m, (w0 / w_new, h0 / h_new)


def match_jamma(t0, m0, s0, t1, m1, s1, backbone, model, conf_thresh, ransac_thresh):
    data = {"imagec_0": t0, "imagec_1": t1, "mask0": m0, "mask1": m1}
    with torch.inference_mode():
        backbone(data)  # mutates data in-place
        model(data)

    kp0  = data["mkpts0_f"].cpu().numpy().astype(np.float32)
    kp1  = data["mkpts1_f"].cpu().numpy().astype(np.float32)
    conf = data["mconf_f"].cpu().numpy()
    kp0[:, 0] *= s0[0]; kp0[:, 1] *= s0[1]
    kp1[:, 0] *= s1[0]; kp1[:, 1] *= s1[1]

    r = dict(sat_kp=len(kp1), drone_kp=len(kp0), raw=len(kp0), good=0, inliers=0,
             H=None, _kp0=kp0, _kp1=kp1, _conf=conf, _mask=None)
    mask = conf >= conf_thresh
    r["good"], r["_mask"] = int(mask.sum()), mask
    if r["good"] < 4:
        return r
    H, mh = cv2.findHomography(kp0[mask].reshape(-1, 1, 2),
                               kp1[mask].reshape(-1, 1, 2),
                               cv2.USAC_MAGSAC, ransac_thresh,
                               maxIters=5000, confidence=0.9999)
    if H is not None and mh is not None:
        r["inliers"], r["H"] = int(mh.sum()), H
    return r


def _load_jamma(device):
    # Deferred import: JamMa repo must be on sys.path and mamba-ssm installed.
    from src.jamma.jamma import JamMa as JamMaMatcher
    from src.jamma.backbone import CovNextV2_nano

    backbone = CovNextV2_nano()
    model    = JamMaMatcher(JAMMA_CFG, profiler=None)

    ckpt    = torch.hub.load_state_dict_from_url(JAMMA_URL,    map_location="cpu")
    bb_ckpt = torch.hub.load_state_dict_from_url(BACKBONE_URL, map_location="cpu")
    ckpt    = ckpt.get("state_dict", ckpt)
    bb_ckpt = bb_ckpt.get("model", bb_ckpt.get("state_dict", bb_ckpt))

    backbone.load_state_dict(bb_ckpt, strict=False)
    model.load_state_dict(ckpt, strict=False)
    return backbone.eval().to(device), model.eval().to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit",         type=int,   default=400)
    ap.add_argument("--dist",          type=float, default=25.0)
    ap.add_argument("--conf",          type=float, default=0.2)
    ap.add_argument("--resize",        type=int,   default=832)
    ap.add_argument("--df",            type=int,   default=16)
    ap.add_argument("--ransac-thresh", type=float, default=None)
    ap.add_argument("--min-inl",       type=int,   default=None)
    ap.add_argument("--clahe",         action="store_true")
    ap.add_argument("--visualize",     action="store_true")
    args = ap.parse_args()

    ransac_t = args.ransac_thresh if args.ransac_thresh is not None else RANSAC_THRESH
    min_inl  = args.min_inl       if args.min_inl       is not None else MIN_INL

    if not torch.cuda.is_available():
        raise RuntimeError(
            "JamMa requires CUDA (mamba-ssm has no CPU/MPS kernels). "
            "Run this pipeline on Kaggle/Colab with a GPU enabled."
        )

    device = torch.device("cuda")
    print(f"  Device: {device}")
    print(f"  Loading JamMa ({args.resize}px) ... ", end="", flush=True)
    backbone, model = _load_jamma(device)
    print("done")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if args.clahe else None
    sat, geo = load_satellite(SAT_TIF, SAT_CSV)
    df = pd.read_csv(DRONE_CSV).head(args.limit)
    print(f"  Method: JamMa | Resize: {args.resize} | DF: {args.df} | "
          f"CLAHE: {args.clahe} | Conf: {args.conf} | RANSAC: {ransac_t}px | "
          f"MinInl: {min_inl} | Dist: {args.dist}m | {len(df)} images\n")

    def match_factory(drone):
        t0, m0, s0 = jamma_preprocess(drone, args.resize, args.df, device)
        return lambda p: match_jamma(
            t0, m0, s0,
            *jamma_preprocess(p, args.resize, args.df, device),
            backbone, model, args.conf, ransac_t,
        )

    run_pipeline(sat, geo, df, match_factory, OUT_CSV, args.dist,
                 min_inl=min_inl, clahe=clahe, drone_dir=DRONE_DIR,
                 viz_fn=save_dense_viz if args.visualize else None,
                 viz_dir=VIZ_DIR if args.visualize else None)


if __name__ == "__main__":
    main()
