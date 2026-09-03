"""XoFTR (cross-modal LoFTR-family semi-dense matcher) + RANSAC.

XoFTR (OnderT/XoFTR) is a LoFTR-derived detector-free matcher designed for
cross-modal matching (thermal<->visible), which makes it a natural fit for the
drone<->satellite domain gap. It is a pure-python research repo cloned to
/opt/XoFTR and bind-mounted at runtime (kept off the global PYTHONPATH so its
generic top-level `src` package can't shadow anything — same isolation as the
EfficientLoFTR pipeline). No container rebuild is needed: its inference deps
(einops, kornia, yacs, loguru, joblib, opencv, numpy, torch) are already present.

Its DataIOWrapper.from_cv_imgs() grayscales, resizes (keeping aspect, divisible
by `df`), runs the matcher, and rescales matches back to the original crop
coordinates — so we feed it the BGR drone + satellite crops directly and get
`mkpts0` (drone) / `mkpts1` (sat) / `mconf` in the SZ_W x SZ_H frame, then a
homography exactly as in loftr/eloftr.

NOTE: DataIOWrapper hardcodes cuda:0, so run single-GPU (the SLURM script
requests --gpus=1; run_pipeline then runs in-process on cuda:0).
"""

import os
import sys

import torch

torch.manual_seed(0)

sys.path.insert(0, "/opt/XoFTR")
from src.xoftr import XoFTR  # noqa: E402
from src.config.default import get_cfg_defaults  # noqa: E402
from src.utils.data_io import DataIOWrapper, lower_config  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.utils import dense_match_result  # noqa: E402
from helpers.workers import run_pipeline  # noqa: E402


def match_xoftr(model, drone_bgr, sat_bgr):
    # from_cv_imgs grayscales, resizes and rescales back to the SZ_W x SZ_H frame.
    out = model.from_cv_imgs(drone_bgr, sat_bgr)
    return dense_match_result(out["mkpts0"], out["mkpts1"], out["mconf"])


def load_model(device, args):
    cfg = lower_config(get_cfg_defaults(inference=True))
    cfg["xoftr"]["match_coarse"]["thr"] = args.coarse_thr
    cfg["xoftr"]["fine"]["thr"]         = args.fine_thr
    cfg["xoftr"]["fine"]["denser"]      = False
    cfg["test"]["img0_resize"] = args.resize
    cfg["test"]["img1_resize"] = args.resize
    matcher = XoFTR(config=cfg["xoftr"])
    return DataIOWrapper(matcher, config=cfg["test"], ckpt=args.weights)


def make_match_factory(model, device, _args):
    def match_factory(drone):
        return lambda p: match_xoftr(model, drone, p)
    return match_factory


def add_args(p):
    p.add_argument("--weights", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "weights", "weights_xoftr_640.ckpt"),
        help="XoFTR checkpoint (weights_xoftr_640.ckpt | weights_xoftr_840.ckpt).")
    p.add_argument("--resize", type=int, default=640,
                   help="Square resize fed to XoFTR (match the checkpoint's train res: 640 or 840).")
    p.add_argument("--coarse-thr", type=float, default=0.3,
                   help="XoFTR coarse-match confidence threshold.")
    p.add_argument("--fine-thr", type=float, default=0.1,
                   help="XoFTR fine-match confidence threshold.")


def main():
    run_pipeline(
        name="xoftr",
        label=lambda a: f"XoFTR (resize {a.resize})",
        add_args=add_args,
        load_model=load_model,
        make_match_factory=make_match_factory,
    )


if __name__ == "__main__":
    main()
