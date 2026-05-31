import argparse
import json
import os
import re
import sys
import time

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.utils import (
    FLIGHTS_AVAILABLE, SZ_W, SZ_H, crop_gt_patch, get_flight_paths,
    load_flight, split_flight_rows,
)

CAPTION_DIR    = "cache/captions"
JPEG_W         = 768          # downscale crops before captioning to cut compute
JPEG_QUALITY   = 85
OLLAMA_MODEL = "qwen3.5:9b"

# Static instruction shared by every backend.
SYSTEM_PROMPT = (
    "List the permanent physical features of this ground patch in 10-15 words. "
    "Include: road shape (T-junction, curve, straight), building density, "
    "water bodies, land cover (farmland, forest, bare ground). "
    "Add cardinal direction (north, south, east, west) only for key features. "
    "Comma-separated noun phrases only. No verbs, no articles, no preamble. "
    "Never use: north-up, view, shows, patch, image, photo, satellite, aerial, drone, "
    "or any color/brightness word (bright, dark, light, pale, grey, green, brown, etc.)."
)
USER_TEXT = "Describe this ground patch."


# ---------- backends: each returns a `caption(pil_rgb) -> str` callable -------

def _downscale(pil):
    w, h = pil.size
    if w > JPEG_W:
        pil = pil.resize((JPEG_W, int(round(h * JPEG_W / w))))
    return pil


def _strip_think(text):
    """Drop any <think>...</think> reasoning a thinking model may emit."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# Color/brightness words are appearance cues that do NOT survive the
# drone<->satellite gap, so we strip any the VLM leaks despite the prompt.
_COLOR_RE = re.compile(
    r",?\s*\b(?:dark|light|bright|pale|deep|white|black|grey|gray|green|brown|"
    r"red|blue|yellow|orange|golden|tan|beige|silver|reddish|greenish|brownish)"
    r"(?:er|est)?\b\s*,?",
    re.IGNORECASE)


def _clean_caption(text):
    text = _strip_think(text)
    text = _COLOR_RE.sub(" ", text)
    text = re.sub(r"\s+,", ",", text)            # " ," -> ","
    text = re.sub(r",\s*,", ",", text)           # ", ," -> ","
    text = re.sub(r"\s{2,}", " ", text)          # collapse spaces
    text = re.sub(r"\s+\.", ".", text)           # " ." -> "."
    return text.strip().strip(",").strip()


def make_ollama(model_id, max_tokens, _device, host=None, retries=4):
    """Caption via a local Ollama server (default http://localhost:11434)."""
    try:
        import ollama
    except ImportError:
        sys.exit("ollama package not installed. Run: pip install ollama")
    host = host or os.environ.get("OLLAMA_HOST")
    client = ollama.Client(host=host) if host else ollama.Client()
    try:
        resp = client.list()
        # ollama >= 0.4 returns a ListResponse object; older versions return a dict.
        models_list = resp.models if hasattr(resp, "models") else resp.get("models", [])
        have = set()
        for m in models_list:
            have.add(m.model if hasattr(m, "model") else m.get("model", m.get("name", "")))
        if not any(model_id == h or h.startswith(model_id + ":") for h in have):
            print(f"  WARN: '{model_id}' not pulled. Run: ollama pull {model_id}",
                  file=sys.stderr)
    except Exception:                                            # noqa: BLE001
        print("  WARN: could not reach Ollama; is `ollama serve` running?",
              file=sys.stderr)

    def caption(pil):
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(
            np.asarray(_downscale(pil)), cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        img_bytes = buf.tobytes()
        kwargs = dict(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEXT, "images": [img_bytes]},
            ],
            options={"temperature": 0.0, "num_predict": max_tokens})
        for attempt in range(retries):
            try:
                try:  # think=False keeps reasoning out of the caption (qwen3.5)
                    resp = client.chat(**kwargs, think=False)
                except TypeError:                  # older ollama client: no `think`
                    resp = client.chat(**kwargs)
                # ollama >= 0.4 returns a ChatResponse object; older versions return a dict.
                content = resp.message.content if hasattr(resp, "message") else resp["message"]["content"]
                return _strip_think(content)
            except Exception as e:                               # noqa: BLE001
                if attempt == retries - 1:
                    raise
                wait = 2 ** attempt
                print(f"\n  Ollama error ({e}); retry in {wait}s", file=sys.stderr)
                time.sleep(wait)
    return caption


def build_captioner():
    print(f"  Backend ollama | model {OLLAMA_MODEL}")
    return make_ollama(OLLAMA_MODEL, 40, None)


# ---------- crop sources ----------------------------------------------------

def _bgr_to_pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


TILE_SIZE = 1024
TILE_STRIDE = 512
TEST_FRAC = 0.25


def iter_sat_patches(flight, limit):
    import pandas as pd
    tiles, _, drone_csv, _ = load_flight(flight)
    df = split_flight_rows(pd.read_csv(drone_csv), which="train",
                           test_frac=TEST_FRAC, axis="auto", buffer_frac=0.0)
    if limit is not None:
        df = df.iloc[:limit]
    for _, row in df.iterrows():
        # yaw_deg=0 keeps the crop north-up, matching gallery tile orientation
        # so cardinal directions in captions are consistent across sat and tile targets.
        patch = crop_gt_patch(tiles, float(row["lat"]), float(row["lon"]),
                              float(row["height"]), yaw_deg=0.0, flight=flight)
        yield row["filename"], (None if patch is None else _bgr_to_pil(patch))


def iter_drone_images(flight, limit):
    import pandas as pd
    _, drone_dir, drone_csv, _ = get_flight_paths(flight)
    df = split_flight_rows(pd.read_csv(drone_csv), which="test",
                           test_frac=TEST_FRAC, axis="auto", buffer_frac=0.0)
    if limit is not None:
        df = df.iloc[:limit]
    for _, row in df.iterrows():
        img = cv2.imread(os.path.join(drone_dir, row["filename"]))
        if img is not None:
            img = cv2.resize(img, (SZ_W, SZ_H))
            yaw = float(row["Phi1"]) if "Phi1" in row.index else 0.0
            if yaw != 0.0:
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w / 2, h / 2), yaw, 1.0)
                img = cv2.warpAffine(img, M, (w, h))
        yield row["filename"], (None if img is None else _bgr_to_pil(img))


def iter_tile_patches(flight, limit):
    """Yield (tile_id, patch) over the SAME grid the retrieval gallery uses.
    tile_id is the global index matching clip_pipeline.build_flight_gallery."""
    from pipelines.clip_pipeline import iter_tiles
    tiles, _, _, _ = load_flight(flight)
    gid = 0
    for sat, geo in tiles:
        for _tid, x0, y0, _lat, _lon in iter_tiles(geo, TILE_SIZE, TILE_STRIDE):
            if limit is not None and gid >= limit:
                return
            patch = sat[y0:y0 + TILE_SIZE, x0:x0 + TILE_SIZE]
            yield gid, _bgr_to_pil(patch)
            gid += 1


# ---------- per-flight driver ----------------------------------------------

def _source_and_path(flight, target, limit):
    if target == "tile":
        out = os.path.join(CAPTION_DIR,
                           f"{flight}_tile_ts{TILE_SIZE}_st{TILE_STRIDE}.jsonl")
        return iter_tile_patches(flight, limit), "tile_id", out
    gen = iter_sat_patches(flight, limit) if target == "sat" \
        else iter_drone_images(flight, limit)
    return gen, "filename", os.path.join(CAPTION_DIR, f"{flight}_{target}.jsonl")


def run_flight(caption, flight, target, limit):
    os.makedirs(CAPTION_DIR, exist_ok=True)
    source, id_field, out_path = _source_and_path(flight, target, limit)
    done = set()
    if os.path.isfile(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)[id_field])
                except (json.JSONDecodeError, KeyError):
                    pass

    n_new = n_skip = 0
    with open(out_path, "a", buffering=1) as f:
        for key, pil in tqdm(source, desc=f"  flight {flight} [{target}]", unit="img"):
            if key in done:
                continue
            if pil is None:
                n_skip += 1
                continue
            text = _clean_caption(caption(pil))
            f.write(json.dumps({"flight": flight, id_field: key,
                                "caption": text}) + "\n")
            n_new += 1
    print(f"  flight {flight} [{target}]: +{n_new} captions "
          f"({len(done)} already done, {n_skip} skipped) -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["sat", "drone", "tile"], default="sat",
                    help="sat=GT crops (train), drone=query images (test), tile=gallery grid.")
    ap.add_argument("--flights", nargs="+", default=["all"])
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap rows per flight (smoke test).")
    args = ap.parse_args()

    flights = FLIGHTS_AVAILABLE if args.flights == ["all"] else args.flights
    caption = build_captioner()
    print(f"  Captioning [{args.target}] | flights {' '.join(flights)}")
    for flight in flights:
        run_flight(caption, flight, args.target, args.limit)


if __name__ == "__main__":
    main()
