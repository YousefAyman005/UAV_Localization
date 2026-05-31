"""View-invariant captioning of satellite crops (and test drone images).

For every drone-image row we crop the ground-truth satellite patch
(``--target sat``) — or load the drone image itself (``--target drone``) — and
ask a VLM for a short description of the *scene layout* that should look the same
from a drone or a satellite. Captions go to a resumable JSONL cache the offline
cluster consumes during LoRA training / fusion evaluation.

Backends:
  - ``ollama`` (default, FREE & easiest): talks to a local Ollama server, so it
    runs on your Mac with no GPU/HF setup. `ollama pull llama3.2-vision` first.
  - ``qwen2vl``  (FREE): local Qwen2-VL via transformers on a GPU, fully offline.
  - ``moondream`` (FREE): tiny/fast local VLM, weaker instruction-following.
  - ``anthropic``: Claude API (needs internet + ANTHROPIC_API_KEY).

Output: cache/captions/{flight}_{sat|drone}.jsonl  rows {flight, filename, caption}

Example (free, on your Mac via Ollama):
    ollama pull llama3.2-vision
    python pipelines/caption_crops.py --target sat \
        --flights 01 02 03 04 05 06 08 09
    python pipelines/caption_crops.py --target drone --flights 10 11
"""

import argparse
import base64
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
OLLAMA_MODEL   = "qwen3.5:9b"
QWEN_MODEL     = "Qwen/Qwen2-VL-7B-Instruct"
MOONDREAM_MODEL = "vikhyatk/moondream2"
ANTHROPIC_MODEL = "claude-sonnet-4-6"

# Static instruction shared by every backend.
SYSTEM_PROMPT = (
    "You describe the fixed physical layout of a piece of ground so it can be "
    "recognized from any altitude or viewing angle. Focus ONLY on permanent "
    "structure: road network and intersections (T-junction, crossroads, curves, "
    "roundabouts), buildings and their rough shape/size, water (rivers, lakes, "
    "ponds, canals), and land cover (farmland/fields, forest, bare ground, "
    "built-up area). Note the spatial arrangement using cardinal directions "
    "(north/south/east/west, center). Write one or two plain sentences, 25-45 "
    "words total. "
    "CRITICAL: never use ANY color or brightness word (no dark, light, bright, "
    "white, black, grey, gray, green, brown, red, blue, yellow, pale, etc.) and "
    "never mention lighting, shadows, season, texture, image quality, or the "
    "words 'satellite', 'aerial', 'drone', 'photo', or 'image'. Describe shape, "
    "type, and position only. Output only the description, no preamble."
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


def make_qwen2vl(model_id, max_tokens, device):
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype).to(device).eval()
    proc = AutoProcessor.from_pretrained(model_id)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image"},
                                     {"type": "text", "text": USER_TEXT}]},
    ]
    chat = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def caption(pil):
        inputs = proc(text=[chat], images=[_downscale(pil)],
                      return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        trimmed = out[:, inputs.input_ids.shape[1]:]
        return proc.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    return caption


def make_moondream(model_id, max_tokens, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = torch.float16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=dtype).to(device).eval()
    tok = AutoTokenizer.from_pretrained(model_id)
    prompt = SYSTEM_PROMPT + " " + USER_TEXT  # moondream has no system role

    def caption(pil):
        with torch.inference_mode():
            enc = model.encode_image(_downscale(pil))
            return model.answer_question(enc, prompt, tok).strip()
    return caption


def make_ollama(model_id, max_tokens, _device, host=None, retries=4):
    """Caption via a local Ollama server (default http://localhost:11434)."""
    try:
        import ollama
    except ImportError:
        sys.exit("ollama package not installed. Run: pip install ollama")
    host = host or os.environ.get("OLLAMA_HOST")
    client = ollama.Client(host=host) if host else ollama.Client()
    try:
        have = {m.get("model", m.get("name", "")) for m in client.list()["models"]}
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
                return _strip_think(resp["message"]["content"])
            except Exception as e:                               # noqa: BLE001
                if attempt == retries - 1:
                    raise
                wait = 2 ** attempt
                print(f"\n  Ollama error ({e}); retry in {wait}s", file=sys.stderr)
                time.sleep(wait)
    return caption


def make_anthropic(model_id, max_tokens, _device, retries=4):
    try:
        import anthropic
    except ImportError:
        sys.exit("anthropic SDK not installed. Run: pip install anthropic")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set.")
    client = anthropic.Anthropic()

    def caption(pil):
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(
            np.asarray(_downscale(pil)), cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        content = [
            {"type": "image", "source": {"type": "base64",
             "media_type": "image/jpeg", "data": b64}},
            {"type": "text", "text": USER_TEXT},
        ]
        for attempt in range(retries):
            try:
                msg = client.messages.create(
                    model=model_id, max_tokens=max_tokens,
                    system=[{"type": "text", "text": SYSTEM_PROMPT,
                             "cache_control": {"type": "ephemeral"}}],
                    messages=[{"role": "user", "content": content}])
                return " ".join(b.text for b in msg.content
                                if b.type == "text").strip()
            except Exception as e:                                # noqa: BLE001
                if attempt == retries - 1:
                    raise
                wait = 2 ** attempt
                print(f"\n  API error ({e}); retry in {wait}s", file=sys.stderr)
                time.sleep(wait)
    return caption


def build_captioner(args):
    device = "cpu"
    if args.backend in ("qwen2vl", "moondream"):
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    default_model = {"ollama": OLLAMA_MODEL, "qwen2vl": QWEN_MODEL,
                     "moondream": MOONDREAM_MODEL,
                     "anthropic": ANTHROPIC_MODEL}[args.backend]
    model_id = args.model or default_model
    factory = {"ollama": make_ollama, "qwen2vl": make_qwen2vl,
               "moondream": make_moondream, "anthropic": make_anthropic}[args.backend]
    print(f"  Backend {args.backend} | model {model_id} | device {device}")
    return factory(model_id, args.max_tokens, device)


# ---------- crop sources ----------------------------------------------------

def _bgr_to_pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _split_df(df, which, args):
    """Restrict to the band each caption target actually uses: sat->train,
    drone->test. `args.no_split` disables this (caption every row)."""
    if args.no_split:
        return df.reset_index(drop=True)
    return split_flight_rows(df, which=which, test_frac=args.test_frac,
                             axis=args.split_axis, buffer_frac=args.split_buffer)


def iter_sat_patches(flight, args):
    import pandas as pd
    tiles, _, drone_csv, _ = load_flight(flight)
    df = _split_df(pd.read_csv(drone_csv), "train", args)   # GT crops train the bridge
    if args.limit is not None:
        df = df.iloc[:args.limit]
    for _, row in df.iterrows():
        yaw = float(row["Phi1"]) if "Phi1" in row.index else 0.0
        patch = crop_gt_patch(tiles, float(row["lat"]), float(row["lon"]),
                              float(row["height"]), yaw_deg=yaw, flight=flight)
        yield row["filename"], (None if patch is None else _bgr_to_pil(patch))


def iter_drone_images(flight, args):
    import pandas as pd
    _, drone_dir, drone_csv, _ = get_flight_paths(flight)
    df = _split_df(pd.read_csv(drone_csv), "test", args)    # query images are the test band
    if args.limit is not None:
        df = df.iloc[:args.limit]
    for _, row in df.iterrows():
        img = cv2.imread(os.path.join(drone_dir, row["filename"]))
        if img is not None:
            img = cv2.resize(img, (SZ_W, SZ_H))
        yield row["filename"], (None if img is None else _bgr_to_pil(img))


def iter_tile_patches(flight, tile_size, stride, limit):
    """Yield (tile_id, patch) over the SAME grid the retrieval gallery uses.
    tile_id is the global index matching clip_pipeline.build_flight_gallery."""
    from pipelines.clip_pipeline import iter_tiles
    tiles, _, _, _ = load_flight(flight)
    gid = 0
    for sat, geo in tiles:
        for _tid, x0, y0, _lat, _lon in iter_tiles(geo, tile_size, stride):
            if limit is not None and gid >= limit:
                return
            patch = sat[y0:y0 + tile_size, x0:x0 + tile_size]
            yield gid, _bgr_to_pil(patch)
            gid += 1


# ---------- per-flight driver ----------------------------------------------

def _source_and_path(flight, target, args):
    """(generator, id_field, out_path) for the requested caption target."""
    if target == "tile":
        out = os.path.join(args.out_dir,
                           f"{flight}_tile_ts{args.tile_size}_st{args.stride}.jsonl")
        return (iter_tile_patches(flight, args.tile_size, args.stride, args.limit),
                "tile_id", out)
    gen = iter_sat_patches(flight, args) if target == "sat" \
        else iter_drone_images(flight, args)
    return gen, "filename", os.path.join(args.out_dir, f"{flight}_{target}.jsonl")


def run_flight(caption, flight, target, args):
    os.makedirs(args.out_dir, exist_ok=True)
    source, id_field, out_path = _source_and_path(flight, target, args)
    done = set()
    if os.path.isfile(out_path) and not args.overwrite:
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)[id_field])
                except (json.JSONDecodeError, KeyError):
                    pass
    mode = "w" if args.overwrite else "a"

    n_new = n_skip = 0
    with open(out_path, mode, buffering=1) as f:
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
    ap.add_argument("--backend",
                    choices=["ollama", "qwen2vl", "moondream", "anthropic"],
                    default="ollama",
                    help="ollama (default) is free & runs on your Mac.")
    ap.add_argument("--model", default=None, help="Override the backend's model id.")
    ap.add_argument("--target", choices=["sat", "drone", "tile"], default="sat",
                    help="sat=GT crops (train), drone=query images (test), "
                         "tile=gallery grid (satellite database).")
    ap.add_argument("--flights", nargs="+", default=["all"])
    ap.add_argument("--out-dir", default=CAPTION_DIR)
    ap.add_argument("--tile-size", type=int, default=1024,
                    help="Gallery tile size for --target tile (match clip_pipeline).")
    ap.add_argument("--stride", type=int, default=512,
                    help="Gallery tile stride for --target tile.")
    ap.add_argument("--max-tokens", type=int, default=120)
    # Caption only the band each target uses (sat->train, drone->test) to avoid
    # captioning rows the experiment never consumes. Keep in sync with training/eval.
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--split-axis", choices=["auto", "lat", "lon"], default="auto")
    ap.add_argument("--split-buffer", type=float, default=0.0)
    ap.add_argument("--no-split", action="store_true",
                    help="Caption every row regardless of train/test band.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap drone rows per flight (for quick tests).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Recaption from scratch instead of resuming.")
    args = ap.parse_args()

    flights = FLIGHTS_AVAILABLE if args.flights == ["all"] else args.flights
    caption = build_captioner(args)
    print(f"  Captioning [{args.target}] | flights {' '.join(flights)}")
    for flight in flights:
        run_flight(caption, flight, args.target, args)


if __name__ == "__main__":
    main()
