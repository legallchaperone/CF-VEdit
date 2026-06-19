#!/usr/bin/env python3
"""OpenRouter VLM judge for CF-VEdit (the only built-in VLM backend).

By design this is the single VLM judging path: it calls the OpenRouter chat
API with ``google/gemini-2.5-pro`` and requires an OpenRouter API key
(``OPENROUTER_API_KEY`` or ``--api-key``). Each video is sent as a short
sequence of evenly sampled JPEG frames (source frames first, then edited
frames), extracted with ffmpeg. The model is asked to return the shared
per-sample judge schema (see ``judges/vlm_prompts.jsonl``), which ``bench.py``
then normalizes exactly like ``--judge-output`` rows.

Frames (rather than native video) keep the request portable across the
OpenAI-compatible API; temporal_consistency is therefore judged from sampled
frames, which is a known limitation of this v0.1 backend.
"""
import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-2.5-pro"
DEFAULT_FRAMES = 8
FRAME_WIDTH = 512
JUDGE_FLAGS = (
    "target_success",
    "preservation_success",
    "physical_effect_success",
    "temporal_consistency",
    "major_artifacts",
    "overall_pass",
)


class JudgeError(RuntimeError):
    pass


def require_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise JudgeError(
            "ffmpeg not found on PATH; the OpenRouter VLM judge extracts video frames with ffmpeg"
        )


def even_indices(num_frames, count):
    n = max(1, int(num_frames))
    count = max(1, min(int(count), n))
    if count == 1:
        return [0]
    step = (n - 1) / (count - 1)
    return sorted({int(round(k * step)) for k in range(count)})


def probe_num_frames(video_path):
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-count_frames", "-show_entries", "stream=nb_read_frames",
                "-of", "csv=p=0", str(video_path),
            ],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        return int(out)
    except Exception:
        return None


def extract_frames(video_path, num_frames, count=DEFAULT_FRAMES, width=FRAME_WIDTH):
    require_ffmpeg()
    video_path = Path(video_path)
    if not video_path.exists():
        raise JudgeError(f"missing video for frame extraction: {video_path}")
    idxs = even_indices(num_frames, count)
    select = "+".join(f"eq(n\\,{i})" for i in idxs)
    vf = f"select={select},scale={width}:-2"
    tmp = Path(tempfile.mkdtemp(prefix="cfvedit_frames_"))
    try:
        cmd = [
            "ffmpeg", "-v", "error", "-i", str(video_path),
            "-vf", vf, "-frames:v", str(len(idxs)),
            str(tmp / "f_%03d.jpg"),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise JudgeError(f"ffmpeg failed on {video_path}: {exc.stderr.strip()[:300]}") from exc
        frames = [p.read_bytes() for p in sorted(tmp.glob("f_*.jpg"))]
        if not frames:
            raise JudgeError(f"ffmpeg produced no frames for {video_path}")
        return frames
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def data_url(jpeg_bytes):
    return "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode("ascii")


def build_messages(prompt_text, source_frames, edited_frames):
    content = [
        {
            "type": "text",
            "text": prompt_text
            + "\n\nThe SOURCE video frames are shown first, then the EDITED candidate "
            "video frames. Each block is in temporal order.",
        },
        {"type": "text", "text": "SOURCE video frames:"},
    ]
    for frame in source_frames:
        content.append({"type": "image_url", "image_url": {"url": data_url(frame)}})
    content.append({"type": "text", "text": "EDITED candidate video frames:"})
    for frame in edited_frames:
        content.append({"type": "image_url", "image_url": {"url": data_url(frame)}})
    content.append({"type": "text", "text": "Respond with only the JSON object."})
    return [{"role": "user", "content": content}]


def call_openrouter(api_key, messages, model=DEFAULT_MODEL, timeout=180, retries=3):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/cf-vedit/benchmark",
        "X-Title": "CF-VEdit Benchmark",
    }
    last_error = None
    for attempt in range(retries):
        request = urllib.request.Request(OPENROUTER_URL, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            last_error = JudgeError(f"OpenRouter HTTP {exc.code}: {detail[:400]}")
            if exc.code in (408, 409, 429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise last_error
        except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError) as exc:
            last_error = JudgeError(f"OpenRouter call failed: {exc}")
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise last_error
    raise last_error


def parse_judge_json(text):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise JudgeError(f"no JSON object in judge response: {text[:200]!r}")
    obj = json.loads(text[start : end + 1])
    row = {flag: (1 if obj.get(flag) else 0) for flag in JUDGE_FLAGS}
    hits = obj.get("effect_hits", [])
    if isinstance(hits, str):
        hits = [hits]
    row["effect_hits"] = [str(h) for h in hits]
    row["short_reason"] = str(obj.get("short_reason", ""))[:500]
    if not row["target_success"]:
        row["physical_effect_success"] = 0
        row["effect_hits"] = []
    return row


def load_prompts(root):
    path = Path(root) / "judges" / "vlm_prompts.jsonl"
    if not path.exists():
        raise JudgeError(f"missing judge prompts: {path}")
    prompts = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                prompts[row["sample_id"]] = row["prompt"]
    return prompts


def judge_predictions(root, rows, pred_by_id, run_dir, api_key, model=DEFAULT_MODEL,
                      num_frames=DEFAULT_FRAMES, log=print):
    """Return {sample_id: raw judge row} for every prediction with status == ok."""
    root = Path(root)
    run_dir = Path(run_dir)
    prompts = load_prompts(root)
    results = {}
    ok_ids = [r["sample_id"] for r in rows if pred_by_id.get(r["sample_id"], {}).get("status") == "ok"]
    for position, manifest_row in enumerate(rows):
        sample_id = manifest_row["sample_id"]
        prediction = pred_by_id.get(sample_id, {})
        if prediction.get("status") != "ok" or not prediction.get("video"):
            continue
        prompt_text = prompts.get(sample_id)
        if prompt_text is None:
            raise JudgeError(f"no vlm prompt for {sample_id}")
        source = root / manifest_row["source_video"]
        edited = run_dir / prediction["video"]
        source_frames = manifest_row.get("video_meta", {}).get("num_frames") or probe_num_frames(source) or num_frames
        edited_frames = probe_num_frames(edited) or source_frames
        log(f"[vlm] judging {sample_id} ({ok_ids.index(sample_id) + 1}/{len(ok_ids)})")
        messages = build_messages(
            prompt_text,
            extract_frames(source, source_frames, num_frames),
            extract_frames(edited, edited_frames, num_frames),
        )
        results[sample_id] = parse_judge_json(call_openrouter(api_key, messages, model=model))
    return results


def write_raw(path, results):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample_id, row in results.items():
            out = dict(row)
            out["sample_id"] = sample_id
            handle.write(json.dumps(out, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _resolve_api_key(explicit):
    api_key = explicit or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise JudgeError(
            "no OpenRouter API key: set OPENROUTER_API_KEY or pass --api-key. "
            "The only built-in VLM backend is OpenRouter with google/gemini-2.5-pro."
        )
    return api_key


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the OpenRouter (Gemini 2.5 Pro) VLM judge for a run")
    parser.add_argument("run")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--out", default=None, help="output JSONL (default results/<run>/vlm_raw_judge.jsonl)")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import bench

    api_key = _resolve_api_key(args.api_key)
    rows, predictions, _run_meta, run_dir = bench.validate_predictions(args.run)
    pred_by_id = {row["sample_id"]: row for row in predictions}
    results = judge_predictions(
        root, rows, pred_by_id, run_dir, api_key,
        model=args.model, num_frames=args.frames,
        log=lambda message: print(message, file=sys.stderr),
    )
    out_path = Path(args.out) if args.out else root / "results" / args.run / "vlm_raw_judge.jsonl"
    write_raw(out_path, results)
    print(f"wrote {len(results)} judge rows: {out_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except JudgeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
