#!/usr/bin/env python3
"""Gemini-official VLM judge for CF-VEdit — a drop-in when OpenRouter is unreachable.

The built-in `--judge vlm` calls OpenRouter, which is region/IP-blocked from some hosts.
This script reuses judges/vlm_judge.py's frame-extraction / prompt-building / parsing
verbatim and only swaps the API call to Gemini's OpenAI-compatible endpoint (which works
through a local proxy). It writes a raw JSONL identical in shape to vlm_raw_judge.jsonl,
which you then import through the normal path — the read-only judges/ dir is untouched:

    python3 gemini_judge.py <run> --root /path/to/physics_iq_for_simple_eval --out /tmp/<run>.jsonl
    (cd /path/to/physics_iq_for_simple_eval && python3 bench.py score <run> --judge vlm --judge-output /tmp/<run>.jsonl)

Needs GEMINI_API_KEY. Honors http(s)_proxy env (urllib picks it up automatically).
"""
import argparse, json, os, sys, time, urllib.request, urllib.error
from pathlib import Path

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"


def call_gemini(api_key, messages, model, timeout=180, retries=3):
    payload = {"model": model, "messages": messages, "temperature": 0,
               "response_format": {"type": "json_object"}}
    headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
    last = None
    for attempt in range(retries):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(GEMINI_URL, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:  # uses http(s)_proxy env
                body = json.loads(r.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            last = RuntimeError(f"Gemini HTTP {e.code}: {detail[:300]}")
            if e.code == 400 and "response_format" in payload:
                payload.pop("response_format", None)  # some models reject json_object
                continue
            if e.code in (408, 409, 429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 * (attempt + 1)); continue
            raise last
        except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError) as e:
            last = RuntimeError(f"Gemini call failed: {e}")
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1)); continue
            raise last
    raise last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--root", default=os.getcwd(),
                    help="path to physics_iq_for_simple_eval (default: cwd)")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not (root / "bench.py").exists():
        raise SystemExit(f"--root does not look like the benchmark (no bench.py): {root}")
    sys.path.insert(0, str(root))
    import bench
    from judges import vlm_judge as vj

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not set")

    rows, predictions, _meta, run_dir = bench.validate_predictions(args.run)
    pred_by_id = {r["sample_id"]: r for r in predictions}
    prompts = vj.load_prompts(root)
    ok_ids = [r["sample_id"] for r in rows if pred_by_id.get(r["sample_id"], {}).get("status") == "ok"]
    results = {}
    for mrow in rows:
        sid = mrow["sample_id"]
        pred = pred_by_id.get(sid, {})
        if pred.get("status") != "ok" or not pred.get("video"):
            continue
        source = root / mrow["source_video"]
        edited = run_dir / pred["video"]
        sframes = mrow.get("video_meta", {}).get("num_frames") or args.frames
        eframes = vj.probe_num_frames(edited) or sframes
        print(f"[gemini] judging {sid} ({ok_ids.index(sid)+1}/{len(ok_ids)})", file=sys.stderr)
        messages = vj.build_messages(
            prompts[sid],
            vj.extract_frames(source, sframes, args.frames),
            vj.extract_frames(edited, eframes, args.frames),
        )
        results[sid] = vj.parse_judge_json(call_gemini(api_key, messages, args.model))
    vj.write_raw(args.out, results)
    print(f"wrote {len(results)} judge rows -> {args.out}")


if __name__ == "__main__":
    main()
