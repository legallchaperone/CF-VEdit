"""Name DAVIS objects for remove-only training rows with one VLM call per sequence."""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "gemini-3-pro-preview"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Name highlighted DAVIS objects with a VLM")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    out_path = Path(args.out_json)
    rows = json.loads(config_path.read_text(encoding="utf-8"))
    cache = _read_json(out_path)
    client, model = _client(args.model)
    for row in rows:
        sequence = row["sequence"]
        cache[sequence] = _name_sequence(client, model, row)
    _write_json(out_path, cache)
    return 0


def _client(default_model: str):
    import openai

    cf_project_id = os.environ.get("CF_PROJECT_ID")
    cf_user_id = os.environ.get("CF_USER_ID")
    if not cf_project_id or not cf_user_id:
        raise RuntimeError("CF_PROJECT_ID and CF_USER_ID environment variables must be set")
    metadata = json.dumps({"project_id": cf_project_id, "user_id": cf_user_id})
    return openai.OpenAI(
        api_key=os.environ.get("GEMINI_API_KEY", "placeholder"),
        base_url="https://ai-gateway.plain-flower-4887.workers.dev/compat",
        default_headers={"cf-aig-metadata": metadata},
    ), os.environ.get("MODEL_ID", default_model)


def _name_sequence(client: Any, model: str, row: dict[str, Any]) -> dict[str, str]:
    content: list[dict[str, Any]] = []
    for obj in row["objects"]:
        content.append({"type": "text", "text": f"Object {obj['index']}:"})
        content.append({"type": "image_url", "image_url": {"url": _image_data_url(Path(obj["image_path"]))}})
    content.append({
        "type": "text",
        "text": (
            "Each image highlights one object in red. For each, give a short distinct "
            "referring expression (4 words or fewer, lowercase noun phrase) uniquely "
            "identifying it against the others. JSON only: "
            "{\"objects\":[{\"index\":1,\"noun\":\"man in blue shirt\"}]}"
        ),
    })
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You name highlighted video objects. Output valid JSON only."},
            {"role": "user", "content": content},
        ],
    )
    parsed = _parse_json(resp.choices[0].message.content)
    by_index = {int(obj["index"]): str(obj["noun"]).strip().lower() for obj in parsed.get("objects", [])}
    return {
        str(obj["object_id"]): by_index.get(int(obj["index"]), "")
        for obj in row["objects"]
    }


def _parse_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end > start:
            return json.loads(raw[start:end + 1])
        raise


def _image_data_url(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{data}"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
