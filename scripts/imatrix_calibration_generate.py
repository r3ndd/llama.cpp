#!/usr/bin/env python3

import argparse
import json
import os
import re
from typing import Any

import requests
from requests import RequestException


PASS_THROUGH_FIELDS = {
    "temperature",
    "top_p",
    "max_tokens",
    "seed",
    "presence_penalty",
    "frequency_penalty",
    "stop",
    "n",
    "min_p",
    "top_k",
    "repeat_penalty",
    "repetition_penalty",
}


def load_prompt_specs(input_jsonl: str) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []

    with open(input_jsonl, "r", encoding="utf-8") as fin:
        for line_no, raw_line in enumerate(fin, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_no}: {exc}") from exc

            if not isinstance(row, dict):
                raise ValueError(f"line {line_no}: expected a JSON object")

            if "prompt" not in row and "messages" not in row:
                raise ValueError(f"line {line_no}: expected either 'prompt' or 'messages'")

            if "prompt" in row and not isinstance(row["prompt"], str):
                raise ValueError(f"line {line_no}: 'prompt' must be a string")

            if "messages" in row and not isinstance(row["messages"], list):
                raise ValueError(f"line {line_no}: 'messages' must be a list")

            specs.append(row)

    if not specs:
        raise ValueError("no prompt entries found in input file")

    return specs


def build_messages(spec: dict[str, Any]) -> list[dict[str, str]]:
    if "messages" in spec:
        messages = spec["messages"]
        if not messages:
            raise ValueError("'messages' must contain at least one entry")

        out: list[dict[str, str]] = []
        for entry in messages:
            if not isinstance(entry, dict):
                raise ValueError("each 'messages' entry must be an object")

            role = entry.get("role")
            content = entry.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                raise ValueError("each message requires string 'role' and string 'content'")
            out.append({"role": role, "content": content})
        return out

    return [{"role": "user", "content": spec["prompt"]}]


def extract_prompt_text(spec: dict[str, Any]) -> str:
    if "prompt" in spec and isinstance(spec["prompt"], str):
        return spec["prompt"]

    messages = build_messages(spec)
    user_lines = [m["content"] for m in messages if m["role"] == "user"]
    if user_lines:
        return "\n".join(user_lines)

    return "\n".join(m["content"] for m in messages)


def build_payload(spec: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "messages": build_messages(spec),
        "stream": False,
    }

    if args.model:
        payload["model"] = args.model

    for key in PASS_THROUGH_FIELDS:
        if key in spec:
            payload[key] = spec[key]

    if "max_tokens" not in payload and args.default_max_tokens is not None:
        payload["max_tokens"] = args.default_max_tokens
    if "temperature" not in payload and args.default_temperature is not None:
        payload["temperature"] = args.default_temperature
    if "top_p" not in payload and args.default_top_p is not None:
        payload["top_p"] = args.default_top_p

    extra = spec.get("extra")
    if isinstance(extra, dict):
        payload.update(extra)

    return payload


def parse_generated_text(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("response did not include any choices")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError("unexpected response structure for first choice")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("response missing message in first choice")

    content = message.get("content", "")

    if isinstance(content, str) and content.strip():
        return content

    if isinstance(content, list):
        text_chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                text_chunks.append(item["text"])
        joined = "".join(text_chunks)
        if joined.strip():
            return joined

    # Some reasoning-capable chat models return generated text in
    # message.reasoning_content with empty message.content.
    reasoning_content = message.get("reasoning_content")
    if isinstance(reasoning_content, str):
        return reasoning_content

    return ""


def require_non_empty_text(text: str) -> str:
    if text.strip():
        return text
    raise ValueError("model response content was empty")


def safe_slug(value: str, fallback: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-")
    if not slug:
        return fallback
    return slug[:80]


def write_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as fout:
        fout.write(content)
        if content and not content.endswith("\n"):
            fout.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate calibration text files for llama-imatrix by sending prompts to "
            "a llama-server OpenAI-compatible chat completions endpoint."
        ),
        epilog=(
            "Example:\n"
            "  python scripts/imatrix_calibration_generate.py "
            "--input-jsonl scripts/imatrix_calibration_prompts_40.jsonl "
            "--output-dir ./imatrix-calibration\n\n"
            "Then use the combined output with llama-imatrix:\n"
            "  ./llama-imatrix -m model.gguf -f ./imatrix-calibration/calibration_all.txt"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--input-jsonl", required=True, help="JSONL prompt file with per-prompt inference parameters")
    parser.add_argument("--output-dir", required=True, help="Directory for calibration outputs")
    parser.add_argument("--server-url", default="http://127.0.0.1:8080", help="Base llama-server URL")
    parser.add_argument("--endpoint", default="/v1/chat/completions", help="OpenAI-compatible endpoint path")
    parser.add_argument("--model", default=None, help="Optional model identifier for the request payload")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds")
    parser.add_argument("--api-key", default=os.environ.get("LLAMA_API_KEY"), help="Optional Bearer token")
    parser.add_argument("--default-max-tokens", type=int, default=None, help="Fallback max_tokens when omitted in JSONL")
    parser.add_argument("--default-temperature", type=float, default=None, help="Fallback temperature when omitted in JSONL")
    parser.add_argument("--default-top-p", type=float, default=None, help="Fallback top_p when omitted in JSONL")
    parser.add_argument("--combined-file", default="calibration_all.txt", help="Filename for combined calibration text")
    parser.add_argument("--prompts-file", default="prompts_used.txt", help="Filename for exported prompt text")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Skip failed prompts and continue; default is to stop on first request error",
    )
    args = parser.parse_args()

    specs = load_prompt_specs(args.input_jsonl)

    os.makedirs(args.output_dir, exist_ok=True)

    endpoint = args.endpoint if args.endpoint.startswith("/") else f"/{args.endpoint}"
    url = f"{args.server_url.rstrip('/')}{endpoint}"

    headers: dict[str, str] = {}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    all_generated: list[str] = []
    prompts_used: list[str] = []

    with requests.Session() as session:
        for index, spec in enumerate(specs, start=1):
            prompt_text = extract_prompt_text(spec)
            prompt_id = spec.get("id") if isinstance(spec.get("id"), str) else f"prompt-{index:03d}"
            prompt_slug = safe_slug(prompt_id, f"prompt-{index:03d}")
            payload = build_payload(spec, args)

            try:
                response = session.post(url, headers=headers, json=payload, timeout=args.timeout)
                response.raise_for_status()
                response_json = response.json()
                generated = require_non_empty_text(parse_generated_text(response_json))
            except (RequestException, ValueError, json.JSONDecodeError) as exc:
                if args.continue_on_error:
                    print(f"[warn] prompt {index} ({prompt_slug}) failed: {exc}")
                    continue
                raise RuntimeError(f"request failed for prompt {index} ({prompt_slug}): {exc}") from exc

            per_prompt_path = os.path.join(args.output_dir, f"calibration_{index:03d}_{prompt_slug}.txt")
            write_text(per_prompt_path, generated)

            all_generated.append(generated)
            prompts_used.append(prompt_text)
            print(f"[ok] wrote {per_prompt_path}")

    if not all_generated:
        raise RuntimeError("no calibration text generated")

    combined_path = os.path.join(args.output_dir, args.combined_file)
    write_text(combined_path, "\n\n".join(all_generated))
    print(f"[ok] wrote {combined_path}")

    prompts_path = os.path.join(args.output_dir, args.prompts_file)
    prompts_text = "\n\n".join(prompts_used)
    write_text(prompts_path, prompts_text)
    print(f"[ok] wrote {prompts_path}")


if __name__ == "__main__":
    main()
