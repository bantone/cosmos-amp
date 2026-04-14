#!/usr/bin/env python3
"""
Interactive NVIDIA Cosmos LLM — video Q&A interface.

Usage:
  python test_llm.py                         # prompts for video URL interactively
  python test_llm.py --video <url>           # supply video URL upfront
  python test_llm.py --host http://host:port --model <model>
"""

import json
import sys
import argparse
import urllib.request
import urllib.error


DEFAULT_HOST = "http://0.0.0.0:8000"
DEFAULT_MODEL = "nvidia/cosmos-reason2-8b"
DEFAULT_NUM_FRAMES = 10


def chat(host: str, model: str, messages: list, num_frames: int) -> dict:
    url = f"{host}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "media_io_kwargs": {"video": {"num_frames": num_frames}},
        "stream": False,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {"error": {"message": f"HTTP {e.code}: {body}"}}
    except urllib.error.URLError as e:
        return {"error": {"message": str(e.reason)}}


def print_response(result: dict) -> str | None:
    """Pretty-print the API response. Returns assistant text or None on error."""
    if "error" in result:
        msg = result["error"]
        if isinstance(msg, dict):
            msg = msg.get("message", str(msg))
        print(f"\nERROR: {msg}\n")
        return None

    if "choices" not in result:
        print(f"\nUnexpected response: {json.dumps(result, indent=2)}\n")
        return None

    text = result["choices"][0]["message"]["content"]
    finish = result["choices"][0].get("finish_reason", "?")
    usage = result.get("usage", {})
    total_tokens = usage.get("total_tokens", "?")

    print(f"\n{text}")
    print(f"\n[finish: {finish}  |  tokens: {total_tokens}]\n")
    return text


def main():
    parser = argparse.ArgumentParser(description="Interactive Cosmos video Q&A.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--video", default=None, help="Video URL to analyze.")
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    args = parser.parse_args()

    print(f"\nNVIDIA Cosmos — {args.model}  [{args.host}]")
    print("=" * 60)

    # Get video URL
    video_url = args.video
    if not video_url:
        try:
            video_url = input("Video URL: ").strip()
        except (KeyboardInterrupt, EOFError):
            sys.exit(0)
    if not video_url:
        print("No video URL provided. Exiting.")
        sys.exit(1)

    print(f"\nVideo loaded: {video_url}")
    print("Ask questions about the video. Type 'quit' or Ctrl-C to exit.\n")

    # Conversation history — video is attached only to the first user message
    history: list = []

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Exiting.")
            break

        # Attach the video only on the first user message; subsequent turns use text only
        if not history:
            user_message = {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": video_url}},
                    {"type": "text", "text": question},
                ],
            }
        else:
            user_message = {
                "role": "user",
                "content": question,
            }

        history.append(user_message)
        print("\nModel: ", end="", flush=True)

        result = chat(args.host, args.model, history, args.num_frames)
        assistant_text = print_response(result)

        if assistant_text is not None:
            history.append({"role": "assistant", "content": assistant_text})
        else:
            # Remove the failed user turn so history stays consistent
            history.pop()


if __name__ == "__main__":
    main()
