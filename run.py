"""Minimal Llama playground.

Loads a small instruction-tuned Llama model and either answers a single prompt
passed on the command line, or starts an interactive chat loop.

Usage:
    python run.py "your question here"   # one-shot
    python run.py                         # interactive chat
"""

import sys

import torch
from transformers import pipeline

# Swap this for another gated Llama model you have access to, e.g.
# "meta-llama/Llama-3.2-3B-Instruct".
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

SYSTEM_PROMPT = "You are a concise, helpful assistant."


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # Apple Silicon
        return "mps"
    return "cpu"


def build_pipe():
    device = pick_device()
    print(f"Loading {MODEL_ID} on {device} (first run downloads the weights)...")
    return pipeline(
        "text-generation",
        model=MODEL_ID,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        device_map=device,
    )


def generate(pipe, user_text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    out = pipe(messages, max_new_tokens=256, do_sample=True, temperature=0.7)
    return out[0]["generated_text"][-1]["content"].strip()


def main() -> None:
    pipe = build_pipe()

    if len(sys.argv) > 1:  # one-shot mode
        print(generate(pipe, " ".join(sys.argv[1:])))
        return

    print("Interactive chat — type 'exit' or Ctrl-C to quit.\n")
    try:
        while True:
            user_text = input("you> ").strip()
            if user_text.lower() in {"exit", "quit"}:
                break
            if user_text:
                print(f"llama> {generate(pipe, user_text)}\n")
    except (KeyboardInterrupt, EOFError):
        print("\nbye!")


if __name__ == "__main__":
    main()
