"""Quick smoke test with a tiny, ungated model.

Proves the transformers pipeline works and produces text output, without needing
Hugging Face login or the gated Llama download. Once this works, run.py uses the
real Llama model.

    python smoke_test.py
"""

import torch
from transformers import pipeline

# Small, openly-licensed model — no gating, ~270 MB download.
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    device = pick_device()
    print(f"Loading {MODEL_ID} on {device} ...")
    pipe = pipeline("text-generation", model=MODEL_ID, device_map=device)

    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "In one sentence, what is an open-weight model?"},
    ]
    out = pipe(messages, max_new_tokens=60, do_sample=False)
    print("\n--- model output ---")
    print(out[0]["generated_text"][-1]["content"].strip())


if __name__ == "__main__":
    main()
