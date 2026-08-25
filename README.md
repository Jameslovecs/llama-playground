# llama-playground

A tiny playground for experimenting with Meta's open-weight **Llama** models locally.

This uses Hugging Face [`transformers`](https://huggingface.co/docs/transformers) to load a
small instruction-tuned model (`meta-llama/Llama-3.2-1B-Instruct`) and chat with it from the
command line. The 1B model is small enough to run on a laptop (CPU works; Apple Silicon MPS or a
GPU is faster).

## 1. Set up

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Get access to the model (one-time)

Llama weights are gated on Hugging Face:

1. Create a free account at https://huggingface.co
2. Open https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct and click **"Agree and access"**
   (Meta requires accepting their license — approval is usually instant).
3. Create a read token at https://huggingface.co/settings/tokens and log in:

```bash
huggingface-cli login
```

## 3. Run it

```bash
python run.py "Explain what an open-weight model is in two sentences."
```

Or start an interactive chat:

```bash
python run.py
```

## Notes

- First run downloads ~2.5 GB of weights (cached afterward in `~/.cache/huggingface`).
- To try a different size, change `MODEL_ID` in `run.py` (e.g. `meta-llama/Llama-3.2-3B-Instruct`).
- Weights and virtualenv are git-ignored — only code is tracked.
