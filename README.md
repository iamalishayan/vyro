# Pocket-Agent — On-Device Mobile Assistant

This repository contains an end-to-end pipeline to fine-tune `meta-llama/Llama-3.2-1B-Instruct` into a JSON tool-calling agent that operates entirely offline on CPU with high inference speed.

## Quickstart (Colab CPU / T4 GPU)

The entire project is reproducible.

```bash
# 1. Install dependencies
make setup

# 2. Synthesize dataset (requires GEMINI_API_KEY in .env)
make data

# 3. Fine-tune LoRA on Llama-3.2-1B-Instruct (requires GPU)
make train

# 4. Export to Q4_K_M GGUF format
make quantize

# 5. Launch the Offline Chatbot UI
make demo
```

## Strategy & Design Decisions

- **Base Model:** We selected `meta-llama/Llama-3.2-1B-Instruct` as our 1B parameter base model. It exhibits strong base instruction-following capabilities but easily fits within a `<500 MB` GGUF footprint and naturally executes tool calls at `≤ 200 ms` on a standard CPU node.
- **Data Formulation:** Our training uses `google-genai` to synthetically bootstrap a rich subset of all expected tool schemas and edge cases (adversarial prompts, refusals, multi-turn context switching). We enforce strict constraint decoding during synthetic generation.
- **Fine-Tuning System:** We apply QLoRA (via bitsandbytes and peft) using `trl/SFTTrainer`, targeting all linear projection matrices to maximize performance recovery. The chat template specifically ensures adherence to `<tool_call>` output structures.
- **Quantization:** We employ `llama.cpp` to quantize the model to `Q4_K_M`. Our final footprint is dramatically suppressed to fit well below the requisite `<500 MB`.

## Artifacts

- Model file: `model-q4_k_m.gguf`
- Interface script (for graders): `inference.py`
- Chatbot Demo: `app.py`
