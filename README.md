# Pocket-Agent — On-Device Mobile Assistant

Fine-tuned **Qwen/Qwen2.5-1.5B-Instruct** (1.5B parameters) for structured JSON tool-calling, designed to run entirely offline on CPU.

## Model

- **Base Model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Parameters:** 1.5B (within the ≤ 2B limit)
- **Fine-Tuning:** QLoRA (4-bit) with LoRA rank 16, targeting all linear projections
- **Quantization:** GGUF Q4_K_M via llama.cpp

## Quickstart (Colab T4)

```bash
# 1. Install dependencies
make setup

# 2. Generate synthetic training data (requires GEMINI_API_KEY in .env)
make data

# 3. Fine-tune with LoRA on T4 GPU (~30 min)
make train

# 4. Quantize to GGUF Q4_K_M
make quantize

# 5. Launch chatbot demo (gives a public gradio.live URL)
make demo
```

Or run everything at once:
```bash
make all
```

## Project Structure

```
├── starter/                  # Starter pack (schemas, dev set, teacher examples)
│   ├── public_test.jsonl     # 40 dev-set examples
│   ├── eval_harness_contract.py
│   ├── tool_schemas.json
│   └── teacher_examples.jsonl
├── data_generate.py          # Synthetic data generation via Gemini API
├── train.py                  # QLoRA fine-tuning script
├── quantize.sh               # GGUF quantization via llama.cpp
├── inference.py              # Grader interface: run(prompt, history) -> str
├── app.py                    # Gradio chatbot demo
├── requirements.txt
├── Makefile
└── README.md
```

## Design Decisions

- **Qwen2.5-1.5B-Instruct** was chosen over Llama-3.2-1B because it has stronger out-of-the-box instruction following and structured output capabilities, while still fitting comfortably under the 2B parameter and 500MB quantized size limits.
- **QLoRA** (4-bit base + LoRA adapters) allows full fine-tuning on a free Colab T4 with 16GB VRAM.
- **Synthetic data** was generated using the Gemini API to create diverse examples covering all 5 tools, multi-turn conversations, refusals, adversarial prompts (typos, code-switching), and edge cases.
- **llama-cpp-python** is used for inference to guarantee fast CPU performance (< 200ms/turn) with zero network dependencies.

## What Worked

- Loss converged well from 2.5 → 0.09 over 200 steps (~7 epochs on 230 examples)
- The model learned the `<tool_call>` JSON format reliably
- Refusals for out-of-scope requests work consistently

## What Didn't / Could Be Better

- More diverse training data (1000+ examples) would improve paraphrase handling
- Code-switched (multilingual) prompts could use more training examples
- Could explore Q3_K_S quantization for the < 250MB bonus
