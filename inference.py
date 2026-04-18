from llama_cpp import Llama

# Load model globally so it only loads once
try:
    llm = Llama(
        model_path="./model-q4_k_m.gguf",
        n_ctx=2048,
        verbose=False,
        n_gpu_layers=0,  # CPU-only for grading
        n_threads=4
    )
except Exception:
    print("WARNING: model-q4_k_m.gguf not found. run() will fail until model is present.")
    llm = None

SYSTEM_PROMPT = """You are a helpful on-device assistant. You have access to these tools:
- weather: Get weather for a location. Args: location (string), unit (C or F)
- calendar: List or create calendar events. Args: action (list or create), date (YYYY-MM-DD), title (string, optional)
- convert: Convert units. Args: value (number), from_unit (string), to_unit (string)
- currency: Convert currency. Args: amount (number), from (ISO3 code), to (ISO3 code)
- sql: Run a SQL query. Args: query (string)

When a tool fits, reply ONLY with: <tool_call>{"tool": "name", "args": {...}}</tool_call>
When no tool fits, reply in plain natural language. Never invent tools that don't exist."""

def run(prompt: str, history: list[dict]) -> str:
    if llm is None:
        raise RuntimeError("Model was not loaded.")

    # Build Qwen/ChatML style prompt
    formatted = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    formatted += f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    output = llm(
        formatted,
        max_tokens=256,
        stop=["<|im_end|>"],
        temperature=0.0
    )

    return output["choices"][0]["text"].strip()

if __name__ == "__main__":
    if llm is not None:
        print("User: What is the weather in Tokyo?")
        res = run("What is the weather in Tokyo?", [])
        print("Agent:", res)
