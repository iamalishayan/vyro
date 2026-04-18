import os
import json
import random
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in the environment.")

genai.configure(api_key=api_key)

generation_config = {
  "temperature": 0.9,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "application/json",
}

model = genai.GenerativeModel(
  model_name="gemini-2.5-flash",
  generation_config=generation_config,
)

prompt_template = """
You are an expert training data generator for a tool-calling LLM.
We need to generate synthetic training data loops for an LLM that is restricted to the following 5 tools:
```json
[
  {{"tool": "weather",  "args": {{"location": "string", "unit": "C|F"}}}},
  {{"tool": "calendar", "args": {{"action": "list|create", "date": "YYYY-MM-DD", "title": "string?"}}}},
  {{"tool": "convert",  "args": {{"value": "number", "from_unit": "string", "to_unit": "string"}}}},
  {{"tool": "currency", "args": {{"amount": "number", "from": "ISO3", "to": "ISO3"}}}},
  {{"tool": "sql",      "args": {{"query": "string"}}}}
]
```

Please generate exactly 10 unique examples of type: "{example_type}".

Rules for the assistant:
1. When calling a tool, it MUST reply with EXACTLY this string format (no other text): `<tool_call>{{"tool": "tool_name", "args": {{...}}}}</tool_call>`
2. When refusing or engaging in chitchat because no tool fits, it MUST reply in plain natural language WITHOUT generating a `<tool_call>` tag.
3. Multi-turn contexts must resolve references perfectly.

Output a JSON object with a single key "dataset" which is a list of lists of messages. 
Each item in "dataset" is a single conversation/example. A conversation is a list of messages.
A message is a dictionary with "role" (either "user", "assistant", or "user" simulating tool response if multi-turn) and "content" (string).

If "example_type" is multi-turn, provide 3 to 5 messages in the exchange. Make sure the ASSISTANT calls a tool when appropriate and speaks naturally when not.

Example output format:
{{
  "dataset": [
    [
      {{"role": "user", "content": "What is 50 USD in EUR?"}},
      {{"role": "assistant", "content": "<tool_call>{{\"tool\": \"currency\", \"args\": {{\"amount\": 50, \"from\": \"USD\", \"to\": \"EUR\"}}}}</tool_call>"}}
    ]
  ]
}}
"""

types_to_generate = [
    "Simple clear single-turn weather tool calls",
    "Simple clear single-turn calendar tool calls",
    "Simple clear single-turn convert tool calls",
    "Simple clear single-turn currency tool calls",
    "Simple clear single-turn sql tool calls",
    "Refusal: User asks for a timer or alarm (we don't have that tool)",
    "Refusal: User asks to send a text message or email (we don't have that)",
    "Refusal: Plain chitchat and greetings",
    "Refusal: Ambiguous requests with no context (e.g., 'convert it', 'what about tomorrow?')",
    "Multi-turn: User asks for weather, then asks to convert the temperature to the other unit",
    "Multi-turn: User asks for currency conversion, then asks for another related conversion",
    "Adversarial: Typos in the tool request ('wether in londen')",
    "Adversarial: Code-switched prompts (Spanish/English mixed) asking for a tool",
    "Adversarial: User implies a tool but it's actually just a general question about units or currencies (e.g. 'what is the symbol for Euro?') -> Should be refusal/plain text",
]

def generate_data():
    all_data = []
    
    # We will loop multiple times to generate a decent dataset size.
    # 3 loops * 14 types * 10 examples = 420 examples. Let's do 5 loops for ~700 examples.
    print("Starting generation...")
    for loop in range(5):
        print(f"Loop {loop + 1}/5")
        for ex_type in types_to_generate:
            print(f"  Generating: {ex_type}")
            try:
                response = model.generate_content(prompt_template.format(example_type=ex_type))
                data = json.loads(response.text)
                conversations = data.get("dataset", [])
                
                for conv in conversations:
                    # Enforce that it's a dict with "messages"
                    all_data.append({"messages": conv})
            except Exception as e:
                print(f"  Error generating for {ex_type}: {e}")
                
    # Save to train.jsonl
    with open("train.jsonl", "w") as f:
        for item in all_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"Generated {len(all_data)} items and saved to train.jsonl")

if __name__ == "__main__":
    generate_data()
