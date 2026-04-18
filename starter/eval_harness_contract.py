"""
Evaluation harness contract — the grader calls run() from inference.py.
This file documents the exact interface the grader expects.
"""
import importlib
import time

def grade_example(module, prompt: str, history: list[dict], expected: str):
    """
    Calls module.run(prompt, history) and returns (response, latency_ms).
    """
    start = time.perf_counter()
    response = module.run(prompt, history)
    latency_ms = (time.perf_counter() - start) * 1000
    return response, latency_ms

def load_submission():
    """
    Loads inference.py as a module and verifies the run() function exists.
    """
    mod = importlib.import_module("inference")
    assert hasattr(mod, "run"), "inference.py must expose a `run(prompt, history)` function"
    assert callable(mod.run), "run must be callable"
    return mod

if __name__ == "__main__":
    mod = load_submission()
    # Smoke test
    resp, lat = grade_example(mod, "What's the weather in Paris?", [], "")
    print(f"Response: {resp}")
    print(f"Latency: {lat:.1f} ms")
