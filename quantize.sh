#!/bin/bash
set -e

echo "Setting up llama.cpp..."
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp
fi

cd llama.cpp

# llama.cpp now uses CMake instead of Make
echo "Building llama.cpp with CMake..."
cmake -B build
cmake --build build --config Release -j$(nproc)

echo "Installing convert requirements..."
pip install -r requirements.txt 2>/dev/null || pip install gguf numpy sentencepiece protobuf transformers torch

echo "Converting HuggingFace model to GGUF (F16)..."
python convert_hf_to_gguf.py ../merged_model --outfile ../model-f16.gguf

echo "Quantizing to Q4_K_M..."
./build/bin/llama-quantize ../model-f16.gguf ../model-q4_k_m.gguf Q4_K_M

echo "Done! Quantized model saved as model-q4_k_m.gguf"
ls -lh ../model-q4_k_m.gguf
rm -f ../model-f16.gguf
