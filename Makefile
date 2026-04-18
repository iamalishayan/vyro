.PHONY: all setup data train quantize demo

all: setup data train quantize demo

setup:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

data:
	@echo "Generating synthetic data..."
	python data_generate.py

train:
	@echo "Fine-tuning model..."
	python train.py

quantize:
	@echo "Quantizing model..."
	bash quantize.sh

demo:
	@echo "Launching demo app..."
	python app.py
