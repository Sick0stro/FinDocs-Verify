.PHONY: train eval test serve compare all clean pipeline-stages

# Use sam_env Python (has CUDA + torch for GPU acceleration)
PYTHON ?= /c/Users/mutaw/miniconda3/envs/sam_env/python.exe
export PYTHONPATH :=

train:
	$(PYTHON) src/train.py

eval:
	$(PYTHON) src/evaluate.py

compare:
	$(PYTHON) src/compare_models.py

test:
	$(PYTHON) -m pytest tests/ -v

serve:
	$(PYTHON) src/serve.py

# Pipeline stages for WasteFlow Document AI
pipeline-stages:
	$(PYTHON) stages/1_pdf_to_png.py
	$(PYTHON) stages/2_layoutlmv3_train.py

all: train eval test

clean:
	rm -rf models/checkpoints artifacts/.compare_checkpoint.json
