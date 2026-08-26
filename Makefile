.PHONY: train eval test serve all clean

PYTHON ?= python

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

all: train eval test

clean:
	rm -rf models/checkpoints artifacts/.compare_checkpoint.json
