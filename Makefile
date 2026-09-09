# Venv project ini. numpy 1.26.4 + pandas 2.2.3 (lihat README soal pinning).
PYTHON ?= ./venv/Scripts/python.exe

MODEL ?= lgbm_baseline_full

.PHONY: data split eda features train train-all test lint format

data:
	$(PYTHON) -m src.data.load

split:
	$(PYTHON) -m src.data.split

eda:
	$(PYTHON) -m src.data.eda

features:
	$(PYTHON) -m src.features.build

train:
	$(PYTHON) -m src.models.gbdt --model $(MODEL) --save

train-all:
	$(PYTHON) -m src.models.gbdt --model lgbm_baseline_noC --save
	$(PYTHON) -m src.models.gbdt --model lgbm_baseline_full --save

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m black --check src tests

format:
	$(PYTHON) -m ruff check --fix src tests
	$(PYTHON) -m black src tests
