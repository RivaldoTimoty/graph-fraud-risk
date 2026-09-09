PYTHON ?= python

.PHONY: data split eda test lint format

data:
	$(PYTHON) -m src.data.load

split:
	$(PYTHON) -m src.data.split

eda:
	$(PYTHON) -m src.data.eda

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m black --check src tests

format:
	$(PYTHON) -m ruff check --fix src tests
	$(PYTHON) -m black src tests
