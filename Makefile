UV ?= uv
MODEL ?= densenet169
WEIGHTS ?= imagenet
EPOCHS ?= 10
BATCH_SIZE ?= 32
LR ?= 0.001
DATA_DIR ?=
CHECKPOINT ?=

.PHONY: help sync lint format test check regression clustering dl-train dl-eval clean

help:
	@echo "Available targets:"
	@echo "  make sync          Install/sync project dependencies using uv"
	@echo "  make lint          Run ruff linter"
	@echo "  make format        Format code using ruff"
	@echo "  make test          Run pytest test suite"
	@echo "  make check         Run lint and test together"
	@echo "  make regression    Run Linear Regression on Wine Quality"
	@echo "  make clustering    Run Mean Shift Clustering on Wine Quality"
	@echo "  make dl-train      Train Deep Learning model (MODEL=$(MODEL), WEIGHTS=$(WEIGHTS))"
	@echo "  make dl-eval       Evaluate a saved Deep Learning checkpoint (CHECKPOINT=...)"
	@echo "  make clean         Remove temporary caches"

sync:
	$(UV) sync

lint:
	$(UV) run ruff check src tests

format:
	$(UV) run ruff format src tests

test:
	$(UV) run pytest

check: lint test

regression:
	$(UV) run ml-project linear-regression --no-show-plots

clustering:
	$(UV) run ml-project mean-shift --no-show-plots

dl-train:
	$(UV) run ml-project dl train \
		--model $(MODEL) \
		--weights $(WEIGHTS) \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--lr $(LR) \
		$(if $(DATA_DIR),--data-dir $(DATA_DIR),)

dl-eval:
	@test -n "$(CHECKPOINT)" || (echo "Error: CHECKPOINT is required. Example: make dl-eval CHECKPOINT=outputs/deep_learning/.../best_model.pth" && exit 1)
	$(UV) run ml-project dl evaluate \
		--checkpoint $(CHECKPOINT) \
		$(if $(DATA_DIR),--data-dir $(DATA_DIR),)

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache dist build
	find src tests -type d -name "__pycache__" -exec rm -rf {} +
