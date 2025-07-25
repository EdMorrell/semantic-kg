DATA_DIR:=datasets
datasets := $(wildcard datasets/*/)


.PHONY: download_dataset
download_dataset:
	@echo "Downloading dataset $(DATASET_NAME)"
	@./scripts/run_download_$(DATASET_NAME).sh "$(DATA_DIR)/$(DATASET_NAME)"

# .PHONY: download_all_datasets
# download_datasets:
# 	@echo "Downloading all datasets. This may take a while..."
# 	@$(foreach dataset,$(datasets),$(MAKE) download_dataset DATASET_NAME=$(notdir $(patsubst %/,%,$(dataset)));)

.PHONY: init
init: download_datasets
	@echo "Installing package"
	@uv sync; source .venv/bin/activate; pre-commit install
