DATA_DIR:=datasets
DOWNLOAD_OREGANO:=true
DOWNLOAD_PRIME_KG:=true


.PHONY: download_datasets
download_datasets:
	@echo "Downloading datasets"
	@if [ "$(DOWNLOAD_OREGANO)" = "true" ]; then \
		./scripts/run_download_oregano.sh "$(DATA_DIR)/oregano"; \
	fi
	@if [ "$(DOWNLOAD_PRIME_KG)" = "true" ]; then \
		./scripts/run_download_prime_kg.sh "$(DATA_DIR)/prime_kg"; \
	fi
	
.PHONY: init
init: download_datasets
	@echo "Installing package"
	@uv sync; source .venv/bin/activate; pre-commit install