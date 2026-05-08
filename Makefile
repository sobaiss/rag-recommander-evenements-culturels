.PHONY: help index lint lint-fix run api test clean install reset

# Load environment variables from .env file
ifneq (,$(wildcard .env))
    include .env
    export $(shell sed 's/=.*//' .env)
endif

help:
	@echo "Available commands:"
	@echo "  make index       - Run the indexer to process documents"
	@echo "                     Usage:"
	@echo "                       make index                                    # Use defaults"
	@echo "                       make index input-file=/path/to/file           # Custom input file"
	@echo "                       make index data-url=https://example.com/data   # Custom data URL"
	@echo "  make chat        - Run the Streamlit chat UI"
	@echo "  make api         - Run the FastAPI REST server (http://localhost:8000)"
	@echo "  make test        - Run functional tests"
	@echo "  make lint        - Run Ruff linter on the project"
	@echo "  make lint-fix    - Auto-fix linting issues"
	@echo "  make install     - Install project dependencies"
	@echo "  make clean       - Remove cache files and artifacts"
	@echo "  make reset       - Reset the application state (asks for confirmation)"

index:
	uv run indexer.py $(if $(input-file),--input-file $(input-file)) $(if $(data-url),--data-url $(data-url))
# 	make index data-url="https://public.opendatasoft.com/api/records/1.0/search/?rows=40&disjunctive.keywords_fr=true&disjunctive.location_region=true&disjunctive.location_countrycode=true&disjunctive.location_department=true&disjunctive.location_city=true&refine.location_region=%C3%8Ele-de-France&refine.firstdate_begin=2025%2F04&start=0&dataset=evenements-publics-openagenda&timezone=Europe%2FBerlin&lang=fr"

chat:
	uv run streamlit run Chat.py

feedback:
	uv run streamlit run pages/Feedback_Viewer.py

run:
	uv run main.py

api:
	uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload

test:
	uv run pytest tests/ -v

lint:
	ruff check .

lint-fix:
	ruff check --fix .

install:
	uv sync

reset:
	@read -p "Are you sure you want to reset? [y/N] " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		uv run reset.py; \
	else \
		echo "Reset cancelled."; \
	fi

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "Cleanup complete"
